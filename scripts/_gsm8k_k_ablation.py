#!/usr/bin/env python3
"""GSM8K K-ablation: run hybrid with FORCED K (not adaptive). 
Reuses spec_rebase_bench but patches gen_hybrid to use a fixed K.
Usage: _gsm8k_k_ablation.py <K> <shard> <nshard>
Env: GSM8K_N (default 1319), SAMPLES (default 4), GSM8K_OUTDIR"""
import os, sys, time, json, re, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.spec_rebase_bench as B
from transformers import AutoModelForCausalLM, AutoTokenizer

FORCED_K = int(sys.argv[1]); SHARD = int(sys.argv[2]); NSH = int(sys.argv[3])
N = int(os.environ.get("GSM8K_N", "1319")); SAMPLES = int(os.environ.get("SAMPLES", "4"))
MAXNEW = int(os.environ.get("MAX_NEW", "400"))
_BASE = os.environ.get("GSM8K_OUTDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "gsm8k_kablation"))
OUTD = f"{_BASE}/K{FORCED_K}"; os.makedirs(OUTD, exist_ok=True)
OUT = f"{OUTD}/shard{SHARD}.jsonl"
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(B.MODEL)
model = AutoModelForCausalLM.from_pretrained(B.MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
t_dec = B.calibrate_dec_step(model, tok)
roles = B.roles_for("gsm8k"); NA = B.N_AGENTS
items = B.load_data("gsm8k", N)
print(f"[K-ablation K={FORCED_K}] shard={SHARD}/{NSH} N={len(items)} t_dec={t_dec:.1f}ms", flush=True)

# Monkey-patch gen_hybrid to use FORCED_K instead of adaptive
_orig_gen_hybrid = B.gen_hybrid
def gen_hybrid_forced(model, tok, ids_fresh, stale_kv, last_tok, t_dec_step, max_new, temp, timed):
    # Override: always use FORCED_K
    import time as _time
    B.sync(); tp = _time.perf_counter()
    with torch.no_grad():
        fresh_kv = model(ids_fresh, use_cache=True).past_key_values
    B.sync(); t_prefill = (_time.perf_counter() - tp) * 1000
    K = FORCED_K
    B.sync(); t0 = _time.perf_counter()
    cur = last_tok; k = stale_kv; draft = []; ttft = None
    for i in range(K):
        with torch.no_grad():
            out = model(cur, past_key_values=k, use_cache=True)
        nt = B._sample(out.logits, temp)
        if ttft is None:
            B.sync(); ttft = (_time.perf_counter() - t0) * 1000
        tid = int(nt.item()); draft.append(tid); cur = nt; k = out.past_key_values
        if tid == tok.eos_token_id:
            info = {"t_prefill": t_prefill, "K": K, "t_rebase": 0.0, "n_gen": len(draft)} if timed else None
            return tok.decode(draft, skip_special_tokens=True), ttft, info
    rin = torch.tensor([draft], device="cuda")
    B.sync(); tr = _time.perf_counter()
    with torch.no_grad():
        out = model(rin, past_key_values=fresh_kv, use_cache=True)
    B.sync(); t_rebase = (_time.perf_counter() - tr) * 1000
    k = out.past_key_values; cur = rin[:, -1:]; gen = list(draft)
    if B.EXACT_REPLAY:
        cur = B._sample(out.logits, temp); gen.append(int(cur.item()))
        if gen[-1] == tok.eos_token_id:
            info = {"t_prefill": t_prefill, "K": K, "t_rebase": t_rebase, "n_gen": len(gen)} if timed else None
            return tok.decode(gen, skip_special_tokens=True), ttft, info
    for _ in range(max_new - len(gen)):
        with torch.no_grad():
            out = model(cur, past_key_values=k, use_cache=True)
        nt = B._sample(out.logits, temp); tid = int(nt.item())
        gen.append(tid); cur = nt; k = out.past_key_values
        if tid == tok.eos_token_id:
            break
    info = {"t_prefill": t_prefill, "K": K, "t_rebase": t_rebase, "n_gen": len(gen)} if timed else None
    return tok.decode(gen, skip_special_tokens=True), ttft, info
B.gen_hybrid = gen_hybrid_forced

def do_q(qi, item):
    q = item["q"]
    sample_ok = []; agent_ttft = [0.0]*NA; agent_ngen = [0]*NA
    sum_prefill = 0.0; sum_rebase = 0.0; a3_code = False
    for si in range(SAMPLES):
        temp = 0.0 if si == 0 else 0.7; timed = (si == 0)
        peers = []; proles = []; texts = []
        for ai in range(NA):
            rf = roles[ai]; rs = roles[(ai-1) % NA]
            text, ttft, info = B.run_agent("gsm8k", model, tok, rf, rs, q, peers, proles, qi,
                                           "hybrid", t_dec, MAXNEW, temp, timed)
            peers.append(text); proles.append(rf); texts.append(text)
            if timed:
                agent_ttft[ai] = ttft; agent_ngen[ai] = info["n_gen"]
                sum_prefill += info["t_prefill"]; sum_rebase += info["t_rebase"]
        if si == 0 and len(texts) > 2:
            a3_code = bool("```" in texts[2] or re.search(r"\bdef \w+\s*\(", texts[2]))
        pred = B.extract_answer("gsm8k", texts[-1])
        sample_ok.append(bool(B.is_correct("gsm8k", item, pred, texts[-1])))
    return dict(qi=qi, method=f"hybrid_K{FORCED_K}", sample_ok=sample_ok,
                agent_ttft=agent_ttft, agent_ngen=agent_ngen,
                sum_prefill=sum_prefill, sum_rebase=sum_rebase, K=FORCED_K,
                t_dec=t_dec, a3_code=a3_code)

def main():
    idxs = [i for i in range(len(items)) if i % NSH == SHARD]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: done.add(json.loads(l)["qi"])
            except: pass
    f = open(OUT, "a"); t0 = time.time()
    for k, qi in enumerate(idxs):
        if qi in done: continue
        try:
            r = do_q(qi, items[qi]); f.write(json.dumps(r)+"\n"); f.flush()
            if (k+1) % 10 == 0:
                acc = sum(sum(json.loads(l)["sample_ok"])/4 for l in open(OUT) if l.strip()) / (k+1)
                print(f"[K={FORCED_K} s{SHARD}] {k+1}/{len(idxs)} ({time.time()-t0:.0f}s) acc={acc*100:.1f}%", flush=True)
        except Exception as e:
            import traceback; print(f"[K={FORCED_K} s{SHARD}] q{qi} ERR {e}", flush=True); traceback.print_exc()
    print(f"[K={FORCED_K} s{SHARD}] DONE", flush=True)

if __name__ == "__main__":
    main()
