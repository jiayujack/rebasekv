#!/usr/bin/env python3
"""Full-GSM8K (1319), 4-sample, sharded runner for ONE faithful method (recompute|stale|hybrid),
reusing spec_rebase_bench (B) prompts/agents/decoders (Qwen2.5-7B-Instruct by default, 5 agents, KVCOMM prompts).
Per question: 4 samples (si=0 greedy+timed, si>0 temp=0.7). Records per-agent TTFT/ngen and
prefill/rebase/K on the timed sample; sample_ok for all 4. Output jsonl per shard; aggregate later.
Usage: _gsm8k_full_methods.py <method> <shard> <nshard>"""
import os, sys, time, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.spec_rebase_bench as B
from transformers import AutoModelForCausalLM, AutoTokenizer

METHOD = sys.argv[1]; SHARD = int(sys.argv[2]); NSH = int(sys.argv[3])
N = int(os.environ.get("GSM8K_N", "1319")); SAMPLES = int(os.environ.get("SAMPLES", "4"))
MAXNEW = int(os.environ.get("MAX_NEW", "400"))
_BASE = os.environ.get("GSM8K_OUTDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "gsm8k_full"))
OUTD = f"{_BASE}/{METHOD}"; os.makedirs(OUTD, exist_ok=True)
OUT = f"{OUTD}/shard{SHARD}.jsonl"
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(B.MODEL)
model = AutoModelForCausalLM.from_pretrained(B.MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
t_dec = B.calibrate_dec_step(model, tok)
roles = B.roles_for("gsm8k"); NA = B.N_AGENTS
items = B.load_data("gsm8k", N)
print(f"[gsm8k_full/{METHOD}] N={len(items)} shard={SHARD}/{NSH} samples={SAMPLES} t_dec={t_dec:.1f}ms", flush=True)

def do_q(qi, item):
    q = item["q"]
    sample_ok = []; agent_ttft = [0.0]*NA; agent_ngen = [0]*NA
    sum_prefill = 0.0; sum_rebase = 0.0; Ks = []; a3_code = False
    for si in range(SAMPLES):
        temp = 0.0 if si == 0 else 0.7; timed = (si == 0)
        peers = []; proles = []; texts = []
        for ai in range(NA):
            rf = roles[ai]; rs = roles[(ai-1) % NA]
            text, ttft, info = B.run_agent("gsm8k", model, tok, rf, rs, q, peers, proles, qi,
                                           METHOD, t_dec, MAXNEW, temp, timed)
            peers.append(text); proles.append(rf); texts.append(text)
            if timed:
                agent_ttft[ai] = ttft; agent_ngen[ai] = info["n_gen"]
                if METHOD == "recompute": sum_prefill += info["t_prefill"]
                elif METHOD == "hybrid":
                    sum_prefill += info["t_prefill"]; sum_rebase += info["t_rebase"]; Ks.append(info["K"])
        # Agent-3 code probe (si==0 only for consistency)
        if si == 0 and len(texts) > 2:
            import re
            a3_code = bool("```" in texts[2] or re.search(r"\bdef \w+\s*\(", texts[2]))
        pred = B.extract_answer("gsm8k", texts[-1])
        sample_ok.append(bool(B.is_correct("gsm8k", item, pred, texts[-1])))
    return dict(qi=qi, method=METHOD, sample_ok=sample_ok, agent_ttft=agent_ttft, agent_ngen=agent_ngen,
                sum_prefill=sum_prefill, sum_rebase=sum_rebase, K=(sum(Ks)/len(Ks) if Ks else 0),
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
                print(f"[{METHOD} s{SHARD}] {k+1}/{len(idxs)} ({time.time()-t0:.0f}s) qi={qi} ok={r['sample_ok']}", flush=True)
        except Exception as e:
            import traceback; print(f"[{METHOD} s{SHARD}] q{qi} ERR {e}", flush=True); traceback.print_exc()
    print(f"[{METHOD} s{SHARD}] DONE", flush=True)

if __name__ == "__main__":
    main()
