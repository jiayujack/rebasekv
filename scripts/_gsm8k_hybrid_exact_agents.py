#!/usr/bin/env python3
"""GSM8K full run (1319 q, 4 samples) for RebaseKV with an EXACT-PATH policy for selected agents.
Same protocol and output format as scripts/_gsm8k_full_methods.py (method name 'hybrid_exact<agents>'),
so results/gsm8k_full/recompute can be reused as the paired baseline and rebuttal_stats.py logic applies.

Motivation (rebuttal): under RebaseKV the answer-first Decision Maker (agent 4) inherits the Inspector's
verbose style for its first K tokens; 6.4% of pipelines hit the 400-token cap vs 0.2% under recompute.
MMLU/HumanEval pipelines already recompute the final agent; this run applies the same policy to GSM8K.

Env: EXACT_AGENTS (default "4"; "0,4" also puts the first agent on the exact path, matching the paper's
     'every agent after the first' description), GSM8K_N, SAMPLES, MAX_NEW, GSM8K_OUTDIR, REBASE_PKG
Usage: python _gsm8k_hybrid_exact_agents.py <shard> <nshard>
Place in <package>/scripts/.
"""
import os, sys, time, json, re, torch
PKG = os.environ.get("REBASE_PKG", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PKG)
import scripts.spec_rebase_bench as B
from transformers import AutoModelForCausalLM, AutoTokenizer

SHARD = int(sys.argv[1]); NSH = int(sys.argv[2])
EXACT = {int(x) for x in os.environ.get("EXACT_AGENTS", "4").split(",") if x.strip()}
METHOD = "hybrid_exact" + "-".join(map(str, sorted(EXACT)))
N = int(os.environ.get("GSM8K_N", "1319")); SAMPLES = int(os.environ.get("SAMPLES", "4"))
MAXNEW = int(os.environ.get("MAX_NEW", "400"))
_BASE = os.environ.get("GSM8K_OUTDIR", os.path.join(PKG, "runs", "gsm8k_full"))
OUTD = f"{_BASE}/{METHOD}"; os.makedirs(OUTD, exist_ok=True); OUT = f"{OUTD}/shard{SHARD}.jsonl"
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(B.MODEL)
model = AutoModelForCausalLM.from_pretrained(B.MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
t_dec = B.calibrate_dec_step(model, tok)
roles = B.roles_for("gsm8k"); NA = B.N_AGENTS
items = B.load_data("gsm8k", N)
print(f"[gsm8k_full/{METHOD}] N={len(items)} shard={SHARD}/{NSH} samples={SAMPLES} t_dec={t_dec:.1f}ms exact_agents={sorted(EXACT)}", flush=True)

def do_q(qi, item):
    q = item["q"]
    sample_ok = []; agent_ttft = [0.0] * NA; agent_ngen = [0] * NA
    sum_prefill = 0.0; sum_rebase = 0.0; sum_exact = 0.0; Ks = []; a3_code = False
    for si in range(SAMPLES):
        temp = 0.0 if si == 0 else 0.7; timed = (si == 0)
        peers = []; proles = []; texts = []
        for ai in range(NA):
            rf = roles[ai]; rs = roles[(ai - 1) % NA]
            if ai in EXACT:
                ids = B.build_ids("gsm8k", tok, rf, q, peers, proles, qi)
                text, ttft, info = B.gen_recompute(model, tok, ids, MAXNEW, temp, timed)
                if timed: sum_exact += info["t_prefill"]; sum_prefill += info["t_prefill"]
            else:
                text, ttft, info = B.run_agent("gsm8k", model, tok, rf, rs, q, peers, proles, qi, "hybrid", t_dec, MAXNEW, temp, timed)
                if timed: sum_prefill += info["t_prefill"]; sum_rebase += info["t_rebase"]; Ks.append(info["K"])
            peers.append(text); proles.append(rf); texts.append(text)
            if timed: agent_ttft[ai] = ttft; agent_ngen[ai] = info["n_gen"]
        if si == 0 and len(texts) > 2:
            a3_code = bool("```" in texts[2] or re.search(r"\bdef \w+\s*\(", texts[2]))
        pred = B.extract_answer("gsm8k", texts[-1])
        sample_ok.append(bool(B.is_correct("gsm8k", item, pred, texts[-1])))
    return dict(qi=qi, method=METHOD, sample_ok=sample_ok, agent_ttft=agent_ttft, agent_ngen=agent_ngen,
                sum_prefill=sum_prefill, sum_rebase=sum_rebase, sum_exact_prefill=sum_exact,
                K=(sum(Ks) / len(Ks) if Ks else 0), t_dec=t_dec, a3_code=a3_code, exact_agents=sorted(EXACT))

def main():
    idxs = [i for i in range(len(items)) if i % NSH == SHARD]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: done.add(json.loads(l)["qi"])
            except Exception: pass
    f = open(OUT, "a"); t0 = time.time()
    for k, qi in enumerate(idxs):
        if qi in done: continue
        try:
            r = do_q(qi, items[qi]); f.write(json.dumps(r) + "\n"); f.flush()
            if (k + 1) % 10 == 0:
                print(f"[{METHOD} s{SHARD}] {k+1}/{len(idxs)} ({time.time()-t0:.0f}s) qi={qi} ok={r['sample_ok']} dec_ngen={r['agent_ngen'][-1]}", flush=True)
        except Exception as e:
            import traceback; print(f"[{METHOD} s{SHARD}] q{qi} ERR {e}", flush=True); traceback.print_exc()
    print(f"[{METHOD} s{SHARD}] DONE", flush=True)

if __name__ == "__main__":
    main()
