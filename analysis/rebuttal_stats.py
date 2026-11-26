#!/usr/bin/env python3
"""Reproduce every number quoted in the rebuttal from the raw per-question outputs in the
repository.  Usage (from the repository root):
    python analysis/rebuttal_stats.py .
Reads:  results/gsm8k_full/{recompute,stale,hybrid}/shard*.jsonl   (Qwen2.5-7B, 1319 q, 4 samples)
        results/gsm8k_kablation/K*/shard*.jsonl                     (forced-K ablation)
        results/warmupkv/gsm8k/shard*.jsonl                          (real two-stream WarmupKV)
        scripts/kvcomm_fresh/stale4/{humaneval,mmlu}/shard*.jsonl    (stale, 4 samples)
        scripts/kvcomm_fresh/he_agent_sweep/N5/{hybrid,recompute}.json
"""
import sys, os, json, glob, random, statistics as st
from math import comb

PKG = sys.argv[1] if len(sys.argv) > 1 else "."
random.seed(0)

def load(pat):
    by = {}
    for p in glob.glob(os.path.join(PKG, pat)):
        for l in open(p):
            l = l.strip()
            if l:
                d = json.loads(l); by[d["qi"]] = d
    return by

def pct(v): return 100 * st.mean(v)
def se(v): return 100 * st.pstdev(v) / len(v) ** 0.5
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]

def paired_ci(diffs, nb=10000):
    n = len(diffs); bs = []
    for _ in range(nb):
        bs.append(st.mean(diffs[random.randrange(n)] for _ in range(n)))
    bs.sort()
    return st.mean(diffs), bs[int(0.025 * nb)], bs[int(0.975 * nb)]

def mcnemar(win, lose):
    n = win + lose; k = min(win, lose)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

R = load("results/gsm8k_full/recompute/shard*.jsonl")
S = load("results/gsm8k_full/stale/shard*.jsonl")
H = load("results/gsm8k_full/hybrid/shard*.jsonl")
common = sorted(set(R) & set(S) & set(H))
print(f"GSM8K paired questions: {len(common)}  (recompute {len(R)}, stale {len(S)}, rebasekv {len(H)})")

metrics = {
    "greedy (timed sample)": lambda d: float(d["sample_ok"][0]),
    "mean of 3 @T=0.7":      lambda d: sum(d["sample_ok"][1:]) / 3,
    "mean of 4 (Table 1)":   lambda d: sum(d["sample_ok"]) / 4,
    "any of 4 (true pass@4)": lambda d: float(any(d["sample_ok"])),
}
print("\n== accuracy (%, ±SE) ==")
print(f"{'metric':26s} {'Recompute':>14s} {'Stale':>14s} {'RebaseKV':>14s}")
for name, fn in metrics.items():
    row = []
    for D in (R, S, H):
        v = [fn(D[qq]) for qq in common]; row.append(f"{pct(v):5.1f} ±{se(v):.1f}")
    print(f"{name:26s} " + " ".join(f"{x:>14s}" for x in row))

print("\n== paired differences vs Recompute (pp, 95% bootstrap CI) ==")
for name in ("mean of 4 (Table 1)", "greedy (timed sample)"):
    fn = metrics[name]
    for lab, D in (("RebaseKV", H), ("Stale", S)):
        diffs = [fn(D[qq]) - fn(R[qq]) for qq in common]
        m, lo, hi = paired_ci(diffs)
        extra = ""
        if name.startswith("greedy"):
            win = sum(d > 0 for d in diffs); lose = sum(d < 0 for d in diffs)
            extra = f"  discordant: {lab}-only={win}, Recompute-only={lose}, McNemar p={mcnemar(win, lose):.2g}"
        print(f"{lab:9s} {name:22s} {100*m:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]{extra}")

print("\n== TTFT (ms) over all per-agent greedy measurements ==")
for lab, D in (("Recompute", R), ("Stale", S), ("RebaseKV", H)):
    allv = [x for qq in common for x in D[qq]["agent_ttft"]]
    per = [st.median(D[qq]["agent_ttft"][a] for qq in common) for a in range(5)]
    print(f"{lab:9s} median={st.median(allv):5.1f} p90={q(allv,.9):5.1f} p99={q(allv,.99):5.1f} per-agent median={[round(x,1) for x in per]}")

print("\n== RebaseKV window / replay (per agent, ms) ==")
rep = [H[qq]["sum_rebase"] / 5 for qq in common]
pf = [H[qq]["sum_prefill"] / 5 for qq in common]
win = [H[qq]["K"] * H[qq]["t_dec"] for qq in common]
print(f"replay median={st.median(rep):.1f} p90={q(rep,.9):.1f} (= {st.median(r/H[qq]['t_dec'] for r,qq in zip(rep,common)):.2f} decode steps)")
print(f"exact prefill median={st.median(pf):.1f}; draft window K*t_dec median={st.median(win):.1f}; window/prefill median={st.median(w/p for w,p in zip(win,pf)):.2f}; K mean={st.mean(H[qq]['K'] for qq in common):.2f}")
tdec = 26.2
print(f"per-pipeline prefill region: recompute {st.median(R[qq]['sum_prefill'] for qq in common):.0f} ms vs RebaseKV replay {st.median(H[qq]['sum_rebase'] for qq in common):.0f} ms; "
      f"decode ~ {st.median(sum(R[qq]['agent_ngen']) for qq in common):.0f} tok x {tdec} ms = {st.median(sum(R[qq]['agent_ngen']) for qq in common)*tdec/1000:.1f} s")

print("\n== per-agent output length (greedy): median / p90 / cap-hit(>=400) ==")
names = ["MathSolver", "Analyst", "ProgExpert", "Inspector", "Decision"]
for a in range(5):
    row = []
    for lab, D in (("Rec", R), ("Stale", S), ("Reb", H)):
        v = [D[qq]["agent_ngen"][a] for qq in common]
        row.append(f"{lab}: {st.median(v):4.0f}/{q(v,.9):4.0f}/{100*st.mean(x>=400 for x in v):4.1f}%")
    print(f"  {names[a]:11s} " + " | ".join(row))
g = lambda D, qq: D[qq]["sample_ok"][0]
cap = [qq for qq in common if H[qq]["agent_ngen"][4] >= 400]
nocap = [qq for qq in common if H[qq]["agent_ngen"][4] < 400 and R[qq]["agent_ngen"][4] < 400]
ro = [qq for qq in common if g(R, qq) and not g(H, qq)]
print(f"RebaseKV decision cap-hits: n={len(cap)} ({100*len(cap)/len(common):.1f}%), greedy acc on them={100*st.mean(g(H,qq) for qq in cap):.1f}")
print(f"Recompute-only-correct (greedy): n={len(ro)}, of which RebaseKV decision cap-hit={100*st.mean(H[qq]['agent_ngen'][4]>=400 for qq in ro):.0f}%")
print(f"excluding decision cap-hits (n={len(nocap)}): greedy Rec={100*st.mean(g(R,qq) for qq in nocap):.1f} Reb={100*st.mean(g(H,qq) for qq in nocap):.1f}; "
      f"mean-of-4 Rec={100*st.mean(sum(R[qq]['sample_ok'])/4 for qq in nocap):.1f} Reb={100*st.mean(sum(H[qq]['sample_ok'])/4 for qq in nocap):.1f}")
print("Programming-Expert code-block rate: " + ", ".join(f"{lab}={100*st.mean(float(D[qq]['a3_code']) for qq in common):.1f}%" for lab, D in (("Rec", R), ("Stale", S), ("Reb", H))))

print("\n== forced-K ablation (same-subset Recompute for reference) ==")
for K in (3, 5, 10, 20, 50):
    D = load(f"results/gsm8k_kablation/K{K}/shard*.jsonl"); qs = sorted(qq for qq in D if qq in R)
    if not qs: continue
    print(f"K={K:2d} n={len(qs):3d} mean-of-4={100*st.mean(sum(D[qq]['sample_ok'])/4 for qq in qs):.1f} greedy={100*st.mean(float(D[qq]['sample_ok'][0]) for qq in qs):.1f} "
          f"code={100*st.mean(float(D[qq]['a3_code']) for qq in qs):.1f} | Recompute same subset mean-of-4={100*st.mean(sum(R[qq]['sample_ok'])/4 for qq in qs):.1f} greedy={100*st.mean(float(R[qq]['sample_ok'][0]) for qq in qs):.1f}")

W = load("results/warmupkv/gsm8k/shard*.jsonl")
if W:
    qs = sorted(W)
    rt = [x for qq in qs for x in W[qq]["rec_ttft"]]; pt = [x for qq in qs for x in W[qq]["pipe_ttft"]]
    print(f"\n== WarmupKV real two-stream (n={len(qs)}) ==")
    print(f"per-agent TTFT median: recompute {st.median(rt):.1f} -> warmupkv {st.median(pt):.1f} ms")
    print(f"e2e median: recompute {st.median(W[qq]['rec_e2e'] for qq in qs)/1000:.2f} s, warmupkv {st.median(W[qq]['pipe_e2e'] for qq in qs)/1000:.2f} s, paired median ratio {st.median(W[qq]['pipe_e2e']/W[qq]['rec_e2e'] for qq in qs):.4f}")
    print(f"greedy text identical to recompute: {100*st.mean(float(W[qq]['exact_match']) for qq in qs):.1f}% (bf16 split-forward nondeterminism)")

print("\n== other datasets (as present in the package) ==")
for ds in ("humaneval", "mmlu"):
    D = load(f"scripts/kvcomm_fresh/stale4/{ds}/shard*.jsonl"); qs = sorted(D)
    if qs:
        print(f"stale4/{ds}: n={len(qs)} mean-of-4={100*st.mean(sum(D[qq]['sample_ok'])/4 for qq in qs):.1f} greedy={100*st.mean(float(D[qq]['sample_ok'][0]) for qq in qs):.1f}")
for m in ("recompute", "hybrid"):
    p = os.path.join(PKG, f"scripts/kvcomm_fresh/he_agent_sweep/N5/{m}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        print(f"humaneval N5 {m}: mean-of-4={100*d['pass']:.1f} code-block rate={100*d['a3_code_rate']:.1f} ttft_med={d['ttft_med']:.1f} per-agent ttft={[round(x) for x in d['per_agent_ttft_med']]} per-agent ngen={d['per_agent_ngen_med']} K={d['K_mean']:.2f}")
