#!/usr/bin/env python3
"""Paired comparison of the exact-Decision-Maker RebaseKV run (E) against the paper's per-question Recompute and
RebaseKV (all-agents optimistic) results on the same GSM8K questions.
Usage: python exact_decision_report.py <package_root> <hybrid_exact4_dir> [same_box_recompute_dir]"""
import sys, os, json, glob, random, statistics as st
random.seed(0)
PKG, EX = sys.argv[1], sys.argv[2]; RECB = sys.argv[3] if len(sys.argv) > 3 else None

def load(pat):
    by = {}
    for p in glob.glob(pat):
        for l in open(p):
            l = l.strip()
            if l:
                d = json.loads(l); by[d["qi"]] = d
    return by
R = load(os.path.join(PKG, "results/gsm8k_full/recompute/shard*.jsonl"))
H = load(os.path.join(PKG, "results/gsm8k_full/hybrid/shard*.jsonl"))
E = load(os.path.join(EX, "shard*.jsonl"))
RB = load(os.path.join(RECB, "shard*.jsonl")) if RECB else {}
qs = sorted(set(R) & set(H) & set(E)); n = len(qs)
avg4 = lambda d: sum(d["sample_ok"]) / len(d["sample_ok"]); greedy = lambda d: float(d["sample_ok"][0])
def ci(diffs, nb=10000):
    bs = sorted(st.mean(random.choice(diffs) for _ in range(len(diffs))) for _ in range(nb)); return st.mean(diffs), bs[int(.025*nb)], bs[int(.975*nb)]
print(f"paired GSM8K questions: {n}  (exact agents in E: {E[qs[0]].get('exact_agents')})\n")
print("| method (same questions) | mean-of-4 | greedy | Δ mean-of-4 vs Recompute (95% CI) | Δ greedy vs Recompute (95% CI) | Decision ngen median | Decision cap-hit (>=400) |")
print("|---|---|---|---|---|---|---|")
rows = [("Recompute (paper run, A100)", R), ("RebaseKV, all agents optimistic (paper run, A100)", H), ("RebaseKV, Decision Maker exact (new run)", E)]
if RB: rows.append(("Recompute (same box as new run)", RB)); 
for name, D in rows:
    sub = [q for q in qs if q in D]
    a = 100 * st.mean(avg4(D[q]) for q in sub); g = 100 * st.mean(greedy(D[q]) for q in sub)
    if D is R: da = dg = "—"
    else:
        m, lo, hi = ci([avg4(D[q]) - avg4(R[q]) for q in sub]); da = f"{100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]"
        m, lo, hi = ci([greedy(D[q]) - greedy(R[q]) for q in sub]); dg = f"{100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]"
    dn = st.median(D[q]["agent_ngen"][4] for q in sub); cap = 100 * st.mean(D[q]["agent_ngen"][4] >= 400 for q in sub)
    print(f"| {name} | {a:.1f} | {g:.1f} | {da} | {dg} | {dn:.0f} | {cap:.1f}% |")
m, lo, hi = ci([avg4(E[q]) - avg4(H[q]) for q in qs]); print(f"\nΔ mean-of-4, exact-Decision minus all-optimistic RebaseKV: {100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
m, lo, hi = ci([greedy(E[q]) - greedy(H[q]) for q in qs]); print(f"Δ greedy, exact-Decision minus all-optimistic RebaseKV: {100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
print(f"code-block rate (Programming Expert): Recompute {100*st.mean(float(R[q]['a3_code']) for q in qs):.1f} | RebaseKV-all {100*st.mean(float(H[q]['a3_code']) for q in qs):.1f} | RebaseKV-exactDecision {100*st.mean(float(E[q]['a3_code']) for q in qs):.1f}")
print(f"TTFT medians (E run, this box): per agent {[round(st.median(E[q]['agent_ttft'][a] for q in qs),1) for a in range(5)]}; K mean {st.mean(E[q]['K'] for q in qs):.2f}; t_dec {st.median(E[q]['t_dec'] for q in qs):.1f}")
