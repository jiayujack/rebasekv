#!/usr/bin/env python3
"""Summarise RelayCaching-harness runs (their result JSON: list of records with 'Solved' and 'Agent traces').
Usage: python relaycaching_report.py <runs_dir> [dir1 dir2 ...]   (dirs default: gsm8k_full_greedy gsm8k_relay_greedy gsm8k_zero_greedy)
TTFT is their per-agent 'ttft' (seconds, wall time inside generate() until the first new token; for kvcomm mode
their harness adds preprocess latency).  Agent 1 has no upstream output to reuse, so the reuse-eligible agents are
2..N (compare with our 'rebased agents' medians)."""
import sys, os, json, glob, statistics as st
root = sys.argv[1]; dirs = sys.argv[2:] or ["gsm8k_full_greedy", "gsm8k_relay_greedy", "gsm8k_zero_greedy"]
label = {"gsm8k_full_greedy": "Full prefill (their FULL baseline)", "gsm8k_relay_greedy": "RelayCaching", "gsm8k_zero_greedy": "Direct reuse, RoPE-corrected (their ZERO)"}
q = lambda v, p: sorted(v)[int(p * len(v))]
rows = {}
for d in dirs:
    js = [p for p in glob.glob(os.path.join(root, d, "*.json")) if not p.endswith("Latency.json")]
    if not js: print(f"{d}: no result json"); continue
    recs = json.load(open(js[0])); n = len(recs)
    acc = 100 * sum(bool(r["Solved"]) for r in recs) / n
    NA = max(len(r["Agent traces"]) for r in recs)
    per = [[1000 * t["ttft"] for r in recs for t in [r["Agent traces"][a]] if a < len(r["Agent traces"])] for a in range(NA)]
    reuse = [x for r in recs for t in r["Agent traces"][1:] for x in [1000 * t["ttft"]]]
    final = [1000 * r["Agent traces"][-1]["ttft"] for r in recs]
    rows[d] = dict(n=n, acc=acc, per=[st.median(v) for v in per], reuse_med=st.median(reuse), reuse_p90=q(reuse, .9), final_med=st.median(final))
    print(f"{label.get(d,d)}: n={n} accuracy={acc:.1f}% | per-agent TTFT median (ms) {[round(x) for x in rows[d]['per']]} | agents 2..{NA} median {rows[d]['reuse_med']:.0f} p90 {rows[d]['reuse_p90']:.0f} | final-agent median {rows[d]['final_med']:.0f}")
if "gsm8k_full_greedy" in rows:
    f = rows["gsm8k_full_greedy"]
    print("\n| method (RelayCaching harness, Qwen2.5-7B, 5 agents, 200 GSM8K q, greedy, A6000) | accuracy | TTFT agents 2-5 (median) | speedup vs full |")
    print("|---|---|---|---|")
    for d, r in rows.items():
        print(f"| {label.get(d,d)} | {r['acc']:.1f} | {r['reuse_med']:.0f} ms (p90 {r['reuse_p90']:.0f}) | {f['reuse_med']/r['reuse_med']:.2f}x |")
