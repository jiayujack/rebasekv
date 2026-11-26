#!/usr/bin/env python3
"""Unified aggregator for all paper tables. Reads JSONL shard files from result dirs.
Usage: _aggregate_all.py <results_base_dir>
Expected structure:
  <base>/gsm8k/{recompute,stale,hybrid}/shard*.jsonl
  <base>/gsm8k_warmupkv/shard*.jsonl  
  <base>/gsm8k_kablation/K{5,10,20,50}/shard*.jsonl
  <base>/mmlu/{recompute,stale,hybrid}/shard*.jsonl
  <base>/mmlu_warmupkv/shard*.jsonl
"""
import glob, json, statistics as st, sys, os

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def load_dedup(pattern):
    byqi = {}
    for p in glob.glob(pattern):
        for l in open(p):
            l = l.strip()
            if l:
                d = json.loads(l); byqi[d["qi"]] = d
    return list(byqi.values())

def acc_avg4(rows):
    return sum(sum(r["sample_ok"])/len(r["sample_ok"]) for r in rows) / len(rows)

def ttft_med(rows):
    return st.median([x for r in rows for x in r["agent_ttft"]])

def a3_code_rate(rows):
    return sum(1 for r in rows if r.get("a3_code", False)) / len(rows)

print("\n" + "="*90)
print("TABLE 1: Main Results (tab:gsm8k_main)")
print("="*90)
print(f"{'Dataset':<10} {'Method':<11} {'Acc%':>6} {'TTFT':>7} {'K':>5} {'saved/pipe':>11} {'saved/%':>8} {'a3_code':>8} {'N':>5}")
print("-"*90)

for ds in ["gsm8k", "mmlu"]:
    # Recompute
    rc = load_dedup(f"{BASE}/{ds}/recompute/shard*.jsonl")
    if not rc: print(f"  {ds}/recompute: NO DATA"); continue
    rc_acc = acc_avg4(rc); rc_ttft = ttft_med(rc)
    rc_pipe = st.median([sum(r["agent_ttft"]) for r in rc])
    # e2e for saved/%
    e2e = st.median([sum(r["agent_ttft"]) + sum(r["agent_ngen"])*r["t_dec"] for r in rc])
    a3r = a3_code_rate(rc) if ds == "gsm8k" else None
    print(f"{ds:<10} {'Recompute':<11} {rc_acc*100:>5.1f}% {rc_ttft:>6.1f} {'--':>5} {'--':>11} {'--':>8} {(f'{a3r*100:.0f}%' if a3r is not None else '--'):>8} {len(rc):>5}")

    for method, label in [("stale","Stale"), ("hybrid","RebaseKV")]:
        rows = load_dedup(f"{BASE}/{ds}/{method}/shard*.jsonl")
        if not rows: print(f"  {ds}/{method}: NO DATA"); continue
        acc = acc_avg4(rows); ttft = ttft_med(rows)
        K = sum(r.get("K",0) for r in rows)/len(rows) if method=="hybrid" else 0
        if method == "stale":
            saved = rc_pipe  # stale saves all prefill
        else:
            saved = st.median([r["sum_prefill"] for r in rows]) - st.median([r["sum_rebase"] for r in rows])
        savedp = saved / e2e * 100
        a3 = a3_code_rate(rows) if ds == "gsm8k" else None
        print(f"{'':<10} {label:<11} {acc*100:>5.1f}% {ttft:>6.1f} {K:>5.1f} {saved:>8.0f}ms {savedp:>7.2f}% {(f'{a3*100:.0f}%' if a3 is not None else '--'):>8} {len(rows):>5}")

    # WarmupKV
    wk = load_dedup(f"{BASE}/{ds}_warmupkv/shard*.jsonl")
    if wk:
        wk_ttft = st.median([x for r in wk for x in r["pipe_ttft"]])
        wk_saved = st.median([sum(r["rec_ttft"])-sum(r["pipe_ttft"]) for r in wk])
        print(f"{'':<10} {'WarmupKV':<11} {rc_acc*100:>5.1f}% {wk_ttft:>6.1f} {'--':>5} {wk_saved:>8.0f}ms {wk_saved/e2e*100:>7.2f}% {'--':>8} {len(wk):>5}")
    print()

print("\n" + "="*90)
print("TABLE 3: K-Ablation (tab:k_ablation)")
print("="*90)
print(f"{'K':<6} {'pass@4':>8} {'a3_code':>8} {'N':>5}")
print("-"*40)
for K in [3, 5, 10, 20, 50]:
    # K=3 is the default adaptive → use hybrid results
    if K == 3:
        rows = load_dedup(f"{BASE}/gsm8k/hybrid/shard*.jsonl")
    else:
        rows = load_dedup(f"{BASE}/gsm8k_kablation/K{K}/shard*.jsonl")
    if not rows: print(f"  K={K}: NO DATA"); continue
    acc = acc_avg4(rows); a3 = a3_code_rate(rows)
    print(f"K={K:<4} {acc*100:>7.1f}% {a3*100:>7.0f}% {len(rows):>5}")

print("\n" + "="*90)
print("TABLE 4: KL Divergence (tab:klladder)")
print("="*90)
kl_file = f"{BASE}/gsm8k_kl/kl_results.json"
if os.path.exists(kl_file):
    kl = json.load(open(kl_file))
    for cond, vals in kl.items():
        print(f"  {cond}: {vals}")
else:
    print("  KL results not yet available")

print("\nDONE")
