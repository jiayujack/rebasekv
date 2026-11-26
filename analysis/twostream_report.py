#!/usr/bin/env python3
"""Cross-variant report for the real two-stream RebaseKV runs -> markdown tables for the rebuttal.
Usage: python twostream_report.py <results_root> [dir1 dir2 ...]
  results_root contains variant dirs (A_fixed_all_naive, B_fixed_all_prio, ...) each with shard*.jsonl
Only questions present in ALL listed variants are used for the paired comparison (same GSM8K questions,
same recompute protocol), unless --all is given."""
import sys, os, json, glob, statistics as st

root = sys.argv[1]; dirs = [a for a in sys.argv[2:] if not a.startswith("--")]
if not dirs:
    dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and glob.glob(os.path.join(root, d, "shard*.jsonl")))
ALL = "--all" in sys.argv
med = lambda v: st.median(v) if v else float("nan")

def load(d):
    rows = {}
    for p in glob.glob(os.path.join(root, d, "shard*.jsonl")):
        for l in open(p):
            l = l.strip()
            if l:
                r = json.loads(l); rows[r["qi"]] = r
    return rows

V = {d: load(d) for d in dirs}
common = set.intersection(*(set(v) for v in V.values())) if not ALL else None
def sel(rows): return [rows[q] for q in sorted(rows) if ALL or q in common]

def label(r):
    pol = r["policy"]; ex = "exact" + "".join(map(str, r["exact_agents"])) if r["exact_agents"] else "all"
    hw = ("prio" if r.get("fg_priority") else "default") + ("+thread" if r.get("bg_thread") else "") + (f"+chunk{r['prefill_chunk']}" if r.get("prefill_chunk") else "")
    return f"{pol} K, agents={ex}, {hw}"

print(f"paired questions: {len(common) if common is not None else 'all'}\n")
print("| variant | n | Recompute TTFT | RebaseKV TTFT | ratio | prefill solo→contended | K | stall | replay | step in window (solo) | draft agree | 1st tok agree | greedy acc R / B | non-decode Δ |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in dirs:
    rows = sel(V[d]); 
    if not rows: continue
    r0 = rows[0]; NA = len(r0["rec_ttft"])
    reb_agents = [a for a in range(NA) if r0["mode"][a] == "rebase"]
    rec_t = [r["rec_ttft"][a] for r in rows for a in reb_agents]; reb_t = [r["reb_ttft"][a] for r in rows for a in reb_agents]
    pf_s = [r["pf_solo"][a] for r in rows for a in reb_agents]; pf_b = [r["pf_bg"][a] for r in rows for a in reb_agents]
    K = [r["K"][a] for r in rows for a in reb_agents]; stall = [r["stall_ms"][a] for r in rows for a in reb_agents]
    rep = [r["replay_ms"][a] for r in rows for a in reb_agents]; step = [r["step_ms"][a] for r in rows for a in reb_agents]
    agree = sum(r["agree"][a] for r in rows for a in reb_agents) / max(1, sum(K)); first = st.mean(r["first_agree"][a] for r in rows for a in reb_agents)
    acc_r = 100 * st.mean(r["rec_ok"] for r in rows); acc_b = 100 * st.mean(r["reb_ok"] for r in rows)
    nd = [ (r["reb_e2e_adj"] - sum(r["reb_ngen"]) * r["t_dec"]) - (r["rec_e2e"] - sum(r["rec_ngen"]) * r["t_dec"]) for r in rows]
    tdec = med([r["t_dec"] for r in rows])
    print(f"| {label(r0)} | {len(rows)} | {med(rec_t):.0f} ms | {med(reb_t):.0f} ms | {med(rec_t)/med(reb_t):.2f}x | {med(pf_s):.0f}→{med(pf_b):.0f} ms | {st.mean(K):.1f} | {med(stall):.0f} ms | {med(rep):.0f} ms | {med(step):.0f} ({tdec:.0f}) ms | {100*agree:.0f}% | {100*first:.0f}% | {acc_r:.1f} / {acc_b:.1f} | {med(nd):+.0f} ms |")

print("\nColumns: medians over rebased agents (agent 0 and exact-path agents excluded from TTFT); 'prefill solo→contended' = exact prefill of the new prompt alone vs. while the draft runs; "
      "'stall' = wait for the fresh cache after the draft window; 'replay' = K-token forward on the fresh cache; 'step in window' = per-token decode time during the window vs. solo t_dec; "
      "'draft agree' = drafted tokens equal to the fresh-cache greedy token; 'non-decode Δ' = paired median of (e2e − tokens×t_dec) RebaseKV minus Recompute, i.e. total overhead attributable to prefill/draft/replay/contention.")

# per-agent detail for each variant
for d in dirs:
    rows = sel(V[d])
    if not rows: continue
    NA = len(rows[0]["rec_ttft"]); print(f"\n### {d}  ({label(rows[0])}, n={len(rows)})")
    print("| agent | mode | Recompute TTFT | RebaseKV TTFT | K | stall | replay | 1st agree | ngen R / B |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a in range(NA):
        mode = rows[0]["mode"][a]; f = lambda k: med([r[k][a] for r in rows if r[k][a] is not None])
        fa = [r["first_agree"][a] for r in rows if r["first_agree"][a] is not None]
        g = lambda k: f"{f(k):.1f}" if mode == "rebase" else "-"
        print(f"| {a} | {mode} | {f('rec_ttft'):.0f} | {f('reb_ttft'):.0f} | {g('K')} | {g('stall_ms')} | {g('replay_ms')} | {100*st.mean(fa) if fa else float('nan'):.0f}% | {f('rec_ngen'):.0f} / {f('reb_ngen'):.0f} |")
