#!/usr/bin/env python3
"""Paired accuracy comparison of two per-question result directories (shard*.jsonl).

Usage (from the repository root):
    python analysis/paired.py <baseline_dir> <method_dir> [<method_dir> ...]

Each line of a shard holds {"qi": ..., "sample_ok": [greedy, T=0.7, T=0.7, T=0.7]} (or a single greedy
entry). Reports greedy and avg@4 accuracy on the questions present in every directory, and the paired
difference to the baseline with a 95% bootstrap CI (10,000 resamples, seed 0).
"""
import glob, json, os, random, statistics as st, sys


def load(d):
    by = {}
    for p in sorted(glob.glob(os.path.join(d, "shard*.jsonl"))):
        for line in open(p):
            if line.strip():
                r = json.loads(line); by[r["qi"]] = r
    return by


def paired_ci(diffs, nb=10000, seed=0):
    rng = random.Random(seed); n = len(diffs)
    bs = sorted(st.mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(nb))
    return st.mean(diffs), bs[int(0.025 * nb)], bs[int(0.975 * nb)]


METRICS = {"greedy": lambda r: float(r["sample_ok"][0]),
           "avg@4": lambda r: sum(r["sample_ok"]) / len(r["sample_ok"])}

if __name__ == "__main__":
    dirs = sys.argv[1:]
    if len(dirs) < 2:
        sys.exit(__doc__)
    runs = [load(d) for d in dirs]
    common = sorted(set.intersection(*(set(r) for r in runs)))
    n_samples = min(len(r[q]["sample_ok"]) for r in runs for q in common)
    names = ["greedy"] + (["avg@4"] if n_samples == 4 else [])
    print(f"paired questions: {len(common)}\n")
    print("| run | " + " | ".join(f"{m} (%) | Δ {m} vs baseline [95% CI]" for m in names) + " |")
    print("|---|" + "---|---|" * len(names))
    base = runs[0]
    for d, run in zip(dirs, runs):
        cells = []
        for m in names:
            f = METRICS[m]
            acc = 100 * st.mean(f(run[q]) for q in common)
            if run is base:
                cells += [f"{acc:.1f}", "—"]
            else:
                mu, lo, hi = paired_ci([f(run[q]) - f(base[q]) for q in common])
                cells += [f"{acc:.1f}", f"{100 * mu:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"]
        print(f"| {d} | " + " | ".join(cells) + " |")
