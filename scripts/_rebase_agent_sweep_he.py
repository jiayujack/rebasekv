#!/usr/bin/env python3
"""RebaseKV (hybrid) HumanEval agent-number sweep, one N per process (for multi-GPU parallelism).
KVCOMM humaneval_prompt_set cycles ['Project Manager','Algorithm Designer','Programming Expert',
'Test Analyst','Bug Fixer']; agent_nums=N uses first N + FinalWriteCode. Patch H.WORKER_ROLES=first N,
H.N_AGENTS=N+1, per-N OUTDIR; run hybrid (=RebaseKV) THEN recompute (baseline), samples=4 (avg-of-4).
Usage: _rebase_agent_sweep_he.py <N>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import scripts.spec_rebase_humaneval_faithful as H

N = int(sys.argv[1])
N_Q = int(os.environ.get("SWEEP_N", "161"))
SAMPLES = int(os.environ.get("SWEEP_SAMPLES", "4"))
FULL = H.WORKER_ROLES[:]
H.WORKER_ROLES = FULL[:N]
H.N_AGENTS = N + 1
H.OUTDIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "runs" / "he_agent_sweep" / f"N{N}"
H.OUTDIR.mkdir(parents=True, exist_ok=True)
print(f"########## HE SWEEP N={N} workers ({H.WORKER_ROLES}) + FinalWriteCode = {H.N_AGENTS} agents ##########", flush=True)
for method in ["hybrid", "recompute"]:
    print(f"---- N={N} {method} ----", flush=True)
    H.run_method(method, N_Q, SAMPLES)
print(f"N={N} SWEEP DONE", flush=True)
