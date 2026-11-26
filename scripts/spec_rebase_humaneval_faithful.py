#!/usr/bin/env python3
"""FAITHFUL KVCOMM HumanEval pipeline in OUR framework (recompute/stale/hybrid), Qwen2.5-Coder-7B.
Replicates KVCOMM exactly (prompt/humaneval_prompt_set.py + agents/code_writing.py +
agents/final_decision.py FinalWriteCode).

Pipeline (6 agents): Project Manager -> Algorithm Designer -> Programming Expert ->
Test Analyst -> Bug Fixer -> FinalWriteCode.
- Worker system = ROLE_DESCRIPTION[role] VERBATIM.
- Peers whose output is a ```python block are run against internal_tests (doctests parsed
  from the prompt) and shown with pass/fail + feedback, exactly like KVCOMM.
- FinalWriteCode: get_decision_role + get_decision_constraint; its code is the scored answer.
Model = Qwen2.5-Coder-7B-Instruct for ALL methods (matches KVCOMM's HumanEval setup).

  CUDA_VISIBLE_DEVICES=2 python scripts/spec_rebase_humaneval_faithful.py --method recompute --n 161
  python scripts/spec_rebase_humaneval_faithful.py --aggregate
"""
from __future__ import annotations
import os, sys, json, re, time, statistics, argparse, tempfile, subprocess
from pathlib import Path
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.spec_rebase_bench as B
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTDIR = Path(os.environ.get("HE_OUTDIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "humaneval_faithful")))
HE_JSONL = os.environ.get("HE_JSONL", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "humaneval-py.jsonl"))

# ---- KVCOMM verbatim HE prompts (prompt/humaneval_prompt_set.py) ----
ROLE_DESCRIPTION = {
 "Project Manager":
  "You are a project manager. You will be given a function signature and its docstring by the user. "
  "You are responsible for overseeing the overall structure of the code, ensuring that the code is "
  "structured to complete the task Implement code concisely and correctly without pursuing over-engineering."
  "You need to suggest optimal design patterns to ensure that the code follows best practices for "
  "maintainability and flexibility. You can specify the overall design of the code, including the classes "
  "that need to be defined(maybe none) and the functions used(maybe only one function) ."
  "I hope your reply will be more concise. Preferably within fifty words. Don\u2019t list too many points.",
 "Algorithm Designer":
  "You are an algorithm designer. You will be given a function signature and its docstring by the user. "
  "You need to specify the specific design of the algorithm, including the classes that may be defined "
  "and the functions used. You need to generate the detailed documentation, including explanations of the "
  "algorithm, usage instructions, and API references. When the implementation logic is complex, you can "
  "give the pseudocode logic of the main algorithm.I hope your reply will be more concise. Preferably "
  "within fifty words. Don\u2019t list too many points.",
 "Programming Expert":
  "You are a programming expert. You will be given a function signature and its docstring by the user. "
  "You may be able to get the output results of other agents. They may have passed internal tests, but "
  "they may not be completely correct. Write your full implementation (restate the function signature). "
  "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
  "Do not include anything other than Python code blocks in your response. "
  "Do not change function names and input variable types in tasks.",
 "Test Analyst":
  "You are a test analyst. You will be given a function signature and its docstring by the user. "
  "You need to provide problems in the current code or solution based on the test data and possible test "
  "feedback in the question. You need to provide additional special use cases, boundary conditions, etc. "
  "that should be paid attention to when writing code. You can point out any potential errors in the code."
  "I hope your reply will be more concise. Preferably within fifty words. Don\u2019t list too many points.",
 "Bug Fixer":
  "You are a bug fixer.You will be given a function signature and its docstring by the user. "
  "You need to provide modified and improved python code based on the current overall code design, "
  "algorithm framework, code implementation or test problems. Write your full implementation (restate "
  "the function signature). Use a Python code block to write your response. For example:\n```python\n"
  "print('Hello world!')\n```Do not include anything other than Python code blocks in your response "
  "Do not change function names and input variable types in tasks",
}
WORKER_ROLES = ["Project Manager", "Algorithm Designer", "Programming Expert", "Test Analyst", "Bug Fixer"]
DECISION_ROLE = ("You are the top decision-maker and are good at analyzing and summarizing other people's "
 "opinions, finding errors and giving final answers. And you are an AI that only responds with only python code.")
DECISION_CONSTRAINT = (
 "You will be given a function signature and its docstring by the user."
 "You may be given the overall code design, algorithm framework, code implementation or test problems."
 "Write your full implementation (restate the function signature). "
 "If the prompt given to you contains code that passed internal testing, you can choose the most reliable reply."
 "If there is no code that has passed internal testing in the prompt, you can change it yourself according to the prompt."
 "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
 "Do not include anything other than Python code blocks in your response")
N_AGENTS = len(WORKER_ROLES) + 1  # 6


def extract_example(task):
    lines = (l.strip() for l in task.split("\n") if l.strip())
    res = []; it = iter(lines)
    for line in it:
        if line.startswith(">>>"):
            call = line[4:]; exp = next(it, None)
            if exp:
                res.append(f"assert {call} == {exp}")
    return res


def run_internal(code, tests):
    """Return (all_passed_bool, feedback_str). Executes each 'assert' test in a subprocess."""
    if not tests:
        return True, "No internal tests."
    passed, failed = [], []
    for t in tests:
        prog = f"from typing import *\n{code}\n{t}\n"
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(prog); path = f.name
            r = subprocess.run([sys.executable, path], capture_output=True, timeout=8)
            (passed if r.returncode == 0 else failed).append(t)
        except Exception:
            failed.append(t)
        finally:
            try: os.unlink(path)
            except Exception: pass
    fb = "Tests passed:\n" + "\n".join(passed) + "\n\nTests failed:\n" + "\n".join(failed)
    return (len(failed) == 0), fb


def is_code_block(t):
    t = t.strip()
    return t.startswith("```python") and t.endswith("```")


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text


def worker_msgs(role, task, peers, proles, internal_tests):
    system = ROLE_DESCRIPTION[role]
    spatial = ""
    for i, (o, r) in enumerate(zip(peers, proles)):
        if is_code_block(o):
            code = o.split("```python\n")[-1].split("\n```")[0]
            ok, fb = run_internal(code, internal_tests)
            spatial += (f"Agent {i+1} as a {r}:\n\nThe code written by the agent is:\n\n{o}\n\n "
                        f"Whether it passes internal testing?\n{ok}.\n\nThe feedback is:\n\n {fb}.\n\n")
        else:
            spatial += f"Agent {i+1} as a {r} provides the following info: {o}\n\n"
    user = f"The task is:\n\n{task}\n"
    if spatial:
        user += f"At the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial} \n\n"
    return system, user


def decision_msgs(task, peers, proles, internal_tests):
    system = f"{DECISION_ROLE}.\n {DECISION_CONSTRAINT}"
    spatial = ""
    for i, (o, r) in enumerate(zip(peers, proles)):
        if is_code_block(o):
            code = o.split("```python\n")[-1].split("\n```")[0]
            ok, fb = run_internal(code, internal_tests)
            spatial += (f"Agent {i+1} as a {r}:\n\nThe code written by the agent is:\n\n{o}\n\n "
                        f"Whether it passes internal testing?\n{ok}.\n\nThe feedback is:\n\n {fb}.\n\n")
        else:
            spatial += f"Agent {i+1} as a {r} provides the following info: {o}\n\n"
    user = f"  {task}\n At the same time, the output of other agents is as follows:\n\n{spatial}\n\n"
    return system, user


def role_of(ai):
    return WORKER_ROLES[ai] if ai < len(WORKER_ROLES) else "FinalWriteCode"


def sys_user_for(ai, task, peers, proles, itests):
    if ai < len(WORKER_ROLES):
        return worker_msgs(WORKER_ROLES[ai], task, peers, proles, itests)
    return decision_msgs(task, peers, proles, itests)


def stale_sys_for(ai):
    prev = (ai - 1) % N_AGENTS
    if prev < len(WORKER_ROLES):
        return ROLE_DESCRIPTION[WORKER_ROLES[prev]]
    return f"{DECISION_ROLE}.\n {DECISION_CONSTRAINT}"


def to_ids(tok, system, user, qi):
    msgs = [{"role": "system", "content": f"[task:{qi}] " + system},
            {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids.cuda()


def score(item, final_text):
    code = extract_code(final_text)
    # KVCOMM's PyExecutor prepends 'from typing import *' so type hints (Dict/List/...) resolve.
    # Omitting it makes correct typed solutions fail with NameError -> must match.
    program = "from typing import *\n" + code + "\n\n" + item["test"]
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program); path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=20)
        ok = (r.returncode == 0)
    except Exception:
        ok = False
    finally:
        try: os.unlink(path)
        except Exception: pass
    return ok


def run_method(method, n, samples):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"{method}.json"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    torch.set_grad_enabled(False)
    rows = [json.loads(l) for l in open(HE_JSONL) if l.strip()][:n]
    items = [{"task": r["prompt"], "test": r["test"], "entry": r["entry_point"]} for r in rows]
    t_dec = B.calibrate_dec_step(model, tok)
    print(f"[he_faithful/{method}] N={len(items)} samples={samples} t_dec={t_dec:.1f}ms model={MODEL}", flush=True)

    correct = 0; total = 0; a3 = 0
    agent_ttft = [[] for _ in range(N_AGENTS)]; agent_ngen = [[] for _ in range(N_AGENTS)]
    sum_prefill = []; sum_rebase = []; Ks = []
    t0 = time.time()
    for qi, item in enumerate(items):
        task = item["task"]; itests = extract_example(task)
        for si in range(samples):
            temp = 0.0 if si == 0 else 0.7
            timed = (si == 0)
            total += 1
            peers = []; proles = []
            for ai in range(N_AGENTS):
                is_decision = (ai == N_AGENTS - 1)  # FinalWriteCode = synthesis/decision agent
                mx = 512
                system, user = sys_user_for(ai, task, peers, proles, itests)
                ids_fresh = to_ids(tok, system, user, qi)
                # POLICY (RebaseKV/hybrid ONLY): the final synthesis agent (FinalWriteCode) recomputes
                # (full prefill), because drafting code from a wrong-role stale cache corrupts the
                # synthesized program into garbled/empty output (FINDINGS §98). This mirrors the MMLU
                # decision-recompute policy and is part of the METHOD. STALE stays pure reuse
                # everywhere (incl. the synthesis agent) -> honest naive-reuse baseline.
                if method == "recompute" or (is_decision and method == "hybrid"):
                    text, ttft, info = B.gen_recompute(model, tok, ids_fresh, mx, temp, timed)
                    if timed and "t_prefill" in info:
                        if method == "recompute":
                            sum_prefill.append(info["t_prefill"])
                        else:  # hybrid synthesis agent: real (un-hidden) prefill-region cost
                            sum_rebase.append(info["t_prefill"])
                else:
                    ids_stale = to_ids(tok, stale_sys_for(ai), user, qi)
                    stale_kv, last = B.stale_prefill(model, ids_stale)
                    if method == "stale":
                        text, ttft, info = B.gen_stale(model, tok, stale_kv, last, mx, temp, timed)
                    else:  # hybrid worker
                        text, ttft, info = B.gen_hybrid(model, tok, ids_fresh, stale_kv, last, t_dec, mx, temp, timed)
                        if timed:
                            sum_rebase.append(info["t_rebase"]); Ks.append(info["K"])
                peers.append(text); proles.append(role_of(ai))
                if timed:
                    agent_ttft[ai].append(ttft); agent_ngen[ai].append(info["n_gen"])
            if score(item, peers[-1]):
                correct += 1
            # A3 = agent index 2 (Programming Expert) emits code
            if si == 0 and ("```" in peers[2] or re.search(r"\bdef \w+\s*\(", peers[2])):
                a3 += 1
        if (qi + 1) % 10 == 0:
            _dec_ng = agent_ngen[N_AGENTS - 1]
            _dec_max = max(_dec_ng) if _dec_ng else 0
            _dec_tr = sum(1 for x in _dec_ng if x >= 513)
            print(f"[he_faithful/{method}] Q{qi+1}/{len(items)} ({time.time()-t0:.0f}s) "
                  f"pass@{samples}={correct/total*100:.1f}% K~{statistics.median(Ks) if Ks else '-'} "
                  f"decAgent_maxNgen={_dec_max} decAgent_capHits={_dec_tr}", flush=True)
            _save(out_path, method, len(items), total, correct, a3, agent_ttft, agent_ngen, sum_prefill, sum_rebase, Ks, t_dec, samples)
    _save(out_path, method, len(items), total, correct, a3, agent_ttft, agent_ngen, sum_prefill, sum_rebase, Ks, t_dec, samples)
    print(f"[he_faithful/{method}] DONE pass@{samples}={correct/total*100:.1f}% saved {out_path}", flush=True)


def _save(p, method, n, total, correct, a3, agent_ttft, agent_ngen, sum_prefill, sum_rebase, Ks, t_dec, samples):
    json.dump({
        "dataset": "humaneval_faithful", "method": method, "n": n, "samples": samples, "total_samples": total,
        "pass": correct / total if total else 0, "a3_code_rate": a3 / n if n else 0,
        "per_agent_ttft_med": [statistics.median(v) if v else 0 for v in agent_ttft],
        "ttft_med": statistics.median([x for v in agent_ttft for x in v]) if any(agent_ttft) else 0,
        "per_agent_ngen_med": [statistics.median(v) if v else 0 for v in agent_ngen],
        "per_agent_ngen_max": [max(v) if v else 0 for v in agent_ngen],
        # cap-hit: gen_recompute/gen_hybrid do 1 pre-loop token + range(512) -> n_gen==513 means the
        # 512-token cap was reached with no EOS (truncated). Count per agent.
        "per_agent_trunc": [sum(1 for x in v if x >= 513) for v in agent_ngen],
        "t_dec_step": t_dec,
        "prefill_sum_per_pipeline": (sum(sum_prefill) / (len(sum_prefill) / N_AGENTS)) if sum_prefill else 0,
        "rebase_sum_per_pipeline": (sum(sum_rebase) / (len(sum_rebase) / N_AGENTS)) if sum_rebase else 0,
        "K_mean": (sum(Ks) / len(Ks)) if Ks else 0,
    }, open(p, "w"), indent=1)


def aggregate():
    data = {}
    for m in ["recompute", "stale", "hybrid"]:
        p = OUTDIR / f"{m}.json"
        if p.exists():
            data[m] = json.load(open(p))
    if "recompute" not in data:
        print("need recompute.json"); return
    rc = data["recompute"]; t_dec = rc["t_dec_step"]
    e2e = rc["prefill_sum_per_pipeline"] + sum(rc["per_agent_ngen_med"]) * t_dec
    print(f"\nHumanEval FAITHFUL ({rc['n']} q, {N_AGENTS} agents, Qwen-Coder). recompute e2e ~{e2e/1000:.2f}s")
    print(f"{'method':<10}{'pass@4':>8}{'A3code':>8}{'TTFT':>8}{'K':>6}{'saved/pipe':>12}{'%e2e':>8}")
    for m in ["recompute", "stale", "hybrid"]:
        if m not in data:
            continue
        d = data[m]
        if m == "recompute":
            saved = 0.0; K = "-"
        elif m == "stale":
            saved = rc["prefill_sum_per_pipeline"]; K = "0"
        else:
            saved = rc["prefill_sum_per_pipeline"] - d["rebase_sum_per_pipeline"]; K = f"{d['K_mean']:.1f}"
        sv = "-" if m == "recompute" else f"+{saved:.0f}ms"
        pe = "-" if m == "recompute" else f"{saved/e2e*100:.1f}%"
        print(f"{m:<10}{d['pass']*100:>7.1f}%{d['a3_code_rate']*100:>7.0f}%{d['ttft_med']:>7.0f}m{K:>6}{sv:>12}{pe:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["recompute", "stale", "hybrid"])
    ap.add_argument("--n", type=int, default=161); ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        run_method(a.method, a.n, a.samples)


if __name__ == "__main__":
    main()
