#!/usr/bin/env python3
"""Generalized speculative-rebasing benchmark: GSM8K / MMLU / HumanEval.
5-agent KVCOMM pipeline, methods recompute / stale / hybrid (adaptive K).
Same metrics as the GSM8K final table: pass@4 (or pass@k via samples), per-agent
median TTFT, adaptive K, and the prefill-region time saved (vs full e2e and vs
recompute's prefill+first-K-decodes). KVCOMM-faithful roles/constraints taken from
KVCOMM's KVCOMM/prompt/{mmlu,humaneval,gsm8k}_prompt_set.py.

  CUDA_VISIBLE_DEVICES=0 python scripts/spec_rebase_bench.py --dataset mmlu --method recompute --n 150
  python scripts/spec_rebase_bench.py --dataset mmlu --aggregate
"""
from __future__ import annotations
import os, json, time, re, argparse, statistics, math, subprocess, tempfile, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("REBASEKV_MODEL", "Qwen/Qwen2.5-7B-Instruct")   # Qwen/Qwen3-8B for the Qwen3 rows
N_AGENTS = 5
BASE = Path(os.environ.get("REBASE_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "_multiagent")))

# Qwen3: suppress <think> reasoning tokens for fair comparison
_CHAT_KWARGS = {"enable_thinking": False} if "Qwen3" in MODEL else {}

# REBASEKV_EXACT_REPLAY=1 feeds every token to the model exactly once: the inherited cache stops one token
# before the end of the prompt (the first draft step consumes it), and decoding after the replay continues from
# the replay's last logits. The default (0) reproduces the released results, where the inherited cache already
# holds the last prompt token and the continuation starts by feeding y_K once more.
EXACT_REPLAY = os.environ.get("REBASEKV_EXACT_REPLAY", "0") == "1"


def stale_prefill(model, ids_stale):
    """Inherited (stale-role) cache and the token that opens the optimistic draft."""
    with torch.no_grad():
        kv = model(ids_stale[:, :-1] if EXACT_REPLAY else ids_stale, use_cache=True).past_key_values
    return kv, ids_stale[:, -1:]

# ---------------- GSM8K (KVCOMM native prompts) ----------------
GSM8K_PROMPTS = json.load(open(os.environ.get("GSM8K_PROMPTS_JSON", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "_kvcomm_gsm8k_prompts.json"))))
GSM8K_ROLES = ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector", "__decision__"]

# ---------------- MMLU (KVCOMM roles) ----------------
MMLU_ROLE_DESC = {
    "Knowledgeable Expert": "You are a knowledgeable expert in question answering.",
    "Critic": "You are an excellent critic.\nPlease point out potential issues in other agent's analysis point by point.",
    "Mathematician": "You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.",
    "Psychologist": "You are a psychologist.\nYou are good at psychology, sociology, and philosophy.",
}
MMLU_ROLES = ["Knowledgeable Expert", "Critic", "Mathematician", "Psychologist", "__decision__"]
MMLU_CONSTRAINT = (
    "I will ask you a question and 4 answers enumerated as A, B, C and D.\n"
    "Only one answer out of the offered 4 is correct.\n"
    "Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?\n"
    "Your reply must be less than 100 words but include your answer and a brief step by step analysis.\n"
    "The first line of your reply must contain only one letter(for example : A, B, C or D)")
MMLU_DECISION = (
    "You are the top decision-maker, good at analyzing and summarizing other people's opinions, finding errors and giving final answers.\n"
    "You will receive a question followed by four possible answers labeled A, B, C, and D. Only one answer is correct.\n"
    "Choose the correct option based on the analysis from other agents.\n"
    "Your response must be exactly one of the letters A, B, C, or D, with no additional characters or text.")

# ---------------- HumanEval (KVCOMM roles = 5 agents, no separate decision) ----------------
HE_ROLE_DESC = {
    "Project Manager": ("You are a project manager. You will be given a function signature and its docstring. "
        "Oversee the overall structure of the code, ensuring it is structured to complete the task. Implement concisely "
        "and correctly without over-engineering. Suggest optimal design. Reply concisely, preferably within fifty words."),
    "Algorithm Designer": ("You are an algorithm designer. You will be given a function signature and its docstring. "
        "Specify the design of the algorithm, classes and functions used. When logic is complex, give pseudocode. "
        "Reply concisely, preferably within fifty words."),
    "Programming Expert": ("You are a programming expert. You will be given a function signature and its docstring. "
        "You may get outputs of other agents; they may have passed internal tests but may not be correct. "
        "Write your full implementation (restate the function signature). Use a Python code block. "
        "Do not include anything other than Python code blocks. Do not change function names or input types."),
    "Test Analyst": ("You are a test analyst. You will be given a function signature and its docstring. "
        "Provide problems in the current code based on test data and feedback, additional special use cases and "
        "boundary conditions to pay attention to. Point out potential errors. Reply concisely, within fifty words."),
    "Bug Fixer": ("You are a bug fixer. You will be given a function signature and its docstring. "
        "Provide modified and improved python code based on the current overall design, framework, implementation or "
        "test problems. Write your full implementation (restate the function signature). Use a Python code block. "
        "Do not include anything other than Python code blocks. Do not change function names or input types."),
}
HE_ROLES = ["Project Manager", "Algorithm Designer", "Programming Expert", "Test Analyst", "Bug Fixer"]


# ================= dataset adapters =================
def roles_for(dataset):
    return {"gsm8k": GSM8K_ROLES, "mmlu": MMLU_ROLES, "humaneval": HE_ROLES}[dataset]


def role_system(dataset, role):
    if dataset == "gsm8k":
        if role == "__decision__":
            return GSM8K_PROMPTS["decision_constraint"]
        return GSM8K_PROMPTS["role_description"][role]
    if dataset == "mmlu":
        if role == "__decision__":
            return MMLU_DECISION
        return MMLU_ROLE_DESC[role] + "\n" + MMLU_CONSTRAINT
    if dataset == "humaneval":
        return HE_ROLE_DESC[role]


def role_fewshot(dataset, role):
    if dataset == "gsm8k":
        if role == "__decision__":
            return GSM8K_PROMPTS["decision_few_shot"]
        return GSM8K_PROMPTS["few_shot"].get(role, "")
    return ""  # mmlu / humaneval: no few-shot (KVCOMM uses none)


def build_user(dataset, role, q, peers, proles):
    fs = role_fewshot(dataset, role)
    u = (f"{fs}\n\nQ: {q}\n" if fs else (f"{q}\n" if dataset != "gsm8k" else f"Q: {q}\n"))
    if peers:
        s = ""
        for pi, (po, pr) in enumerate(zip(peers, proles)):
            prd = "Decision Maker" if pr == "__decision__" else pr
            s += f"Agent {pi+1} as a {prd} his answer to this question is:\n\n{po}\n\n"
        u += f"At the same time, there are the following responses to the same question for your reference:\n\n{s}\n\n"
    return u


def build_ids(dataset, tok, role, q, peers, proles, qi):
    msgs = [{"role": "system", "content": f"[task:{qi}] " + role_system(dataset, role)},
            {"role": "user", "content": build_user(dataset, role, q, peers, proles)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **_CHAT_KWARGS)
    return tok(text, return_tensors="pt").input_ids.cuda()


def load_data(dataset, n):
    from datasets import load_dataset
    items = []
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for i in range(min(n, len(ds))):
            g = ds[i]["answer"].split("####")[-1].strip()
            items.append({"q": ds[i]["question"], "gold": _norm_num(g)})
    elif dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42)
        for i in range(min(n, len(ds))):
            ch = ds[i]["choices"]
            qtext = ds[i]["question"] + "\n" + "".join(f"{'ABCD'[j]}. {c}\n" for j, c in enumerate(ch))
            items.append({"q": qtext, "gold": "ABCD"[ds[i]["answer"]]})
    elif dataset == "humaneval":
        path = os.environ.get("HE_JSONL", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "humaneval-py.jsonl"))
        rows = [json.loads(l) for l in open(path) if l.strip()]
        for r in rows[:n]:
            items.append({"q": r["prompt"], "gold": None, "test": r["test"],
                          "entry": r["entry_point"], "prompt_code": r["prompt"]})
    return items


def _norm_num(s):
    if s is None:
        return None
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s); return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_answer(dataset, text):
    if dataset == "gsm8k":
        mm = re.findall(r"[Tt]he answer is\s*[:\s]*\$?([-+]?\d[\d,]*(?:\.\d+)?)", text)
        if mm:
            return _norm_num(mm[-1])
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        return _norm_num(nums[-1]) if nums else None
    if dataset == "mmlu":
        head = text.strip()[:5]
        m = re.search(r"\b([A-D])\b", head)
        if m:
            return m.group(1)
        m = re.search(r"[Tt]he answer is\s*[:\s]*([A-D])", text)
        if m:
            return m.group(1)
        ls = re.findall(r"\b([A-D])\b", text)
        return ls[-1] if ls else None
    return text  # humaneval: scored by execution


def _extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text


def score_humaneval(item, final_text):
    code = _extract_code(final_text)
    program = code + "\n\n" + item["test"] + f"\n\ncheck({item['entry']})\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program); path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=15)
        ok = (r.returncode == 0)
    except Exception:
        ok = False
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return ok


def is_correct(dataset, item, pred, final_text):
    if dataset == "humaneval":
        return score_humaneval(item, final_text)
    return str(pred) == str(item["gold"])


# ================= engine (same as final_table, parameterized) =================
def sync():
    torch.cuda.synchronize()


def _sample(logits, temp):
    if temp <= 0:
        return logits[:, -1:].argmax(-1)
    p = torch.softmax(logits[:, -1, :] / temp, dim=-1)
    return torch.multinomial(p, 1)


def gen_recompute(model, tok, ids_fresh, max_new, temp, timed):
    sync(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model(ids_fresh, use_cache=True)
    k = out.past_key_values
    sync(); t_prefill = (time.perf_counter() - t0) * 1000
    nt = _sample(out.logits, temp)
    sync(); ttft = (time.perf_counter() - t0) * 1000
    gen = [int(nt.item())]; cur = nt
    for _ in range(max_new):
        with torch.no_grad():
            out = model(cur, past_key_values=k, use_cache=True)
        nt = _sample(out.logits, temp); tid = int(nt.item())
        gen.append(tid); cur = nt; k = out.past_key_values
        if tid == tok.eos_token_id:
            break
    info = {"t_prefill": t_prefill, "n_gen": len(gen)} if timed else None
    return tok.decode(gen, skip_special_tokens=True), ttft, info


def gen_stale(model, tok, stale_kv, last_tok, max_new, temp, timed):
    sync(); t0 = time.perf_counter()
    cur = last_tok; k = stale_kv
    with torch.no_grad():
        out = model(cur, past_key_values=k, use_cache=True)
    nt = _sample(out.logits, temp)
    sync(); ttft = (time.perf_counter() - t0) * 1000
    gen = [int(nt.item())]; cur = nt; k = out.past_key_values
    for _ in range(max_new):
        with torch.no_grad():
            out = model(cur, past_key_values=k, use_cache=True)
        nt = _sample(out.logits, temp); tid = int(nt.item())
        gen.append(tid); cur = nt; k = out.past_key_values
        if tid == tok.eos_token_id:
            break
    info = {"n_gen": len(gen)} if timed else None
    return tok.decode(gen, skip_special_tokens=True), ttft, info


def gen_hybrid(model, tok, ids_fresh, stale_kv, last_tok, t_dec_step, max_new, temp, timed):
    sync(); tp = time.perf_counter()
    with torch.no_grad():
        fresh_kv = model(ids_fresh, use_cache=True).past_key_values
    sync(); t_prefill = (time.perf_counter() - tp) * 1000
    K = max(2, math.ceil(t_prefill / t_dec_step))
    sync(); t0 = time.perf_counter()
    cur = last_tok; k = stale_kv; draft = []; ttft = None
    for i in range(K):
        with torch.no_grad():
            out = model(cur, past_key_values=k, use_cache=True)
        nt = _sample(out.logits, temp)
        if ttft is None:
            sync(); ttft = (time.perf_counter() - t0) * 1000
        tid = int(nt.item()); draft.append(tid); cur = nt; k = out.past_key_values
        if tid == tok.eos_token_id:
            info = {"t_prefill": t_prefill, "K": K, "t_rebase": 0.0, "n_gen": len(draft)} if timed else None
            return tok.decode(draft, skip_special_tokens=True), ttft, info
    rin = torch.tensor([draft], device="cuda")
    sync(); tr = time.perf_counter()
    with torch.no_grad():
        out = model(rin, past_key_values=fresh_kv, use_cache=True)
    sync(); t_rebase = (time.perf_counter() - tr) * 1000
    k = out.past_key_values; cur = rin[:, -1:]; gen = list(draft)
    if EXACT_REPLAY:
        cur = _sample(out.logits, temp); gen.append(int(cur.item()))
        if gen[-1] == tok.eos_token_id:
            info = {"t_prefill": t_prefill, "K": K, "t_rebase": t_rebase, "n_gen": len(gen)} if timed else None
            return tok.decode(gen, skip_special_tokens=True), ttft, info
    for _ in range(max_new - len(gen)):
        with torch.no_grad():
            out = model(cur, past_key_values=k, use_cache=True)
        nt = _sample(out.logits, temp); tid = int(nt.item())
        gen.append(tid); cur = nt; k = out.past_key_values
        if tid == tok.eos_token_id:
            break
    info = {"t_prefill": t_prefill, "K": K, "t_rebase": t_rebase, "n_gen": len(gen)} if timed else None
    return tok.decode(gen, skip_special_tokens=True), ttft, info


def run_agent(dataset, model, tok, rf, rs, q, peers, proles, qi, method, t_dec_step, max_new, temp, timed):
    ids_fresh = build_ids(dataset, tok, rf, q, peers, proles, qi)
    if method == "recompute":
        return gen_recompute(model, tok, ids_fresh, max_new, temp, timed)
    ids_stale = build_ids(dataset, tok, rs, q, peers, proles, qi)
    stale_kv, last_tok = stale_prefill(model, ids_stale)
    if method == "stale":
        return gen_stale(model, tok, stale_kv, last_tok, max_new, temp, timed)
    return gen_hybrid(model, tok, ids_fresh, stale_kv, last_tok, t_dec_step, max_new, temp, timed)


def calibrate_dec_step(model, tok):
    ids = tok("Solve: " + "the cat sat. " * 100, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        k = model(ids, use_cache=True).past_key_values
    cur = ids[:, -1:]
    for _ in range(3):
        with torch.no_grad():
            out = model(cur, past_key_values=k, use_cache=True)
        cur = out.logits[:, -1:].argmax(-1); k = out.past_key_values
    sync(); ts = []
    for _ in range(10):
        sync(); t = time.perf_counter()
        with torch.no_grad():
            out = model(cur, past_key_values=k, use_cache=True)
        cur = out.logits[:, -1:].argmax(-1); k = out.past_key_values
        sync(); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def outdir(dataset):
    d = BASE / f"final_table_{dataset}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_method(dataset, method, n, samples, max_new):
    out_path = outdir(dataset) / f"{method}.json"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    torch.set_grad_enabled(False)
    items = load_data(dataset, n)
    roles = roles_for(dataset)
    t_dec = calibrate_dec_step(model, tok)
    print(f"[{dataset}/{method}] N={len(items)} samples={samples} max_new={max_new} t_dec={t_dec:.1f}ms", flush=True)

    correct = 0; total = 0; a3 = 0
    agent_ttft = [[] for _ in range(N_AGENTS)]; agent_ngen = [[] for _ in range(N_AGENTS)]
    sum_prefill = []; sum_rebase = []; Ks = []
    t0 = time.time()
    for qi, item in enumerate(items):
        q = item["q"]
        for si in range(samples):
            temp = 0.0 if si == 0 else 0.7
            timed = (si == 0)
            total += 1
            peers = []; proles = []; texts = []
            for ai in range(N_AGENTS):
                rf = roles[ai]; rs = roles[(ai - 1) % N_AGENTS]
                text, ttft, info = run_agent(dataset, model, tok, rf, rs, q, peers, proles, qi,
                                             method, t_dec, max_new, temp, timed)
                peers.append(text); proles.append(rf); texts.append(text)
                if timed:
                    agent_ttft[ai].append(ttft); agent_ngen[ai].append(info["n_gen"])
                    if method == "recompute":
                        sum_prefill.append(info["t_prefill"])
                    elif method == "hybrid":
                        sum_prefill.append(info["t_prefill"]); sum_rebase.append(info["t_rebase"]); Ks.append(info["K"])
            pred = extract_answer(dataset, texts[-1])
            if is_correct(dataset, item, pred, texts[-1]):
                correct += 1
            if si == 0 and ("```" in texts[2] or bool(re.search(r"\bdef \w+\s*\(", texts[2]))):
                a3 += 1
        if (qi + 1) % 5 == 0:
            print(f"[{dataset}/{method}] Q{qi+1}/{len(items)} ({time.time()-t0:.0f}s) "
                  f"pass@{samples}={correct/total*100:.1f}% K~{statistics.median(Ks) if Ks else '-'} "
                  f"TTFT~{statistics.median([x for v in agent_ttft for x in v]):.0f}ms", flush=True)
            _save(out_path, dataset, method, len(items), total, correct, a3, agent_ttft, agent_ngen, sum_prefill, sum_rebase, Ks, t_dec, samples)
    _save(out_path, dataset, method, len(items), total, correct, a3, agent_ttft, agent_ngen, sum_prefill, sum_rebase, Ks, t_dec, samples)
    print(f"[{dataset}/{method}] DONE pass@{samples}={correct/total*100:.1f}% saved {out_path}", flush=True)


def _save(p, dataset, method, n, total, correct, a3, agent_ttft, agent_ngen, sum_prefill, sum_rebase, Ks, t_dec, samples):
    json.dump({
        "dataset": dataset, "method": method, "n": n, "samples": samples, "total_samples": total,
        "pass": correct / total if total else 0,
        "a3_code_rate": a3 / n if n else 0,
        "per_agent_ttft_med": [statistics.median(v) if v else 0 for v in agent_ttft],
        "ttft_med": statistics.median([x for v in agent_ttft for x in v]) if any(agent_ttft) else 0,
        "per_agent_ngen_med": [statistics.median(v) if v else 0 for v in agent_ngen],
        "t_dec_step": t_dec,
        "prefill_sum_per_pipeline": (sum(sum_prefill) / (len(sum_prefill) / N_AGENTS)) if sum_prefill else 0,
        "rebase_sum_per_pipeline": (sum(sum_rebase) / (len(sum_rebase) / N_AGENTS)) if sum_rebase else 0,
        "K_mean": (sum(Ks) / len(Ks)) if Ks else 0,
    }, open(p, "w"), indent=1)


def aggregate(dataset):
    data = {}
    for mth in ["recompute", "stale", "hybrid"]:
        p = outdir(dataset) / f"{mth}.json"
        if p.exists():
            data[mth] = json.load(open(p))
    if "recompute" not in data:
        print(f"[{dataset}] need recompute.json"); return
    rc = data["recompute"]; t_dec = rc["t_dec_step"]
    rc_ngen = rc["per_agent_ngen_med"]; rc_prefill = rc["prefill_sum_per_pipeline"]
    rc_e2e = rc_prefill + sum(rc_ngen) * t_dec
    Khyb = data.get("hybrid", {}).get("K_mean", 0)
    rc_front = rc_prefill + Khyb * N_AGENTS * t_dec
    samples = rc.get("samples", 4)
    print(f"\n{'='*104}")
    print(f"{dataset.upper()} ({rc['n']} problems), {MODEL}, 5 agents (KVCOMM prompts). avg@{samples}; timing=det. sample.")
    print(f"decode ~{t_dec:.0f}ms/tok | recompute e2e ~{rc_e2e/1000:.2f}s | prefill {rc_prefill:.0f}ms/pipeline | front(pf+K·dec) {rc_front:.0f}ms")
    hdr = (f"{'Approach':<12}{'#ag':>4}{'pass@'+str(samples):>8}{'TTFT':>7}{'K':>6}"
           f"{'saved/pipe':>13}{'% full e2e':>12}{'% front':>10}")
    print(hdr); print("-" * len(hdr))
    for mth in ["recompute", "stale", "hybrid"]:
        if mth not in data:
            continue
        d = data[mth]; ttft = d["ttft_med"]; K = d.get("K_mean", 0)
        if mth == "recompute":
            saved = 0.0; Kshow = "-"; fr_den = rc_front
        elif mth == "stale":
            saved = rc_prefill; Kshow = "0"; fr_den = rc_prefill
        else:
            saved = rc_prefill - d["rebase_sum_per_pipeline"]; Kshow = f"{K:.1f}"; fr_den = rc_front
        sv = "-" if mth == "recompute" else f"+{saved:.0f}ms"
        pe = "-" if mth == "recompute" else f"{saved/rc_e2e*100:.1f}%"
        pf = "-" if mth == "recompute" else f"{saved/fr_den*100:.0f}%"
        print(f"{mth:<12}{N_AGENTS:>4}{d['pass']*100:>7.1f}%{ttft:>6.0f}m{Kshow:>6}{sv:>13}{pe:>12}{pf:>10}")
    print("Note: prefill-hiding idealized (full overlap); realistic single-stream ~50% (FINDINGS §91).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "mmlu", "humaneval"], required=True)
    ap.add_argument("--method", choices=["recompute", "stale", "hybrid"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.dataset)
    else:
        run_method(args.dataset, args.method, args.n, args.samples, args.max_new)


if __name__ == "__main__":
    main()
