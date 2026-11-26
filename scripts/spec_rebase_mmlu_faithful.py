#!/usr/bin/env python3
"""FAITHFUL KVCOMM MMLU pipeline in OUR framework (recompute/stale/hybrid).
Replicates KVCOMM exactly (KVCOMM/prompt/mmlu_prompt_set.py + agents/analyze_agent.py +
agents/final_decision.py), so recompute/stale/RebaseKV run on identical agents+prompts
and we can cross-check our recompute vs KVCOMM-harness `default`.

Pipeline (agent_nums=3 -> 4 total): Knowledgeable Expert -> Wiki Searcher -> Critic -> FinalRefer.
- Worker system prompt = ROLE_DESCRIPTION[role] VERBATIM (KVCOMM's get_analyze_constraint returns
  only the role desc for known roles due to its operator-precedence quirk -> replicated).
- Wiki Searcher: KE emits @entities@; we look them up in the offline cache and inject
  "The key entities of the problem are explained in Wikipedia as follows:{summaries}".
- FinalRefer: get_decision_role + get_decision_constraint; outputs a single letter.
Wiki source = offline cache (KVCOMM uses live wikipedia; documented deviation).

  CUDA_VISIBLE_DEVICES=2 python scripts/spec_rebase_mmlu_faithful.py --method recompute --n 150
  python scripts/spec_rebase_mmlu_faithful.py --aggregate
"""
from __future__ import annotations
import os, sys, json, re, time, statistics, argparse, urllib.request, urllib.parse
from pathlib import Path
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.spec_rebase_bench as B
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = B.MODEL
OUTDIR = Path(os.environ.get("MMLU_OUTDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "mmlu_faithful")))
WIKI = json.load(open(os.environ.get("MMLU_WIKI_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mmlu_wiki_cache.json"))))
MMLU_JSONL = None  # unused

# ---- KVCOMM verbatim prompts (KVCOMM/prompt/mmlu_prompt_set.py) ----
ROLE_DESCRIPTION = {
"Knowledgeable Expert":
"""
You are a knowledgeable expert in question answering.
Please give at most six key entities that need to be searched in wikipedia to solve the problem. 
Key entities that need to be searched are included between two '@' when output, for example: @catfish effect@, @broken window effect@, @Shakespeare@.
If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it
""",
"Wiki Searcher":
"""
You will be given a question and a wikipedia overview of the key entities within it.
Please refer to them step by step to give your answer.
And point out potential issues in other agent's analysis.
""",
"Critic":
"""
You are an excellent critic.
Please point out potential issues in other agent's analysis point by point.
""",
"Mathematician":
"""
You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.
""",
"Psychologist":
"""
You are a psychologist.
You are good at psychology, sociology, and philosophy.
You give people scientific suggestions that will make them feel better.
""",
}
# Paper's tuned MMLU config = 5 workers (AnalyzeAgent) + FinalRefer (README: "tuned the agent
# number ... for each benchmark"; their 5-agent original = 69.9%). Match it.
WORKER_ROLES = ["Knowledgeable Expert", "Wiki Searcher", "Critic", "Mathematician", "Psychologist"]
DECISION_ROLE = ("You are the top decision-maker and are good at analyzing and summarizing other "
                 "people's opinions, finding errors and giving final answers.")
DECISION_CONSTRAINT = """
You will receive a question followed by four possible answers labeled A, B, C, and D. Only one answer is correct.
Choose the correct option based on the analysis and recommendations provided by the output of other agents.
Your response must be exactly one of the letters A, B, C, or D, with no additional characters or text.
        """
N_AGENTS = len(WORKER_ROLES) + 1  # 3 workers + FinalRefer = 4
WIKI_TOKEN_LEN = 1024


def find_entities(text):
    return re.findall(r"@+([^@]+?)@+", text)


# ---- LIVE Wikipedia retrieval (replicates KVCOMM search_wiki: top-2 matching pages per
# entity + intro summaries) via the MediaWiki API, with a persistent cache for
# reproducibility and to avoid re-fetching across pass@4 samples / reruns. ----
_UA = {"User-Agent": "RebaseKV-research/1.0 (research contact)"}
_WIKI_API = "https://en.wikipedia.org/w/api.php"
LIVE_CACHE_PATH = Path(os.environ.get("MMLU_LIVE_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "mmlu_wiki_live_cache.json")))
LIVE_CACHE = json.load(open(LIVE_CACHE_PATH)) if LIVE_CACHE_PATH.exists() else {}
USE_LIVE = os.environ.get("MMLU_LIVE_WIKI", "1") == "1"


def _api(params):
    u = _WIKI_API + "?" + urllib.parse.urlencode(params)
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=_UA), timeout=15))


def _live_entity_block(entity):
    """Search top-2 pages for entity, return concatenated 'The summary of T in Wikipedia is: S\\n'."""
    key = entity.lower().strip()
    if key in LIVE_CACHE:
        return LIVE_CACHE[key]
    block = ""
    try:
        d = _api({"action": "query", "list": "search", "srsearch": entity, "format": "json",
                  "srlimit": 2, "formatversion": "2"})
        titles = [r["title"] for r in d.get("query", {}).get("search", [])]
        for t in titles:
            e = _api({"action": "query", "prop": "extracts", "exintro": "true",
                      "explaintext": "true", "titles": t, "format": "json"})
            summ = ""
            for _, v in e.get("query", {}).get("pages", {}).items():
                summ = v.get("extract", ""); break
            if summ:
                block += f"The summary of {t} in Wikipedia is: {summ}\n"
    except Exception:
        block = None  # network error -> signal fallback
    if block is not None:
        LIVE_CACHE[key] = block
    return block


def _save_live_cache():
    try:
        json.dump(LIVE_CACHE, open(LIVE_CACHE_PATH, "w"))
    except Exception:
        pass


def build_wiki(entities, tok):
    parts = []
    for e in entities[:6]:
        block = _live_entity_block(e) if USE_LIVE else None
        if block:
            parts.append(block)
        elif block is None or not block:  # live failed/empty -> offline exact-key fallback
            c = WIKI.get(e.lower().strip())
            if c and c.get("summary"):
                parts.append(f"The summary of {c['title']} in Wikipedia is: {c['summary']}\n")
    if not parts:
        return ""
    summary = ".\n".join(parts)
    ids = tok(summary, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] > WIKI_TOKEN_LEN:
        summary = tok.decode(ids[0, :WIKI_TOKEN_LEN], skip_special_tokens=True)
    return summary


def load_kvcomm_val(n):
    """Replicate KVCOMM's MMLUDataset('val') EXACTLY: concat sorted val CSVs, shuffle with
    numpy rng(888) permutation, take first n. Same questions KVCOMM evaluates its 69.9% on."""
    import glob, pandas as pd, numpy as np
    names = ["question", "A", "B", "C", "D", "correct_answer"]
    paths = sorted(glob.glob(os.environ.get("MMLU_VAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "MMLU", "data", "val")) + "/*.csv"))
    df = pd.concat([pd.read_csv(p, header=None, names=names, encoding="utf-8") for p in paths]).reset_index(drop=True)
    rng = np.random.default_rng(888)
    df = df.reindex(rng.permutation(df.index)).head(n)
    return [{"question": r["question"], "choices": [r["A"], r["B"], r["C"], r["D"]],
             "gold": str(r["correct_answer"]).strip()} for _, r in df.iterrows()]


def fmt_task(item):
    ch = item["choices"]
    return (f"{item['question']}\n"
            f"Option A: {ch[0]}\nOption B: {ch[1]}\nOption C: {ch[2]}\nOption D: {ch[3]}\n")


def worker_msgs(role, task, peers, proles, wiki):
    """peers/proles: prior agents' outputs+roles. wiki: injected for Wiki Searcher."""
    system = ROLE_DESCRIPTION[role]
    user = f"The task is: {task}\n"
    if role == "Wiki Searcher" and wiki:
        user += f"The key entities of the problem are explained in Wikipedia as follows:{wiki}"
    spatial = ""
    for i, (o, r) in enumerate(zip(peers, proles)):
        spatial += f"Agent {i+1}, role is {r}, output is:\n\n {o}\n\n"
    if spatial:
        user += f"At the same time, the outputs of other agents are as follows:\n\n{spatial} \n\n"
    return system, user


def decision_msgs(task, peers, proles):
    system = f"{DECISION_ROLE}.\n {DECISION_CONSTRAINT}"
    spatial = ""
    for i, (o, r) in enumerate(zip(peers, proles)):
        spatial += f"Agent {i+1}, role is {r}, output is:\n\n {o}\n\n"
    user = f"  {task}\n At the same time, the output of other agents is as follows:\n\n{spatial}\n\n"
    return system, user


def to_ids(tok, system, user, qi):
    msgs = [{"role": "system", "content": f"[task:{qi}] " + system},
            {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **B._CHAT_KWARGS)
    return tok(text, return_tensors="pt").input_ids.cuda()


def extract(text):
    h = text.strip()[:5]
    m = re.search(r"\b([A-D])\b", h)
    if m:
        return m.group(1)
    m = re.search(r"[Tt]he answer is\s*[:\s]*([A-D])", text)
    if m:
        return m.group(1)
    ls = re.findall(r"\b([A-D])\b", text)
    return ls[-1] if ls else None


def role_of(ai):
    return WORKER_ROLES[ai] if ai < len(WORKER_ROLES) else "FinalRefer"


def sys_user_for(ai, task, peers, proles, wiki, qi):
    """Return (system, user) for agent ai's OWN role."""
    if ai < len(WORKER_ROLES):
        return worker_msgs(WORKER_ROLES[ai], task, peers, proles, wiki)
    return decision_msgs(task, peers, proles)


def stale_sys_for(ai):
    """System prompt of the PREVIOUS agent (the reused KV's role)."""
    prev = (ai - 1) % N_AGENTS
    if prev < len(WORKER_ROLES):
        return ROLE_DESCRIPTION[WORKER_ROLES[prev]]
    return f"{DECISION_ROLE}.\n {DECISION_CONSTRAINT}"


def run_method(method, n, samples):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"{method}.json"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    torch.set_grad_enabled(False)
    items = load_kvcomm_val(n)
    t_dec = B.calibrate_dec_step(model, tok)
    print(f"[mmlu_faithful/{method}] N={len(items)} samples={samples} t_dec={t_dec:.1f}ms", flush=True)

    correct = 0; total = 0; n_timed = 0
    agent_ttft = [[] for _ in range(N_AGENTS)]; agent_ngen = [[] for _ in range(N_AGENTS)]
    sum_prefill = []; sum_rebase = []; Ks = []; wiki_hit = 0; wiki_tot = 0
    DEC = len(WORKER_ROLES)  # index of the final decision agent
    t0 = time.time()
    for qi, item in enumerate(items):
        task = fmt_task(item)
        for si in range(samples):
            temp = 0.0 if si == 0 else 0.7
            timed = (si == 0)
            total += 1
            if timed:
                n_timed += 1
            peers = []; proles = []; wiki = ""
            for ai in range(N_AGENTS):
                is_decision = (ai == DEC)
                mx = 320 if is_decision else 512
                system, user = sys_user_for(ai, task, peers, proles, wiki, qi)
                ids_fresh = to_ids(tok, system, user, qi)
                # POLICY (RebaseKV/hybrid ONLY): the final decision agent recomputes (full prefill),
                # because its answer is a front-loaded single letter and drafting from a wrong-role
                # cache corrupts it into runaway critique (FINDINGS diag). This is part of the METHOD.
                # STALE stays pure reuse everywhere (incl. decision) -> honest naive-reuse baseline.
                if method == "recompute" or (is_decision and method == "hybrid"):
                    text, ttft, info = B.gen_recompute(model, tok, ids_fresh, mx, temp, timed)
                    if timed and "t_prefill" in info:
                        if method == "recompute":
                            sum_prefill.append(info["t_prefill"])
                        else:  # hybrid decision: prefill is the (un-hidden) prefill-region cost
                            sum_rebase.append(info["t_prefill"])
                else:
                    stale_sys = stale_sys_for(ai)
                    ids_stale = to_ids(tok, stale_sys, user, qi)
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
                if ai == 0:  # KE -> entities -> wiki for Wiki Searcher
                    ents = find_entities(text); wiki = build_wiki(ents, tok)
                    if timed:
                        wiki_tot += min(len(ents), 6)
                        wiki_hit += sum(1 for e in ents[:6]
                                        if LIVE_CACHE.get(e.lower().strip())
                                        or WIKI.get(e.lower().strip(), {}).get("summary"))
            pred = extract(peers[-1])
            if str(pred) == str(item["gold"]):
                correct += 1
        if (qi + 1) % 10 == 0:
            print(f"[mmlu_faithful/{method}] Q{qi+1}/{len(items)} ({time.time()-t0:.0f}s) "
                  f"pass@{samples}={correct/total*100:.1f}% wiki_hit={wiki_hit}/{wiki_tot} "
                  f"K~{statistics.median(Ks) if Ks else '-'}", flush=True)
            _save(out_path, method, len(items), total, correct, agent_ttft, agent_ngen, sum_prefill, sum_rebase, Ks, t_dec, samples, wiki_hit, wiki_tot, n_timed)
            _save_live_cache()
    _save(out_path, method, len(items), total, correct, agent_ttft, agent_ngen, sum_prefill, sum_rebase, Ks, t_dec, samples, wiki_hit, wiki_tot, n_timed)
    _save_live_cache()
    print(f"[mmlu_faithful/{method}] DONE pass@{samples}={correct/total*100:.1f}% saved {out_path}", flush=True)


def _save(p, method, n, total, correct, agent_ttft, agent_ngen, sum_prefill, sum_rebase, Ks, t_dec, samples, wh, wt, n_timed):
    nt = max(1, n_timed)
    json.dump({
        "dataset": "mmlu_faithful", "method": method, "n": n, "samples": samples, "total_samples": total,
        "pass": correct / total if total else 0, "a3_code_rate": None,  # no code agent in MMLU
        "decision_agent_recompute": True,  # final agent always full-prefill recompute
        "per_agent_ttft_med": [statistics.median(v) if v else 0 for v in agent_ttft],
        "ttft_med": statistics.median([x for v in agent_ttft for x in v]) if any(agent_ttft) else 0,
        "per_agent_ngen_med": [statistics.median(v) if v else 0 for v in agent_ngen],
        "t_dec_step": t_dec,
        "prefill_sum_per_pipeline": sum(sum_prefill) / nt,   # recompute: total prefill/pipeline
        "rebase_sum_per_pipeline": sum(sum_rebase) / nt,     # stale/hybrid: prefill-region cost/pipeline
        "K_mean": (sum(Ks) / len(Ks)) if Ks else 0, "wiki_hit": wh, "wiki_tot": wt,
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
    print(f"\nMMLU FAITHFUL ({rc['n']} q, {N_AGENTS} agents). recompute e2e ~{e2e/1000:.2f}s  "
          f"wiki_hit={rc.get('wiki_hit')}/{rc.get('wiki_tot')}")
    print(f"{'method':<10}{'pass@4':>8}{'TTFT':>8}{'K':>6}{'saved/pipe':>12}{'%e2e':>8}")
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
        print(f"{m:<10}{d['pass']*100:>7.1f}%{d['ttft_med']:>7.0f}m{K:>6}{sv:>12}{pe:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["recompute", "stale", "hybrid"])
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        run_method(a.method, a.n, a.samples)


if __name__ == "__main__":
    main()
