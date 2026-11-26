#!/usr/bin/env python3
"""warmupKV: REAL two-stream pipelined execution (not analytical). The design:
 - DURING agent i's decode, stream B prefills agent i+1's prompt EXCEPT agent i's (not-yet-known) output,
   i.e. the shared prefix (constraints/question/earlier peers/role) -> hidden, off the critical path.
 - AT the START of agent i+1 (BEFORE its decode), we prefill the one missing block: agent i's output
   (the exact re-tokenized residual), appended on top of the already-warmed prefix. This short "warmup"
   prefill is the only exposed work and defines warmupKV's TTFT. Then decode with fully-correct KV.
NOTE: the residual is prefilled UP FRONT at agent start (not chunked into decode). warmupKV is EXACT:
outputs are verified identical to full recompute. Also measures recompute TTFT (full prefill+1tok) as
baseline, and both e2e wall-clocks. Greedy. MMLU=faithful+wiki, HE=faithful (Qwen2.5-Coder), GSM8K=spec_rebase_bench.
Usage: _warmupkv_run.py <dataset> <shard> <nshard>"""
import os, sys, time, json, torch
os.environ.setdefault("MMLU_LIVE_WIKI", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.spec_rebase_bench as B
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

DS = sys.argv[1]; SHARD = int(sys.argv[2]); NSH = int(sys.argv[3])
NQ = int(os.environ.get("NQ", "60"))   # TTFT is stable; subset is enough
_BASE = os.environ.get("WARMUP_OUTDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "warmupkv"))
OUTD = f"{_BASE}/{DS}"; os.makedirs(OUTD, exist_ok=True)
OUT = f"{OUTD}/shard{SHARD}.jsonl"
torch.set_grad_enabled(False); dev = "cuda"
if DS == "mmlu":
    import scripts.spec_rebase_mmlu_faithful as F; MODEL = F.MODEL
elif DS == "humaneval":
    import scripts.spec_rebase_humaneval_faithful as H; MODEL = H.MODEL
else: MODEL = B.MODEL
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa").eval()
sB = torch.cuda.Stream()
def sync(): torch.cuda.synchronize()
def cpl(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
    return n

def load_items():
    if DS == "mmlu": return F.load_kvcomm_val(153)
    if DS == "humaneval":
        rows = [json.loads(l) for l in open(H.HE_JSONL) if l.strip()]
        return [{"task": r["prompt"], "test": r["test"], "entry": r["entry_point"]} for r in rows]
    return B.load_data("gsm8k", NQ)
NA = (lambda: F.N_AGENTS if DS=="mmlu" else (H.N_AGENTS if DS=="humaneval" else len(B.roles_for("gsm8k"))))()

def build_agent(ai, item, peers, proles, wiki, qi):
    if DS == "mmlu":
        mx = 320 if ai == F.N_AGENTS-1 else 512
        s, u = F.sys_user_for(ai, F.fmt_task(item), peers, proles, wiki, qi); return F.to_ids(tok,s,u,qi), u, s, mx
    if DS == "humaneval":
        s, u = H.sys_user_for(ai, item["task"], peers, proles, H.extract_example(item["task"])); return H.to_ids(tok,s,u,qi), u, s, 512
    role = B.roles_for("gsm8k")[ai]
    return B.build_ids("gsm8k", tok, role, item["q"], peers, proles, qi), B.build_user("gsm8k", role, item["q"], peers, proles), role, 400
def role_name(ai): return F.role_of(ai) if DS=="mmlu" else (H.role_of(ai) if DS=="humaneval" else B.roles_for("gsm8k")[ai])
def wikiup(ai, text, wiki): return F.build_wiki(F.find_entities(text), tok) if (DS=="mmlu" and ai==0) else wiki
def correct(item, ft): return (str(F.extract(ft))==str(item["gold"])) if DS=="mmlu" else (H.score(item,ft) if DS=="humaneval" else B.is_correct("gsm8k",item,B.extract_answer("gsm8k",ft),ft))
def decode_from(logits, kv, mx):
    cur = logits[:, -1:].argmax(-1); g=[int(cur.item())]; k=kv
    for _ in range(mx):
        o = model(cur, past_key_values=k, use_cache=True); cur=o.logits[:,-1:].argmax(-1); k=o.past_key_values
        t=int(cur.item()); g.append(t)
        if t==tok.eos_token_id: break
    return g

def do_q(qi, item):
    # ---- pass 1: recompute (baseline) -> exact outputs + full-prefill TTFT + e2e ----
    peers=[]; proles=[]; wiki=""; prompts=[]; rec_ttft=[]; ngen=[]
    sync(); e0=time.perf_counter()
    for ai in range(NA):
        ids,u,s,mx = build_agent(ai, item, peers, proles, wiki, qi); prompts.append((ids,u,s,mx))
        sync(); t0=time.perf_counter(); full=model(ids, use_cache=True); _=full.logits[:,-1:].argmax(-1); sync()
        rec_ttft.append((time.perf_counter()-t0)*1000)
        g=decode_from(full.logits, full.past_key_values, mx); ngen.append(len(g))
        txt=tok.decode(g, skip_special_tokens=True); peers.append(txt); proles.append(role_name(ai)); wiki=wikiup(ai,txt,wiki)
    sync(); rec_e2e=(time.perf_counter()-e0)*1000; rec_ok=bool(correct(item, peers[-1]))
    rec_peers=peers[:]

    # ---- pass 2: warmupKV (prefix warmed on sB during prev decode; prev-output residual warmup-prefilled at agent start) ----
    # precompute prefix/residual token splits from exact prompts
    splits=[]
    for ai in range(NA):
        ids=prompts[ai][0]; L=ids.shape[1]
        if ai==0: splits.append((ids[:, :1], ids))
        else:
            prev=rec_peers[ai-1]; u=prompts[ai][1]; s=prompts[ai][2]
            idx=u.find(prev[:40]) if prev.strip() else -1; ku=u[:idx] if idx>0 else u
            if DS=="mmlu": kids=F.to_ids(tok,s,ku,qi)
            elif DS=="humaneval": kids=H.to_ids(tok,s,ku,qi)
            else: kids=B.build_ids("gsm8k",tok,s,item["q"],[],[],qi)
            m=max(1,min(cpl(ids[0].tolist(), kids[0].tolist()), L-1)); splits.append((ids[:, :m], ids[:, m:]))
    pipe_ttft=[]; prev_prefix=None; ok_match=True
    sync(); p0=time.perf_counter()
    for ai in range(NA):
        ids,u,s,mx = prompts[ai]; prefix_ids,resid_ids=splits[ai]
        sync(); t0=time.perf_counter()
        if ai==0 or prev_prefix is None: out=model(ids, use_cache=True)
        else: out=model(resid_ids, past_key_values=prev_prefix, use_cache=True)   # warmup: prefill prev-output residual at agent start = TTFT
        cur=out.logits[:,-1:].argmax(-1); sync(); pipe_ttft.append((time.perf_counter()-t0)*1000)
        # warm the NEXT agent's shared prefix on stream B (hidden under this decode)
        nxt=None
        if ai < NA-1:
            with torch.cuda.stream(sB): nxt=model(splits[ai+1][0], use_cache=True).past_key_values
        g=[int(cur.item())]; k=out.past_key_values
        for _ in range(mx):
            o=model(cur, past_key_values=k, use_cache=True); cur=o.logits[:,-1:].argmax(-1); k=o.past_key_values
            t=int(cur.item()); g.append(t)
            if t==tok.eos_token_id: break
        sync(); prev_prefix=nxt
        if tok.decode(g, skip_special_tokens=True).strip()!=rec_peers[ai].strip(): ok_match=False
    sync(); pipe_e2e=(time.perf_counter()-p0)*1000
    return dict(qi=qi, rec_ok=rec_ok, rec_ttft=rec_ttft, pipe_ttft=pipe_ttft, ngen=ngen,
                rec_e2e=rec_e2e, pipe_e2e=pipe_e2e, exact_match=ok_match)

def main():
    items=load_items()[:NQ]
    idxs=[i for i in range(len(items)) if i%NSH==SHARD]
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: done.add(json.loads(l)["qi"])
            except: pass
    f=open(OUT,"a")
    for qi in idxs:
        if qi in done: continue
        try:
            r=do_q(qi, items[qi]); f.write(json.dumps(r)+"\n"); f.flush()
            print(f"{DS} q{qi} rec_ok={r['rec_ok']} exact_match={r['exact_match']} "
                  f"recTTFT={[round(x) for x in r['rec_ttft']]} warmupTTFT={[round(x) for x in r['pipe_ttft']]}", flush=True)
        except Exception as e:
            import traceback; print(f"{DS} q{qi} ERR {e}",flush=True); traceback.print_exc()

if __name__ == "__main__":
    main()
