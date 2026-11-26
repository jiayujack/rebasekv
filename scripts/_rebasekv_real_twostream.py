#!/usr/bin/env python3
"""REAL two-stream RebaseKV execution on the GSM8K 5-agent pipeline (rebuttal experiment).

Answers reviewers 75sK / ATUj: what happens when the exact prefill really runs concurrently with the
optimistic draft on the same GPU?  Per question we run two full pipelines (greedy):

  pass 1  Recompute : exact prefill per agent (timed, exposed)  -> baseline TTFT / e2e / accuracy
  pass 2  RebaseKV  : agent 0 (and any agent in EXACT_AGENTS) exact; every other agent:
            - background exact prefill of the NEW prompt launched on CUDA stream B
            - foreground greedy draft from the inherited cache on the default stream, event-timed
            - policy fixed : draft K = max(K_MIN, ceil(t_prefill_recent / t_dec)) tokens (paper rule)
              policy event : draft until stream B signals the fresh cache is ready (>= K_MIN, <= K_MAX)
            - default stream waits for B, replay the draft on the fresh cache (timed), continue decoding
            - zero-cost draft-agreement: fresh top-1 at every draft position comes from the prefill's
              last logits (pos 1) and the replay logits (pos 2..K)

Inherited-cache proxy: as in the paper's harness, the inherited state is the exact prefill of the
previous role's prompt minus its last token (computed BEFORE the timed region and subtracted from e2e as
'overhead_ms'); the timed bootstrap step is one forward over that last token on the inherited cache.
The bootstrap (RoPE shift of resident spans) is therefore not timed here; say so when reporting.

Env: NQ (default 300), MAX_NEW (400), REBASE_POLICY fixed|event, EXACT_AGENTS "4" (comma list),
     K_MIN 2, K_MAX 64, LOGITS_TO_KEEP 1|0, OUTDIR, REBASE_PKG (package root containing scripts/)
     FG_PRIORITY 1|0   : run the foreground draft/replay/continue on a HIGH-priority CUDA stream (default 1);
                         0 = naive overlap on the default stream (background prefill kernels starve the draft)
     PREFILL_CHUNK N   : background prefill in N-token chunks (default 0 = one shot); chunking gives the GPU
                         scheduler kernel boundaries at which high-priority draft kernels can run
     BG_THREAD 1|0     : launch the background prefill from a separate CPU thread (default 1), as a serving
                         scheduler would; 0 = launch from the foreground thread (HF eager is CPU-launch-bound,
                         so the draft then waits until every prefill kernel has been enqueued)
Usage: python _rebasekv_real_twostream.py <shard> <nshard>      |  --aggregate <jsonl-glob>
Place this file in <package>/scripts/ (it imports scripts.spec_rebase_bench).
"""
import os, sys, time, json, math, re, glob, inspect, threading, statistics as st

if len(sys.argv) > 1 and sys.argv[1] == "--aggregate":
    rows = {}
    for p in glob.glob(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("OUTDIR", "."), "shard*.jsonl")):
        for l in open(p):
            l = l.strip()
            if l:
                d = json.loads(l); rows[d["qi"]] = d
    rows = [rows[k] for k in sorted(rows)]
    NA = len(rows[0]["rec_ttft"])
    med = lambda v: st.median(v) if v else float("nan")
    print(f"n={len(rows)} policy={rows[0]['policy']} exact_agents={rows[0]['exact_agents']} fg_priority={rows[0].get('fg_priority')} prefill_chunk={rows[0].get('prefill_chunk')} bg_thread={rows[0].get('bg_thread')} t_dec_solo={med([r['t_dec'] for r in rows]):.1f} ms")
    print(f"greedy acc: recompute {100*st.mean(r['rec_ok'] for r in rows):.1f}  rebasekv {100*st.mean(r['reb_ok'] for r in rows):.1f}  "
          f"(paired diff {100*st.mean(r['reb_ok']-r['rec_ok'] for r in rows):+.1f} pp)")
    print(f"e2e median: recompute {med([r['rec_e2e'] for r in rows])/1000:.2f} s | rebasekv {med([r['reb_e2e_adj'] for r in rows])/1000:.2f} s "
          f"| paired median ratio {med([r['reb_e2e_adj']/r['rec_e2e'] for r in rows]):.4f}")
    nd_rec = [r['rec_e2e'] - sum(r['rec_ngen']) * r['t_dec'] for r in rows]
    nd_reb = [r['reb_e2e_adj'] - sum(r['reb_ngen']) * r['t_dec'] for r in rows]
    print(f"tokens per pipeline: recompute {med([sum(r['rec_ngen']) for r in rows]):.0f} | rebasekv {med([sum(r['reb_ngen']) for r in rows]):.0f}  (e2e is confounded by output length)")
    print(f"non-decode time per pipeline (e2e - tokens x t_dec_solo): recompute {med(nd_rec):.0f} ms | rebasekv {med(nd_reb):.0f} ms | paired diff {med([b-a for a,b in zip(nd_rec,nd_reb)]):+.0f} ms")
    print(f"code-block rate (Programming Expert): recompute {100*st.mean(r['rec_a3'] for r in rows):.1f} rebasekv {100*st.mean(r['reb_a3'] for r in rows):.1f}")
    hdr = f"{'agent':>5} {'mode':>6} {'recTTFT':>8} {'rebTTFT':>8} {'pfSolo':>7} {'pfBg':>7} {'K':>4} {'draft':>6} {'stall':>6} {'replay':>6} {'step/win':>8} {'agree':>6} {'1st':>5} {'ngenR':>5} {'ngenB':>5}"
    print(hdr); print("-" * len(hdr))
    for a in range(NA):
        sel = [r for r in rows]
        mode = rows[0]["mode"][a]
        f = lambda key: med([r[key][a] for r in sel if r[key][a] is not None])
        agree = [r["agree"][a] / r["K"][a] for r in sel if r["K"][a]]
        first = [r["first_agree"][a] for r in sel if r["first_agree"][a] is not None]
        print(f"{a:>5} {mode:>6} {f('rec_ttft'):8.1f} {f('reb_ttft'):8.1f} {f('pf_solo'):7.1f} {f('pf_bg'):7.1f} {f('K'):4.1f} {f('draft_ms'):6.1f} {f('stall_ms'):6.1f} {f('replay_ms'):6.1f} {f('step_ms'):8.1f} "
              f"{100*st.mean(agree) if agree else float('nan'):5.1f}% {100*st.mean(first) if first else float('nan'):4.0f}% {f('rec_ngen'):5.0f} {f('reb_ngen'):5.0f}")
    print("columns: ms medians; pfSolo/pfBg = exact prefill of the new prompt alone / under the concurrent draft; "
          "draft = wall time of the K-token window; stall = wait for the fresh cache after the window; step/win = per-step decode time inside the window "
          "(compare with t_dec_solo); agree = draft tokens equal to fresh-cache greedy tokens; 1st = first token agreement.")
    sys.exit(0)

import torch
PKG = os.environ.get("REBASE_PKG", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PKG)
import scripts.spec_rebase_bench as B
from transformers import AutoModelForCausalLM, AutoTokenizer

SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NSH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NQ = int(os.environ.get("NQ", "300")); MAXNEW = int(os.environ.get("MAX_NEW", "400"))
POLICY = os.environ.get("REBASE_POLICY", "fixed"); assert POLICY in ("fixed", "event")
EXACT = {int(x) for x in os.environ.get("EXACT_AGENTS", "").split(",") if x.strip()}
KMIN = int(os.environ.get("K_MIN", "2")); KMAX = int(os.environ.get("K_MAX", "64"))
FG_PRIORITY = os.environ.get("FG_PRIORITY", "1") == "1"; CHUNK = int(os.environ.get("PREFILL_CHUNK", "0"))
BG_THREAD = os.environ.get("BG_THREAD", "1") == "1"
OUTD = os.environ.get("OUTDIR", os.path.join(PKG, "runs", f"rebase_twostream_{POLICY}_exact{'-'.join(map(str, sorted(EXACT))) or 'none'}_prio{int(FG_PRIORITY)}_chunk{CHUNK}_thr{int(BG_THREAD)}"))
os.makedirs(OUTD, exist_ok=True); OUT = os.path.join(OUTD, f"shard{SHARD}.jsonl")
torch.set_grad_enabled(False); dev = "cuda"

tok = AutoTokenizer.from_pretrained(B.MODEL)
model = AutoModelForCausalLM.from_pretrained(B.MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
HAS_LTK = "logits_to_keep" in inspect.signature(model.forward).parameters and os.environ.get("LOGITS_TO_KEEP", "1") == "1"
sB = torch.cuda.Stream(priority=0)                                   # background prefill: lowest priority
sFG = torch.cuda.Stream(priority=-1) if FG_PRIORITY else torch.cuda.default_stream()   # foreground draft
def sync(): torch.cuda.synchronize()
def Ev(): return torch.cuda.Event(enable_timing=True)

def fwd(ids, kv=None, keep_last_only=False):
    kw = dict(use_cache=True)
    if kv is not None: kw["past_key_values"] = kv
    if keep_last_only and HAS_LTK: kw["logits_to_keep"] = 1
    return model(ids, **kw)

def bg_prefill(ids):
    """Exact prefill of the new prompt on the current (background) stream, optionally in CHUNK-token pieces."""
    if CHUNK <= 0 or ids.shape[1] <= CHUNK:
        return fwd(ids, keep_last_only=True)
    kv = None; out = None
    for s0 in range(0, ids.shape[1], CHUNK):
        out = fwd(ids[:, s0:s0 + CHUNK], kv, keep_last_only=True); kv = out.past_key_values
    return out

t_dec = B.calibrate_dec_step(model, tok)
# warm-up: full-logits replay path + two-stream path once, so first-call costs (cuBLAS init, allocator) are not timed
with torch.cuda.stream(sFG):
    _w_ids = tok("warm up " * 64, return_tensors="pt").input_ids.cuda()
    with torch.cuda.stream(sB):
        _w_out = bg_prefill(_w_ids)
    torch.cuda.current_stream().wait_stream(sB)
    _ = fwd(_w_ids[:, :4], _w_out.past_key_values)
    sync(); del _w_ids, _w_out
roles = B.roles_for("gsm8k"); NA = B.N_AGENTS
items = B.load_data("gsm8k", NQ)
EOS = tok.eos_token_id
print(f"[twostream/{POLICY}] N={len(items)} shard={SHARD}/{NSH} exact_agents={sorted(EXACT)} fg_priority={FG_PRIORITY} prefill_chunk={CHUNK} bg_thread={BG_THREAD} t_dec={t_dec:.1f} ms logits_to_keep={HAS_LTK}", flush=True)

def decode_rest(cur, kv, gen, budget):
    for _ in range(budget):
        out = fwd(cur, kv, keep_last_only=True); kv = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1); tid = int(cur.item()); gen.append(tid)
        if tid == EOS: break
    return gen

def exact_agent(ids):
    """Exact prefill + greedy decode; returns text, ttft_ms, prefill_ms(=ttft), n_gen."""
    sync(); t0 = time.perf_counter()
    out = fwd(ids, keep_last_only=True); kv = out.past_key_values
    cur = out.logits[:, -1:].argmax(-1); tid = int(cur.item())      # .item() syncs the default stream
    ttft = (time.perf_counter() - t0) * 1000
    gen = [tid]
    if tid != EOS: gen = decode_rest(cur, kv, gen, MAXNEW)
    return tok.decode(gen, skip_special_tokens=True), ttft, len(gen)

def a3_code(texts):
    return bool(len(texts) > 2 and ("```" in texts[2] or re.search(r"\bdef \w+\s*\(", texts[2])))

def run_recompute(item, qi):
    peers, proles, texts = [], [], []
    ttft, ngen = [], []
    sync(); e0 = time.perf_counter()
    for ai in range(NA):
        ids = B.build_ids("gsm8k", tok, roles[ai], item["q"], peers, proles, qi)
        text, tt, ng = exact_agent(ids)
        peers.append(text); proles.append(roles[ai]); texts.append(text); ttft.append(tt); ngen.append(ng)
    sync(); e2e = (time.perf_counter() - e0) * 1000
    pred = B.extract_answer("gsm8k", texts[-1])
    return dict(ok=bool(B.is_correct("gsm8k", item, pred, texts[-1])), ttft=ttft, ngen=ngen, e2e=e2e, a3=a3_code(texts))

def run_rebasekv(item, qi, recent_prefill):
    peers, proles, texts = [], [], []
    R = {k: [None] * NA for k in ("ttft", "ngen", "pf_solo", "pf_bg", "K", "draft_ms", "stall_ms", "replay_ms", "step_ms", "agree", "first_agree")}
    mode = []; overhead = 0.0
    sync(); e0 = time.perf_counter()
    for ai in range(NA):
        rf = roles[ai]; rs = roles[(ai - 1) % NA]
        ids_fresh = B.build_ids("gsm8k", tok, rf, item["q"], peers, proles, qi)
        if ai == 0 or ai in EXACT:
            text, tt, ng = exact_agent(ids_fresh)
            R["ttft"][ai] = tt; R["ngen"][ai] = ng; R["pf_solo"][ai] = tt; mode.append("exact")
            peers.append(text); proles.append(rf); texts.append(text); continue
        mode.append("rebase")
        # ---- untimed harness overhead: inherited-cache proxy + solo prefill reference (subtracted from e2e) ----
        sync(); o0 = time.perf_counter()
        ids_stale = B.build_ids("gsm8k", tok, rs, item["q"], peers, proles, qi)
        # inherited cache = all but the last prompt token; the bootstrap step then feeds that last token, so the
        # first draft logits come from a forward over the inherited context
        stale_kv = fwd(ids_stale[:, :-1], keep_last_only=True).past_key_values; last = ids_stale[:, -1:]
        sync(); s0 = time.perf_counter(); fwd(ids_fresh, keep_last_only=True); sync(); pf_solo = (time.perf_counter() - s0) * 1000
        overhead += (time.perf_counter() - o0) * 1000
        K_fixed = max(KMIN, math.ceil(recent_prefill[ai] / t_dec))
        fg = torch.cuda.stream(sFG); fg.__enter__()                  # foreground work on the (high-priority) stream
        # ---- timed region ----
        evA0, evBs, evBe, evAd = Ev(), Ev(), Ev(), Ev()
        bg = {}
        def _bg():                                                # background exact prefill (optionally chunked)
            with torch.no_grad(), torch.cuda.stream(sB):          # grad mode + stream context are thread-local
                evBs.record(); bg["out"] = bg_prefill(ids_fresh); evBe.record(); bg["recorded"] = True
        sync(); t0 = time.perf_counter(); evA0.record()
        if BG_THREAD:
            th = threading.Thread(target=_bg, daemon=True); th.start()
        else:
            th = None; _bg()
        cur = last; k = stale_kv; draft = []; ttft = None; steps = []
        while True:
            es = Ev(); es.record()
            out = fwd(cur, k, keep_last_only=True); k = out.past_key_values
            cur = out.logits[:, -1:].argmax(-1); tid = int(cur.item())
            ee = Ev(); ee.record(); ee.synchronize(); steps.append(es.elapsed_time(ee))
            if ttft is None: ttft = (time.perf_counter() - t0) * 1000
            draft.append(tid)
            if tid == EOS: break
            if POLICY == "fixed":
                if len(draft) >= K_fixed: break
            else:
                if len(draft) >= KMIN and bg.get("recorded") and evBe.query(): break   # query() of an unrecorded event is True
                if len(draft) >= KMAX: break
        evAd.record()
        if th is not None: th.join()
        fresh_out = bg["out"]
        torch.cuda.current_stream().wait_stream(sB)               # foreground waits for the fresh cache
        evW = Ev(); evW.record(); evW.synchronize()
        draft_ms = evA0.elapsed_time(evAd); pf_bg = evBs.elapsed_time(evBe)
        stall_ms = max(0.0, evA0.elapsed_time(evBe) - draft_ms)
        # ---- replay on the fresh cache (full logits: needed for the zero-cost agreement check) ----
        fresh_kv = fresh_out.past_key_values; fresh_first = fresh_out.logits[:, -1, :]
        rin = torch.tensor([draft], device=dev)
        er0, er1 = Ev(), Ev(); er0.record()
        rout = fwd(rin, fresh_kv); er1.record(); er1.synchronize(); replay_ms = er0.elapsed_time(er1)
        fresh_top = [int(fresh_first.argmax(-1).item())] + rout.logits[0, :-1, :].argmax(-1).tolist()
        agree = sum(int(a == b) for a, b in zip(draft, fresh_top))
        gen = list(draft)
        if B.EXACT_REPLAY and draft[-1] != EOS:                  # continue from the replay's last logits
            nxt = rout.logits[:, -1:].argmax(-1); gen.append(int(nxt.item()))
            if gen[-1] != EOS:
                gen = decode_rest(nxt, rout.past_key_values, gen, MAXNEW - len(gen))
        elif draft[-1] != EOS:
            gen = decode_rest(rin[:, -1:], rout.past_key_values, gen, MAXNEW - len(draft))
        text = tok.decode(gen, skip_special_tokens=True)
        R["ttft"][ai] = ttft; R["ngen"][ai] = len(gen); R["pf_solo"][ai] = pf_solo; R["pf_bg"][ai] = pf_bg
        R["K"][ai] = len(draft); R["draft_ms"][ai] = draft_ms; R["stall_ms"][ai] = stall_ms; R["replay_ms"][ai] = replay_ms
        R["step_ms"][ai] = st.mean(steps); R["agree"][ai] = agree; R["first_agree"][ai] = int(draft[0] == fresh_top[0])
        peers.append(text); proles.append(rf); texts.append(text)
        sync(); fg.__exit__(None, None, None); del fresh_out, rout
    sync(); e2e = (time.perf_counter() - e0) * 1000
    pred = B.extract_answer("gsm8k", texts[-1])
    R.update(ok=bool(B.is_correct("gsm8k", item, pred, texts[-1])), e2e=e2e, e2e_adj=e2e - overhead, overhead=overhead, mode=mode, a3=a3_code(texts))
    return R

def main():
    idxs = [i for i in range(len(items)) if i % NSH == SHARD]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: done.add(json.loads(l)["qi"])
            except Exception: pass
    f = open(OUT, "a"); t0 = time.time()
    for n, qi in enumerate(idxs):
        if qi in done: continue
        try:
            rec = run_recompute(items[qi], qi)
            reb = run_rebasekv(items[qi], qi, recent_prefill=rec["ttft"])
            row = dict(qi=qi, policy=POLICY, exact_agents=sorted(EXACT), fg_priority=int(FG_PRIORITY), prefill_chunk=CHUNK, bg_thread=int(BG_THREAD), t_dec=t_dec,
                       rec_ok=rec["ok"], rec_ttft=rec["ttft"], rec_ngen=rec["ngen"], rec_e2e=rec["e2e"], rec_a3=rec["a3"],
                       reb_ok=reb["ok"], reb_ttft=reb["ttft"], reb_ngen=reb["ngen"], reb_e2e=reb["e2e"], reb_e2e_adj=reb["e2e_adj"],
                       overhead_ms=reb["overhead"], reb_a3=reb["a3"], mode=reb["mode"],
                       pf_solo=reb["pf_solo"], pf_bg=reb["pf_bg"], K=reb["K"], draft_ms=reb["draft_ms"], stall_ms=reb["stall_ms"],
                       replay_ms=reb["replay_ms"], step_ms=reb["step_ms"], agree=reb["agree"], first_agree=reb["first_agree"])
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"[s{SHARD}] {n+1}/{len(idxs)} ({time.time()-t0:.0f}s) q{qi} rec_ok={rec['ok']} reb_ok={reb['ok']} "
                  f"recTTFT={[round(x) for x in rec['ttft']]} rebTTFT={[round(x) if x else None for x in reb['ttft']]} "
                  f"K={reb['K']} stall={[round(x,1) if x is not None else None for x in reb['stall_ms']]} replay={[round(x,1) if x is not None else None for x in reb['replay_ms']]}", flush=True)
        except Exception as e:
            import traceback; print(f"[s{SHARD}] q{qi} ERR {e}", flush=True); traceback.print_exc()
    print(f"[s{SHARD}] DONE", flush=True)

if __name__ == "__main__":
    main()
