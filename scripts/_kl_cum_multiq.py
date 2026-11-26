#!/usr/bin/env python3
"""Cumulative same-trial KL over MULTIPLE GSM8K questions, averaged over all
(question x agent x sample) trials. RoPE-corrected reuse vs fresh; teacher-forced on
a shared fresh-sampled trajectory (identical history each step). Full-vocab, top-5
union, top-10 union. Reports cumulative KL at t=1/3/5/10/20/40/60."""
import os, sys, json, statistics, argparse
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.spec_rebase_fused_pipeline as P
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ap = argparse.ArgumentParser(); ap.add_argument("--nq", type=int, default=12); ap.add_argument("--S", type=int, default=4)
args = ap.parse_args()
NQ, S, T = args.nq, args.S, 60
torch.manual_seed(0)
P.m_tok = AutoTokenizer.from_pretrained(P.MODEL)
P.m = AutoModelForCausalLM.from_pretrained(P.MODEL, dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa").eval()
P.NL = P.m.config.num_hidden_layers; P.NH = P.m.config.num_attention_heads
P.NKV = P.m.config.num_key_value_heads
P.HD = getattr(P.m.config, "head_dim", P.m.config.hidden_size // P.NH)
P.rotary = P.m.model.rotary_emb; P.NEG = torch.finfo(torch.bfloat16).min; P.SCALE = P.HD ** -0.5
torch.set_grad_enabled(False)
m, NL = P.m, P.NL

# Monkey-patch: handle DynamicCache in transformers 5.x
@torch.no_grad()
def _full_prefill_kv(ids):
    t = torch.tensor([ids], device="cuda")
    out = m(t, use_cache=True)
    kv = out.past_key_values
    if hasattr(kv, 'layers'):  # transformers 5.14+
        return [(kv.layers[li].keys, kv.layers[li].values) for li in range(NL)]
    if hasattr(kv, 'key_cache'):  # transformers 4.x
        return [(kv.key_cache[li], kv.value_cache[li]) for li in range(NL)]
    return [(kv[li][0], kv[li][1]) for li in range(NL)]
P.full_prefill_kv = _full_prefill_kv

# Fix build_ids for transformers 5.x (returns BatchEncoding instead of list)
def _build_ids(role, q, peers, proles, qi):
    msgs = [{"role": "system", "content": f"[task:{qi}] " + P.rsys(role)},
            {"role": "user", "content": P.build_user(role, q, peers, proles)}]
    return P.m_tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False)
P.build_ids = _build_ids

# Fix stack_cache
def _stack_cache(kv):
    if hasattr(kv, 'layers'):
        kv = [(kv.layers[i].keys, kv.layers[i].values) for i in range(len(kv.layers))]
    elif hasattr(kv, 'key_cache'):
        kv = [(kv.key_cache[i], kv.value_cache[i]) for i in range(len(kv.key_cache))]
    K = torch.stack([kv[li][0] for li in range(NL)], dim=0)
    V = torch.stack([kv[li][1] for li in range(NL)], dim=0)
    return K, V
P.stack_cache = _stack_cache

# Fix assemble_bootstrap: its internal model call also returns DynamicCache
_orig_assemble = P.assemble_bootstrap
@torch.no_grad()
def _assemble_bootstrap(ids, src, bankK, bankV, bank_origpos):
    from transformers.cache_utils import DynamicCache as DC
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    import time as _time
    L = len(ids); src = list(src); src[L - 1] = -1
    torch.cuda.synchronize(); t0 = _time.perf_counter()
    bK = torch.zeros(NL, 1, P.NKV, L, P.HD, device="cuda", dtype=torch.bfloat16)
    bV = torch.zeros_like(bK)
    matched_abs = [p for p in range(L) if src[p] != -1]
    new_abs = [p for p in range(L) if src[p] == -1]
    if matched_abs:
        m_abs = torch.tensor(matched_abs, device="cuda")
        m_idx = torch.tensor([src[p] for p in matched_abs], device="cuda")
        M = len(matched_abs)
        mK = bankK[:, :, :, m_idx, :]; mV = bankV[:, :, :, m_idx, :]
        delta = (m_abs - bank_origpos[m_idx]).view(1, M)
        cos, sin = P.rotary(mK, delta)
        mKr = apply_rotary_pos_emb(mK.view(NL, P.NKV, M, P.HD), mK.view(NL, P.NKV, M, P.HD), cos, sin)[1].view(NL, 1, P.NKV, M, P.HD)
        bK[:, :, :, m_abs, :] = mKr; bV[:, :, :, m_abs, :] = mV
    Nn = len(new_abs)
    n_abs = torch.tensor(new_abs, device="cuda")
    new_ids = torch.tensor([[ids[p] for p in new_abs]], device="cuda")
    is_suffix = (new_abs == list(range(L - Nn, L)))
    past = None; mask = None
    if matched_abs:
        M = len(matched_abs)
        past = DC()
        for li in range(NL):
            past.update(bK[li, :, :, m_abs, :].contiguous(), bV[li, :, :, m_abs, :].contiguous(), li)
        if not is_suffix:
            mask = torch.full((1, 1, Nn, M + Nn), P.NEG, device="cuda", dtype=torch.bfloat16)
            z = torch.tensor(0.0, dtype=torch.bfloat16, device="cuda")
            vis = (m_abs.view(1, 1, 1, M) <= n_abs.view(1, 1, Nn, 1))
            mask[:, :, :, :M] = torch.where(vis, z, mask[:, :, :, :M])
            tri = torch.tril(torch.ones(Nn, Nn, device="cuda")).bool().view(1, 1, Nn, Nn)
            mask[:, :, :, M:] = torch.where(tri, z, mask[:, :, :, M:])
    out = m(new_ids, past_key_values=past, attention_mask=mask, position_ids=n_abs.view(1, -1), use_cache=True)
    first_logits = out.logits[0, -1]
    nc_raw = out.past_key_values
    if hasattr(nc_raw, 'layers'):
        nc = [(nc_raw.layers[li].keys, nc_raw.layers[li].values) for li in range(len(nc_raw.layers))]
    elif hasattr(nc_raw, 'key_cache'):
        nc = [(nc_raw.key_cache[li], nc_raw.value_cache[li]) for li in range(len(nc_raw.key_cache))]
    else:
        nc = [(nc_raw[li][0], nc_raw[li][1]) for li in range(NL)]
    for li in range(NL):
        bK[li, :, :, n_abs, :] = nc[li][0][:, :, -Nn:, :]
        bV[li, :, :, n_abs, :] = nc[li][1][:, :, -Nn:, :]
    torch.cuda.synchronize(); ttft = (_time.perf_counter() - t0) * 1000
    stale_kv = [(bK[li].contiguous(), bV[li].contiguous()) for li in range(NL)]
    return stale_kv, int(first_logits.argmax()), ttft
P.assemble_bootstrap = _assemble_bootstrap
# ====== End monkey-patches ======

from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")


def cache(kv, upto):
    c = DynamicCache()
    for li in range(NL):
        c.update(kv[li][0][:, :, :upto, :].contiguous(), kv[li][1][:, :, :upto, :].contiguous(), li)
    return c


def kl_full(lF, lR):
    return float((lF.exp() * (lF - lR)).sum())


def kl_topk(lF, lR, k):
    U = torch.unique(torch.cat([lF.topk(k).indices, lR.topk(k).indices]))
    pf = lF[U].exp(); pr = lR[U].exp(); pf = pf / pf.sum(); pr = pr / pr.sum()
    return float((pf * (pf.log() - pr.log())).sum())


def one_shared(reuse_kv, fresh_kv, L, last_tok):
    cR = cache(reuse_kv, L - 1); cF = cache(fresh_kv, L - 1)
    cur = last_tok; pos = L - 1; full = []; t5 = []; t10 = []
    for t in range(T):
        oR = m(torch.tensor([[cur]], device="cuda"), past_key_values=cR,
               position_ids=torch.tensor([[pos]], device="cuda"), use_cache=True)
        oF = m(torch.tensor([[cur]], device="cuda"), past_key_values=cF,
               position_ids=torch.tensor([[pos]], device="cuda"), use_cache=True)
        lR = F.log_softmax(oR.logits[0, -1].float(), -1); lF = F.log_softmax(oF.logits[0, -1].float(), -1)
        full.append(kl_full(lF, lR)); t5.append(kl_topk(lF, lR, 5)); t10.append(kl_topk(lF, lR, 10))
        cur = int(torch.multinomial(lF.exp(), 1)); cR = oR.past_key_values; cF = oF.past_key_values; pos += 1
    return full, t5, t10


idxs = [1, 3, 5, 10, 20, 40, 60]
# accumulate cumulative curves over all (q, agent, sample) trials
acc = {"full": [], "top5": [], "top10": []}   # each = list of per-trial cumulative-curve
ntrials = 0
for qi in range(NQ):
    q = ds[qi]["question"]
    peers, proles = [], []
    bankK = bankV = bank_origpos = None; bank_ids = None
    for ai in range(P.N_AGENTS):
        role = P.ROLE_ORDER[ai]
        ids = P.build_ids(role, q, peers, proles, ai); L = len(ids)
        if ai >= 1:
            src = P.match_src(ids, bank_ids)
            reuse_kv, _f, _t = P.assemble_bootstrap(ids, src, bankK, bankV, bank_origpos)
            fresh_kv = P.full_prefill_kv(ids)
            for _ in range(S):
                f, a, b = one_shared(reuse_kv, fresh_kv, L, ids[-1])
                acc["full"].append([sum(f[:t]) for t in range(1, T + 1)])
                acc["top5"].append([sum(a[:t]) for t in range(1, T + 1)])
                acc["top10"].append([sum(b[:t]) for t in range(1, T + 1)])
                ntrials += 1
        fresh_full = P.full_prefill_kv(ids)
        pK, pV = P.stack_cache(fresh_full); bankK, bankV = pK, pV
        bank_ids = ids; bank_origpos = torch.arange(L, device="cuda")
        c = cache(fresh_full, L); cur = ids[-1]; gen = []
        for s in range(150):
            o = m(torch.tensor([[cur]], device="cuda"), past_key_values=c,
                  position_ids=torch.tensor([[L + s]], device="cuda"), use_cache=True)
            c = o.past_key_values; cur = int(o.logits[0, -1].argmax()); gen.append(cur)
            if cur == P.m_tok.eos_token_id:
                break
        peers.append(P.m_tok.decode(gen, skip_special_tokens=True)); proles.append(role)
    print(f"Q{qi} done, trials so far={ntrials}", flush=True)

out = {"nq": NQ, "S": S, "ntrials": ntrials, "idxs": idxs}
print(f"\n=== cumulative KL averaged over {ntrials} (question x agent x sample) trials ===")
for metric in ["full", "top5", "top10"]:
    curves = acc[metric]
    mean_at = [statistics.mean(c[t - 1] for c in curves) for t in idxs]
    std_at = [statistics.pstdev([c[t - 1] for c in curves]) for t in idxs]
    out[metric + "_mean"] = mean_at; out[metric + "_std"] = std_at
    print(f"{metric:6s} cum@{idxs} = " + str([round(x, 3) for x in mean_at]))
    print(f"       std      = " + str([round(x, 3) for x in std_at]))
json.dump(out, open(os.environ.get("KL_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "kl_cum_multiq_results.json")), "w"))
print("saved _kl_cum_multiq_results.json")
