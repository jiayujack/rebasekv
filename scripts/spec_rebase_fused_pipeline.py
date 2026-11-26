#!/usr/bin/env python3
"""FULL hide-prefill flow on GSM8K Q0 (5-agent KVCOMM pipeline).

Per agent:
  1. BOOTSTRAP (reuse bank: prev prompt KV ++ prev output decode-KV; RoPE-shift
     matched spans; compute only new tokens) -> stale KV (length L) + first token.
     TTFT measured here.
  2. HIDE-PREFILL decode: for K_steps = ceil(L / C) steps, each step is ONE fused
     forward over [C fresh-prefill tokens + 1 decode token]: shared GEMMs, but TWO
     is_causal SDPA calls (decode row attends the stale/bootstrap KV; prefill rows
     attend the growing FRESH KV). So the full fresh recompute of the prompt is
     spread over — and hidden behind — the first K_steps stale decodes.
  3. REBASE: replay the K_steps drafted tokens through the now-complete FRESH KV,
     continue decoding from fresh KV to EOS.

Measures per agent: TTFT, fused-phase time vs (K plain decodes + separate full
prefill) [is the prefill hidden?], total, final answer.
Validated primitives: scripts/_fused_split_probe.py.
"""
import os, json, re, time, argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from vllm.vllm_flash_attn import flash_attn_varlen_func

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_AGENTS = 5
MAX_NEW = 400
CHUNK = 256
PROMPTS = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "_kvcomm_gsm8k_prompts.json")))
ROLE_ORDER = ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector", "__decision__"]
ROLE_DESC = PROMPTS["role_description"]; FEW_SHOT = PROMPTS["few_shot"]
DEC_C = PROMPTS["decision_constraint"]; DEC_FS = PROMPTS["decision_few_shot"]
m = None; NL = NH = NKV = HD = None; rotary = None; tok = None; NEG = None


def rsys(r):
    return DEC_C if r == "__decision__" else ROLE_DESC[r]


def rfs(r):
    return DEC_FS if r == "__decision__" else FEW_SHOT.get(r, "")


def build_user(role, q, peers, proles):
    fs = rfs(role)
    u = (f"{fs}\n\nQ: {q}\n" if fs else f"Q: {q}\n")
    if peers:
        s = ""
        for pi, (po, pr) in enumerate(zip(peers, proles)):
            prd = "Decision Maker" if pr == "__decision__" else pr
            s += f"Agent {pi+1} as a {prd} his answer to this question is:\n\n{po}\n\n"
        u += f"At the same time, there are the following responses to the same question for your reference:\n\n{s}\n\n"
    return u


def build_ids(role, q, peers, proles, qi):
    msgs = [{"role": "system", "content": f"[task:{qi}] " + rsys(role)},
            {"role": "user", "content": build_user(role, q, peers, proles)}]
    return m_tok.apply_chat_template(msgs, add_generation_prompt=True)


def _norm(s):
    if s is None:
        return None
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s); return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_answer(t):
    mm = re.findall(r"[Tt]he answer is\s*[:\s]*\$?([-+]?\d[\d,]*(?:\.\d+)?)", t)
    if mm:
        return _norm(mm[-1])
    n = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", t)
    return _norm(n[-1]) if n else None


def sync():
    torch.cuda.synchronize()


def match_src(ids, prev, minlen=8):
    from collections import defaultdict
    idx = defaultdict(list)
    for j in range(len(prev) - 2):
        idx[(prev[j], prev[j + 1], prev[j + 2])].append(j)
    src = [-1] * len(ids); i = 0; n = len(ids)
    while i < n - 2:
        best = 0; bestj = -1
        for j in idx.get((ids[i], ids[i + 1], ids[i + 2]), []):
            L = 0
            while i + L < n and j + L < len(prev) and ids[i + L] == prev[j + L]:
                L += 1
            if L > best:
                best = L; bestj = j
        if best >= minlen:
            for d in range(best):
                src[i + d] = bestj + d
            i += best
        else:
            i += 1
    return src


# ---------- bootstrap: returns stale KV as per-layer (K,V) list + first token ----------
@torch.no_grad()
def assemble_bootstrap(ids, src, bankK, bankV, bank_origpos):
    L = len(ids); src = list(src); src[L - 1] = -1
    sync(); t0 = time.perf_counter()
    bK = torch.zeros(NL, 1, NKV, L, HD, device="cuda", dtype=torch.bfloat16)
    bV = torch.zeros_like(bK)
    matched_abs = [p for p in range(L) if src[p] != -1]
    new_abs = [p for p in range(L) if src[p] == -1]
    if matched_abs:
        m_abs = torch.tensor(matched_abs, device="cuda")
        m_idx = torch.tensor([src[p] for p in matched_abs], device="cuda")
        M = len(matched_abs)
        mK = bankK[:, :, :, m_idx, :]; mV = bankV[:, :, :, m_idx, :]
        delta = (m_abs - bank_origpos[m_idx]).view(1, M)
        cos, sin = rotary(mK, delta)
        mKr = apply_rotary_pos_emb(mK.view(NL, NKV, M, HD), mK.view(NL, NKV, M, HD), cos, sin)[1].view(NL, 1, NKV, M, HD)
        bK[:, :, :, m_abs, :] = mKr; bV[:, :, :, m_abs, :] = mV
    Nn = len(new_abs)
    n_abs = torch.tensor(new_abs, device="cuda")
    new_ids = torch.tensor([[ids[p] for p in new_abs]], device="cuda")
    is_suffix = (new_abs == list(range(L - Nn, L)))
    past = None; mask = None
    if matched_abs:
        M = len(matched_abs)
        past = DynamicCache()
        for li in range(NL):
            past.update(bK[li, :, :, m_abs, :].contiguous(), bV[li, :, :, m_abs, :].contiguous(), li)
        if not is_suffix:
            mask = torch.full((1, 1, Nn, M + Nn), NEG, device="cuda", dtype=torch.bfloat16)
            z = torch.tensor(0.0, dtype=torch.bfloat16, device="cuda")
            vis = (m_abs.view(1, 1, 1, M) <= n_abs.view(1, 1, Nn, 1))
            mask[:, :, :, :M] = torch.where(vis, z, mask[:, :, :, :M])
            tri = torch.tril(torch.ones(Nn, Nn, device="cuda")).bool().view(1, 1, Nn, Nn)
            mask[:, :, :, M:] = torch.where(tri, z, mask[:, :, :, M:])
    out = m(new_ids, past_key_values=past, attention_mask=mask, position_ids=n_abs.view(1, -1), use_cache=True)
    first_logits = out.logits[0, -1]
    nc = out.past_key_values
    for li in range(NL):
        bK[li, :, :, n_abs, :] = nc[li][0][:, :, -Nn:, :]
        bV[li, :, :, n_abs, :] = nc[li][1][:, :, -Nn:, :]
    sync(); ttft = (time.perf_counter() - t0) * 1000
    stale_kv = [(bK[li].contiguous(), bV[li].contiguous()) for li in range(NL)]
    return stale_kv, int(first_logits.argmax()), ttft


# ---------- fused step: 1 decode (stale KV) + C prefill (fresh KV) ----------
# Prefill chunk attention uses FLASH varlen (causal, BOTTOM-RIGHT aligned: query i
# attends fresh[0:Sf] + chunk[0:i]) -> no dense mask, no LSE merge. Decode row is a
# single query over stale KV (is_causal=False SDPA, fast path). Validated:
# scripts/_flash_prefill_probe.py (flash causal == correct dense-mask, diff 0.002).
SCALE = None  # set in main once HD known
@torch.no_grad()
def fused_step(stale_kv, fresh_kv, dec_tok, chunk_ids, S_dec, S_fresh):
    C = chunk_ids.shape[1]
    ids = torch.cat([chunk_ids, dec_tok], dim=1)
    h = m.model.embed_tokens(ids)
    pos = torch.tensor([list(range(S_fresh, S_fresh + C)) + [S_dec]], device="cuda")
    cos, sin = rotary(h, pos)
    cu_q = torch.tensor([0, C], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, S_fresh + C], device="cuda", dtype=torch.int32)
    new_fresh = []; new_stale = []
    for li, layer in enumerate(m.model.layers):
        res = h; x = layer.input_layernorm(h)
        q = layer.self_attn.q_proj(x).view(1, C + 1, NH, HD).transpose(1, 2)
        k = layer.self_attn.k_proj(x).view(1, C + 1, NKV, HD).transpose(1, 2)
        v = layer.self_attn.v_proj(x).view(1, C + 1, NKV, HD).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q_pre, k_pre, v_pre = q[:, :, :C], k[:, :, :C], v[:, :, :C]
        q_dec, k_dec, v_dec = q[:, :, C:], k[:, :, C:], v[:, :, C:]
        kp = torch.cat([fresh_kv[li][0], k_pre], dim=2); vp = torch.cat([fresh_kv[li][1], v_pre], dim=2)
        # FLASH prefill: q (C,NH,HD), k/v (Sf+C,NKV,HD), causal bottom-right aligned
        qf = q_pre.squeeze(0).transpose(0, 1).contiguous()
        kf = kp.squeeze(0).transpose(0, 1).contiguous()
        vf = vp.squeeze(0).transpose(0, 1).contiguous()
        o_pre = flash_attn_varlen_func(qf, kf, vf, max_seqlen_q=C, cu_seqlens_q=cu_q,
                                       max_seqlen_k=S_fresh + C, cu_seqlens_k=cu_k,
                                       softmax_scale=SCALE, causal=True)  # (C, NH, HD)
        o_pre = o_pre.transpose(0, 1).unsqueeze(0)  # (1, NH, C, HD)
        kd = torch.cat([stale_kv[li][0], k_dec], dim=2); vd = torch.cat([stale_kv[li][1], v_dec], dim=2)
        o_dec = F.scaled_dot_product_attention(q_dec, kd, vd, is_causal=False, enable_gqa=True)
        o = torch.cat([o_pre, o_dec], dim=2).transpose(1, 2).reshape(1, C + 1, NH * HD)
        h = res + layer.self_attn.o_proj(o)
        res = h; x = layer.post_attention_layernorm(h)
        h = res + layer.mlp(x)
        new_fresh.append((kp, vp)); new_stale.append((kd, vd))
    h = m.model.norm(h)
    dec_logits = m.lm_head(h)[0, -1]
    return int(dec_logits.argmax()), new_fresh, new_stale


@torch.no_grad()
def decode_step_plain(kv, tok_id, pos):
    """1 plain decode step from kv (per-layer list). Returns next id + updated kv."""
    c = DynamicCache()
    for li in range(NL):
        c.update(kv[li][0], kv[li][1], li)
    out = m(torch.tensor([[tok_id]], device="cuda"), past_key_values=c,
            position_ids=torch.tensor([[pos]], device="cuda"), use_cache=True)
    nc = out.past_key_values
    nkv = [(nc[li][0], nc[li][1]) for li in range(NL)]
    return int(out.logits[0, -1].argmax()), nkv


@torch.no_grad()
def full_prefill_kv(ids):
    t = torch.tensor([ids], device="cuda")
    out = m(t, use_cache=True)
    return [(out.past_key_values[li][0], out.past_key_values[li][1]) for li in range(NL)]


@torch.no_grad()
def stack_cache(kv):
    K = torch.stack([kv[li][0] for li in range(NL)], dim=0)
    V = torch.stack([kv[li][1] for li in range(NL)], dim=0)
    return K, V


def main():
    global m, m_tok, NL, NH, NKV, HD, rotary, NEG, SCALE
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()
    m_tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    NL = m.config.num_hidden_layers; NH = m.config.num_attention_heads
    NKV = m.config.num_key_value_heads; HD = getattr(m.config, "head_dim", m.config.hidden_size // NH)
    rotary = m.model.rotary_emb; NEG = torch.finfo(torch.bfloat16).min
    SCALE = HD ** -0.5
    torch.set_grad_enabled(False)
    for _ in range(2):
        full_prefill_kv(list(range(1, 1500)))
    sync()
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    q = ds[args.qid]["question"]; gold = _norm(ds[args.qid]["answer"].split("####")[-1].strip())
    print(f"Q{args.qid} gold={gold} chunk={args.chunk}\n" + "=" * 78)

    bankK = bankV = bank_origpos = None; bank_ids = None
    peers = []; proles = []
    C = args.chunk
    for ai in range(N_AGENTS):
        role = ROLE_ORDER[ai]
        ids = build_ids(role, q, peers, proles, ai)
        L = len(ids)
        src = [-1] * L if bank_ids is None else match_src(ids, bank_ids)
        matched = sum(1 for x in src if x != -1)
        # 1) bootstrap -> stale KV + first token
        stale_kv, first_tok, ttft = assemble_bootstrap(ids, src, bankK, bankV, bank_origpos)
        # 2) hide-prefill: K_steps fused steps (decode from stale + prefill fresh chunks)
        K_steps = (L + C - 1) // C
        draft = [first_tok]
        fresh_kv = [(torch.zeros(1, NKV, 0, HD, device="cuda", dtype=torch.bfloat16),
                     torch.zeros(1, NKV, 0, HD, device="cuda", dtype=torch.bfloat16)) for _ in range(NL)]
        sync(); tf = time.perf_counter()
        cur = first_tok; S_dec = L; S_fresh = 0
        for s in range(K_steps):
            chunk = torch.tensor([ids[s * C:(s + 1) * C]], device="cuda")
            cc = chunk.shape[1]
            nt, fresh_kv, stale_kv = fused_step(stale_kv, fresh_kv, torch.tensor([[cur]], device="cuda"),
                                                chunk, S_dec, S_fresh)
            draft.append(nt); cur = nt; S_dec += 1; S_fresh += cc
        sync(); fused_phase = (time.perf_counter() - tf) * 1000
        # DEBUG: verify fused-built fresh_kv == true full prefill
        if ai == 0:
            tk = full_prefill_kv(ids)
            fl = fresh_kv[0][0].shape[2]
            d0 = (fresh_kv[0][0].float() - tk[0][0][:, :, :fl, :].float()).abs().max().item()
            dL = (fresh_kv[NL - 1][0].float() - tk[NL - 1][0][:, :, :fl, :].float()).abs().max().item()
            scaleL = tk[NL - 1][0][:, :, :fl, :].float().abs().max().item()
            relL = dL / scaleL if scaleL > 0 else 0
            print(f"  [DBG] fresh_kv len={fl} vs L={L} | L0 K diff={d0:.4f} L{NL-1} K diff={dL:.4f} "
                  f"(|K|max={scaleL:.1f}, rel={relL*100:.2f}%)")
        # fresh_kv now == full fresh prefill of the prompt (length L)
        # 3) rebase: replay draft through fresh KV, continue to EOS
        sync(); tr = time.perf_counter()
        c = DynamicCache()
        for li in range(NL):
            c.update(fresh_kv[li][0], fresh_kv[li][1], li)
        di = torch.tensor([draft], device="cuda")
        pos = torch.arange(L, L + len(draft), device="cuda").view(1, -1)
        out = m(di, past_key_values=c, position_ids=pos, use_cache=True); c = out.past_key_values
        nt = int(out.logits[0, -1].argmax()); gen = list(draft) + [nt]; cur = nt; base = L + len(draft)
        for s in range(MAX_NEW - len(draft)):
            if nt == m_tok.eos_token_id:
                break
            out = m(torch.tensor([[cur]], device="cuda"), past_key_values=c,
                    position_ids=torch.tensor([[base + s]], device="cuda"), use_cache=True)
            c = out.past_key_values; nt = int(out.logits[0, -1].argmax()); gen.append(nt); cur = nt
        sync(); rebase_cont = (time.perf_counter() - tr) * 1000

        # measure floor (K plain decodes, happen anyway) and full prefill separately
        sync(); tp = time.perf_counter()
        _ = full_prefill_kv(ids)
        sync(); t_prefill = (time.perf_counter() - tp) * 1000
        kk = [(stale_kv[li][0][:, :, :L, :], stale_kv[li][1][:, :, :L, :]) for li in range(NL)]
        cur2 = draft[0]
        sync(); td0 = time.perf_counter()
        for s in range(K_steps):
            cur2, kk = decode_step_plain(kk, cur2, L + s)
        sync(); t_kdec = (time.perf_counter() - td0) * 1000
        separate = t_prefill + t_kdec
        hidden_pct = (separate - fused_phase) / t_prefill * 100 if t_prefill > 0 else 0

        text = m_tok.decode(gen, skip_special_tokens=True); pred = extract_answer(text)
        rd = "Decision Maker" if role == "__decision__" else role
        print(f"\n--- Agent {ai+1} ({rd}) L={L} reuse={matched}({matched/L*100:.0f}%) K_steps={K_steps} ---")
        print(f"  draft(first8)={draft[:8]}")
        print(f"  out[:160]={text[:160]!r}")
        print(f"  TTFT(bootstrap->1st)={ttft:.0f}ms | C={C} K_steps={K_steps} fused-phase={fused_phase:.0f}ms")
        print(f"  floor(K plain decodes)={t_kdec:.0f}ms  full-prefill={t_prefill:.0f}ms  separate={separate:.0f}ms")
        print(f"  -> prefill hidden%={hidden_pct:.0f}%  (fused vs floor: {fused_phase:.0f} vs {t_kdec:.0f})")
        print(f"  rebase+continue={rebase_cont:.0f}ms gen_len={len(gen)} pred={pred}")
        # bank for next agent
        fresh_full = full_prefill_kv(ids)
        pK, pV = stack_cache(fresh_full)
        # capture output decode-KV by decoding is skipped here; reuse prompt-only bank for simplicity
        bankK, bankV = pK, pV; bank_ids = ids; bank_origpos = torch.arange(L, device="cuda")
        peers.append(text); proles.append(role)
    print(f"\n{'='*78}\nFINAL pred={pred} gold={gold} -> {'CORRECT' if str(pred)==str(gold) else 'wrong'}")


if __name__ == "__main__":
    main()
