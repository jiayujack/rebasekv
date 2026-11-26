"""Measured per-agent handoff timeline (column width).

Medians over the four rebased agents of the serving-style two-stream run
(GSM8K, Qwen2.5-7B-Instruct, 200 questions, RTX A6000; rebuttal_draft/results_server/B_fixed_all_prio):
  Recompute TTFT 135 ms; RebaseKV TTFT 46 ms; exact prefill 137 ms alone -> 235 ms while the
  draft runs; draft window 258 ms (K ~ 6, ~41 ms/step in the window); replay 28 ms; t_dec 24 ms.
Usage: python make_measured_timeline.py ../figures/rebasekv_measured_timeline.pdf
"""
import sys
from figstyle import setup, canvas, box, arrow, C, INK, INK2, GRID

setup()
T_DEC, REC_TTFT, PF_SOLO, PF_BG = 24.0, 135.0, 137.0, 235.0
REB_TTFT, DRAFT_END, REPLAY, K, T_END = 46.0, 258.0, 28.0, 6, 335.0

W, H = 100.0, 59.0
fig, ax = canvas(3.03, 3.03 * H / W, W, H)
X0, XS = 19.0, (98.5 - 19.0) / T_END  # left margin, units per ms
tx = lambda t: X0 + t * XS
lane = {"rec": 40.0, "fg": 25.5, "bg": 13.0}
LH = 6.4

# grid + axis
for t in range(0, 351, 50):
    if t > T_END:
        break
    ax.plot([tx(t), tx(t)], [10.6, 49.5], color=GRID, lw=0.5, zorder=0)
    ax.text(tx(t), 10.0, f"{t}", ha="center", va="top", fontsize=6.6, color=INK2)
ax.text(tx(T_END / 2), 0.4, "time since handoff (ms)", ha="center", va="bottom", fontsize=6.8, color=INK2)

for key, lab, bold in (("rec", "Recompute", False), ("fg", "RebaseKV\nforeground", True), ("bg", "RebaseKV\nbackground", True)):
    ax.text(0.8, lane[key] + LH / 2, lab, ha="left", va="center", fontsize=7.0, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.05)

# Recompute: exact prefill blocks the first token, then fresh decoding.
y = lane["rec"]
box(ax, tx(0), y, tx(REC_TTFT - T_DEC) - tx(0), LH, "prefill", "exact prefill", size=6.8, pad=0.2, rounding=0.6)
t = REC_TTFT - T_DEC
while t + T_DEC <= T_END + 0.1:
    box(ax, tx(t), y, T_DEC * XS, LH, "fresh", "", pad=0.15, rounding=0.4)
    t += T_DEC

# RebaseKV foreground: optimistic draft, replay, fresh decoding.
y = lane["fg"]
step = (DRAFT_END - (REB_TTFT - (DRAFT_END - REB_TTFT) / (K - 1))) / K
t0 = DRAFT_END - K * step
box(ax, tx(0), y, tx(t0) - tx(0), LH, "stale", "", pad=0.12, rounding=0.3)
for i in range(K):
    box(ax, tx(t0 + i * step), y, step * XS, LH, "draft", "", pad=0.15, rounding=0.4)
ax.text(tx((t0 + DRAFT_END) / 2), y + LH / 2, f"optimistic draft, $K={K}$", ha="center", va="center",
        fontsize=6.7, color=C["draft"]["ink"], zorder=6,
        bbox=dict(boxstyle="round,pad=0.15", facecolor=C["draft"]["fill"], edgecolor="none"))
box(ax, tx(DRAFT_END), y, REPLAY * XS, LH, "fresh", "", pad=0.15, rounding=0.4, fill="#9fdcc8")
t = DRAFT_END + REPLAY
while t + T_DEC <= T_END + 0.1:
    box(ax, tx(t), y, T_DEC * XS, LH, "fresh", "", pad=0.15, rounding=0.4)
    t += T_DEC
ax.text(tx(DRAFT_END + REPLAY / 2), y - 0.6, "replay", ha="center", va="top", fontsize=6.3, color=C["fresh"]["ink"])

# RebaseKV background: exact prefill slowed by the concurrent draft.
y = lane["bg"]
box(ax, tx(0), y, PF_BG * XS, LH, "prefill", "", pad=0.2, rounding=0.6)
ax.plot([tx(PF_SOLO)] * 2, [y + 0.8, y + LH - 0.8], color=C["prefill"]["edge"], lw=0.8, ls=(0, (2, 1.4)), zorder=5)
ax.text(tx(PF_SOLO / 2), y + LH / 2, "exact prefill", ha="center", va="center", fontsize=6.8, color=C["prefill"]["ink"], zorder=6)
ax.text(tx(PF_BG) + 1.2, y + LH / 2, f"{PF_SOLO:.0f} ms alone\n{PF_BG:.0f} ms concurrent", ha="left", va="center",
        fontsize=6.2, color=INK2, linespacing=1.05)
arrow(ax, tx(PF_BG), y + LH + 0.2, tx(DRAFT_END + 4), lane["fg"] - 0.2, color=C["prefill"]["edge"], lw=0.7, ms=4.5, rad=-0.2)

# First-token markers and the gain.
for t, key, lab in ((REC_TTFT, "rec", f"first token {REC_TTFT:.0f} ms"), (REB_TTFT, "fg", f"first token {REB_TTFT:.0f} ms")):
    yy = lane[key] + LH + 1.1
    ax.plot(tx(t), yy, marker="v", markersize=3.4, color=INK, zorder=7)
    ax.text(tx(t) + 1.2, yy + 0.2, lab, ha="left", va="center", fontsize=6.6, color=INK, fontweight="bold" if key == "fg" else "normal")
arrow(ax, tx(REC_TTFT), 53.5, tx(REB_TTFT), 53.5, color=INK, lw=0.8, style="<|-|>", ms=4.5)
ax.text(tx((REC_TTFT + REB_TTFT) / 2), 54.6, r"$\mathbf{2.95\times}$ earlier first token", ha="center", va="bottom",
        fontsize=6.9, color=INK)
ax.plot([tx(REC_TTFT)] * 2, [lane["rec"] + LH + 0.2, 53.5], color=INK2, lw=0.5, ls=(0, (1.5, 1.2)), zorder=3)
ax.plot([tx(REB_TTFT)] * 2, [lane["fg"] + LH + 2.4, 53.5], color=INK2, lw=0.5, ls=(0, (1.5, 1.2)), zorder=3)

out = sys.argv[1] if len(sys.argv) > 1 else "rebasekv_measured_timeline.pdf"
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=300)
