"""Appendix figure: schematic handoff from agent m-1 to agent m for every method (full text width).
Usage: python make_schematic_timeline.py ../figures/rebasekv_schematic_timeline.pdf
"""
import sys
from figstyle import setup, canvas, box, arrow, C, INK, INK2, GRID

setup()
W, H = 100.0, 44.0
fig, ax = canvas(6.3, 6.3 * H / W, W, H)
RH = 4.4
rows = {"rec": 33.0, "stale": 26.4, "fg": 19.8, "bg": 13.6, "warm": 6.8}
XP0, XP1, XH = 12.0, 30.0, 31.0

# headers and handoff line
ax.text((XP0 + XP1) / 2, 40.6, r"agent $m{-}1$ decoding", ha="center", va="center", fontsize=7.4, color=INK, fontweight="bold")
ax.text((XH + 99) / 2, 40.6, r"agent $m$", ha="center", va="center", fontsize=7.4, color=INK, fontweight="bold")
ax.plot([XH, XH], [5.8, 39.4], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(XH, 39.2, "handoff", ha="center", va="bottom", fontsize=6.4, color=INK2,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
for y in (rows["stale"] + RH + 1.1, rows["fg"] + RH + 1.1, rows["warm"] + RH + 1.3):
    ax.plot([1, 99], [y, y], color=GRID, lw=0.5, zorder=0)

labels = {"rec": "Recompute", "stale": "Stale", "warm": "WarmupKV"}
for k, lab in labels.items():
    ax.text(1.0, rows[k] + RH / 2, lab, ha="left", va="center", fontsize=7.4, color=INK)
ax.text(1.0, (rows["fg"] + rows["bg"] + RH) / 2, "RebaseKV", ha="left", va="center", fontsize=7.4, color=INK, fontweight="bold")
ax.text(9.6, rows["fg"] + RH / 2, "FG", ha="right", va="center", fontsize=6.4, color=INK2)
ax.text(9.6, rows["bg"] + RH / 2, "BG", ha="right", va="center", fontsize=6.4, color=INK2)


def chain(y, segs, sz=6.7):
    for i, (x0, x1, kind, text, extra) in enumerate(segs):
        box(ax, x0, y, x1 - x0, RH, kind, text, size=sz, **extra)
        if i + 1 < len(segs) and segs[i + 1][0] - x1 > 1.0:
            arrow(ax, x1 + 0.1, y + RH / 2, segs[i + 1][0] - 0.1, y + RH / 2, ms=4.5, lw=0.7)


dashed = dict(ls=(0, (2.5, 1.5)))
replay_fill = dict(fill="#9fdcc8")
chain(rows["rec"], [(XP0, XP1, "neutral", "previous agent decodes", {}),
                    (XH + 1.2, 56, "prefill", r"blocking exact prefill of $s_m$", {}),
                    (56.5, 64.5, "fresh", "first token", {}),
                    (65.0, 99, "fresh", "standard decoding on exact KV", {})])
chain(rows["stale"], [(XP0, XP1, "neutral", "previous agent decodes", {}),
                      (XH + 1.2, 44.5, "stale", "reuse inherited KV", {}),
                      (45.0, 53.0, "stale", "first token", {}),
                      (53.5, 99, "stale", "entire response decoded from the inherited cache", dashed)])
chain(rows["fg"], [(XP0, XP1, "neutral", "previous agent decodes", {}),
                   (XH + 1.2, 41.0, "stale", "bootstrap", {}),
                   (41.5, 67.0, "draft", r"optimistic draft $y_{1:K}$ (first token here)", {}),
                   (67.5, 76.0, "fresh", r"replay $y_{1:K}$", replay_fill),
                   (76.5, 99, "fresh", "continue on exact KV", {})])
box(ax, 41.5, rows["bg"], 67.0 - 41.5, RH, "prefill", r"exact prefill $\rightarrow \mathrm{KV}(s_m)$", size=6.7)
arrow(ax, 67.0, rows["bg"] + RH / 2, 70.5, rows["fg"] - 0.1, color=C["prefill"]["edge"], lw=0.8, ms=5, rad=0.3)
ax.text(71.6, rows["bg"] + RH / 2, "fresh cache ready: replay and switch", ha="left", va="center", fontsize=6.5,
        color=C["prefill"]["ink"])
chain(rows["warm"], [(XP0, XP1, "prefill", r"prefill known part of $s_m$", {}),
                     (XH + 1.2, 44.5, "prefill", "residual prefill", {}),
                     (45.0, 53.0, "fresh", "first token", {}),
                     (53.5, 99, "fresh", "standard decoding on exact KV", {})])

# legend
lx = 12.0
for k, lab, wdt in (("neutral", "previous agent", 15.5), ("stale", "inherited KV", 15.0), ("draft", "optimistic draft", 16.5),
                    ("prefill", "exact prefill", 14.5), ("fresh", "exact continuation", 18.0)):
    box(ax, lx, 0.6, 2.4, 2.3, k, "", pad=0.1, rounding=0.35)
    ax.text(lx + 3.0, 1.75, lab, ha="left", va="center", fontsize=6.7, color=INK)
    lx += wdt

out = sys.argv[1] if len(sys.argv) > 1 else "rebasekv_schematic_timeline.pdf"
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=250)
