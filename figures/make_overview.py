"""Figure 1: Optimistic Cache Alignment overview (full text width).
Usage: python make_overview.py figures/rebasekv_overview.pdf
"""
import sys
from figstyle import setup, canvas, box, arrow, bracket, C, INK, INK2

setup()
W, H = 100.0, 54.0
fig, ax = canvas(6.3, 6.3 * H / W, W, H)


def panel(x, y, w, h, title):
    box(ax, x, y, w, h, "panel", pad=0.0, rounding=1.2, lw=0.8)
    ax.text(x + 1.6, y + h - 2.1, title, ha="left", va="center", fontsize=8.2, fontweight="bold", color=INK)


# ---------------------------------------------------------------- (A)
panel(0.4, 21.0, 41.2, 32.6, "(A) Role prefixes break exact prefix caching")
ya, yb, hh = 41.6, 32.4, 5.2
for y, who in ((ya, r"Agent $m{-}1$"), (yb, r"Agent $m$")):
    ax.text(2.0, y + hh / 2, who, ha="left", va="center", fontsize=7.4, color=INK)
box(ax, 10.5, ya, 6.6, hh, "neutral", r"$p_{m-1}$", size=7.6)
box(ax, 17.1, ya, 18.0, hh, "stale", r"shared context $c,\,q,\,o_{<m-1}$", size=7.2)
box(ax, 10.5, yb, 6.6, hh, "prefill", r"$p_m$", size=7.6)
box(ax, 17.1, yb, 18.0, hh, "stale", r"shared context $c,\,q,\,o_{<m-1}$", size=7.2)
box(ax, 35.1, yb, 5.6, hh, "neutral", r"$o_{m-1}$", size=7.4, ls=(0, (2, 1.5)))
ax.text(26.1, (ya + yb + hh) / 2, "same tokens, shifted positions,\nnew preceding role",
        ha="center", va="center", fontsize=6.6, color=C["prefill"]["ink"], style="italic", linespacing=1.1)
arrow(ax, 13.8, ya - 0.2, 13.8, yb + hh + 0.2, color=C["prefill"]["edge"], lw=0.8, ms=5)
# outcomes
box(ax, 1.6, 22.4, 19.4, 8.0, "neutral", "", lw=0.7)
ax.text(11.3, 28.4, "Recompute", ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
ax.text(11.3, 25.1, "exact, but the first token\nwaits for the full prefill", ha="center", va="center",
        fontsize=6.6, color=INK2, linespacing=1.1)
box(ax, 21.4, 22.4, 19.4, 8.0, "neutral", "", lw=0.7)
ax.text(31.1, 28.4, "Stale reuse", ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
ax.text(31.1, 25.1, "fast, but the old role persists\n(code rate 100% $\\rightarrow$ 3%)", ha="center",
        va="center", fontsize=6.6, color=INK2, linespacing=1.1)

# ---------------------------------------------------------------- (B)
panel(42.6, 21.0, 57.0, 32.6, "(B) RebaseKV: draft on the inherited cache while the exact prefill runs")
import matplotlib.patches as mpatches
x0 = 52.6
lane_fg, lane_bg, lh = 38.6, 29.4, 5.4
ax.text(44.2, lane_fg + lh / 2, "Foreground", ha="left", va="center", fontsize=7.4, color=INK)
ax.text(44.2, lane_bg + lh / 2, "Background", ha="left", va="center", fontsize=7.4, color=INK)
box(ax, x0, lane_fg, 7.6, lh, "stale", "inherited\nKV", size=6.8)
cells = [r"$y_1$", r"$y_2$", r"$\cdots$", r"$y_{K-1}$", r"$y_K$"]
cx, cw = x0 + 7.9, 3.55
for i, t in enumerate(cells):
    box(ax, cx + i * cw, lane_fg, cw, lh, "draft", t, size=7.2, pad=0.18, rounding=0.5)
xr = cx + 5 * cw + 0.3
box(ax, xr, lane_fg, 10.0, lh, "fresh", "replay $y_{1:K}$\non $\\mathrm{KV}(s_m)$", size=6.7)
box(ax, xr + 10.2, lane_fg, 99.0 - xr - 10.2, lh, "fresh", "continue on\nexact KV", size=6.7)
for x, n, lab in ((x0 + 0.6, "1", "bootstrap"), (cx + 2.4, "2", "optimistic draft"), (xr + 0.6, "3", "replay & continue")):
    ax.add_patch(mpatches.Circle((x + 1.0, 47.3), 1.05, facecolor=INK, edgecolor="none", zorder=5))
    ax.text(x + 1.0, 47.3, n, ha="center", va="center", fontsize=6.4, color="white", fontweight="bold", zorder=6)
    ax.text(x + 2.5, 47.3, lab, ha="left", va="center", fontsize=7.0, color=INK)
arrow(ax, x0, 45.3, 98.8, 45.3, color=INK2, lw=0.7, ms=5)
bracket(ax, cx + 0.2, cx + 5 * cw - 0.2, lane_fg - 0.9,
        r"$K=\max\!\left(2,\lceil t_{\mathrm{prefill}}/t_{\mathrm{decode}}\rceil\right)$", size=6.8)
box(ax, cx, lane_bg, 5 * cw, lh, "prefill", r"exact prefill $\rightarrow\mathrm{KV}(s_m)$", size=6.9)
arrow(ax, cx + 5 * cw - 0.3, lane_bg + lh / 2, xr + 4.0, lane_fg + 0.25, color=C["prefill"]["edge"], lw=0.8,
      ms=5, rad=0.25)
ax.text(xr + 5.6, lane_bg + 3.2, "fresh cache\nready", ha="left", va="center", fontsize=6.4,
        color=C["prefill"]["ink"], linespacing=1.05)
ys, hs = 22.4, 4.2
ax.text(44.2, ys + hs / 2, "Decoding\nstate", ha="left", va="center", fontsize=6.8, color=INK, linespacing=1.05)
box(ax, x0, ys, 7.6, hs, "stale", "inherited", size=6.5, ls=(0, (2, 1.2)), lw=0.7)
box(ax, cx, ys, 5 * cw, hs, "draft", "optimistic prefix", size=6.5, ls=(0, (2, 1.2)), lw=0.7)
box(ax, xr, ys, 99.0 - xr, hs, "fresh", "exact, long-lived", size=6.5, ls=(0, (2, 1.2)), lw=0.7)

# ---------------------------------------------------------------- (C)
panel(0.4, 0.4, 99.2, 19.6, "(C) The exact prefill leaves the first-token critical path")
r1, r2, rh = 9.8, 2.6, 4.4
ax.text(2.0, r1 + rh / 2, "Recompute", ha="left", va="center", fontsize=7.4, color=INK)
ax.text(2.0, r2 + rh / 2, "RebaseKV", ha="left", va="center", fontsize=7.4, color=INK, fontweight="bold")
u, xs = 0.165, 12.0
T_REC, T_REB, T_DEC, T_BOOT, WIN, REPLAY, K = 135, 46, 24, 5, 258, 28, 6
pf = T_REC * u
box(ax, xs, r1, pf - T_DEC * u, rh, "prefill", "exact prefill", size=6.8)
n_rec = 7
for i in range(n_rec):
    box(ax, xs + pf - T_DEC * u + i * T_DEC * u, r1, T_DEC * u, rh, "fresh", "", pad=0.12, rounding=0.35)
box(ax, xs, r2, T_BOOT * u, rh, "stale", "", pad=0.08, rounding=0.2)
step = (WIN - T_BOOT) / K * u
labs = [r"$y_1$", r"$y_2$", r"$y_3$", r"$\cdots$", r"$y_{K-1}$", r"$y_K$"]
for i in range(K):
    box(ax, xs + T_BOOT * u + i * step, r2, step, rh, "draft", labs[i], size=6.6, pad=0.12, rounding=0.35)
xrp = xs + WIN * u
box(ax, xrp, r2, REPLAY * u, rh, "fresh", "", pad=0.12, rounding=0.35, fill="#9fdcc8")
for i in range(2):
    box(ax, xrp + REPLAY * u + i * T_DEC * u, r2, T_DEC * u, rh, "fresh", "", pad=0.12, rounding=0.35)
ax.text(xrp + REPLAY * u / 2, r2 - 0.5, "replay", ha="center", va="top", fontsize=6.2, color=C["fresh"]["ink"])
ax.text(xs + 0.4, r2 - 0.5, "boot", ha="center", va="top", fontsize=6.2, color=C["stale"]["ink"])
for x, y, lab in ((xs + pf, r1 + rh, "first token: 135 ms"), (xs + T_REB * u, r2 + rh, "first token: 46 ms")):
    ax.plot(x, y + 0.75, marker="v", markersize=3.4, color=INK, zorder=6)
    ax.text(x + 0.8, y + 0.8, lab, ha="left", va="center", fontsize=6.6, color=INK)
xt = xs + (WIN + REPLAY + 2 * T_DEC) * u + 3.0
ax.text(xt, r1 + rh / 2, r"TTFT $\approx t_{\mathrm{prefill}}+t_{\mathrm{decode}}$", ha="left", va="center", fontsize=7.2, color=INK2)
ax.text(xt, r2 + rh / 2, r"TTFT $\approx t_{\mathrm{boot}}+t_{\mathrm{decode}}$  ($\mathbf{2.95\times}$ lower)", ha="left",
        va="center", fontsize=7.2, color=INK)
lx = 47.6
items = [("stale", "inherited KV"), ("draft", "optimistic draft"), ("prefill", "exact prefill"), ("fresh", "exact continuation")]
widths = [11.9, 13.0, 11.3, 14.8]
for (k, lab), wdt in zip(items, widths):
    box(ax, lx, 16.9, 2.2, 2.1, k, "", pad=0.1, rounding=0.35)
    ax.text(lx + 2.7, 17.95, lab, ha="left", va="center", fontsize=6.5, color=INK)
    lx += wdt

out = sys.argv[1] if len(sys.argv) > 1 else "rebasekv_overview.pdf"
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=250)
