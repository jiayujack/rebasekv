"""Shared style for the RebaseKV camera-ready figures.

Semantic colors (validated for CVD separation with the dataviz palette validator):
  inherited / stale KV  -> blue      optimistic draft -> amber
  exact prefill         -> vermilion exact / fresh continuation -> teal green
Every colored box carries a text label, so identity never relies on color alone.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {
    "stale": dict(edge="#257de0", fill="#e4effc", ink="#1b5fb0"),
    "draft": dict(edge="#e09a00", fill="#fff2d1", ink="#8a5a00"),
    "prefill": dict(edge="#e2541b", fill="#fde5db", ink="#a83a0c"),
    "fresh": dict(edge="#00a07a", fill="#d8f3ea", ink="#006b51"),
    "neutral": dict(edge="#9aa3ad", fill="#f3f4f6", ink="#374151"),
    "panel": dict(edge="#d5d9de", fill="#fbfbfc", ink="#111827"),
}
INK, INK2, GRID = "#111827", "#4b5563", "#e5e7eb"


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    })


def box(ax, x, y, w, h, kind, text="", size=7.2, weight="normal", ls="-", lw=0.9, pad=0.25,
        rounding=0.9, color=None, fill=None, text_color=None, zorder=2):
    c = C[kind]
    p = FancyBboxPatch((x + pad, y + pad), w - 2 * pad, h - 2 * pad,
                       boxstyle=f"round,pad={pad},rounding_size={rounding}",
                       facecolor=fill or c["fill"], edgecolor=color or c["edge"], linewidth=lw,
                       linestyle=ls, zorder=zorder)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                color=text_color or (c["ink"] if kind != "neutral" else INK), fontweight=weight,
                zorder=zorder + 1, linespacing=1.15)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.8, ls="-", style="-|>", ms=6, zorder=4, rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color,
                        linewidth=lw, linestyle=ls, zorder=zorder,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def bracket(ax, x0, x1, y, text, color=INK2, size=6.8, down=True, tick=0.8, text_gap=0.6):
    s = -1 if down else 1
    ax.plot([x0, x0, x1, x1], [y - s * tick, y, y, y - s * tick], color=color, lw=0.7,
            solid_capstyle="butt", zorder=4)
    ax.text((x0 + x1) / 2, y + s * text_gap, text, ha="center", va="bottom" if not down else "top",
            fontsize=size, color=color)


def canvas(w_in, h_in, xmax, ymax):
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax
