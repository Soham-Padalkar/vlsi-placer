"""Shared matplotlib style: one palette, recessive axes, used by every figure."""

from __future__ import annotations

import matplotlib.pyplot as plt

# categorical slots, always assigned in this order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)  # fmt: skip
SERIES = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def apply() -> None:
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": INK,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.family": "DejaVu Sans",
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.prop_cycle": plt.cycler(color=SERIES),
        }
    )
