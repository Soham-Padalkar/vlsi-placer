"""Run every experiment in the README and regenerate its figures.

    python -m placer.experiments            # a few minutes of pure-Python annealing
    python -m placer.experiments --quick    # 2 seeds, smaller budgets

Quality is reported as HPWL divided by the known optimum of each benchmark
(1.0 is optimal). Seed s is used for both the benchmark and the annealer.
Outputs go to docs/figures/ (PNG figures and results.json).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import plotstyle as ps
from .anneal import AnnealConfig, anneal
from .benchmark import planted
from .netlist import Netlist, hpwl, random_placement
from .spectral import spectral_placement

OUT = Path("docs/figures")
COLD = 0.1  # initial temperature scale used when refining an existing placement


def _stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0}


def _ratio(netlist: Netlist, slots) -> float:
    return hpwl(netlist, slots) / netlist.optimal_hpwl


def _methods(budget: int) -> dict:
    """name -> function(netlist, seed) returning (placement, HPWL history or None)."""

    def sa(nl, s, **kw):
        r = anneal(nl, s, AnnealConfig(moves_per_cell=budget, **kw))
        return r.slot_of, r.history

    def hybrid(nl, s, t=COLD, **kw):
        r = anneal(
            nl, s, AnnealConfig(moves_per_cell=budget, t_start_scale=t, **kw), initial=spectral_placement(nl)
        )
        return r.slot_of, r.history

    return {
        "random": lambda nl, s: (random_placement(nl, np.random.default_rng(s)), None),
        "spectral": lambda nl, s: (spectral_placement(nl), None),
        "annealing": sa,
        "spectral + annealing": hybrid,
        "annealing, no range limit": lambda nl, s: sa(nl, s, range_limit=False),
        "spectral + annealing, hot start": lambda nl, s: hybrid(nl, s, t=1.0),
    }


def run(seeds: list[int], budget: int, quick: bool) -> dict:
    res: dict = {"seeds": seeds, "budget_moves_per_cell": budget}
    size = 16
    bench = {s: planted(size, size, s)[0] for s in seeds}
    methods = _methods(budget)

    # 1 + 2: main comparison and ablations on 16 x 16
    res["comparison"], res["history"] = {}, {}
    for name, fn in methods.items():
        ratios, times, hist = [], [], None
        for s in seeds:
            t0 = time.perf_counter()
            slots, history = fn(bench[s], s)
            times.append(time.perf_counter() - t0)
            ratios.append(_ratio(bench[s], slots))
            if s == seeds[0] and history:
                hist = [(m / bench[s].n_cells, h / bench[s].optimal_hpwl) for m, h in history]
        res["comparison"][name] = {**_stats(ratios), "seconds": float(np.mean(times))}
        if hist:
            res["history"][name] = hist
        print(f"{name:34s} {res['comparison'][name]['mean']:.3f} x optimal  ({np.mean(times):.2f} s)")

    # 3: sensitivity to the move budget
    budgets = [30, 100, 300, 1000, 3000] if quick else [30, 100, 300, 1000, 3000, 10000]
    res["budget_sweep"] = {"budgets": budgets, "annealing": [], "spectral + annealing": []}
    for b in budgets:
        m = _methods(b)
        for name in ("annealing", "spectral + annealing"):
            res["budget_sweep"][name].append(
                _stats([_ratio(bench[s], m[name](bench[s], s)[0]) for s in seeds])
            )

    # 4: sensitivity to the initial temperature, for both starting points
    scales = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    res["temperature_sweep"] = {"scales": scales, "random start": [], "spectral start": []}
    for t in scales:
        cfg = AnnealConfig(moves_per_cell=budget, t_start_scale=t)
        res["temperature_sweep"]["random start"].append(
            _stats([_ratio(bench[s], anneal(bench[s], s, cfg).slot_of) for s in seeds])
        )
        res["temperature_sweep"]["spectral start"].append(
            _stats(
                [
                    _ratio(bench[s], anneal(bench[s], s, cfg, spectral_placement(bench[s])).slot_of)
                    for s in seeds
                ]
            )
        )

    # 5: scaling with problem size at a fixed budget per cell
    sizes = [8, 12, 16, 24] if quick else [8, 12, 16, 24, 32]
    res["scaling"] = {"sizes": sizes, "spectral": [], "annealing": [], "spectral + annealing": []}
    for n in sizes:
        nls = {s: planted(n, n, s)[0] for s in seeds}
        for name in ("spectral", "annealing", "spectral + annealing"):
            res["scaling"][name].append(_stats([_ratio(nls[s], methods[name](nls[s], s)[0]) for s in seeds]))
    return res


# ------------------------------------------------------------------- figures


def _draw_layout(ax, netlist: Netlist, slots, title: str) -> None:
    rows, cols = netlist.rows, netlist.cols
    r, c = slots // cols, slots % cols
    for net in netlist.nets:
        pins = list(net)
        cr, cc = r[pins].mean(), c[pins].mean()
        for p in pins:
            ax.plot([cc, c[p]], [cr, r[p]], color=ps.BLUE, linewidth=0.7, alpha=0.45, solid_capstyle="round")
    ax.scatter(c, r, s=7, color=ps.INK, zorder=3, linewidths=0)
    ax.set(xlim=(-0.7, cols - 0.3), ylim=(-0.7, rows - 0.3), aspect="equal", xticks=[], yticks=[])
    ax.set_title(f"{title}\n{_ratio(netlist, slots):.2f} x optimal", fontsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)


def plot(res: dict) -> None:
    ps.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    budget = res["budget_moves_per_cell"]
    seed = res["seeds"][0]

    netlist, optimal = planted(16, 16, seed)
    spec = spectral_placement(netlist)
    hybrid = anneal(
        netlist, seed, AnnealConfig(moves_per_cell=budget, t_start_scale=COLD), initial=spec
    ).slot_of
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7))
    layouts = (
        ("Random", random_placement(netlist, np.random.default_rng(seed))),
        ("Spectral", spec),
        ("Spectral + annealing", hybrid),
        ("Constructed optimum", optimal),
    )
    for ax, (title, slots) in zip(axes, layouts):
        _draw_layout(ax, netlist, slots, title)
    fig.savefig(OUT / "layouts.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), width_ratios=(1.15, 1))
    names = list(res["comparison"])[::-1]
    mean = [res["comparison"][n]["mean"] for n in names]
    std = [res["comparison"][n]["std"] for n in names]
    axes[0].barh(
        names, mean, xerr=std, color=ps.BLUE, height=0.55, error_kw={"ecolor": ps.MUTED, "elinewidth": 1.2}
    )
    axes[0].axvline(1.0, color=ps.MUTED, linewidth=1, linestyle="--")
    for y, (m, sd) in enumerate(zip(mean, std)):
        axes[0].text((m + sd) * 1.05, y, f"{m:.2f}", va="center", fontsize=9, color=ps.MUTED)
    axes[0].set(xscale="log", xlim=(0.9, 14), xlabel="HPWL / optimal (log scale; dashed line = optimal)",
                title=f"16 x 16 grid, {budget} moves per cell")  # fmt: skip
    axes[0].grid(axis="y", visible=False)
    for color, name in zip(ps.SERIES, res["history"]):
        h = np.array(res["history"][name])
        axes[1].plot(h[:, 0], h[:, 1], color=color, label=name)
    axes[1].set(
        yscale="log",
        xlabel="moves per cell",
        ylabel="HPWL / optimal",
        title="Annealing trajectories (one seed)",
    )
    axes[1].legend()
    fig.savefig(OUT / "comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = (
        (axes[0], "budget_sweep", "budgets", "moves per cell", "Sensitivity to move budget", True),
        (axes[1], "temperature_sweep", "scales", "initial temperature (x std of random-move deltas)",
         "Sensitivity to initial temperature", True),
        (axes[2], "scaling", "sizes", "grid side (cells = side²)", "Scaling with problem size", False),
    )  # fmt: skip
    for ax, key, xkey, xlabel, title, logx in panels:
        x = res[key][xkey]
        for color, name in zip(ps.SERIES, [k for k in res[key] if k != xkey]):
            m = np.array([s["mean"] for s in res[key][name]])
            sd = np.array([s["std"] for s in res[key][name]])
            ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, m, "o-", color=color, label=name, markeredgecolor=ps.SURFACE, markeredgewidth=1.2)
        ax.axhline(1.0, color=ps.MUTED, linewidth=1, linestyle="--")
        ax.set(xlabel=xlabel, ylabel="HPWL / optimal", title=title, xscale="log" if logx else "linear")
        if not logx:
            ax.set_xticks(x)
        ax.set_ylim(bottom=0.9)
        ax.legend()
    fig.savefig(OUT / "sensitivity.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="2 seeds and a smaller budget")
    parser.add_argument("--plot-only", action="store_true", help="re-draw figures from results.json")
    args = parser.parse_args()
    path = OUT / "results.json"
    if args.plot_only:
        res = json.loads(path.read_text())
    else:
        res = run(
            [0, 1] if args.quick else [0, 1, 2, 3, 4], budget=300 if args.quick else 2000, quick=args.quick
        )
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=2) + "\n")
    plot(res)


if __name__ == "__main__":
    main()
