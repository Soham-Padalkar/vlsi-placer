"""Simulated-annealing placement by pairwise swaps with incremental HPWL.

Each move swaps the cells in two slots. Only nets touching the two cells are
re-evaluated, so a move costs O(pins on those nets) instead of O(all nets).

Two standard refinements are switchable so their contribution can be measured:

* ``range_limit``: the second slot is drawn from a window around the first
  whose radius shrinks with the temperature, so late moves are local.
* ``initial``: start from a given placement (e.g. spectral) rather than random.

Plain Python lists are used in the inner loop; they are several times faster
than NumPy scalars for this kind of pointer-chasing code.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .netlist import Netlist


@dataclass(frozen=True)
class AnnealConfig:
    moves_per_cell: int = 1000  # total move budget = moves_per_cell * n_cells
    n_temperatures: int = 100
    t_start_scale: float = 1.0  # initial temperature in units of the std of random-move deltas
    t_end_ratio: float = 1e-3  # final temperature relative to the initial one
    range_limit: bool = True


@dataclass
class AnnealResult:
    slot_of: np.ndarray
    hpwl: int
    history: list[tuple[int, int]] = field(default_factory=list)  # (moves done, HPWL)
    accepted: int = 0


class _State:
    """Mutable placement with cached per-net HPWL."""

    def __init__(self, netlist: Netlist, slot_of: np.ndarray):
        self.cols = netlist.cols
        self.nets = [list(net) for net in netlist.nets]
        self.cell_nets = [list(n) for n in netlist.cell_nets]
        self.row = [int(s) // netlist.cols for s in slot_of]
        self.col = [int(s) % netlist.cols for s in slot_of]
        self.cell_at = [0] * netlist.n_cells
        for cell, slot in enumerate(slot_of):
            self.cell_at[int(slot)] = cell
        self.net_hpwl = [self._net_hpwl(i) for i in range(len(self.nets))]
        self.total = sum(self.net_hpwl)

    def _net_hpwl(self, i: int) -> int:
        row, col = self.row, self.col
        pins = self.nets[i]
        r = [row[p] for p in pins]
        c = [col[p] for p in pins]
        return max(r) - min(r) + max(c) - min(c)

    def _swap_positions(self, a: int, b: int) -> None:
        self.row[a], self.row[b] = self.row[b], self.row[a]
        self.col[a], self.col[b] = self.col[b], self.col[a]

    def try_swap(self, a: int, b: int) -> tuple[int, list[tuple[int, int]]]:
        """Swap cells a and b; return (delta HPWL, updates to commit or discard)."""
        self._swap_positions(a, b)
        delta, updates = 0, []
        for i in set(self.cell_nets[a] + self.cell_nets[b]):
            new = self._net_hpwl(i)
            delta += new - self.net_hpwl[i]
            updates.append((i, new))
        return delta, updates

    def commit(self, a: int, b: int, delta: int, updates) -> None:
        for i, new in updates:
            self.net_hpwl[i] = new
        self.total += delta
        sa, sb = self.row[a] * self.cols + self.col[a], self.row[b] * self.cols + self.col[b]
        self.cell_at[sa], self.cell_at[sb] = a, b

    def revert(self, a: int, b: int) -> None:
        self._swap_positions(a, b)

    def slot_of(self) -> np.ndarray:
        return np.array([r * self.cols + c for r, c in zip(self.row, self.col)], dtype=np.int64)


def anneal(
    netlist: Netlist,
    seed: int,
    config: AnnealConfig | None = None,
    initial: np.ndarray | None = None,
) -> AnnealResult:
    """Anneal from ``initial`` (or a random placement) and return the best placement seen."""
    cfg = config or AnnealConfig()
    rng = random.Random(seed)
    n, rows, cols = netlist.n_cells, netlist.rows, netlist.cols
    if initial is None:
        initial = np.random.default_rng(seed).permutation(n)
    state = _State(netlist, initial)

    # temperature scale: spread of cost changes for random swaps at the starting point
    deltas = []
    for _ in range(min(200, 4 * n)):
        a, b = rng.sample(range(n), 2)
        deltas.append(state.try_swap(a, b)[0])
        state.revert(a, b)
    spread = float(np.std(deltas)) or 1.0
    t_start = cfg.t_start_scale * spread
    alpha = cfg.t_end_ratio ** (1.0 / max(1, cfg.n_temperatures - 1))
    moves_per_temp = max(1, cfg.moves_per_cell * n // cfg.n_temperatures)
    max_radius = max(rows, cols)

    best, best_slots = state.total, state.slot_of()
    result = AnnealResult(best_slots, best, history=[(0, state.total)])
    temperature, moves = t_start, 0
    for step in range(cfg.n_temperatures):
        radius = max(1, round(max_radius * alpha**step)) if cfg.range_limit else max_radius
        for _ in range(moves_per_temp):
            a = rng.randrange(n)
            ra, ca = state.row[a], state.col[a]
            rb = rng.randint(max(0, ra - radius), min(rows - 1, ra + radius))
            cb = rng.randint(max(0, ca - radius), min(cols - 1, ca + radius))
            b = state.cell_at[rb * cols + cb]
            if a == b:
                continue
            delta, updates = state.try_swap(a, b)
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                state.commit(a, b, delta, updates)
                result.accepted += 1
            else:
                state.revert(a, b)
        moves += moves_per_temp
        if state.total < best:
            best, best_slots = state.total, state.slot_of()
        result.history.append((moves, state.total))
        temperature *= alpha
    result.slot_of, result.hpwl = best_slots, best
    return result
