"""Netlists, placements and wirelength for unit cells on a rows x cols grid.

A placement is a permutation: ``slot_of[cell]`` is the grid slot (row * cols +
col) that the cell occupies. Wirelength is the half-perimeter wirelength
(HPWL): for each net, the half perimeter of the bounding box of its pins.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Netlist:
    rows: int
    cols: int
    nets: tuple[tuple[int, ...], ...]  # each net lists the cells it connects
    optimal_hpwl: int | None = None  # known optimum, for constructed benchmarks

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    @cached_property
    def cell_nets(self) -> tuple[tuple[int, ...], ...]:
        """For each cell, the nets it belongs to."""
        table: list[list[int]] = [[] for _ in range(self.n_cells)]
        for i, net in enumerate(self.nets):
            for cell in net:
                table[cell].append(i)
        return tuple(tuple(t) for t in table)


def hpwl(netlist: Netlist, slot_of: np.ndarray) -> int:
    """Total half-perimeter wirelength of a placement."""
    total = 0
    for net in netlist.nets:
        slots = slot_of[list(net)]
        r, c = slots // netlist.cols, slots % netlist.cols
        total += int(r.max() - r.min() + c.max() - c.min())
    return total


def is_legal(netlist: Netlist, slot_of: np.ndarray) -> bool:
    """Every cell on its own slot."""
    return slot_of.shape == (netlist.n_cells,) and np.array_equal(
        np.sort(slot_of), np.arange(netlist.n_cells)
    )


def net_lower_bound(n_pins: int) -> int:
    """Smallest possible HPWL of a net with ``n_pins`` pins on distinct unit slots.

    The pins must fit in a w x h box with w * h >= n_pins; the bound is the
    smallest (w - 1) + (h - 1) over such boxes.
    """
    best = n_pins - 1
    for w in range(1, n_pins + 1):
        h = -(-n_pins // w)
        best = min(best, w + h - 2)
    return best


def lower_bound(netlist: Netlist) -> int:
    """Sum of per-net lower bounds: no placement can do better."""
    return sum(net_lower_bound(len(net)) for net in netlist.nets)


def random_placement(netlist: Netlist, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(netlist.n_cells)
