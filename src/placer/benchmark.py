"""Benchmarks with a known optimal wirelength.

Follows the idea of the PEKO suite (Chang, Cong, Xie, "Optimality and
scalability study of existing placement algorithms", ASP-DAC 2003): start
from a hidden placement and only create nets whose pins sit in the tightest
possible bounding box there. Every net is then at its individual lower
bound, so the hidden placement is provably optimal and its HPWL is the exact
optimum. Cell indices are shuffled before the netlist is returned, so the
answer cannot be read off the numbering.
"""

from __future__ import annotations

import numpy as np

from .netlist import Netlist, net_lower_bound

# pin patterns (row, col offsets) that achieve the lower bound for 2, 3 and 4 pins
_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)
_SHAPE_WEIGHTS = np.array([4, 4, 1, 1, 1, 1, 2], dtype=float)


def planted(rows: int, cols: int, seed: int, nets_per_cell: float = 1.5) -> tuple[Netlist, np.ndarray]:
    """Return (netlist, optimal placement) for a rows x cols grid.

    Roughly ``nets_per_cell * rows * cols`` nets of 2-4 pins are generated;
    every cell is guaranteed to be on at least one net.
    """
    if rows < 2 or cols < 2:
        raise ValueError("grid must be at least 2 x 2")
    rng = np.random.default_rng(seed)
    n = rows * cols
    hidden_cell_at = rng.permutation(n)  # hidden_cell_at[slot] = cell placed there
    nets: set[tuple[int, ...]] = set()

    def add(shape, r0, c0) -> None:
        nets.add(tuple(sorted(int(hidden_cell_at[(r0 + dr) * cols + c0 + dc]) for dr, dc in shape)))

    # a serpentine chain of 2-pin nets touches every cell and keeps the netlist connected
    order = [r * cols + (c if r % 2 == 0 else cols - 1 - c) for r in range(rows) for c in range(cols)]
    for a, b in zip(order, order[1:]):
        nets.add(tuple(sorted((int(hidden_cell_at[a]), int(hidden_cell_at[b])))))

    target = int(nets_per_cell * n)
    while len(nets) < target:
        shape = _SHAPES[rng.choice(len(_SHAPES), p=_SHAPE_WEIGHTS / _SHAPE_WEIGHTS.sum())]
        h = max(dr for dr, _ in shape) + 1
        w = max(dc for _, dc in shape) + 1
        add(shape, int(rng.integers(0, rows - h + 1)), int(rng.integers(0, cols - w + 1)))

    net_tuple = tuple(sorted(nets))
    optimum = sum(net_lower_bound(len(net)) for net in net_tuple)
    slot_of = np.empty(n, dtype=np.int64)
    slot_of[hidden_cell_at] = np.arange(n)
    return Netlist(rows, cols, net_tuple, optimal_hpwl=optimum), slot_of
