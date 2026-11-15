"""Analytic placement: Hall's quadratic (spectral) placement plus legalization.

Minimising the quadratic wirelength sum_ij w_ij (x_i - x_j)^2 subject to a
unit-norm, zero-mean constraint gives the Laplacian eigenvectors with the
smallest non-zero eigenvalues (K. M. Hall, 1970). The two lowest give x and y
coordinates. Nets are expanded to cliques with weight 1 / (pins - 1).

The continuous solution overlaps heavily, so it is legalized by sorting:
cells are cut into columns by x, then ordered by y inside each column.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

from .netlist import Netlist


def laplacian(netlist: Netlist):
    rows, cols, vals = [], [], []
    for net in netlist.nets:
        w = 1.0 / (len(net) - 1)
        for i, a in enumerate(net):
            for b in net[i + 1 :]:
                rows += [a, b]
                cols += [b, a]
                vals += [w, w]
    n = netlist.n_cells
    adj = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return diags(np.asarray(adj.sum(axis=1)).ravel()) - adj


def spectral_coordinates(netlist: Netlist) -> tuple[np.ndarray, np.ndarray]:
    """Continuous (x, y) from the 2nd and 3rd Laplacian eigenvectors."""
    lap = laplacian(netlist)
    n = netlist.n_cells
    if n <= 64:
        _, vecs = np.linalg.eigh(lap.toarray())
    else:
        # shift-invert around a small negative value: the Laplacian is singular at 0
        _, vecs = eigsh(lap.tocsc(), k=3, sigma=-1e-3, which="LM", v0=np.ones(n))
    x, y = vecs[:, 1].copy(), vecs[:, 2].copy()
    for v in (x, y):  # eigenvector signs are arbitrary; fix them for reproducibility
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1
    return x, y


def legalize(netlist: Netlist, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Snap continuous coordinates to grid slots, one cell per slot."""
    slot_of = np.empty(netlist.n_cells, dtype=np.int64)
    by_x = np.argsort(x, kind="stable")
    for col in range(netlist.cols):
        column = by_x[col * netlist.rows : (col + 1) * netlist.rows]
        for row, cell in enumerate(column[np.argsort(y[column], kind="stable")]):
            slot_of[cell] = row * netlist.cols + col
    return slot_of


def spectral_placement(netlist: Netlist) -> np.ndarray:
    return legalize(netlist, *spectral_coordinates(netlist))
