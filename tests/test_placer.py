import itertools
import random

import numpy as np
import pytest

from placer.anneal import AnnealConfig, _State, anneal
from placer.benchmark import planted
from placer.netlist import Netlist, hpwl, is_legal, lower_bound, net_lower_bound, random_placement
from placer.spectral import laplacian, legalize, spectral_coordinates, spectral_placement


def test_hpwl_of_hand_checked_placement():
    # 2 x 3 grid, cells placed in slot order: slot = row * 3 + col
    netlist = Netlist(2, 3, ((0, 1), (0, 5), (1, 2, 4)))
    slot_of = np.arange(6)
    # (0,1): adjacent -> 1;  (0,5): corner to corner -> 1 + 2 = 3;  (1,2,4): rows 0-1, cols 1-2 -> 2
    assert hpwl(netlist, slot_of) == 6


@pytest.mark.parametrize("pins,bound", [(2, 1), (3, 2), (4, 2), (5, 3), (6, 3), (9, 4)])
def test_net_lower_bound(pins, bound):
    assert net_lower_bound(pins) == bound


def test_planted_placement_is_legal_and_meets_the_lower_bound():
    netlist, optimal = planted(8, 8, seed=1)
    assert is_legal(netlist, optimal)
    assert hpwl(netlist, optimal) == netlist.optimal_hpwl == lower_bound(netlist)
    assert all(len(nets) > 0 for nets in netlist.cell_nets)


def test_planted_optimum_agrees_with_exhaustive_search():
    """On a grid small enough to enumerate, no permutation beats the planted placement."""
    netlist, _ = planted(2, 4, seed=3)
    best = min(hpwl(netlist, np.array(p)) for p in itertools.permutations(range(8)))
    assert best == netlist.optimal_hpwl


def test_benchmark_is_deterministic_and_shuffled():
    a, slots_a = planted(6, 6, seed=5)
    b, slots_b = planted(6, 6, seed=5)
    assert a.nets == b.nets and np.array_equal(slots_a, slots_b)
    assert not np.array_equal(slots_a, np.arange(36))  # the identity placement is not the answer


def test_incremental_hpwl_matches_full_recomputation():
    netlist, _ = planted(6, 6, seed=2)
    rng = random.Random(0)
    slots = random_placement(netlist, np.random.default_rng(0))
    state = _State(netlist, slots)
    assert state.total == hpwl(netlist, slots)
    for _ in range(500):
        a, b = rng.sample(range(netlist.n_cells), 2)
        delta, updates = state.try_swap(a, b)
        if rng.random() < 0.5:
            state.commit(a, b, delta, updates)
        else:
            state.revert(a, b)
        current = state.slot_of()
        assert is_legal(netlist, current)
        assert state.total == hpwl(netlist, current)
        assert all(state.cell_at[current[c]] == c for c in range(netlist.n_cells))


def test_laplacian_rows_sum_to_zero_and_is_symmetric():
    netlist, _ = planted(5, 5, seed=0)
    lap = laplacian(netlist).toarray()
    np.testing.assert_allclose(lap.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(lap, lap.T)


def test_spectral_coordinates_recover_a_path():
    """For a chain of cells the Fiedler vector orders them along the chain."""
    order = [3, 0, 4, 1, 5, 2, 7, 6]
    netlist = Netlist(1, 8, tuple((a, b) for a, b in zip(order, order[1:])))
    x, _ = spectral_coordinates(netlist)
    ranked = list(np.argsort(x))
    assert ranked in (order, order[::-1])


def test_legalize_returns_a_permutation():
    netlist, _ = planted(7, 5, seed=4)
    rng = np.random.default_rng(0)
    assert is_legal(netlist, legalize(netlist, rng.normal(size=35), rng.normal(size=35)))


def test_quality_ordering_random_spectral_annealed():
    netlist, _ = planted(8, 8, seed=0)
    rand = hpwl(netlist, random_placement(netlist, np.random.default_rng(0)))
    spec_slots = spectral_placement(netlist)
    spec = hpwl(netlist, spec_slots)
    sa = anneal(netlist, seed=0, config=AnnealConfig(moves_per_cell=300, n_temperatures=40))
    assert is_legal(netlist, spec_slots) and is_legal(netlist, sa.slot_of)
    assert hpwl(netlist, sa.slot_of) == sa.hpwl
    assert netlist.optimal_hpwl <= sa.hpwl < spec < rand


def test_anneal_is_reproducible_and_never_worse_than_its_start():
    netlist, _ = planted(6, 6, seed=7)
    cfg = AnnealConfig(moves_per_cell=100, n_temperatures=20)
    start = spectral_placement(netlist)
    a = anneal(netlist, seed=11, config=cfg, initial=start)
    b = anneal(netlist, seed=11, config=cfg, initial=start)
    assert np.array_equal(a.slot_of, b.slot_of) and a.history == b.history
    assert a.hpwl <= hpwl(netlist, start)
