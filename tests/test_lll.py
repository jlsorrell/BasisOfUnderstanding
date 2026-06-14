import time
from fractions import Fraction

import numpy as np
import pytest

from basis_of_understanding.lll import reduce, _lll_int, _lll_fpylll, _gso


def _is_size_reduced(B, mu):
    n = len(B)
    for i in range(n):
        for j in range(i):
            if abs(mu[i][j]) > Fraction(1, 2):
                return False
    return True


def test_classic_2d_example():
    # Rows (1,1,1),(-1,0,2),(3,5,6) — a standard LLL textbook lattice.
    basis = [[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    out, U = _lll_int(basis, Fraction(3, 4))
    # Reduced basis spans the same lattice: U @ basis == out, |det U| == 1.
    A = np.array(basis)
    assert np.array_equal(np.array(U) @ A, np.array(out))
    assert round(abs(np.linalg.det(np.array(U, dtype=float)))) == 1


def test_invariants_on_random_lattices():
    rng = np.random.default_rng(0)
    delta = Fraction(99, 100)
    for _ in range(20):
        basis = rng.integers(-9, 10, size=(4, 4)).tolist()
        if round(abs(np.linalg.det(np.array(basis, dtype=float)))) == 0:
            continue  # skip singular
        out, U = _lll_int(basis, delta)
        # same lattice
        assert np.array_equal(np.array(U) @ np.array(basis), np.array(out))
        assert round(abs(np.linalg.det(np.array(U, dtype=float)))) == 1
        # size-reduced + Lovász condition
        from basis_of_understanding.lll import _gso
        mu, Bn = _gso([[Fraction(x) for x in row] for row in out])
        assert _is_size_reduced(out, mu)
        for k in range(1, len(out)):
            assert Bn[k] >= (delta - mu[k][k - 1] ** 2) * Bn[k - 1]


def test_reduce_scales_and_unscales():
    M = np.array([[1.0, 0.0], [0.5, 1.0]])
    rb = reduce(M, scale=1000, delta=0.99)
    # reduced_float == reduced_int / scale
    assert np.allclose(rb.reduced_float, np.array(rb.reduced_int) / 1000.0)
    # same lattice over the scaled integer matrix
    scaled = np.rint(M * 1000).astype(int)
    assert np.array_equal(np.array(rb.transform) @ scaled, np.array(rb.reduced_int))


def _random_full_rank_4x4_lattices(n=20, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        basis = rng.integers(-9, 10, size=(4, 4)).tolist()
        if round(abs(np.linalg.det(np.array(basis, dtype=float)))) == 0:
            continue  # skip singular
        out.append(basis)
    return out


def _assert_lll_reduced(reduced_rows, delta, eta=Fraction(51, 100)):
    """Reducedness with the fpylll eta=0.51 bound on mu + Lovász condition."""
    mu, bnorm = _gso([[Fraction(x) for x in row] for row in reduced_rows])
    n = len(reduced_rows)
    for i in range(n):
        for j in range(i):
            assert abs(mu[i][j]) <= eta
    for k in range(1, n):
        assert bnorm[k] >= (delta - mu[k][k - 1] ** 2) * bnorm[k - 1]


def test_fpylll_same_lattice_and_unimodular():
    for basis in _random_full_rank_4x4_lattices():
        reduced, U = _lll_fpylll(basis, 0.99)
        assert np.array_equal(np.array(U) @ np.array(basis), np.array(reduced))
        assert round(abs(np.linalg.det(np.array(U, dtype=float)))) == 1


def test_fpylll_output_is_reduced():
    delta = Fraction(99, 100)
    for basis in _random_full_rank_4x4_lattices():
        reduced, U = _lll_fpylll(basis, 0.99)
        _assert_lll_reduced(reduced, delta)


def test_fpylll_wide_rows():
    # 3 linearly independent rows in ambient dimension 6 (rank 3 < dim 6).
    basis = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1],
    ]
    reduced, U = _lll_fpylll(basis, 0.99)
    assert np.array(U).shape == (3, 3)
    assert np.array(reduced).shape == (3, 6)
    assert np.array_equal(np.array(U) @ np.array(basis), np.array(reduced))
    assert round(abs(np.linalg.det(np.array(U, dtype=float)))) == 1


def test_fpylll_performance():
    rows = np.random.default_rng(0).integers(
        -10**6, 10**6, size=(100, 100)
    ).tolist()
    start = time.perf_counter()
    _lll_fpylll(rows, 0.99)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0


def test_reduce_fpylll_backend():
    M = np.array([[1.0, 0.0], [0.5, 1.0]])
    rb = reduce(M, scale=1000, delta=0.99)  # default backend == fpylll
    scaled = np.rint(M * 1000).astype(int)
    assert np.array_equal(np.array(rb.transform) @ scaled, np.array(rb.reduced_int))
    assert np.allclose(rb.reduced_float, np.array(rb.reduced_int) / 1000.0)


def test_reduce_unknown_backend_raises():
    M = np.array([[1.0, 0.0], [0.5, 1.0]])
    with pytest.raises(ValueError):
        reduce(M, 1000, 0.99, backend="nope")
