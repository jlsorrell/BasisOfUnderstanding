import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from fpylll import IntegerMatrix, LLL, GSO


@dataclass
class ReducedBasis:
    reduced_int: list[list[int]]   # LLL-reduced basis of the scaled integer lattice
    reduced_float: np.ndarray      # reduced_int / scale, shape (rank, dim)
    transform: list[list[int]]     # unimodular U with U @ scaled_input == reduced_int


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _gso(B):
    """Exact Gram-Schmidt. B is rows of Fraction. Returns (mu, Bnorm)."""
    n = len(B)
    bstar = [None] * n
    mu = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    bnorm = [Fraction(0) for _ in range(n)]
    for i in range(n):
        bi = list(B[i])
        for j in range(i):
            mu[i][j] = _dot(B[i], bstar[j]) / bnorm[j]
            bi = [bi[k] - mu[i][j] * bstar[j][k] for k in range(len(bi))]
        bstar[i] = bi
        bnorm[i] = _dot(bi, bi)
    return mu, bnorm


def _lll_int(basis, delta):
    """Classic LLL on integer rows. Returns (reduced_rows, U) with U @ basis == reduced."""
    B = [[int(x) for x in row] for row in basis]
    n = len(B)
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    Bf = [[Fraction(x) for x in row] for row in B]
    mu, bnorm = _gso(Bf)

    def recompute():
        nonlocal mu, bnorm
        mu, bnorm = _gso([[Fraction(x) for x in row] for row in B])

    k = 1
    while k < n:
        # size-reduce b_k against b_{k-1}
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > Fraction(1, 2):
                q = math.floor(mu[k][j] + Fraction(1, 2))
                B[k] = [B[k][i] - q * B[j][i] for i in range(len(B[k]))]
                U[k] = [U[k][i] - q * U[j][i] for i in range(n)]
                recompute()
        if bnorm[k] >= (delta - mu[k][k - 1] ** 2) * bnorm[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            U[k], U[k - 1] = U[k - 1], U[k]
            recompute()
            k = max(k - 1, 1)
    return B, U


def _lll_fpylll(basis, delta, eta=0.51):
    """LLL via fpylll. Returns (reduced_rows, U) with U @ basis == reduced."""
    rows = [[int(x) for x in row] for row in basis]
    if not rows:
        return [], []
    A = IntegerMatrix.from_matrix(rows)
    U = IntegerMatrix.identity(A.nrows)
    M = GSO.Mat(A, U=U)
    M.update_gso()
    LLL.Reduction(M, delta=float(delta), eta=eta)()
    reduced = [[A[i, j] for j in range(A.ncols)] for i in range(A.nrows)]
    Umat = [[U[i, j] for j in range(U.ncols)] for i in range(U.nrows)]
    return reduced, Umat


def reduce(
    matrix: np.ndarray, scale: int, delta: float, backend: str = "fpylll"
) -> ReducedBasis:
    """Scale float rows to integers, run LLL, return ReducedBasis."""
    scaled = np.rint(np.asarray(matrix, dtype=np.float64) * scale).astype(np.int64)
    int_rows = scaled.tolist()
    if backend == "fpylll":
        reduced_int, U = _lll_fpylll(int_rows, float(delta))
    elif backend == "python":
        reduced_int, U = _lll_int(int_rows, Fraction(delta).limit_denominator(10**9))
    else:
        raise ValueError(f"unknown backend: {backend!r}")
    reduced_float = np.array(reduced_int, dtype=np.float64) / scale
    return ReducedBasis(reduced_int=reduced_int, reduced_float=reduced_float, transform=U)
