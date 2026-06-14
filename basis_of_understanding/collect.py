import numpy as np

from .embeddings import embed


def collect_independent(tokens, model, dim, tol):
    """Walk tokens in order, keeping each embedded vector that increases the
    rank of the collected set. Stop at `dim` accepted vectors or end of tokens.

    Rank test: project the candidate onto the orthonormal basis built so far;
    accept iff the residual norm exceeds `tol` relative to the candidate's norm.

    Returns (words, matrix) where matrix has shape (rank, embedding_dim).
    """
    words: list[str] = []
    rows: list[np.ndarray] = []
    ortho: list[np.ndarray] = []  # orthonormal basis of the current span

    for tok in tokens:
        v = embed(model, tok)
        if v is None:
            continue
        vnorm = np.linalg.norm(v)
        if vnorm == 0:
            continue
        residual = v.copy()
        for q in ortho:
            residual = residual - (residual @ q) * q
        if np.linalg.norm(residual) > tol * vnorm:
            ortho.append(residual / np.linalg.norm(residual))
            rows.append(v)
            words.append(tok)
            if len(rows) == dim:
                break

    matrix = np.array(rows, dtype=np.float64) if rows else np.empty((0, model.vector_size))
    return words, matrix
