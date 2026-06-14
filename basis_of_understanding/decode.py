import numpy as np


def decode(reduced_vecs, model):
    """For each reduced vector, return (nearest_word, euclidean_distance).

    Brute-force vectorized nearest neighbor over the full vocabulary matrix.
    """
    vocab = np.asarray(model.vectors, dtype=np.float64)  # (V, dim)
    keys = model.index_to_key
    out = []
    for v in np.asarray(reduced_vecs, dtype=np.float64):
        diff = vocab - v
        d2 = np.einsum("ij,ij->i", diff, diff)
        idx = int(np.argmin(d2))
        out.append((keys[idx], float(np.sqrt(d2[idx]))))
    return out
