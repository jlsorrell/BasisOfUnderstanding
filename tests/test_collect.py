import numpy as np
from gensim.models import KeyedVectors

from basis_of_understanding.collect import collect_independent


def _model():
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(
        ["e1", "e2", "e3", "dup", "combo"],
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],  # duplicate of e1 -> dependent
                [1.0, 1.0, 0.0],  # e1+e2 -> dependent once e1,e2 seen
            ],
            dtype=np.float32,
        ),
    )
    return kv


def test_rejects_dependent_vectors():
    model = _model()
    tokens = ["e1", "dup", "e2", "combo", "e3"]
    words, matrix = collect_independent(tokens, model, dim=3, tol=1e-9)
    assert words == ["e1", "e2", "e3"]
    assert matrix.shape == (3, 3)


def test_stops_at_dim_cap():
    model = _model()
    tokens = ["e1", "e2", "e3"]
    words, matrix = collect_independent(tokens, model, dim=2, tol=1e-9)
    assert words == ["e1", "e2"]
    assert matrix.shape == (2, 3)


def test_short_document_returns_partial_rank():
    model = _model()
    words, matrix = collect_independent(["e1", "zebra"], model, dim=3, tol=1e-9)
    assert words == ["e1"]
    assert matrix.shape == (1, 3)
