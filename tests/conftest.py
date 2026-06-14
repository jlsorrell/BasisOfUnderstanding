import numpy as np
import pytest
from gensim.models import KeyedVectors


@pytest.fixture
def stub_model():
    """Tiny in-memory KeyedVectors so tests never touch the real GloVe file."""
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(
        ["king", "queen", "man", "woman"],
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    return kv
