import numpy as np
from basis_of_understanding import embeddings


def test_tokenize_lowercases_and_strips_punctuation():
    assert embeddings.tokenize("The King, and Queen!") == ["the", "king", "and", "queen"]


def test_embed_returns_vector_for_known_word(stub_model):
    v = embeddings.embed(stub_model, "king")
    assert np.allclose(v, [1.0, 0.0, 0.0])


def test_embed_returns_none_for_oov(stub_model):
    assert embeddings.embed(stub_model, "zebra") is None
