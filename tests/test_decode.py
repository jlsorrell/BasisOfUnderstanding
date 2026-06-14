import numpy as np

from basis_of_understanding.decode import decode


def test_decode_returns_nearest_word(stub_model):
    # Closest vocab word to (0.9,0.05,0.0) is "king" at (1,0,0).
    reduced = np.array([[0.9, 0.05, 0.0], [0.0, 0.9, 0.1]])
    out = decode(reduced, stub_model)
    assert out[0][0] == "king"
    assert out[1][0] == "queen"
    assert out[0][1] < 0.2  # Euclidean distance is small


def test_collisions_allowed(stub_model):
    reduced = np.array([[1.0, 0.0, 0.0], [0.99, 0.0, 0.0]])
    out = decode(reduced, stub_model)
    assert out[0][0] == "king" and out[1][0] == "king"
