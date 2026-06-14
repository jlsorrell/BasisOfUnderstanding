from basis_of_understanding.config import Config


def test_defaults():
    c = Config()
    assert c.embedding_dim == 100
    assert c.scale == 10**6
    assert c.delta == 0.99
    assert c.rank_tol == 1e-9
    assert c.metric == "euclidean"


def test_override():
    c = Config(embedding_dim=300, delta=0.75)
    assert c.embedding_dim == 300
    assert c.delta == 0.75
