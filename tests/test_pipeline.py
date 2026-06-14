from basis_of_understanding.config import Config
from basis_of_understanding.pipeline import run, Result


def test_full_run(stub_model):
    cfg = Config(embedding_dim=3, scale=1000, delta=0.99)
    res = run("king queen man woman", stub_model, cfg)
    assert isinstance(res, Result)
    assert res.input_words == ["king", "queen", "man"]  # 3 independent, then stop
    assert res.rank_achieved == 3
    assert res.reached_dim is True
    assert len(res.output_words) == 3
    assert len(res.distances) == 3


def test_short_document_reports_partial(stub_model):
    cfg = Config(embedding_dim=3, scale=1000, delta=0.99)
    res = run("king zebra", stub_model, cfg)
    assert res.rank_achieved == 1
    assert res.reached_dim is False
    assert len(res.output_words) == 1


def test_all_oov_returns_empty_result(stub_model):
    # No in-vocabulary words: rank-0 guard returns an empty Result without
    # calling reduce/decode (which would choke on an empty matrix).
    cfg = Config(embedding_dim=3, scale=1000, delta=0.99)
    res = run("zzz qqq", stub_model, cfg)
    assert res.rank_achieved == 0
    assert res.reached_dim is False
    assert res.input_words == []
    assert res.output_words == []
    assert res.distances == []
