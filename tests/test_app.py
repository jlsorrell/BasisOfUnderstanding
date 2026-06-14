def test_app_module_imports():
    from basis_of_understanding import app
    assert hasattr(app, "build_ui")
    assert hasattr(app, "format_result")


def test_model_path_tracks_dimension():
    # Selecting a dimension in the UI must load the matching GloVe file,
    # not silently reuse the 100-d model.
    from basis_of_understanding.app import _model_path_for_dim

    assert _model_path_for_dim(100) == "data/glove.6B.100d.txt"
    assert _model_path_for_dim(300) == "data/glove.6B.300d.txt"


def test_format_result_short_document():
    from basis_of_understanding.app import format_result
    from basis_of_understanding.pipeline import Result

    res = Result(["king"], 1, ["queen"], [0.4], reached_dim=False)
    headline, table = format_result(res, embedding_dim=100)
    assert "queen" in headline
    assert "1 of 100" in table  # rank note for short document
