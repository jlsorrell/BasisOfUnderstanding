from dataclasses import dataclass


@dataclass
class Config:
    """All tunable knobs for the pipeline."""

    embedding_dim: int = 100
    scale: int = 10**6
    delta: float = 0.99
    rank_tol: float = 1e-9
    model_path: str = "data/glove.6B.100d.txt"
    metric: str = "euclidean"
    lll_backend: str = "fpylll"
