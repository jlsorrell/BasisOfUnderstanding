from dataclasses import dataclass

from .collect import collect_independent
from .config import Config
from .decode import decode
from .embeddings import tokenize
from .lll import reduce


@dataclass
class Result:
    input_words: list[str]
    rank_achieved: int
    output_words: list[str]
    distances: list[float]
    reached_dim: bool


def run(text: str, model, config: Config) -> Result:
    tokens = tokenize(text)
    words, matrix = collect_independent(
        tokens, model, config.embedding_dim, config.rank_tol
    )
    rank = len(words)
    if rank == 0:
        return Result([], 0, [], [], False)
    rb = reduce(matrix, config.scale, config.delta, backend=config.lll_backend)
    decoded = decode(rb.reduced_float, model)
    return Result(
        input_words=words,
        rank_achieved=rank,
        output_words=[w for w, _ in decoded],
        distances=[d for _, d in decoded],
        reached_dim=(rank == config.embedding_dim),
    )
