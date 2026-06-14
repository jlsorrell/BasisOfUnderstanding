import re

import numpy as np
from gensim.models import KeyedVectors

_TOKEN = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphabetic chars, drop empties."""
    return _TOKEN.findall(text.lower())


def load_model(path: str) -> KeyedVectors:
    """Load GloVe text-format vectors (no header line)."""
    return KeyedVectors.load_word2vec_format(path, no_header=True)


def embed(model: KeyedVectors, word: str):
    """Return the word's vector as float64, or None if out of vocabulary."""
    if word in model.key_to_index:
        return np.asarray(model[word], dtype=np.float64)
    return None
