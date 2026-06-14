# BasisOfUnderstanding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Gradio web app that embeds a document's words with GloVe, collects `EMBEDDING_DIM` linearly independent vectors, runs exact LLL lattice reduction on them, and decodes each reduced basis vector to its Euclidean-nearest vocabulary word.

**Architecture:** One Python package `basis_of_understanding/` with small single-purpose modules (`config`, `embeddings`, `collect`, `lll`, `decode`, `pipeline`) behind narrow interfaces, plus a Gradio `app.py`. The model loads once and is passed in. The LLL engine is a pure-Python exact integer implementation behind a narrow `reduce()` interface so `fpylll` can be swapped in later.

**Tech Stack:** Python 3.11+, gensim (GloVe loading), numpy, gradio, pytest. Pure-Python Fraction-based integer LLL (no native deps).

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata + deps |
| `basis_of_understanding/__init__.py` | Package marker |
| `basis_of_understanding/config.py` | `Config` dataclass — all tunable knobs |
| `basis_of_understanding/embeddings.py` | Load GloVe, tokenize, embed/lookup, skip OOV |
| `basis_of_understanding/collect.py` | Collect rank-increasing vectors until dim/EOF |
| `basis_of_understanding/lll.py` | Scale floats→ints, exact integer LLL, `ReducedBasis` |
| `basis_of_understanding/decode.py` | Euclidean nearest-neighbor decode |
| `basis_of_understanding/pipeline.py` | `Result` dataclass + `run()` wiring |
| `basis_of_understanding/app.py` | Gradio UI |
| `scripts/fetch_glove.py` | One-time GloVe download |
| `tests/test_*.py` | Unit tests per module |
| `tests/conftest.py` | `stub_model` fixture (in-memory KeyedVectors) |

---

### Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `basis_of_understanding/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "basis-of-understanding"
version = "0.1.0"
description = "LLL lattice reduction over word embeddings as generative art"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "gensim>=4.3",
    "gradio>=4.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["basis_of_understanding*"]
```

- [ ] **Step 2: Create empty package + test markers**

```bash
mkdir -p basis_of_understanding tests scripts data
touch basis_of_understanding/__init__.py tests/__init__.py
```

- [ ] **Step 3: Install**

Run: `uv venv && uv pip install -e ".[dev]"`
Expected: gensim, numpy, gradio, pytest installed without errors.

- [ ] **Step 4: Verify pytest discovers nothing yet**

Run: `uv run pytest -q`
Expected: "no tests ran" (exit 5) — confirms harness works.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml basis_of_understanding/__init__.py tests/__init__.py
git commit -m "chore: project scaffolding"
```

---

### Task 1: Config dataclass

**Files:**
- Create: `basis_of_understanding/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: basis_of_understanding.config`

- [ ] **Step 3: Write minimal implementation**

```python
# basis_of_understanding/config.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add basis_of_understanding/config.py tests/test_config.py
git commit -m "feat: Config dataclass"
```

---

### Task 2: Embeddings — tokenize, load, lookup

**Files:**
- Create: `basis_of_understanding/embeddings.py`
- Create: `tests/conftest.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write the stub-model fixture**

```python
# tests/conftest.py
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
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_embeddings.py
import numpy as np
from basis_of_understanding import embeddings


def test_tokenize_lowercases_and_strips_punctuation():
    assert embeddings.tokenize("The King, and Queen!") == ["the", "king", "and", "queen"]


def test_embed_returns_vector_for_known_word(stub_model):
    v = embeddings.embed(stub_model, "king")
    assert np.allclose(v, [1.0, 0.0, 0.0])


def test_embed_returns_none_for_oov(stub_model):
    assert embeddings.embed(stub_model, "zebra") is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`

- [ ] **Step 4: Write minimal implementation**

```python
# basis_of_understanding/embeddings.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add basis_of_understanding/embeddings.py tests/conftest.py tests/test_embeddings.py
git commit -m "feat: tokenize, load, embed"
```

---

### Task 3: Collect linearly independent vectors

**Files:**
- Create: `basis_of_understanding/collect.py`
- Test: `tests/test_collect.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_collect.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# basis_of_understanding/collect.py
import numpy as np

from .embeddings import embed


def collect_independent(tokens, model, dim, tol):
    """Walk tokens in order, keeping each embedded vector that increases the
    rank of the collected set. Stop at `dim` accepted vectors or end of tokens.

    Rank test: project the candidate onto the orthonormal basis built so far;
    accept iff the residual norm exceeds `tol` relative to the candidate's norm.

    Returns (words, matrix) where matrix has shape (rank, embedding_dim).
    """
    words: list[str] = []
    rows: list[np.ndarray] = []
    ortho: list[np.ndarray] = []  # orthonormal basis of the current span

    for tok in tokens:
        v = embed(model, tok)
        if v is None:
            continue
        vnorm = np.linalg.norm(v)
        if vnorm == 0:
            continue
        residual = v.copy()
        for q in ortho:
            residual = residual - (residual @ q) * q
        if np.linalg.norm(residual) > tol * vnorm:
            ortho.append(residual / np.linalg.norm(residual))
            rows.append(v)
            words.append(tok)
            if len(rows) == dim:
                break

    matrix = np.array(rows, dtype=np.float64) if rows else np.empty((0, model.vector_size))
    return words, matrix
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_collect.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add basis_of_understanding/collect.py tests/test_collect.py
git commit -m "feat: collect linearly independent vectors"
```

---

### Task 4: Exact integer LLL

**Files:**
- Create: `basis_of_understanding/lll.py`
- Test: `tests/test_lll.py`

This is the mathematical core. We implement the classic LLL with incremental
rational (Fraction) Gram–Schmidt updates — efficient enough for 100×100 and
fully exact. We track a unimodular transform `U` so tests can assert the reduced
basis spans the same lattice (`U @ original == reduced`, `|det U| == 1`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lll.py
from fractions import Fraction

import numpy as np

from basis_of_understanding.lll import reduce, _lll_int


def _is_size_reduced(B, mu):
    n = len(B)
    for i in range(n):
        for j in range(i):
            if abs(mu[i][j]) > Fraction(1, 2):
                return False
    return True


def test_classic_2d_example():
    # Rows (1,1,1),(-1,0,2),(3,5,6) — a standard LLL textbook lattice.
    basis = [[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    out, U = _lll_int(basis, Fraction(3, 4))
    # Reduced basis spans the same lattice: U @ basis == out, |det U| == 1.
    A = np.array(basis)
    assert np.array_equal(np.array(U) @ A, np.array(out))
    assert round(abs(np.linalg.det(np.array(U, dtype=float)))) == 1


def test_invariants_on_random_lattices():
    rng = np.random.default_rng(0)
    delta = Fraction(99, 100)
    for _ in range(20):
        basis = rng.integers(-9, 10, size=(4, 4)).tolist()
        if round(abs(np.linalg.det(np.array(basis, dtype=float)))) == 0:
            continue  # skip singular
        out, U = _lll_int(basis, delta)
        # same lattice
        assert np.array_equal(np.array(U) @ np.array(basis), np.array(out))
        assert round(abs(np.linalg.det(np.array(U, dtype=float)))) == 1
        # size-reduced + Lovász condition
        from basis_of_understanding.lll import _gso
        mu, Bn = _gso([[Fraction(x) for x in row] for row in out])
        assert _is_size_reduced(out, mu)
        for k in range(1, len(out)):
            assert Bn[k] >= (delta - mu[k][k - 1] ** 2) * Bn[k - 1]


def test_reduce_scales_and_unscales():
    M = np.array([[1.0, 0.0], [0.5, 1.0]])
    rb = reduce(M, scale=1000, delta=0.99)
    # reduced_float == reduced_int / scale
    assert np.allclose(rb.reduced_float, np.array(rb.reduced_int) / 1000.0)
    # same lattice over the scaled integer matrix
    scaled = np.rint(M * 1000).astype(int)
    assert np.array_equal(np.array(rb.transform) @ scaled, np.array(rb.reduced_int))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lll.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# basis_of_understanding/lll.py
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass
class ReducedBasis:
    reduced_int: list[list[int]]   # LLL-reduced basis of the scaled integer lattice
    reduced_float: np.ndarray      # reduced_int / scale, shape (rank, dim)
    transform: list[list[int]]     # unimodular U with U @ scaled_input == reduced_int


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _gso(B):
    """Exact Gram–Schmidt. B is rows of Fraction. Returns (mu, Bnorm)."""
    n = len(B)
    bstar = [None] * n
    mu = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    bnorm = [Fraction(0) for _ in range(n)]
    for i in range(n):
        bi = list(B[i])
        for j in range(i):
            mu[i][j] = _dot(B[i], bstar[j]) / bnorm[j]
            bi = [bi[k] - mu[i][j] * bstar[j][k] for k in range(len(bi))]
        bstar[i] = bi
        bnorm[i] = _dot(bi, bi)
    return mu, bnorm


def _lll_int(basis, delta):
    """Classic LLL on integer rows. Returns (reduced_rows, U) with U @ basis == reduced."""
    B = [[int(x) for x in row] for row in basis]
    n = len(B)
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    Bf = [[Fraction(x) for x in row] for row in B]
    mu, bnorm = _gso(Bf)

    def recompute():
        nonlocal mu, bnorm
        mu, bnorm = _gso([[Fraction(x) for x in row] for row in B])

    k = 1
    while k < n:
        # size-reduce b_k against b_{k-1}
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > Fraction(1, 2):
                q = math.floor(mu[k][j] + Fraction(1, 2))
                B[k] = [B[k][i] - q * B[j][i] for i in range(len(B[k]))]
                U[k] = [U[k][i] - q * U[j][i] for i in range(n)]
                recompute()
        if bnorm[k] >= (delta - mu[k][k - 1] ** 2) * bnorm[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            U[k], U[k - 1] = U[k - 1], U[k]
            recompute()
            k = max(k - 1, 1)
    return B, U


def reduce(matrix: np.ndarray, scale: int, delta: float) -> ReducedBasis:
    """Scale float rows to integers, run exact LLL, return ReducedBasis."""
    scaled = np.rint(np.asarray(matrix, dtype=np.float64) * scale).astype(np.int64)
    reduced_int, U = _lll_int(scaled.tolist(), Fraction(delta).limit_denominator(10**9))
    reduced_float = np.array(reduced_int, dtype=np.float64) / scale
    return ReducedBasis(reduced_int=reduced_int, reduced_float=reduced_float, transform=U)
```

> Note: `recompute()` re-runs full GSO after each modification — simplest correct
> form. Tests use small matrices so it is instant. If the real 100-d run is too
> slow, replace `recompute()` with incremental μ/`bnorm` updates (RED updates only
> `mu[k][*]`; SWAP updates locally) — same interface, behind this module only.
> The `len(B[k])` expression keeps row width = dim even when rank < dim.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lll.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add basis_of_understanding/lll.py tests/test_lll.py
git commit -m "feat: exact integer LLL with unimodular transform"
```

---

### Task 5: Nearest-neighbor decode

**Files:**
- Create: `basis_of_understanding/decode.py`
- Test: `tests/test_decode.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decode.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decode.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# basis_of_understanding/decode.py
import numpy as np


def decode(reduced_vecs, model):
    """For each reduced vector, return (nearest_word, euclidean_distance).

    Brute-force vectorized nearest neighbor over the full vocabulary matrix.
    """
    vocab = np.asarray(model.vectors, dtype=np.float64)  # (V, dim)
    keys = model.index_to_key
    out = []
    for v in np.asarray(reduced_vecs, dtype=np.float64):
        diff = vocab - v
        d2 = np.einsum("ij,ij->i", diff, diff)
        idx = int(np.argmin(d2))
        out.append((keys[idx], float(np.sqrt(d2[idx]))))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decode.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add basis_of_understanding/decode.py tests/test_decode.py
git commit -m "feat: Euclidean nearest-neighbor decode"
```

---

### Task 6: Pipeline wiring

**Files:**
- Create: `basis_of_understanding/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# basis_of_understanding/pipeline.py
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
    rb = reduce(matrix, config.scale, config.delta)
    decoded = decode(rb.reduced_float, model)
    return Result(
        input_words=words,
        rank_achieved=rank,
        output_words=[w for w, _ in decoded],
        distances=[d for _, d in decoded],
        reached_dim=(rank == config.embedding_dim),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add basis_of_understanding/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline wiring"
```

---

### Task 7: GloVe download script

**Files:**
- Create: `scripts/fetch_glove.py`

No unit test (network/IO). Verified by a smoke import + manual run.

- [ ] **Step 1: Write the script**

```python
# scripts/fetch_glove.py
"""Download GloVe vectors into ./data once. Usage: python scripts/fetch_glove.py [100|300]"""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://nlp.stanford.edu/data/glove.6B.zip"
DATA = Path("data")


def main(dim: str = "100") -> None:
    DATA.mkdir(exist_ok=True)
    target = DATA / f"glove.6B.{dim}d.txt"
    if target.exists():
        print(f"{target} already present.")
        return
    zip_path = DATA / "glove.6B.zip"
    if not zip_path.exists():
        print(f"Downloading {URL} (~822 MB)...")
        urllib.request.urlretrieve(URL, zip_path)
    print(f"Extracting glove.6B.{dim}d.txt...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(f"glove.6B.{dim}d.txt", DATA)
    print(f"Ready: {target}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "100")
```

- [ ] **Step 2: Smoke-check it imports and is runnable**

Run: `uv run python -c "import ast; ast.parse(open('scripts/fetch_glove.py').read()); print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: (Manual, optional now) fetch the real vectors**

Run: `uv run python scripts/fetch_glove.py 100`
Expected: `data/glove.6B.100d.txt` exists (~331 MB). Skip if deferring to first app run.

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_glove.py
git commit -m "feat: GloVe download script"
```

---

### Task 8: Gradio app

**Files:**
- Create: `basis_of_understanding/app.py`
- Test: `tests/test_app.py` (import smoke test only)

The UI is thin glue over `pipeline.run`; logic is already covered. We add an
import smoke test and verify launch manually.

- [ ] **Step 1: Write the import smoke test**

```python
# tests/test_app.py
def test_app_module_imports():
    from basis_of_understanding import app
    assert hasattr(app, "build_ui")
    assert hasattr(app, "format_result")


def test_format_result_short_document():
    from basis_of_understanding.app import format_result
    from basis_of_understanding.pipeline import Result

    res = Result(["king"], 1, ["queen"], [0.4], reached_dim=False)
    headline, table = format_result(res, embedding_dim=100)
    assert "queen" in headline
    assert "1 of 100" in table  # rank note for short document
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`

- [ ] **Step 3: Write the app**

```python
# basis_of_understanding/app.py
import gradio as gr

from .config import Config
from .embeddings import load_model
from .pipeline import run

_CONFIG = Config()
_MODEL = None  # lazy-loaded on first request


def _get_model(path: str):
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(path)
    return _MODEL


def format_result(res, embedding_dim: int):
    """Return (headline_poem, markdown_table)."""
    headline = " ".join(res.output_words) if res.output_words else "(no words embedded)"
    lines = [
        f"**Rank achieved:** {res.rank_achieved} of {embedding_dim}"
        + ("" if res.reached_dim else "  _(document too short to reach full dimension)_"),
        "",
        "| # | reduced → word | distance |",
        "|---|---|---|",
    ]
    for i, (w, d) in enumerate(zip(res.output_words, res.distances)):
        lines.append(f"| {i} | {w} | {d:.4f} |")
    lines += ["", f"**Input words embedded:** {', '.join(res.input_words)}"]
    return headline, "\n".join(lines)


def _run(text, uploaded, embedding_dim, delta, scale):
    content = text or ""
    if uploaded:
        with open(uploaded, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    cfg = Config(embedding_dim=int(embedding_dim), delta=float(delta), scale=int(scale))
    res = run(content, _get_model(cfg.model_path), cfg)
    return format_result(res, cfg.embedding_dim)


def build_ui():
    with gr.Blocks(title="BasisOfUnderstanding") as demo:
        gr.Markdown("# BasisOfUnderstanding\nLLL lattice reduction over word embeddings.")
        with gr.Row():
            text = gr.Textbox(label="Paste text", lines=10)
            uploaded = gr.File(label="...or upload .txt", file_types=[".txt"], type="filepath")
        with gr.Accordion("Advanced", open=False):
            embedding_dim = gr.Dropdown([100, 300], value=100, label="EMBEDDING_DIM")
            delta = gr.Slider(0.26, 0.999, value=0.99, step=0.001, label="δ (Lovász)")
            scale = gr.Number(value=10**6, label="SCALE", precision=0)
        run_btn = gr.Button("Run", variant="primary")
        headline = gr.Textbox(label="Decoded poem", interactive=False)
        table = gr.Markdown()
        run_btn.click(
            _run,
            inputs=[text, uploaded, embedding_dim, delta, scale],
            outputs=[headline, table],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_app.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -q`
Expected: ALL tests pass (config, embeddings, collect, lll, decode, pipeline, app).

- [ ] **Step 6: Manual launch verification**

Run: `uv run python -m basis_of_understanding.app`
Expected: Gradio serves at `http://127.0.0.1:7860`. Requires `data/glove.6B.100d.txt`
(run Task 7 Step 3 first). Paste a paragraph, click Run, confirm a decoded poem
and the rank table render.

- [ ] **Step 7: Commit**

```bash
git add basis_of_understanding/app.py tests/test_app.py
git commit -m "feat: Gradio app"
```

---

## Self-Review

**Spec coverage:** Decode semantics (Task 5), GloVe via gensim (Tasks 2/7), Gradio
(Task 8), pure-Python exact LLL with scale-to-integer (Task 4), δ=0.99 default
(Task 1), independence collection with short-doc handling (Tasks 3/6), per-module
tests with stub model (all tasks), config knobs (Task 1) — all covered.

**Placeholder scan:** No TBD/TODO; every code step is complete and runnable.

**Type consistency:** `Config` fields, `ReducedBasis` (`reduced_int`/`reduced_float`/
`transform`), and `Result` (`input_words`/`rank_achieved`/`output_words`/`distances`/
`reached_dim`) are used identically across tasks. `reduce(matrix, scale, delta)`,
`decode(reduced_vecs, model)`, `collect_independent(tokens, model, dim, tol)`, and
`run(text, model, config)` signatures match every call site.
