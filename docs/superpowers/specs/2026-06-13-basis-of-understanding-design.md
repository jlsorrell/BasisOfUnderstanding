# BasisOfUnderstanding — Design

**Date:** 2026-06-13
**Status:** Approved (pending spec review)

## Summary

A Gradio web app that treats word embeddings as lattice basis vectors and runs
LLL lattice basis reduction over them as a generative-art exercise. Given a text
document, the app reads it word by word, embeds each word with GloVe, and collects
vectors until it has `EMBEDDING_DIM` linearly independent ones (or the document
ends). It runs exact LLL reduction on that basis, then decodes each reduced basis
vector back to its Euclidean-nearest vocabulary word. The decoded words, joined in
reduced-basis order, are the output "poem."

This is an art / exploration project. The decode step is inherently approximate
(reduced vectors are integer combinations of the inputs and are essentially never
the embedding of any real word), and that approximation is the point — it is not a
bug to be corrected.

## Goals

- Read text, embed words, collect `EMBEDDING_DIM` linearly independent vectors.
- Run exact LLL reduction on the resulting basis.
- Decode reduced vectors to nearest vocabulary words and display the result.
- Keep `EMBEDDING_DIM` a one-line config swap (100 now, 300 later).
- Small, single-purpose modules behind narrow interfaces; everything unit-tested
  without needing the full GloVe file in CI.

## Non-goals

- Exact invertibility of the reduce step (impossible by construction).
- Semantic meaningfulness of the output.
- Large-scale / high-performance lattice work (the LLL engine is swappable later).
- A hand-built frontend.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Decode semantics | Euclidean nearest-neighbor, 1 word per reduced vector | Most natural; output is a "lattice-reduced poem" |
| Intent | Art / exploration | Output need not be semantically meaningful |
| Embeddings | Pretrained GloVe `glove.6B.{100,300}d` via gensim | 100-d in / 100-d out, no truncation; clean dim swap |
| Web stack | Gradio | A few lines wraps the pipeline; Python-native end to end |
| LLL engine | **`fpylll` (default)**, with the pure-Python exact LLL retained as a reference/oracle backend | Pure-Python exact (Fraction GSO) is correct but ~n^6.5 — a 100×100 run extrapolates to days. `fpylll` does 100×100 in ~0.02s. Native deps (`brew install fplll gmp mpfr qd`) accepted. Engine selectable via `Config.lll_backend`. |
| Real→integer | Scale by `SCALE`, round to integer matrix | Lattices are defined over the integers |
| δ (Lovász) | 0.99 default, configurable | Strongest practical reduction; speed irrelevant at this size |

## Architecture & modules

One package, `basis_of_understanding/`, plus a Gradio entrypoint. Small modules,
narrow interfaces.

| Module | Responsibility | Key interface |
|---|---|---|
| `config.py` | All tunable knobs in one dataclass | `EMBEDDING_DIM=100`, `SCALE=10**6`, `DELTA=0.99`, `RANK_TOL=1e-9`, model path, metric |
| `embeddings.py` | Load GloVe via gensim; tokenize; look up; skip OOV | `load_model()`, `tokenize(text)`, `embed(word) -> vec \| None` |
| `collect.py` | Stream tokens, keep rank-increasing vectors, stop at `EMBEDDING_DIM` or EOF | `collect_independent(tokens, model) -> (words, matrix)` |
| `lll.py` | Scale floats→ints, run exact integer LLL, return reduced basis (int + un-scaled float) | `reduce(matrix) -> ReducedBasis` |
| `decode.py` | Euclidean nearest-neighbor of each reduced vector in full GloVe space | `decode(reduced_vecs, model) -> [(word, distance)]` |
| `pipeline.py` | Wire it together; return structured result | `run(text, config) -> Result` |
| `app.py` | Gradio UI calling `pipeline.run` | — |

**Data flow:** `text → tokenize → embed + collect (until EMBEDDING_DIM independent)
→ scale → LLL → un-scale → nearest-neighbor decode → words`.

The model loads once at startup and is passed in, not reloaded per request. The
`lll.py` interface is deliberately narrow (matrix in, reduced basis out). It ships
two backends behind that interface: `fpylll` (the default, fast enough for the
100-d/300-d targets) and the pure-Python exact LLL (kept as a readable reference
and cross-validation oracle for tests). `Config.lll_backend` selects between them.
Both return the same `ReducedBasis` (reduced integer basis, un-scaled floats, and
the unimodular transform `U` with `U @ scaled_input == reduced_int`).

## Algorithm details & edge cases

**Tokenization & embedding.** Lowercase, strip punctuation, split on whitespace.
Walk tokens in document order. Skip out-of-vocabulary words.

**Independence collection.** As each candidate vector arrives, test whether it
raises the rank of the accumulated matrix. Maintain a QR / Gram–Schmidt
factorization incrementally and accept the vector iff its residual norm after
projection onto the current span exceeds `RANK_TOL` relative to its own norm. Stop
at `EMBEDDING_DIM` accepted vectors or end of document. This naturally handles
duplicates, near-duplicates, and numerical near-dependence (a repeated word
reproduces an existing vector and is rejected).

**Short documents.** If the text runs out before reaching `EMBEDDING_DIM`
independent vectors, reduce the rank-`r` lattice actually collected (an
`r`-dimensional lattice sitting in `EMBEDDING_DIM`-space). The pipeline reports
`rank_achieved = r` and proceeds — no error. LLL handles a rank-`r` basis of `r`
vectors.

**Scaling & LLL.** `M_int = round(SCALE · M)`. Run exact integer LLL (Fraction-based
GSO, δ = `DELTA`). Return the reduced integer basis; divide by `SCALE` to recover
float reduced vectors for decoding. Exact rational GSO ⇒ fully reproducible results
for a given `SCALE` / `δ`.

**Decoding.** For each reduced float vector, find the single Euclidean-nearest GloVe
vocabulary word via a vectorized numpy distance over the full vocab matrix
(400k×100 is fast). Emit `(word, distance)`. Collisions are allowed: two reduced
vectors decoding to the same word simply repeat it in the output.

**Output.** Decoded words joined in reduced-basis order form the "poem." The result
also carries: the input words that got embedded, `rank_achieved`, and per-output
distances, so the user can see how far each reduced point sits from real language.

## UI & configuration

**Gradio UI**, single screen:

- **Input:** a paste text box and a `.txt` file upload; whichever is provided is used.
- **Advanced panel (collapsed), sourced from `config.py`:** `EMBEDDING_DIM` (100/300
  dropdown), `δ` slider, `SCALE`, distance metric (Euclidean for now).
- **Run button.**
- **Output:** the decoded poem as the headline; below it a table of input words
  embedded, `rank_achieved`, and each output word with its nearest-neighbor
  distance. A note appears if the document was too short to reach `EMBEDDING_DIM`
  (shows the rank actually used).

**Config** is a dataclass in `config.py` with the chosen defaults (`EMBEDDING_DIM=100`,
`SCALE=10**6`, `DELTA=0.99`, `RANK_TOL=1e-9`). GloVe files download once to a
gitignored local cache; a `scripts/fetch_glove.py` (or documented step) fetches them.

## Testing

Pure functions with narrow interfaces keep this clean and CI-light:

- **`lll.py`** — test on known small integer lattices with hand-verifiable reduced
  bases (textbook 2-D / 3-D examples); on random inputs assert the LLL invariants:
  output is size-reduced, satisfies the Lovász condition for δ, and spans the same
  lattice (related to the input by a unimodular integer matrix).
- **`collect.py`** — synthetic vectors with known rank structure (including planted
  duplicates / dependencies); assert it accepts exactly the independent ones and
  stops at the cap.
- **`embeddings.py` / `decode.py`** — a small stub model (a handful of known
  vectors) so tests avoid the ~130 MB GloVe file; assert OOV skipping, tokenization,
  and that nearest-neighbor returns the planted closest word.
- **`pipeline.py`** — end-to-end on the stub model: a short text (rank < dim) and a
  normal run.

The heavy real-model path stays out of CI while every unit is covered.

## Open questions / future

- Optional: sort output words by reduced-vector norm ("most reduced" first) instead
  of basis order.
- Optional: `fpylll` backend for scale-up to 300-d / larger vocab.
- Scale to `EMBEDDING_DIM = 300` once the 100-d pipeline is validated.
