# 03 — Embeddings and the On-Disk Cache

## Concept

An **embedding** is a fixed-length vector of numbers that represents the
*meaning* of a piece of text, produced by a neural network trained so that
texts with similar meaning end up close together in vector space (by some
distance metric) and dissimilar texts end up far apart. This is the core
trick that makes semantic search possible: instead of matching keywords, you
compare vectors.

This project uses **`BAAI/bge-small-en-v1.5`**, a small (33M parameter)
open-weight sentence-embedding model from the BGE family, run locally via
`sentence-transformers`. It produces **384-dimensional** vectors. It is
*not* an OpenAI or OpenRouter model — see
[`00-overview.md`](00-overview.md#the-local-vs-openrouter-split-and-why) for
why: OpenRouter doesn't expose an embeddings endpoint at all, and running a
small local bi-encoder is faster, free, and fully offline.

**Normalization and cosine similarity.** Every embedding produced here is
**L2-normalized** (`normalize_embeddings=True` in `local_embedder.py`),
meaning each vector is scaled to unit length. For unit vectors, cosine
similarity — the standard way to compare embedding direction while ignoring
magnitude — reduces to a simple dot product, and cosine *distance*
(what Chroma returns) is `1 - cosine_similarity`. Normalizing at embed time
is what makes the vector store's `hnsw:space: "cosine"` configuration
(see [`04-vector-store-chromadb.md`](04-vector-store-chromadb.md)) behave
correctly and efficiently.

**Asymmetric query/passage embedding.** BGE models are trained with a
recommended asymmetry: queries can be embedded with a special instruction
prefix while passages/documents are embedded plain. This project supports
that via `cfg.embedding.query_prefix` in `config.yaml` (empty by default —
BGE's instruction prefix is optional for this model size, but the plumbing
is there if you want to experiment with it).

**Why a cache.** Embedding is the most CPU-intensive step in the ingest
pipeline. If you re-run ingestion after only adding one new PDF, you don't
want to re-embed every chunk from every other document — that wastes CPU
time for no benefit, since chunks whose text hasn't changed will produce
the exact same vector. A **content-hash cache** keyed by
`sha256(model_name + "\n" + text)` solves this: any chunk whose exact text
(and embedding model) hasn't changed before is a cache hit, skipping the
model entirely. This also means changing `cfg.embedding.model` in
`config.yaml` automatically invalidates the whole cache (different model
name → different hash → cache miss), so you never accidentally mix vectors
from two different models.

## In this repo

- `src/embed/cache.py`
  - `chunk_hash(text, model) -> str` — `sha256(f"{model}\n{text}")` hex
    digest. Used both as the embedding-cache key **and** (in
    `src/store/chroma_client.py`) as the Chroma document id — the same hash
    function ties the cache and the vector store's idempotency together.
  - `EmbeddingCache(cache_dir)` — on-disk cache storing each vector as an
    individual `<hash>.npy` file.
    - `EmbeddingCache.get(hash_) -> list[float] | None` — returns `None` on
      a miss or a corrupt/unreadable file (treated as a miss, not an error).
    - `EmbeddingCache.set(hash_, vector) -> None` — writes/overwrites the
      `.npy` file for that hash.

- `src/embed/local_embedder.py`
  - `get_model() -> SentenceTransformer` — lazily loads a module-level
    singleton `SentenceTransformer(cfg.embedding.model, device="cpu")` so the
    (fairly large) model weights are only loaded into memory once per
    process.
  - `_verify_dim(vector_len)` — raises a `RuntimeError` if the model's
    actual output dimension doesn't match `cfg.embedding.dim` (384) in
    `config.yaml` — a guard against silently shipping mismatched vectors
    into Chroma if you swap `cfg.embedding.model` without updating `dim`.
  - `embed_texts(texts, use_cache=True, is_query=False) -> list[list[float]]`
    — the main entry point. For each text: prepend `cfg.embedding.query_prefix`
    if `is_query=True`; compute `chunk_hash`; check the cache; batch every
    cache miss through `SentenceTransformer.encode(..., batch_size=cfg.embedding.batch_size,
    normalize_embeddings=True)`; write fresh vectors back to the cache.
    Order of the input list is preserved in the output.
  - `embed_query(text) -> list[float]` — convenience wrapper:
    `embed_texts([text], use_cache=True, is_query=True)[0]`. This is what
    `src/search/semantic_search.py` calls to embed a user's question with
    the **same model** used at ingest time — this consistency (same model,
    same normalization) is required for the query vector to live in the
    same vector space as the stored chunk vectors.

## Try it

Embed a small batch of ad-hoc strings from a Python REPL and inspect the
shape/normalization directly:

```powershell
python -c "from src.embed.local_embedder import embed_texts; import numpy as np; vecs = embed_texts(['overdraft fee policy', 'wire transfer limits'], use_cache=True); arr = np.array(vecs); print('shape:', arr.shape); print('norms:', np.linalg.norm(arr, axis=1))"
```

Then run it again — the second run should be near-instant since both
strings are now cache hits:

```powershell
python -c "import time; from src.embed.local_embedder import embed_texts; t0=time.time(); embed_texts(['overdraft fee policy', 'wire transfer limits']); print('elapsed:', time.time()-t0)"
```

Inspect the cache directory directly:

```powershell
python -c "from src.config import get_config; cfg = get_config(); import os; files = os.listdir(cfg.paths.embedding_cache); print(len(files), 'cached vectors'); print(files[:3])"
```

## What to look for / checkpoint

- `arr.shape` should be `(2, 384)` — two vectors, each 384-dimensional,
  matching `cfg.embedding.dim` in `config.yaml`.
- `norms` should print values very close to `1.0` for each row — proof the
  vectors are L2-normalized.
- The second timed run should be dramatically faster than a cold run
  (typically well under a second vs. potentially several seconds for model
  load + encode) — proof the cache is doing its job.
- `.cache/embeddings/` (or whatever `cfg.paths.embedding_cache` resolves
  to) contains one `.npy` file per unique `(model, text)` pair you've ever
  embedded.

## Teaching note

Have trainees delete one specific `.npy` file from the cache directory, then
re-run the same embed call and watch it repopulate — this demonstrates the
cache is a pure performance optimization with no correctness dependency:
losing cache entries never produces wrong vectors, only slower ones. Then
have them change `cfg.embedding.model` in `config.yaml` to a different
`sentence-transformers` model name (without re-ingesting) and predict, before
running anything, what will happen to `_verify_dim` if the new model's
output dimension doesn't match `cfg.embedding.dim` — this previews Exercise 2
in [`10-exercises.md`](10-exercises.md).
