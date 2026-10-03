# 04 — The Vector Store: ChromaDB

## Concept

A **vector store** indexes embedding vectors so that, given a new query
vector, it can quickly find the *k* nearest stored vectors (approximate
nearest-neighbor search) without brute-force comparing against every vector
in the corpus. [ChromaDB](https://www.trychroma.com/) is an open-source
vector database that this project uses as an embedded library (no separate
server process to run).

**Persistent vs. in-memory.** Chroma can run purely in memory (vectors
vanish when the process exits) or backed by files on disk via
`PersistentClient`. This project always uses `PersistentClient`, rooted at
`cfg.paths.chroma_db` (`data/chroma_db/`), so ingestion is a one-time cost:
run it once, and every later search/UI session reads the same on-disk index
without re-embedding anything.

**Cosine space.** Chroma's default distance metric is squared Euclidean
(L2) distance, but this project explicitly configures the collection with
`metadata={"hnsw:space": "cosine"}`. Since embeddings are already
L2-normalized (see [`03-embeddings-and-cache.md`](03-embeddings-and-cache.md)),
cosine distance is the metric that matches how the embedding model was
designed to be compared — configuring the space explicitly (rather than
relying on Chroma's default) makes that choice visible and intentional
rather than accidental.

**Unified collection with `doc_type` metadata.** Rather than one Chroma
collection per document type (one for PDFs, one for CSVs, one for SOPs),
this project uses a **single collection**, `cascade_docs`, and distinguishes
document types with a `doc_type` metadata field (`"pdf"` / `"sop"` /
`"csv"`) on every stored chunk. This is simpler to manage (one collection
to back up, reset, or inspect) and lets a single query search across *all*
document types at once, while `doc_type` metadata filtering (see
[`06-semantic-search.md`](06-semantic-search.md)) still lets you narrow a
search to one type when you want to.

**Idempotent upsert by content-hash id.** Every chunk's Chroma document
`id` is `chunk_hash(text, model)` — the same sha256 hash used for the
embedding cache (see lesson 03). Because Chroma's `upsert` (as opposed to
`add`) overwrites an existing document with the same id rather than
creating a duplicate, re-running ingestion on an unchanged corpus is
perfectly idempotent: chunks whose text hasn't changed get overwritten
with identical data (a no-op in effect), and only genuinely new or changed
chunks add new entries. This is what makes incremental re-ingestion (add
one new PDF, re-run `ingest.py --no-reset`) safe to run repeatedly without
accumulating duplicate chunks.

## In this repo

- `src/store/chroma_client.py`
  - `get_client() -> chromadb.ClientAPI` — returns a
    `chromadb.PersistentClient(path=str(cfg.paths.chroma_db))`.
  - `get_or_create_collection() -> Collection` — gets or creates the
    collection named `cfg.vector_store.collection` (`"cascade_docs"`),
    with `metadata={"hnsw:space": "cosine"}` and `embedding_function=None`
    (embeddings are always supplied explicitly by the caller — Chroma never
    computes its own embeddings in this project).
  - `_chunk_metadata(chunk) -> dict` — builds the metadata dict stored per
    chunk: `source_file`, `doc_type`, `chunk_index`, `chunk_start_offset`,
    `chunk_end_offset`, `page_number` (stored as `-1` when `None`, since
    Chroma metadata values must be scalar), `created_at`.
  - `upsert_chunks(chunks, embeddings) -> None` — computes each chunk's id
    via `chunk_hash(c["text"], model)`, then calls `collection.upsert(ids=...,
    embeddings=..., documents=..., metadatas=...)` in batches of
    `UPSERT_BATCH_SIZE` (1000) to stay under Chroma's per-call limits.
  - `collection_stats() -> dict` — returns
    `{"total": int, "by_doc_type": {...}, "collection": name}` by calling
    `collection.count()` and tallying `metadatas` by `doc_type` with a
    `collections.Counter`.
  - `reset_collection() -> None` — deletes the collection (ignoring errors
    if it doesn't exist yet) and recreates it empty via
    `get_or_create_collection()`. Used for a clean rebuild.

- `src/store/ingest_to_chroma.py`
  - `ingest_all(reset=False) -> dict` — the end-to-end ingest orchestration:
    optionally `reset_collection()`, then `chunk_all()` (lesson 02), then
    `embed_texts(texts, use_cache=True, is_query=False)` (lesson 03), then
    `upsert_chunks(chunks, embeddings)`, then returns `collection_stats()`.
    Runnable standalone with `python -m src.store.ingest_to_chroma [--reset]`.

## Try it

Run a clean ingest (chunk → embed → upsert), wiping any existing
collection first:

```powershell
python -m src.store.ingest_to_chroma --reset
```

Or an incremental ingest (no reset — upserts on top of whatever's already
there, useful after adding a new document to `data/raw/`):

```powershell
python -m src.store.ingest_to_chroma
```

Check the on-disk footprint Chroma created:

```powershell
python -c "from src.config import get_config; import os; print(get_config().paths.chroma_db); print(os.listdir(get_config().paths.chroma_db))"
```

Prove idempotency: run `--reset` ingestion twice in a row and confirm the
total chunk count is identical both times (it must be — the second run
upserts the exact same content-hash ids over the first run's data):

```powershell
python -m src.store.ingest_to_chroma --reset
python -m src.store.ingest_to_chroma --reset
```

## What to look for / checkpoint

- After ingestion, `data/chroma_db/` contains Chroma's SQLite file and
  HNSW index segment directories — this is the persistent, on-disk state
  that survives process restarts.
- `ingest_all`'s final printed line, `Done. Collection stats: {...}`, shows
  `"total"` matching the chunk count from `chunk_all()`'s own summary
  (lesson 02), and `"by_doc_type"` broken down into `pdf` / `sop` / `csv`
  keys.
- Running `--reset` ingestion twice produces the **same** total chunk
  count both times — proof that upsert-by-content-hash prevents
  duplicates, since the second run's chunks have identical text (and thus
  identical ids) to the first run's.

## Teaching note

Ask trainees to predict, before running it: if you edit even one character
of one paragraph in `src/data_gen/common.py`'s templates and regenerate the
corpus, then re-run `ingest_all(reset=False)` (no reset) — will the total
chunk count in `collection_stats()` go up, stay the same, or could it do
either? The honest answer is "it depends": if only some chunks' text
changed, their content hashes change too, so they upsert as *new* ids
alongside the old (now-orphaned) ones still sitting in the collection —
`upsert` does not delete stale entries for a source file, it only
overwrites entries whose id matches exactly. This is why `ingest.py`'s
default behavior is a full `reset=True` rebuild, and `--no-reset` is
explicitly the advanced/incremental option.
