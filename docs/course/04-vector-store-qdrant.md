# 04 — The Vector Store: Qdrant

> Note: earlier versions of this demo used an embedded ChromaDB store. It was
> migrated to a Qdrant server running in Docker; the public API of
> `src/store/` (and every result-dict shape downstream) stayed the same.

## Concept

A **vector store** indexes embedding vectors so that, given a new query
vector, it can quickly find the *k* nearest stored vectors (approximate
nearest-neighbor search) without brute-force comparing against every vector
in the corpus. [Qdrant](https://qdrant.tech/) is an open-source vector
database written in Rust. This project runs it as a **server in Docker** and
talks to it over HTTP.

**Client/server vs. embedded.** An embedded database (like SQLite or the
previous Chroma setup) lives inside your Python process and reads files
directly. A client/server database runs as its own process: your code holds
a thin *client* that sends requests (`QdrantClient(url=...)`), and the server
owns the index and the on-disk storage. Practical consequences: you must
start the server first (`docker compose up -d`); several programs (ingest,
the chat UI, a notebook) can share one store at once; and moving to a hosted
service such as Qdrant Cloud only means changing `QDRANT_URL` and
`QDRANT_API_KEY` in `.env` — no code changes.

**Persistence.** `docker-compose.yml` mounts `./data/qdrant_storage` into the
container, so the index survives `docker compose down` and restarts.
Ingestion stays a one-time cost: run it once, and every later search/UI
session reads the same index without re-embedding anything.

**Cosine distance and `score` semantics.** The collection is created with
`VectorParams(size=384, distance=Distance.COSINE)`. Since embeddings are
already L2-normalized (see [`03-embeddings-and-cache.md`](03-embeddings-and-cache.md)),
cosine is the metric the embedding model was designed for. Importantly,
Qdrant's `score` for cosine is a **similarity** (higher = closer, 1.0 =
identical direction), *not* a distance. Our search code therefore uses
`score = point.score` directly and reports `distance = 1 - score` so result
dicts keep the same shape as before.

**Payload and payload indexes.** Each stored *point* has an id, a vector, and
a JSON **payload**. We put the chunk `text`, its `chunk_id` and the chunk
metadata (`source_file`, `doc_type`, `chunk_index`, offsets, `page_number`,
`created_at`) in the payload. Filtering is done with a payload `Filter`
(e.g. `doc_type == "sop"`) alongside the vector search. To make those filters
fast, we create **keyword payload indexes** on `doc_type` and `source_file`
when the collection is created.

**Unified collection with `doc_type` payload.** Rather than one collection
per document type, there is a **single collection**, `cascade_docs`, with a
`doc_type` payload field (`"pdf"` / `"sop"` / `"csv"`). One collection is
simpler to reset or inspect, and a query can search across *all* types or be
narrowed by filter (see [`06-semantic-search.md`](06-semantic-search.md)).

**Idempotent upsert; UUID ids derived from the content hash.** Qdrant point
ids must be unsigned integers or UUIDs, so a raw sha256 hex string will not
do. Each chunk's content hash is `chunk_hash(text, model)` (the same hash
used for the embedding cache, lesson 03); the point id is
`str(uuid.UUID(hex=hash[:32]))` — the first 128 bits of the hash formatted as
a UUID. The full hash is stored in the payload as `chunk_id` and is what the
rest of the app calls the chunk's `"id"`. Because `upsert` overwrites a point
with the same id, re-running ingestion on an unchanged corpus is idempotent:
unchanged chunks are overwritten with identical data, and only new or changed
chunks add points.

## In this repo

- `docker-compose.yml` — one `qdrant` service (`qdrant/qdrant`, pinned tag),
  REST/dashboard on `6333`, gRPC on `6334`, storage in `data/qdrant_storage`.
- `src/store/qdrant_store.py` (named so it doesn't shadow the `qdrant_client`
  package)
  - `get_client() -> QdrantClient` — cached client for `QDRANT_URL` (env) or
    `vector_store.url` in `config.yaml`, with `QDRANT_API_KEY` if set. If the
    server can't be reached it raises a `RuntimeError` telling you to run
    `docker compose up -d`.
  - `get_or_create_collection() -> str` — creates the collection (cosine,
    384 dims) plus keyword indexes on `doc_type` / `source_file` if missing,
    and returns its name.
  - `_chunk_metadata(chunk) -> dict` — the metadata stored per chunk
    (`page_number` is `-1` when missing).
  - `upsert_chunks(chunks, embeddings)` — builds `PointStruct`s (UUID id,
    vector, payload with `text` + `chunk_id` + metadata) and upserts them in
    batches of `UPSERT_BATCH_SIZE` with `wait=True`.
  - `scroll_all(doc_type=None, source_file=None, with_vectors=False)` —
    pages through points with `client.scroll` (cursor-based, via
    `next_page_offset`) and returns `{id, text, metadata, vector?}` dicts.
    Used by the inspection tools (lesson 05).
  - `collection_stats() -> dict` — `{"total", "by_doc_type", "collection"}`;
    `total` from `client.count(exact=True)`, the breakdown by paging payloads.
  - `reset_collection()` — deletes and recreates the collection.

- `src/store/ingest_to_qdrant.py`
  - `ingest_all(reset=False) -> dict` — optional `reset_collection()`, then
    `chunk_all()` (lesson 02), `embed_texts(...)` (lesson 03),
    `upsert_chunks(...)`, and returns `collection_stats()`.
    Runnable with `python -m src.store.ingest_to_qdrant [--reset]`.

## Try it

Start the server and check it is healthy:

```bash
docker compose up -d
curl localhost:6333/readyz
```

Run a clean ingest (chunk → embed → upsert), wiping any existing collection:

```bash
python -m src.store.ingest_to_qdrant --reset
```

Or an incremental ingest (upserts on top of what is already there):

```bash
python -m src.store.ingest_to_qdrant
```

Browse the data in the Qdrant dashboard at
<http://localhost:6333/dashboard> (Collections → `cascade_docs`).

Prove idempotency: run ingestion twice and confirm the total is unchanged:

```bash
python -m src.store.ingest_to_qdrant --reset
python -m src.store.ingest_to_qdrant
```

## What to look for / checkpoint

- `data/qdrant_storage/` is populated by the container — this is the
  persistent state that survives restarts.
- `ingest_all`'s final line, `Done. Collection stats: {...}`, shows `"total"`
  matching the chunk count from `chunk_all()` (lesson 02), broken down into
  `pdf` / `sop` / `csv`.
- The second ingest (without `--reset`) reports the **same** total — proof
  that upsert-by-content-hash prevents duplicates.
- If Qdrant isn't running you get a clear `RuntimeError` pointing at
  `docker compose up -d`, not a cryptic connection trace.

## Teaching note

Ask trainees to predict: if you edit one character of one paragraph in
`src/data_gen/common.py`'s templates, regenerate the corpus, then re-run
`ingest_all(reset=False)` — will `collection_stats()["total"]` go up, stay
the same, or either? It depends: changed chunks get new content hashes, so
they upsert as *new* points alongside the old (now-orphaned) ones —
`upsert` only overwrites points whose id matches exactly. This is why
`ingest.py` defaults to a full `reset=True` rebuild and `--no-reset` is the
advanced, incremental option.
