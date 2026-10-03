"""
Qdrant vector store wrapper for the RAG teaching demo.

Qdrant runs as a separate *server* (see `docker-compose.yml`) and we talk to
it over HTTP -- unlike an embedded database, the data lives in the server
process, not inside our Python process. For Qdrant Cloud, only the URL and
API key in `.env` change (QDRANT_URL / QDRANT_API_KEY).

Embeddings are computed locally (see `src.embed.local_embedder`) and passed
in explicitly. The collection uses COSINE distance since our embeddings are
L2-normalized. Note that Qdrant's cosine `score` is a *similarity* (higher =
closer), unlike a distance.

Identity: Qdrant point ids must be unsigned ints or UUIDs, so we derive a
deterministic UUID from the content hash (sha256 of model + chunk text, see
`src.embed.cache.chunk_hash`). Re-ingesting identical chunks therefore
upserts in place rather than creating duplicates. The full hash is also kept
in the payload as `chunk_id` and is what the rest of the app calls the
chunk's "id".

Metadata lives in each point's *payload* (a JSON object). We create keyword
payload indexes on `doc_type` and `source_file` so filtered search is fast.
"""

from __future__ import annotations

import uuid
from collections import Counter
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config import get_config, resolved_qdrant_url
from src.embed.cache import chunk_hash

UPSERT_BATCH_SIZE = 256
SCROLL_PAGE_SIZE = 256

# Payload fields we filter on; indexed as keywords when the collection is made.
INDEXED_FIELDS = ("doc_type", "source_file")


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """
    Return a (cached) QdrantClient for the configured server.

    The URL is QDRANT_URL if set, else `vector_store.url` in config.yaml.
    Raises a RuntimeError with a how-to-fix hint if the server is unreachable.
    """
    cfg = get_config()
    url = resolved_qdrant_url()
    client = QdrantClient(url=url, api_key=cfg.qdrant_api_key)
    try:
        client.get_collections()  # cheap round-trip to prove the server is up
    except Exception as exc:
        get_client.cache_clear()
        raise RuntimeError(
            f"Could not connect to Qdrant at {url} ({exc}). "
            "Start it with `docker compose up -d` and try again."
        ) from exc
    return client


def point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk hash (first 128 bits of the sha256)."""
    return str(uuid.UUID(hex=chunk_id[:32]))


def get_or_create_collection() -> str:
    """
    Ensure the configured collection exists (cosine distance, vector size from
    config, keyword payload indexes) and return its name.
    """
    cfg = get_config()
    client = get_client()
    name = cfg.vector_store.collection
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=cfg.embedding.dim, distance=qm.Distance.COSINE
            ),
        )
        # Payload indexes make `doc_type` / `source_file` filters efficient.
        for field in INDEXED_FIELDS:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
    return name


def _chunk_metadata(chunk: dict) -> dict:
    page_number = chunk.get("page_number")
    return {
        "source_file": chunk["source_file"],
        "doc_type": chunk["doc_type"],
        "chunk_index": chunk["chunk_index"],
        "chunk_start_offset": chunk["chunk_start_offset"],
        "chunk_end_offset": chunk["chunk_end_offset"],
        "page_number": page_number if page_number is not None else -1,
        "created_at": chunk["created_at"],
    }


def build_filter(
    doc_type: str | None = None, source_file: str | None = None
) -> qm.Filter | None:
    """Build an AND-of-equalities payload filter, or None if no filters given."""
    must = []
    if doc_type is not None:
        must.append(qm.FieldCondition(key="doc_type", match=qm.MatchValue(value=doc_type)))
    if source_file is not None:
        must.append(
            qm.FieldCondition(key="source_file", match=qm.MatchValue(value=source_file))
        )
    return qm.Filter(must=must) if must else None


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    Upsert `chunks` (with their matching `embeddings`) into the collection.

    Idempotent: each point's id is derived from the content hash of
    (model, chunk text), so re-ingesting the same chunk text overwrites the
    existing point rather than creating a duplicate. Upserts are sent in
    batches of `UPSERT_BATCH_SIZE` and `wait=True` blocks until they're applied.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
            "must be the same length"
        )
    if not chunks:
        return

    cfg = get_config()
    model = cfg.embedding.model
    name = get_or_create_collection()
    client = get_client()

    points = []
    for chunk, vector in zip(chunks, embeddings):
        cid = chunk_hash(chunk["text"], model)
        payload = {"text": chunk["text"], "chunk_id": cid, **_chunk_metadata(chunk)}
        points.append(qm.PointStruct(id=point_id(cid), vector=list(vector), payload=payload))

    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=name,
            points=points[start : start + UPSERT_BATCH_SIZE],
            wait=True,
        )


def scroll_all(
    doc_type: str | None = None,
    source_file: str | None = None,
    with_vectors: bool = False,
) -> list[dict]:
    """
    Page through every point matching the optional filters.

    Returns dicts shaped {"id", "text", "metadata", "vector"?}, where "id" is
    the full chunk hash and "metadata" is the payload minus text/chunk_id.
    `scroll` is cursor-based: each call returns a page plus the offset of the
    next one (None when done).
    """
    client = get_client()
    name = get_or_create_collection()
    flt = build_filter(doc_type, source_file)

    out: list[dict] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=name,
            scroll_filter=flt,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        for rec in records:
            payload = dict(rec.payload or {})
            item = {
                "id": payload.pop("chunk_id", str(rec.id)),
                "text": payload.pop("text", ""),
                "metadata": payload,
            }
            if with_vectors:
                item["vector"] = rec.vector
            out.append(item)
        if offset is None:
            break
    return out


def collection_stats() -> dict:
    """Return {"total": int, "by_doc_type": {...}, "collection": name}."""
    name = get_or_create_collection()
    client = get_client()

    total = client.count(collection_name=name, exact=True).count

    by_doc_type: Counter[str] = Counter()
    if total:
        # Page through payloads only (no vectors, no text needed).
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=name,
                limit=SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=["doc_type"],
                with_vectors=False,
            )
            for rec in records:
                by_doc_type[(rec.payload or {}).get("doc_type", "unknown")] += 1
            if offset is None:
                break

    return {
        "total": total,
        "by_doc_type": dict(by_doc_type),
        "collection": name,
    }


def reset_collection() -> None:
    """Delete the collection (if it exists) and recreate it empty, with the
    same cosine config and payload indexes. Use before a clean re-ingest."""
    cfg = get_config()
    client = get_client()
    name = cfg.vector_store.collection
    if client.collection_exists(name):
        client.delete_collection(collection_name=name)
    get_or_create_collection()
