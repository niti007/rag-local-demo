"""
ChromaDB persistent vector store wrapper for the RAG teaching demo.

Embeddings are computed locally (see `src.embed.local_embedder`) and passed
in explicitly, so the Chroma collection is created *without* an embedding
function attached. The collection uses cosine similarity
(`metadata={"hnsw:space": "cosine"}`) since our embeddings are L2-normalized.

Chunk ids are content-hash based (sha256 of model + chunk text, see
`src.embed.cache.chunk_hash`), so re-ingesting identical chunks upserts in
place rather than creating duplicates.
"""

from __future__ import annotations

from collections import Counter

import chromadb
from chromadb.api.models.Collection import Collection

from src.config import get_config
from src.embed.cache import chunk_hash

UPSERT_BATCH_SIZE = 1000


def get_client() -> chromadb.ClientAPI:
    """Return a PersistentClient rooted at cfg.paths.chroma_db."""
    cfg = get_config()
    return chromadb.PersistentClient(path=str(cfg.paths.chroma_db))


def get_or_create_collection() -> Collection:
    """
    Get (or create) the configured Chroma collection, using cosine distance
    and no attached embedding function (embeddings are always supplied by
    the caller).
    """
    cfg = get_config()
    client = get_client()
    return client.get_or_create_collection(
        name=cfg.vector_store.collection,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


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


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    Upsert `chunks` (with their matching `embeddings`) into the collection.

    Idempotent: each chunk's id is the content hash of (model, chunk text),
    so re-ingesting the same chunk text overwrites the existing entry rather
    than creating a duplicate. Batches upserts in groups of
    `UPSERT_BATCH_SIZE` to stay well under Chroma's per-call limits.
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
    collection = get_or_create_collection()

    ids = [chunk_hash(c["text"], model) for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [_chunk_metadata(c) for c in chunks]

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        end = start + UPSERT_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def collection_stats() -> dict:
    """Return {"total": int, "by_doc_type": {...}, "collection": name}."""
    cfg = get_config()
    collection = get_or_create_collection()

    total = collection.count()

    by_doc_type: Counter[str] = Counter()
    if total:
        result = collection.get(include=["metadatas"])
        for metadata in result.get("metadatas") or []:
            doc_type = (metadata or {}).get("doc_type", "unknown")
            by_doc_type[doc_type] += 1

    return {
        "total": total,
        "by_doc_type": dict(by_doc_type),
        "collection": cfg.vector_store.collection,
    }


def reset_collection() -> None:
    """Delete the collection (if it exists) and recreate it empty, with the
    same cosine-space configuration. Use before a clean re-ingest."""
    cfg = get_config()
    client = get_client()
    try:
        client.delete_collection(name=cfg.vector_store.collection)
    except Exception:
        # Collection didn't exist yet; nothing to delete.
        pass
    get_or_create_collection()
