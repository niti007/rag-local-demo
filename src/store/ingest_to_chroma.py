"""
End-to-end ingest pipeline: chunk source documents, embed them locally
(with caching), and upsert into the Chroma vector store.

Run as:
    python -m src.store.ingest_to_chroma            # incremental (upsert)
    python -m src.store.ingest_to_chroma --reset     # wipe collection first
"""

from __future__ import annotations

import argparse

from src.embed.local_embedder import embed_texts
from src.ingest.chunk import chunk_all
from src.store.chroma_client import collection_stats, reset_collection, upsert_chunks


def ingest_all(reset: bool = False) -> dict:
    """
    Chunk all configured source documents, embed every chunk (using the
    on-disk embedding cache), and upsert them into the Chroma collection.

    If `reset` is True, the collection is deleted and recreated first, so
    the ingest is a clean rebuild rather than an incremental upsert.

    Returns the collection stats dict after ingest.
    """
    if reset:
        print("Resetting collection...")
        reset_collection()

    print("Chunking source documents...")
    chunks = chunk_all()
    print(f"  {len(chunks)} chunks produced")

    print("Embedding chunks (cached)...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, use_cache=True, is_query=False)

    print("Upserting into Chroma...")
    upsert_chunks(chunks, embeddings)

    stats = collection_stats()
    print(f"Done. Collection stats: {stats}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before ingesting (clean rebuild).",
    )
    args = parser.parse_args()
    ingest_all(reset=args.reset)
