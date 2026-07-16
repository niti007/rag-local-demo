"""
Semantic (vector) search over the Chroma collection for the RAG teaching demo.

Embeds the query with the local embedder, queries the Chroma collection
(cosine distance space), and returns a flat, score-sorted list of plain-dict
results:

    {
        "id": str,
        "text": str,
        "metadata": dict,
        "distance": float,   # raw cosine distance from Chroma (lower = closer)
        "score": float,      # 1 - distance (cosine similarity, higher = better)
    }
"""

from __future__ import annotations

import argparse

from src.config import get_config
from src.embed.local_embedder import embed_query
from src.store.chroma_client import get_or_create_collection


def semantic_search(
    query: str,
    top_k: int | None = None,
    doc_type: str | None = None,
) -> list[dict]:
    """
    Run a semantic search for `query` against the Chroma collection.

    Args:
        query: natural-language query string.
        top_k: number of results to request (defaults to cfg.search.top_k).
        doc_type: if given, restricts results to metadata.doc_type == doc_type.

    Returns:
        List of result dicts (see module docstring), sorted by `score` desc.
    """
    cfg = get_config()
    k = top_k or cfg.search.top_k

    query_vec = embed_query(query)

    collection = get_or_create_collection()

    where = {"doc_type": doc_type} if doc_type is not None else None

    query_kwargs = dict(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    if where is not None:
        query_kwargs["where"] = where

    raw = collection.query(**query_kwargs)

    ids = raw.get("ids", [[]])[0] if raw.get("ids") else []
    documents = raw["documents"][0] if raw.get("documents") else []
    metadatas = raw["metadatas"][0] if raw.get("metadatas") else []
    distances = raw["distances"][0] if raw.get("distances") else []

    results: list[dict] = []
    for i in range(len(documents)):
        distance = distances[i]
        results.append(
            {
                "id": ids[i] if i < len(ids) else None,
                "text": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distance,
                "score": 1.0 - distance,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _print_results(results: list[dict]) -> None:
    if not results:
        print("(no results)")
        return
    for rank, r in enumerate(results, start=1):
        meta = r.get("metadata") or {}
        source_file = meta.get("source_file", "?")
        page = meta.get("page", "?")
        snippet = (r.get("text") or "")[:120].replace("\n", " ")
        print(f"[{rank}] score={r['score']:.4f}  source={source_file}  page={page}")
        print(f"    {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over the RAG corpus")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--k", type=int, default=None, help="Number of results (default: cfg.search.top_k)")
    parser.add_argument("--doc-type", default=None, help="Restrict to a specific doc_type metadata value")
    args = parser.parse_args()

    results = semantic_search(args.query, top_k=args.k, doc_type=args.doc_type)
    _print_results(results)


if __name__ == "__main__":
    main()
