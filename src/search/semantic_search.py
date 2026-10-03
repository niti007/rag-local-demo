"""
Semantic (vector) search over the Qdrant collection for the RAG teaching demo.

Embeds the query with the local embedder, queries the Qdrant collection
(cosine distance), and returns a flat, score-sorted list of plain-dict
results:

    {
        "id": str,
        "text": str,
        "metadata": dict,
        "distance": float,   # 1 - score (cosine distance; lower = closer)
        "score": float,      # cosine similarity straight from Qdrant (higher = better)
    }
"""

from __future__ import annotations

import argparse

from src.config import get_config
from src.embed.local_embedder import embed_query
from src.store.qdrant_store import build_filter, get_client, get_or_create_collection


def semantic_search(
    query: str,
    top_k: int | None = None,
    doc_type: str | None = None,
) -> list[dict]:
    """
    Run a semantic search for `query` against the Qdrant collection.

    Args:
        query: natural-language query string.
        top_k: number of results to request (defaults to cfg.search.top_k).
        doc_type: if given, restricts results to payload.doc_type == doc_type.

    Returns:
        List of result dicts (see module docstring), sorted by `score` desc.
    """
    cfg = get_config()
    k = top_k or cfg.search.top_k

    query_vec = embed_query(query)

    collection = get_or_create_collection()
    client = get_client()

    # Optional payload filter (uses the keyword index on doc_type).
    query_filter = build_filter(doc_type=doc_type)

    points = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    results: list[dict] = []
    for point in points:
        payload = dict(point.payload or {})
        chunk_id = payload.pop("chunk_id", str(point.id))
        text = payload.pop("text", "")
        # Qdrant's COSINE score is already a similarity (higher = closer);
        # keep a `distance` field so downstream result shape is unchanged.
        score = float(point.score)
        results.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": payload,
                "distance": 1.0 - score,
                "score": score,
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
