"""
Cross-encoder reranking for the RAG teaching demo.

Takes a shortlist of semantic-search candidates and re-scores them with a
cross-encoder (query, passage) relevance model (cross-encoder/ms-marco-MiniLM-L-6-v2
by default, via config.yaml), which is more accurate than bi-encoder cosine
similarity but too slow to run over the full corpus -- hence "retrieve broad,
then rerank the shortlist".

`search_and_rerank` ties semantic_search + rerank together and returns both
the pre-rerank ("retrieved") and post-rerank ("reranked") orderings so a UI
(or this module's CLI) can show a clear before/after comparison.
"""

from __future__ import annotations

import argparse
import copy

from sentence_transformers import CrossEncoder

from src.config import get_config

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Return the module-level CrossEncoder singleton, loading it (on CPU)
    on first use."""
    global _reranker
    if _reranker is None:
        cfg = get_config()
        _reranker = CrossEncoder(cfg.rerank.model, device="cpu")
    return _reranker


def rerank(query: str, results: list[dict], top_n: int | None = None) -> list[dict]:
    """
    Re-score `results` (as produced by semantic_search) against `query` using
    a cross-encoder, and return the top `top_n` re-sorted by relevance.

    Does not mutate the input list/dicts. Each returned dict is a copy of the
    original with two new keys added:
        "rerank_score": float  (raw cross-encoder logit, higher = better)
        "rerank_rank":  int    (1-based rank after reranking)
    The original "score" / "distance" (from semantic search) are preserved.
    """
    if not results:
        return []

    cfg = get_config()
    n = top_n or cfg.rerank.top_n

    model = get_reranker()
    pairs = [(query, r["text"]) for r in results]
    raw_scores = model.predict(pairs)

    scored = []
    for r, s in zip(results, raw_scores):
        r_copy = copy.deepcopy(r)
        r_copy["rerank_score"] = float(s)
        scored.append(r_copy)

    scored.sort(key=lambda r: r["rerank_score"], reverse=True)

    top = scored[:n]
    for i, r in enumerate(top, start=1):
        r["rerank_rank"] = i

    return top


def search_and_rerank(
    query: str,
    top_k: int | None = None,
    top_n: int | None = None,
    doc_type: str | None = None,
    use_rerank: bool = True,
) -> dict:
    """
    Run semantic_search to get `top_k` candidates, then (optionally) rerank
    them down to `top_n` with the cross-encoder.

    Returns:
        {
            "query": query,
            "retrieved": [...],   # full top_k semantic-search results (pre-rerank order)
            "reranked": [...],    # top_n results (post-rerank order if use_rerank else
                                   # just the top_n of the semantic order)
            "used_rerank": bool,
        }
    """
    # Local import to avoid a hard import-time dependency between the two
    # modules for callers that only need one of them.
    from src.search.semantic_search import semantic_search

    cfg = get_config()
    n = top_n or cfg.rerank.top_n

    retrieved = semantic_search(query, top_k=top_k, doc_type=doc_type)

    if use_rerank:
        reranked = rerank(query, retrieved, top_n=n)
    else:
        reranked = copy.deepcopy(retrieved[:n])
        for i, r in enumerate(reranked, start=1):
            r["rerank_rank"] = i

    return {
        "query": query,
        "retrieved": retrieved,
        "reranked": reranked,
        "used_rerank": use_rerank,
    }


def _fmt_snippet(r: dict) -> str:
    meta = r.get("metadata") or {}
    source_file = meta.get("source_file", "?")
    page = meta.get("page", "?")
    snippet = (r.get("text") or "")[:120].replace("\n", " ")
    return f"source={source_file} page={page} :: {snippet}"


def _print_before_after(result: dict) -> None:
    retrieved = result["retrieved"]
    reranked = result["reranked"]

    print(f"Query: {result['query']!r}")
    print(f"Reranking: {'ON' if result['used_rerank'] else 'OFF'}")
    print()

    print("=== BEFORE (semantic search order) ===")
    for rank, r in enumerate(retrieved[: len(reranked)], start=1):
        print(f"[{rank}] score={r['score']:.4f}  {_fmt_snippet(r)}")

    print()
    print("=== AFTER (cross-encoder rerank order) ===")
    for r in reranked:
        rr = r.get("rerank_score")
        rr_str = f"{rr:.4f}" if rr is not None else "n/a"
        print(f"[{r['rerank_rank']}] rerank_score={rr_str}  score={r['score']:.4f}  {_fmt_snippet(r)}")

    print()
    old_top = retrieved[0]["id"] if retrieved else None
    new_top = reranked[0]["id"] if reranked else None
    if old_top == new_top:
        print(f"Top-1 unchanged: {new_top}")
    else:
        print(f"Top-1 changed: {old_top!r} -> {new_top!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search + cross-encoder rerank")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--k", type=int, default=None, help="Semantic search top_k (default: cfg.search.top_k)")
    parser.add_argument("--n", type=int, default=None, help="Rerank top_n (default: cfg.rerank.top_n)")
    parser.add_argument("--doc-type", default=None, help="Restrict to a specific doc_type metadata value")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")
    args = parser.parse_args()

    result = search_and_rerank(
        args.query,
        top_k=args.k,
        top_n=args.n,
        doc_type=args.doc_type,
        use_rerank=not args.no_rerank,
    )
    _print_before_after(result)


if __name__ == "__main__":
    main()
