"""
End-to-end smoke test: semantic search + cross-encoder rerank against the
already-populated Qdrant collection (cascade_docs on the Docker server).

Marked slow because it loads the local embedder + cross-encoder models. If
Qdrant is unreachable or the collection is empty (e.g. pipeline hasn't been run in this environment
yet), the test skips rather than failing.
"""

from __future__ import annotations

import pytest

from src.config import get_config
from src.store.qdrant_store import collection_stats


@pytest.mark.slow
def test_search_and_rerank_smoke():
    try:
        stats = collection_stats()
    except RuntimeError as exc:  # raised by get_client() when Qdrant is down
        pytest.skip(f"Qdrant unreachable ({exc}); run `docker compose up -d`.")
    if not stats["total"]:
        pytest.skip("Qdrant collection is empty; run the ingest pipeline first.")

    from src.rerank.cross_encoder_rerank import search_and_rerank

    cfg = get_config()
    result = search_and_rerank("wire transfer authorization")

    assert result["retrieved"], "expected non-empty retrieved results"

    reranked = result["reranked"]
    assert len(reranked) <= cfg.rerank.top_n
    assert len(reranked) <= len(result["retrieved"])

    for item in reranked:
        assert "text" in item
        assert "metadata" in item
        assert "score" in item
