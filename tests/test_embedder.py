"""Embedding model smoke test: loads the real local sentence-transformer
model (BAAI/bge-small-en-v1.5), so it's marked slow. Kept fast in practice
because the model is already cached locally on disk."""

from __future__ import annotations

import math

import pytest

from src.config import get_config
from src.embed.local_embedder import embed_texts


@pytest.mark.slow
def test_embed_texts_returns_normalized_384dim_vectors():
    cfg = get_config()

    vectors = embed_texts(
        ["wire transfer authorization limits", "overdraft protection policy"],
        use_cache=True,
        is_query=False,
    )

    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == cfg.embedding.dim == 384
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=1e-3)
