"""Unit tests for the on-disk embedding cache (src.embed.cache)."""

from __future__ import annotations

import pytest

from src.embed.cache import EmbeddingCache, chunk_hash


def test_chunk_hash_is_deterministic():
    h1 = chunk_hash("hello world", "BAAI/bge-small-en-v1.5")
    h2 = chunk_hash("hello world", "BAAI/bge-small-en-v1.5")
    assert h1 == h2


def test_chunk_hash_differs_for_different_text():
    h1 = chunk_hash("hello world", "BAAI/bge-small-en-v1.5")
    h2 = chunk_hash("goodbye world", "BAAI/bge-small-en-v1.5")
    assert h1 != h2


def test_chunk_hash_differs_for_different_model():
    h1 = chunk_hash("hello world", "model-a")
    h2 = chunk_hash("hello world", "model-b")
    assert h1 != h2


def test_cache_round_trip(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb_cache")
    h = chunk_hash("hello world", "model-a")
    vector = [0.1, 0.2, 0.3, 0.4]

    cache.set(h, vector)
    result = cache.get(h)

    assert result is not None
    assert len(result) == len(vector)
    for expected, actual in zip(vector, result):
        assert actual == pytest.approx(expected, abs=1e-6)


def test_cache_missing_key_returns_none(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb_cache")
    assert cache.get(chunk_hash("never cached", "model-a")) is None
