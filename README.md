"""Sanity checks for the central config loader (src.config.get_config)."""

from __future__ import annotations

from pathlib import Path

from src.config import get_config


def test_get_config_loads():
    cfg = get_config()
    assert cfg is not None


def test_paths_are_absolute():
    cfg = get_config()
    for field_name in type(cfg.paths).model_fields:
        value = getattr(cfg.paths, field_name)
        assert isinstance(value, Path)
        assert value.is_absolute(), f"paths.{field_name} is not absolute: {value}"


def test_embedding_dim_is_384():
    cfg = get_config()
    assert cfg.embedding.dim == 384


def test_embedding_model_name():
    cfg = get_config()
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"


def test_collection_name():
    cfg = get_config()
    assert cfg.vector_store.collection == "cascade_docs"


def test_generation_is_openai():
    cfg = get_config()
    assert cfg.generation.provider == "openai"
    assert cfg.generation.base_url == "https://api.openai.com/v1"
