"""
Local embedding model wrapper for the RAG teaching demo.

Wraps a single SentenceTransformer model (BAAI/bge-small-en-v1.5 by default,
via config.yaml) as a lazily-loaded, module-level singleton running on CPU.
Provides batch text embedding with an on-disk cache (see `src.embed.cache`)
and a convenience helper for embedding a single query string.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.config import get_config
from src.embed.cache import EmbeddingCache, chunk_hash

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the module-level SentenceTransformer singleton, loading it
    (on CPU) on first use."""
    global _model
    if _model is None:
        cfg = get_config()
        _model = SentenceTransformer(cfg.embedding.model, device="cpu")
    return _model


def _get_cache() -> EmbeddingCache:
    cfg = get_config()
    return EmbeddingCache(cfg.paths.embedding_cache)


def _verify_dim(vector_len: int) -> None:
    cfg = get_config()
    if vector_len != cfg.embedding.dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: model '{cfg.embedding.model}' "
            f"produced vectors of length {vector_len}, but config.yaml "
            f"declares embedding.dim = {cfg.embedding.dim}. Update config.yaml "
            f"or check the model name."
        )


def embed_texts(
    texts: list[str], use_cache: bool = True, is_query: bool = False
) -> list[list[float]]:
    """
    Embed a list of texts, returning one L2-normalized vector per input text
    (order preserved).

    If `is_query` is True, `cfg.embedding.query_prefix` (if any) is prepended
    to each text before embedding -- this matches bge's recommended usage,
    where queries and passages are embedded asymmetrically.

    If `use_cache` is True, each text's embedding is looked up in the on-disk
    cache (keyed by sha256 of model name + text, *after* prefixing) first;
    only cache misses are sent through the model, in batches of
    `cfg.embedding.batch_size`. Newly computed embeddings are written back to
    the cache.
    """
    cfg = get_config()
    model = cfg.embedding.model

    if is_query and cfg.embedding.query_prefix:
        prepared = [f"{cfg.embedding.query_prefix}{t}" for t in texts]
    else:
        prepared = list(texts)

    cache = _get_cache() if use_cache else None

    results: list[list[float] | None] = [None] * len(prepared)
    hashes = [chunk_hash(t, model) for t in prepared]

    to_encode_indices: list[int] = []
    if cache is not None:
        for i, h in enumerate(hashes):
            cached = cache.get(h)
            if cached is not None:
                results[i] = cached
            else:
                to_encode_indices.append(i)
    else:
        to_encode_indices = list(range(len(prepared)))

    dim_checked = False
    batch_size = cfg.embedding.batch_size
    st_model = get_model()

    for start in range(0, len(to_encode_indices), batch_size):
        batch_indices = to_encode_indices[start : start + batch_size]
        batch_texts = [prepared[i] for i in batch_indices]

        vectors = st_model.encode(
            batch_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        for idx, vector in zip(batch_indices, vectors):
            vec_list = vector.astype(float).tolist()
            if not dim_checked:
                _verify_dim(len(vec_list))
                dim_checked = True
            results[idx] = vec_list
            if cache is not None:
                cache.set(hashes[idx], vec_list)

    # All entries should now be filled in (either from cache or fresh encode).
    return [r for r in results]  # type: ignore[return-value]


def embed_query(text: str) -> list[float]:
    """Convenience wrapper to embed a single query string."""
    return embed_texts([text], use_cache=True, is_query=True)[0]
