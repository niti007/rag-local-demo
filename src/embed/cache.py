"""
On-disk embedding cache for the local RAG teaching demo.

Each cached vector is stored as a `.npy` file named `<hash>.npy` under
`cfg.paths.embedding_cache`, where `<hash>` is the sha256 of
`model_name + "\\n" + chunk_text`. This lets repeated ingests (and repeated
query embeddings) skip re-encoding text that hasn't changed, and keeps the
cache correct across model changes (different model -> different hash).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def chunk_hash(text: str, model: str) -> str:
    """Return the sha256 hex digest of `model + "\\n" + text`, used as the
    cache key (and, in the vector store, as the chunk id)."""
    payload = f"{model}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EmbeddingCache:
    """Simple, robust on-disk cache mapping a content hash to an embedding
    vector, stored as individual `.npy` files."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, hash_: str) -> Path:
        return self.cache_dir / f"{hash_}.npy"

    def get(self, hash_: str) -> list[float] | None:
        """Return the cached vector for `hash_`, or None if not present /
        unreadable."""
        path = self._path_for(hash_)
        if not path.exists():
            return None
        try:
            arr = np.load(path)
        except Exception:
            # Corrupt or partially-written cache file; treat as a miss.
            return None
        return arr.astype(float).tolist()

    def set(self, hash_: str, vector: list[float]) -> None:
        """Persist `vector` under `hash_`, overwriting any existing entry."""
        path = self._path_for(hash_)
        arr = np.asarray(vector, dtype=np.float32)
        np.save(path, arr)
