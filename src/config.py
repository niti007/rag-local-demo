"""
Central configuration loader for the Cascade Bank local-first RAG project.

Loads `config.yaml` (project settings) and `.env` (secrets) and exposes a
single, typed, cached settings object via `get_config()`.

Usage:
    from src.config import get_config, require_openrouter_key

    cfg = get_config()
    cfg.paths.pdfs          # -> absolute pathlib.Path
    cfg.embedding.model     # -> "BAAI/bge-small-en-v1.5"
    require_openrouter_key()  # raises RuntimeError with a clear message if unset
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the `src/` directory containing this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH: Path = PROJECT_ROOT / "config.yaml"
DOTENV_PATH: Path = PROJECT_ROOT / ".env"


class PathsConfig(BaseModel):
    data_raw: Path
    pdfs: Path
    sops: Path
    csvs: Path
    chroma_db: Path
    embedding_cache: Path


class CorpusConfig(BaseModel):
    company_name: str
    num_pdfs: int
    num_sops: int
    num_csvs: int
    seed: int


class ChunkingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    encoding: str


class EmbeddingConfig(BaseModel):
    model: str
    dim: int
    batch_size: int
    query_prefix: str = ""


class VectorStoreConfig(BaseModel):
    collection: str


class SearchConfig(BaseModel):
    top_k: int


class RerankConfig(BaseModel):
    model: str
    top_n: int


class GenerationConfig(BaseModel):
    provider: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int


class Config(BaseSettings):
    """
    Typed, nested settings object combining config.yaml (non-secret project
    config) with environment variables / .env (secrets).
    """

    model_config = SettingsConfigDict(extra="ignore")

    paths: PathsConfig
    corpus: CorpusConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    search: SearchConfig
    rerank: RerankConfig
    generation: GenerationConfig

    # Secret, read from the OPENROUTER_API_KEY env var (via .env or the shell).
    openrouter_api_key: str | None = Field(default=None)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_paths(raw_paths: dict, root: Path) -> dict:
    """Resolve every value in the `paths` section to an absolute Path under `root`."""
    return {key: (root / value).resolve() for key, value in raw_paths.items()}


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Load config.yaml + .env (once, cached) and return a typed Config object.
    All `paths.*` fields are absolute pathlib.Path instances resolved
    relative to PROJECT_ROOT (the RAG folder, parent of `src/`).
    """
    # Load .env into the process environment (does not override existing env vars).
    load_dotenv(dotenv_path=DOTENV_PATH, override=False)

    raw = _load_yaml(CONFIG_YAML_PATH)
    raw = dict(raw)  # shallow copy so we can mutate the `paths` section safely
    raw["paths"] = _resolve_paths(raw.get("paths", {}), PROJECT_ROOT)

    return Config(**raw)


def require_openrouter_key() -> str:
    """
    Return the OpenRouter API key, or raise a clear RuntimeError if it is missing.
    Use this right before making an OpenRouter API call.
    """
    cfg = get_config()
    if not cfg.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and set "
            "OPENROUTER_API_KEY to a key from https://openrouter.ai/keys, "
            "or export it as an environment variable."
        )
    return cfg.openrouter_api_key
