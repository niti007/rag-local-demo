"""
One-shot end-to-end pipeline for the local RAG teaching demo:

    generate_all()          -- write synthetic PDFs/SOPs/CSVs to data/raw
    ingest_all(reset=True)  -- chunk -> embed -> upsert into Chroma
    print_stats()           -- pretty-print collection stats
    embedding_projection()  -- write a 2D embedding scatter (HTML + PNG)

Run as (from project root):
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --skip-gen   # re-embed existing raw files
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `src.*` importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT  # noqa: E402
from src.data_gen.generate_all import generate_all  # noqa: E402
from src.store.ingest_to_chroma import ingest_all  # noqa: E402
from src.store.inspect import embedding_projection, print_stats  # noqa: E402


def _banner(title: str) -> None:
    print()
    print("#" * 70)
    print(f"# {title}")
    print("#" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full RAG demo pipeline: generate -> ingest -> inspect -> project."
    )
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help="Skip data generation and re-embed/ingest whatever is already in data/raw.",
    )
    args = parser.parse_args()

    started = time.time()
    print(f"Project root: {PROJECT_ROOT}")

    gen_summary = None
    if args.skip_gen:
        _banner("PHASE 1/4: Data generation (skipped, --skip-gen)")
    else:
        _banner("PHASE 1/4: Generating synthetic source documents")
        gen_summary = generate_all()

    _banner("PHASE 2/4: Ingesting into Chroma (chunk -> embed -> upsert)")
    stats = ingest_all(reset=True)

    _banner("PHASE 3/4: Collection stats")
    print_stats()

    _banner("PHASE 4/4: Embedding projection")
    scatter_html = embedding_projection()

    elapsed = time.time() - started

    _banner("DONE")
    if gen_summary is not None:
        print(f"Generated files: {gen_summary}")
    print(f"Chunks in collection: {stats['total']}")
    print(f"By doc_type: {stats['by_doc_type']}")
    print(f"Embedding scatter: {scatter_html}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
