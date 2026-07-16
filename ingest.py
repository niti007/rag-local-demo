"""
Project-root convenience CLI for the RAG teaching demo.

Default behavior: generate the synthetic corpus, then do a clean re-ingest
into Chroma (reset=True), then print collection stats.

Usage (from project root):
    python ingest.py                 # generate + ingest(reset=True) + stats
    python ingest.py --gen-only      # only run generate_all(), no ingest
    python ingest.py --no-reset      # skip generation; ingest without resetting
    python ingest.py --no-reset --gen-only   # (--gen-only wins; no ingest happens)
"""

from __future__ import annotations

import argparse

from src.data_gen.generate_all import generate_all
from src.store.ingest_to_chroma import ingest_all
from src.store.inspect import print_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate + ingest the RAG demo corpus.")
    parser.add_argument(
        "--gen-only",
        action="store_true",
        help="Only generate the synthetic source corpus; skip ingest into Chroma.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip data generation and ingest without resetting the collection "
        "(incremental upsert of whatever is in data/raw).",
    )
    args = parser.parse_args()

    if args.gen_only:
        generate_all()
        return

    if args.no_reset:
        ingest_all(reset=False)
    else:
        generate_all()
        ingest_all(reset=True)

    print_stats()


if __name__ == "__main__":
    main()
