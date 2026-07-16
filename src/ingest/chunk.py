"""
Token-bounded chunking for the local RAG teaching demo.

Pipeline:
    iter_source_files(cfg)         -> list[Path]           (src.ingest.extract)
    extract_file(path)             -> list[dict]           (src.ingest.extract)
    chunk_source_file(path, ...)   -> list[dict]           (this module)
    chunk_all(cfg)                 -> list[dict]           (this module)

Each final chunk dict has the shape:
    {
        "text": str,
        "source_file": str,
        "doc_type": "pdf" | "sop" | "csv",
        "chunk_index": int,           # 0-based, global within the source file
        "chunk_start_offset": int,    # char offset of chunk in the file's full_text
        "chunk_end_offset": int,      # char offset (exclusive) of chunk in full_text
        "page_number": int | None,    # 1-indexed for PDFs, None for CSVs
        "created_at": str,            # ISO timestamp
    }

Run as a script (`python -m src.ingest.chunk`) to chunk the whole configured
corpus and print a summary.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_config
from src.ingest.extract import extract_file, iter_source_files


def get_splitter(cfg=None) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter whose chunk_size/chunk_overlap are
    measured in tiktoken tokens (not characters), per cfg.chunking.*.
    """
    cfg = cfg or get_config()
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=cfg.chunking.encoding,
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )


def chunk_document(text: str, base_meta: dict) -> list[dict]:
    """
    Split `text` into chunks and attach metadata, using `base_meta` as both
    the config carrier and the running state for offsets/chunk_index.

    Required keys in base_meta:
        "splitter"          -- a RecursiveCharacterTextSplitter
        "full_text"         -- the full text of the source file (all pages/
                                units concatenated in order), used to locate
                                each chunk's character offsets
        "source_file"       -- str, e.g. the file name
        "doc_type"          -- "pdf" | "sop" | "csv"
        "created_at"        -- ISO timestamp string

    Optional / running-state keys (created if absent, updated in place so
    repeated calls across pages/units keep a single global counter+cursor):
        "search_cursor"     -- int, defaults to 0
        "chunk_index_start" -- int, defaults to 0
        "page_number"       -- int | None, defaults to None

    Offset computation: for each chunk produced by the splitter, we locate it
    in `full_text` via `full_text.find(chunk_text, cursor)`. After a match we
    advance the cursor to `start + 1` (NOT past the end of the chunk) so that
    the *next* chunk -- which typically overlaps the tail of this one because
    of chunk_overlap -- can still be found starting inside this chunk's span.
    This is what makes consecutive same-source chunks have overlapping
    [start, end) ranges, which a later inspection UI can highlight. If a
    chunk can't be found at all (defensive fallback for edge cases in the
    splitter's whitespace handling), we fall back to an unrestricted search
    from the start of full_text, and if that also fails, we fall back to the
    running cursor itself so offsets stay monotonically sane.
    """
    splitter: RecursiveCharacterTextSplitter = base_meta["splitter"]
    full_text: str = base_meta["full_text"]
    cursor: int = base_meta.get("search_cursor", 0)
    chunk_index: int = base_meta.get("chunk_index_start", 0)
    created_at: str = base_meta.get("created_at") or datetime.now().isoformat()
    page_number = base_meta.get("page_number")

    pieces = splitter.split_text(text)
    results: list[dict] = []

    for chunk_text in pieces:
        start = full_text.find(chunk_text, cursor)
        if start == -1:
            start = full_text.find(chunk_text)
        if start == -1:
            start = cursor
        end = start + len(chunk_text)

        results.append(
            {
                "text": chunk_text,
                "source_file": base_meta["source_file"],
                "doc_type": base_meta["doc_type"],
                "chunk_index": chunk_index,
                "chunk_start_offset": start,
                "chunk_end_offset": end,
                "page_number": page_number,
                "created_at": created_at,
            }
        )

        chunk_index += 1
        # Advance by only 1 char past the match start (not past the chunk's
        # end) so overlapping subsequent chunks remain findable.
        cursor = start + 1

    base_meta["search_cursor"] = cursor
    base_meta["chunk_index_start"] = chunk_index
    return results


def chunk_source_file(path: Path, doc_type: str, cfg=None) -> list[dict]:
    """
    Extract `path` and chunk it, producing a flat list of chunk dicts.

    Pages (PDF) / blocks (CSV) are chunked separately (so page_number is
    preserved per chunk) but offsets and chunk_index are tracked globally
    across the whole file via a shared `base_meta` dict passed into repeated
    `chunk_document` calls.
    """
    cfg = cfg or get_config()
    splitter = get_splitter(cfg)

    units = extract_file(path)
    if not units:
        return []

    full_text = "\n\n".join(unit["text"] for unit in units)
    created_at = datetime.now().isoformat()

    base_meta = {
        "splitter": splitter,
        "full_text": full_text,
        "source_file": path.name,
        "doc_type": doc_type,
        "created_at": created_at,
        "search_cursor": 0,
        "chunk_index_start": 0,
        "page_number": None,
    }

    all_chunks: list[dict] = []
    for unit in units:
        base_meta["page_number"] = unit["page_number"]
        all_chunks.extend(chunk_document(unit["text"], base_meta))

    return all_chunks


def _doc_type_for_path(path: Path, cfg) -> str:
    """pdfs/ -> 'pdf', sops/ -> 'sop', csvs/ -> 'csv'."""
    if path.parent == cfg.paths.sops:
        return "sop"
    if path.parent == cfg.paths.csvs:
        return "csv"
    return "pdf"


def chunk_all(cfg=None) -> list[dict]:
    """
    Chunk every source file under cfg.paths.pdfs / sops / csvs and return the
    combined list of chunk dicts. Prints a summary (files, chunks, chunks per
    doc_type, min/avg/max chunk token length).
    """
    cfg = cfg or get_config()
    encoding = tiktoken.get_encoding(cfg.chunking.encoding)

    files = iter_source_files(cfg)
    all_chunks: list[dict] = []
    chunks_per_type: dict[str, int] = {}

    for path in files:
        doc_type = _doc_type_for_path(path, cfg)
        chunks = chunk_source_file(path, doc_type, cfg)
        all_chunks.extend(chunks)
        chunks_per_type[doc_type] = chunks_per_type.get(doc_type, 0) + len(chunks)

    token_lengths = [len(encoding.encode(c["text"])) for c in all_chunks]

    print("=== Chunking summary ===")
    print(f"Total files:  {len(files)}")
    print(f"Total chunks: {len(all_chunks)}")
    print("Chunks per doc_type:")
    for doc_type, count in sorted(chunks_per_type.items()):
        print(f"  {doc_type}: {count}")

    if token_lengths:
        print(
            "Chunk token length: "
            f"min={min(token_lengths)} "
            f"avg={sum(token_lengths) / len(token_lengths):.1f} "
            f"max={max(token_lengths)}"
        )
    else:
        print("Chunk token length: n/a (no chunks produced)")

    return all_chunks


if __name__ == "__main__":
    chunk_all()
