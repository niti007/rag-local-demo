"""
Raw text extraction for the local RAG teaching demo.

Turns source files (PDFs, CSVs) into a flat list of "extraction units":
    {"text": str, "page_number": int | None}

These units are the input to `src.ingest.chunk`, which splits them further
into token-bounded chunks and attaches metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pypdf import PdfReader


ROWS_PER_BLOCK = 20


def extract_pdf(path: Path) -> list[dict]:
    """
    Extract text from a PDF, one extraction unit per page.

    Returns a list of {"text": str, "page_number": int} (1-indexed),
    skipping pages whose extracted text is empty (after stripping
    whitespace).
    """
    reader = PdfReader(str(path))
    units: list[dict] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        units.append({"text": text, "page_number": i + 1})
    return units


def _row_to_text(row: pd.Series) -> str:
    parts = [f"{col}: {row[col]}" for col in row.index]
    return " | ".join(parts)


def extract_csv(path: Path) -> list[dict]:
    """
    Extract text from a CSV, grouping rows into blocks of ~ROWS_PER_BLOCK
    rows per text unit, prefixed with the filename/table name. The first
    unit is always a header summary block (column names + row count).

    Returns a list of {"text": str, "page_number": None}.
    """
    df = pd.read_csv(path)
    table_name = path.stem
    n_rows, n_cols = df.shape

    units: list[dict] = []

    header_text = (
        f"Table: {table_name}\n"
        f"Columns ({n_cols}): {', '.join(str(c) for c in df.columns)}\n"
        f"Row count: {n_rows}"
    )
    units.append({"text": header_text, "page_number": None})

    for start in range(0, n_rows, ROWS_PER_BLOCK):
        block = df.iloc[start : start + ROWS_PER_BLOCK]
        row_lines = [
            f"Row {start + offset + 1}: {_row_to_text(row)}"
            for offset, (_, row) in enumerate(block.iterrows())
        ]
        block_text = f"Table: {table_name} (rows {start + 1}-{start + len(block)})\n" + "\n".join(
            row_lines
        )
        units.append({"text": block_text, "page_number": None})

    return units


def extract_file(path: Path) -> list[dict]:
    """Dispatch extraction by file suffix. Supports .pdf and .csv."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".csv":
        return extract_csv(path)
    raise ValueError(f"Unsupported file type for extraction: {path}")


def iter_source_files(cfg) -> list[Path]:
    """
    Gather all source files under the configured paths:
    - *.pdf under cfg.paths.pdfs and cfg.paths.sops
    - *.csv under cfg.paths.csvs
    """
    files: list[Path] = []
    files.extend(sorted(cfg.paths.pdfs.glob("*.pdf")))
    files.extend(sorted(cfg.paths.sops.glob("*.pdf")))
    files.extend(sorted(cfg.paths.csvs.glob("*.csv")))
    return files
