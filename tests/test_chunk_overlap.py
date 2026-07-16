"""
Teaching-invariant test: RecursiveCharacterTextSplitter (as configured via
src.ingest.chunk.get_splitter) must produce multiple chunks for a long
document, and consecutive chunks from the same source file must overlap
(because chunk_overlap > 0 in config.yaml).
"""

from __future__ import annotations

from datetime import datetime

from src.config import get_config
from src.ingest.chunk import chunk_document, get_splitter


def _make_long_text(num_sentences: int = 400) -> str:
    """Build a long synthetic text made of many distinct sentences (so any
    overlap found between chunks can only come from real splitter overlap,
    not from repeated/duplicate content)."""
    sentences = [
        f"This is distinct synthetic sentence number {i} about Cascade Bank policy."
        for i in range(num_sentences)
    ]
    return " ".join(sentences)


def test_chunker_produces_multiple_chunks():
    cfg = get_config()
    splitter = get_splitter(cfg)
    text = _make_long_text()

    base_meta = {
        "splitter": splitter,
        "full_text": text,
        "source_file": "synthetic_long_doc.txt",
        "doc_type": "pdf",
        "created_at": datetime.now().isoformat(),
    }

    chunks = chunk_document(text, base_meta)

    assert len(chunks) > 1, "expected the long synthetic text to be split into multiple chunks"


def test_consecutive_chunks_overlap():
    cfg = get_config()
    assert cfg.chunking.chunk_overlap > 0, "test assumes chunk_overlap > 0 per config.yaml"

    splitter = get_splitter(cfg)
    text = _make_long_text()

    base_meta = {
        "splitter": splitter,
        "full_text": text,
        "source_file": "synthetic_long_doc.txt",
        "doc_type": "pdf",
        "created_at": datetime.now().isoformat(),
    }

    chunks = chunk_document(text, base_meta)
    assert len(chunks) > 1

    for i in range(len(chunks) - 1):
        a, b = chunks[i], chunks[i + 1]

        # 1. Offset-based overlap: [start, end) ranges (in the shared
        #    full_text coordinate system) should intersect.
        offset_overlap = max(a["chunk_start_offset"], b["chunk_start_offset"]) < min(
            a["chunk_end_offset"], b["chunk_end_offset"]
        )

        # 2. Text-based overlap: the tail of chunk[i]'s text should reappear
        #    at (or near) the head of chunk[i+1]'s text. Since all sentences
        #    are distinct, any shared substring of meaningful length is real
        #    splitter-introduced overlap, not coincidence.
        tail = a["text"][-40:].strip()
        text_overlap = bool(tail) and tail in b["text"]

        assert offset_overlap or text_overlap, (
            f"chunk {i} -> {i + 1} does not appear to overlap: "
            f"a=[{a['chunk_start_offset']},{a['chunk_end_offset']}) "
            f"b=[{b['chunk_start_offset']},{b['chunk_end_offset']})"
        )
