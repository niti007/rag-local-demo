"""
Inspection / visualization utilities for the Chroma vector store.

Teaching-oriented tools to answer "what actually got stored?":
    - print_stats()          -- collection totals, per doc_type
    - sample_chunks()        -- paged preview of stored chunks + metadata
    - show_overlap()         -- visualize chunk_overlap between consecutive
                                 chunks of one source file (the "aha" moment
                                 for how RecursiveCharacterTextSplitter works)
    - embedding_projection() -- 2D UMAP/TSNE scatter of all embeddings,
                                 colored by doc_type (plotly HTML + PNG)

Run as:
    python -m src.store.inspect --stats
    python -m src.store.inspect --sample 10 --doc-type pdf
    python -m src.store.inspect --overlap some_policy.pdf
    python -m src.store.inspect --project
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, get_config
from src.store.chroma_client import collection_stats, get_or_create_collection

DEFAULT_SCATTER_HTML = PROJECT_ROOT / "docs" / "course" / "embedding_scatter.html"
DEFAULT_SCATTER_PNG = PROJECT_ROOT / "docs" / "course" / "embedding_scatter.png"

PREVIEW_CHARS = 200


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def print_stats() -> dict:
    """Pretty-print collection_stats() (total + per doc_type). Returns the stats dict."""
    stats = collection_stats()
    print("=" * 60)
    print(f"Collection: {stats['collection']}")
    print(f"Total chunks: {stats['total']}")
    print("-" * 60)
    if stats["by_doc_type"]:
        for doc_type, count in sorted(stats["by_doc_type"].items()):
            print(f"  {doc_type:<10} {count}")
    else:
        print("  (collection is empty)")
    print("=" * 60)
    return stats


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def sample_chunks(n: int = 10, doc_type: str | None = None, offset: int = 0) -> list[dict]:
    """
    Return a page of stored chunks (text preview + metadata), sorted by
    (source_file, chunk_index).

    Args:
        n: page size.
        doc_type: optional filter, e.g. "pdf" | "sop" | "csv".
        offset: number of sorted rows to skip before taking the page.
    """
    collection = get_or_create_collection()
    where = {"doc_type": doc_type} if doc_type else None

    result = collection.get(include=["documents", "metadatas"], where=where)
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    rows = list(zip(ids, documents, metadatas))
    rows.sort(key=lambda r: (r[2].get("source_file", ""), r[2].get("chunk_index", 0)))

    page = rows[offset : offset + n]

    out: list[dict] = []
    for chunk_id, text, meta in page:
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "..."
        out.append({"id": chunk_id, "text_preview": preview, "metadata": meta})
    return out


def _print_samples(rows: list[dict]) -> None:
    if not rows:
        print("(no chunks matched)")
        return
    for row in rows:
        meta = row["metadata"]
        print("-" * 60)
        print(f"{meta.get('source_file')}  [chunk {meta.get('chunk_index')}]  "
              f"doc_type={meta.get('doc_type')}  page={meta.get('page_number')}  "
              f"offsets=[{meta.get('chunk_start_offset')}, {meta.get('chunk_end_offset')})")
        print(row["text_preview"])
    print("-" * 60)


# --------------------------------------------------------------------------
# Overlap visualization (teaching centerpiece)
# --------------------------------------------------------------------------


def _longest_common_substring_tail_head(a: str, b: str, min_len: int = 8) -> tuple[int, int]:
    """
    Find the longest suffix of `a` that is also a prefix-ish substring
    appearing near the start of `b` (i.e. the longest run of characters at
    the tail of `a` that reappears at the head of `b`). This is a robust,
    offset-independent fallback/confirmation for detecting the textual
    overlap the splitter introduced.

    Returns (overlap_len_in_a_tail, overlap_len_in_b_head); both 0 if no
    match of at least `min_len` chars is found. Uses a simple shrinking
    window from the full possible overlap length down to min_len, checking
    whether a's suffix of that length appears anywhere within b's first
    (len*3) characters (b's head region) -- good enough for teaching-sized
    chunks (a few hundred to ~2000 chars), no need for full DP LCS.
    """
    max_check = min(len(a), len(b))
    for length in range(max_check, min_len - 1, -1):
        suffix = a[-length:]
        # Search within the head region of b (generous window).
        head_region = b[: max(length * 3, length + 50)]
        idx = head_region.find(suffix)
        if idx != -1:
            return length, idx + length
    return 0, 0


def show_overlap(source_file: str) -> None:
    """
    Fetch all chunks for `source_file` ordered by chunk_index and, for each
    consecutive pair, print both chunk texts with the overlapping region
    wrapped in >>> <<< markers.

    Overlap is determined two ways, both reported:
      1. Offset-based: the [start, end) ranges from chunk metadata directly
         tell us the overlapping character span in the *original* full_text.
      2. Text-based: an independent longest-common-substring check between
         chunk[i]'s tail and chunk[i+1]'s head, which confirms the overlap
         is real text (not just coincidentally overlapping offsets) and is
         robust even if offsets were ever approximate.
    """
    collection = get_or_create_collection()
    result = collection.get(
        include=["documents", "metadatas"],
        where={"source_file": source_file},
    )
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    if not documents:
        print(f"No chunks found for source_file={source_file!r}")
        return

    rows = sorted(
        zip(documents, metadatas),
        key=lambda r: r[1].get("chunk_index", 0),
    )

    print("=" * 70)
    print(f"Overlap inspection: {source_file}  ({len(rows)} chunks)")
    print("=" * 70)

    for i in range(len(rows) - 1):
        text_a, meta_a = rows[i]
        text_b, meta_b = rows[i + 1]

        idx_a, idx_b = meta_a.get("chunk_index"), meta_b.get("chunk_index")
        start_a, end_a = meta_a.get("chunk_start_offset"), meta_a.get("chunk_end_offset")
        start_b, end_b = meta_b.get("chunk_start_offset"), meta_b.get("chunk_end_offset")

        print(f"\n--- chunk {idx_a} -> chunk {idx_b} ---")

        # 1. Offset-based overlap span (in the shared full_text coordinate system).
        offset_overlap_start = max(start_a, start_b) if None not in (start_a, start_b) else None
        offset_overlap_end = min(end_a, end_b) if None not in (end_a, end_b) else None
        has_offset_overlap = (
            offset_overlap_start is not None
            and offset_overlap_end is not None
            and offset_overlap_start < offset_overlap_end
        )
        if has_offset_overlap:
            print(
                f"  [offsets] chunk {idx_a} spans [{start_a},{end_a}), "
                f"chunk {idx_b} spans [{start_b},{end_b}) "
                f"-> overlap region [{offset_overlap_start},{offset_overlap_end}) "
                f"({offset_overlap_end - offset_overlap_start} chars)"
            )
        else:
            print(
                f"  [offsets] chunk {idx_a} spans [{start_a},{end_a}), "
                f"chunk {idx_b} spans [{start_b},{end_b}) -> no offset overlap"
            )

        # 2. Text-based longest-common-substring at tail(a) / head(b).
        tail_len, head_len = _longest_common_substring_tail_head(text_a, text_b)

        if tail_len > 0:
            overlap_text = text_a[-tail_len:]
            a_display = text_a[: len(text_a) - tail_len] + ">>>" + overlap_text + "<<<"
            b_display = ">>>" + text_b[:tail_len] + "<<<" + text_b[tail_len:]

            print(f"  [text]    longest common tail/head run: {tail_len} chars")
            print(f"\n  chunk {idx_a} (tail highlighted):")
            print("  " + a_display.replace("\n", "\n  "))
            print(f"\n  chunk {idx_b} (head highlighted):")
            print("  " + b_display.replace("\n", "\n  "))
        else:
            print("  [text]    no common tail/head substring found (>=8 chars)")
            print(f"\n  chunk {idx_a} (end):   ...{text_a[-120:]!r}")
            print(f"  chunk {idx_b} (start): {text_b[:120]!r}...")

    print("\n" + "=" * 70)


# --------------------------------------------------------------------------
# Embedding projection
# --------------------------------------------------------------------------


def embedding_projection(out_path: Path | None = None, method: str = "umap") -> Path:
    """
    Pull all embeddings + metadata from the collection, reduce to 2D
    (UMAP by default, falling back to sklearn TSNE if `umap-learn` isn't
    importable, or if explicitly requested via method="tsne"), and render a
    scatter plot colored by doc_type.

    Writes both:
        - an interactive Plotly HTML file (returned path; default
          docs/course/embedding_scatter.html)
        - a static matplotlib PNG (default docs/course/embedding_scatter.png)

    Returns the HTML path.
    """
    html_path = out_path or DEFAULT_SCATTER_HTML
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = html_path.with_suffix(".png")

    collection = get_or_create_collection()
    result = collection.get(include=["embeddings", "metadatas", "documents"])

    embeddings = result.get("embeddings")
    metadatas = result.get("metadatas") or []
    documents = result.get("documents") or []

    n = 0 if embeddings is None else len(embeddings)

    if n < 3:
        print(
            f"Only {n} embedding(s) in the collection -- need at least 3 points "
            "for a 2D projection. Run ingest first. Writing placeholder files."
        )
        _write_placeholder(html_path, png_path, n)
        return html_path

    X = np.asarray(embeddings, dtype=np.float32)
    doc_types = [(m or {}).get("doc_type", "unknown") for m in metadatas]
    source_files = [(m or {}).get("source_file", "") for m in metadatas]

    coords, used_method = _reduce_2d(X, method)

    _write_plotly_html(coords, doc_types, source_files, documents, html_path, used_method)
    _write_matplotlib_png(coords, doc_types, png_path, used_method)

    print(f"Wrote {html_path}")
    print(f"Wrote {png_path}")
    return html_path


def _reduce_2d(X: np.ndarray, method: str) -> tuple[np.ndarray, str]:
    n = X.shape[0]

    if method == "umap":
        try:
            import umap

            n_neighbors = max(2, min(15, n - 1))
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            coords = reducer.fit_transform(X)
            return coords, "umap"
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"UMAP unavailable ({exc}); falling back to TSNE.")

    from sklearn.manifold import TSNE

    perplexity = max(2, min(30, n - 1))
    reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
    coords = reducer.fit_transform(X)
    return coords, "tsne"


def _write_plotly_html(coords, doc_types, source_files, documents, html_path: Path, method: str) -> None:
    import pandas as pd
    import plotly.express as px

    previews = [
        (d[:120] + "...") if d and len(d) > 120 else (d or "")
        for d in documents
    ]

    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "doc_type": doc_types,
            "source_file": source_files,
            "preview": previews,
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="doc_type",
        hover_data={"source_file": True, "preview": True, "x": False, "y": False},
        title=f"Chunk embeddings ({method.upper()} projection, cosine space)",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    fig.write_html(str(html_path))


def _write_matplotlib_png(coords, doc_types, png_path: Path, method: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    unique_types = sorted(set(doc_types))
    cmap = plt.get_cmap("tab10")

    for i, dt in enumerate(unique_types):
        mask = [t == dt for t in doc_types]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=dt, s=18, alpha=0.75, color=cmap(i % 10))

    ax.set_title(f"Chunk embeddings ({method.upper()} projection, cosine space)")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.legend(title="doc_type")
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)


def _write_placeholder(html_path: Path, png_path: Path, n: int) -> None:
    import matplotlib.pyplot as plt

    message = f"Not enough points to project ({n} < 3). Ingest more chunks first."

    html_path.write_text(
        f"<html><body><p>{message}</p></body></html>", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize the Chroma vector store."
    )
    parser.add_argument("--stats", action="store_true", help="Print collection stats.")
    parser.add_argument(
        "--sample",
        nargs="?",
        const=10,
        type=int,
        default=None,
        metavar="N",
        help="Print a page of N sample chunks (default 10 if flag given with no value).",
    )
    parser.add_argument("--offset", type=int, default=0, help="Offset for --sample paging.")
    parser.add_argument(
        "--doc-type", type=str, default=None, help="Filter --sample to a single doc_type."
    )
    parser.add_argument(
        "--overlap",
        type=str,
        default=None,
        metavar="SOURCE_FILE",
        help="Show chunk-overlap visualization for one source_file.",
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help="Write a 2D embedding scatter (HTML + PNG) to docs/course/.",
    )
    parser.add_argument(
        "--method",
        choices=["umap", "tsne"],
        default="umap",
        help="Dimensionality reduction method for --project.",
    )
    args = parser.parse_args()

    ran_something = False

    if args.stats:
        print_stats()
        ran_something = True

    if args.sample is not None:
        rows = sample_chunks(n=args.sample, doc_type=args.doc_type, offset=args.offset)
        _print_samples(rows)
        ran_something = True

    if args.overlap:
        show_overlap(args.overlap)
        ran_something = True

    if args.project:
        embedding_projection(method=args.method)
        ran_something = True

    if not ran_something:
        print_stats()


if __name__ == "__main__":
    main()
