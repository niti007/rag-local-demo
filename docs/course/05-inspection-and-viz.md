# 05 — Inspection and Visualization

## Concept

A vector store is, by design, opaque at rest: it holds arrays of 384 floats
per chunk, which are meaningless to stare at directly. **Inspection
tooling** exists to answer, in human terms, "what actually got stored, and
does it look right?" — three complementary views:

1. **Stats** — the simplest sanity check: how many chunks total, broken
   down by `doc_type`. Catches gross failures (an empty collection, a
   doc_type with zero chunks) instantly.
2. **Sampling** — read actual stored text + metadata in a paged table, to
   spot-check that extraction and chunking produced sensible-looking
   content (not garbled PDF text, not truncated CSV rows).
3. **2D projection (UMAP/t-SNE)** — 384-dimensional vectors can't be looked
   at directly, but a dimensionality-reduction algorithm can compress them
   down to 2D coordinates for a scatter plot while trying to preserve
   *relative* closeness. **UMAP** (Uniform Manifold Approximation and
   Projection) is used by default; if the `umap-learn` package isn't
   importable (or `--method tsne` is passed explicitly), it falls back to
   scikit-learn's **t-SNE**. Both are nonlinear projections meant for
   visualization, not for further computation — the 2D coordinates
   themselves aren't meaningful in isolation, only the *clustering pattern*
   they reveal is.

**Reading the scatter.** If embeddings are doing their job, chunks that are
semantically similar should land near each other in the 2D projection.
Because this project colors points by `doc_type`, a reasonable expectation
is some visible separation between `pdf`, `sop`, and `csv` clusters — SOPs
share a rigid Purpose/Scope/Procedure structure and CSV row-blocks share a
tabular `"Row N: col: value | col: value"` format, both of which are
stylistically distinct from free-form policy/manual prose. Overlap between
clusters is also informative: it suggests topical content that reads
similarly regardless of source format (e.g. a policy PDF section about wire
transfers may sit near SOP chunks about wire transfer authorization).

## In this repo

All of this lives in `src/store/inspect.py`:

- `print_stats() -> dict` — pretty-prints `collection_stats()` (from
  `src/store/chroma_client.py`, lesson 04): total chunks and a per-`doc_type`
  breakdown.
- `sample_chunks(n=10, doc_type=None, offset=0) -> list[dict]` — fetches
  chunks via `collection.get(include=["documents", "metadatas"], where=...)`,
  sorts them by `(source_file, chunk_index)`, and returns a page of
  `{"id", "text_preview" (truncated to `PREVIEW_CHARS`=200), "metadata"}`
  dicts. `_print_samples(rows)` renders them to the terminal with source
  file, chunk index, doc_type, page number, and offset range.
- `show_overlap(source_file) -> None` — covered in depth in lesson 02; the
  chunk-overlap visualization centerpiece.
- `embedding_projection(out_path=None, method="umap") -> Path` — pulls
  **every** embedding + metadata + document from the collection via
  `collection.get(include=["embeddings", "metadatas", "documents"])`,
  reduces to 2D via `_reduce_2d(X, method)`, then writes:
  - an interactive Plotly scatter (`_write_plotly_html`) to
    `docs/course/embedding_scatter.html` by default (hoverable — shows
    `source_file` and a text preview per point), colored by `doc_type`.
  - a static matplotlib PNG (`_write_matplotlib_png`) to
    `docs/course/embedding_scatter.png`, same coloring.

  Guards against too little data: if fewer than 3 points exist, it writes
  placeholder files (`_write_placeholder`) instead of attempting a
  projection.
- `_reduce_2d(X, method)` — tries `umap.UMAP(n_components=2, n_neighbors=max(2,
  min(15, n-1)), random_state=42)` first if `method == "umap"`, falling back
  to `sklearn.manifold.TSNE(n_components=2, random_state=42,
  perplexity=max(2, min(30, n-1)), init="pca")` on import failure or if
  `method == "tsne"` was requested explicitly.
- CLI (`main()`): flags `--stats`, `--sample [N]` (default 10 if the bare
  flag is given), `--offset`, `--doc-type`, `--overlap SOURCE_FILE`,
  `--project`, `--method {umap,tsne}`. Running with no flags at all falls
  back to `print_stats()`.

## Try it

Print collection totals:

```powershell
python -m src.store.inspect --stats
```

Sample 10 stored PDF chunks:

```powershell
python -m src.store.inspect --sample 10 --doc-type pdf
```

Page through the next 10:

```powershell
python -m src.store.inspect --sample 10 --doc-type pdf --offset 10
```

Generate the 2D embedding scatter (writes both HTML and PNG):

```powershell
python -m src.store.inspect --project
```

Then open `docs/course/embedding_scatter.html` in a browser for the
interactive, hoverable version, or view `docs/course/embedding_scatter.png`
directly.

## What to look for / checkpoint

- `--stats` output totals should match what `ingest_all` printed right
  after ingestion (lesson 04) — if they don't, something is stale (wrong
  `chroma_db` path, or you're looking at a leftover collection from before
  a `--reset`).
- `--sample` rows should show readable, on-topic text previews (banking
  prose for `pdf`/`sop`, `"Row N: col: value | ..."` lines for `csv`) — if
  previews look garbled, that's a sign of a PDF text-extraction problem
  upstream in `extract_pdf`, not a chunking or embedding problem.
- The scatter plot should show at least loose visual grouping by color
  (`doc_type`) — `csv` points in particular should form a fairly distinct
  cluster or clusters, since their `"Table: ... | Row N: ..."` text format
  is structurally very different from PDF/SOP prose. Perfect separation is
  not expected or required; partial overlap between `pdf` and `sop` is
  normal since both are free-form banking prose.
- If you see the placeholder "Not enough points to project" message, the
  collection has fewer than 3 chunks — run ingestion first.

## Teaching note

Run `--project` twice in a row without changing the corpus, and compare the
two scatter plots side by side. UMAP and t-SNE are both non-deterministic
in general, but this project pins `random_state=42` for both, so re-running
`--project` on an *unchanged* collection should produce a visually
consistent layout each time — a good moment to discuss the difference
between "same algorithm, same seed → same output" and "the absolute
x/y coordinates are meaningless, only relative clustering is."
