# 02 — Chunking and Overlap

## Concept

An embedding model and a language model both have a limited "attention
span" — you can't (and shouldn't) embed or feed an entire 10-page PDF as one
unit. **Chunking** splits documents into smaller pieces so each piece:

- fits comfortably inside the embedding model's context window,
- is small enough that a single embedding vector represents one coherent
  idea (not a blur of ten unrelated paragraphs),
- and is small enough to keep the LLM's final context window (and cost)
  bounded when several chunks are retrieved at once.

**Recursive character splitting** (`RecursiveCharacterTextSplitter`) tries
to split on the most natural boundary first (paragraph breaks), then falls
back to sentence breaks, then words, then raw characters, only escalating
when a piece is still too big. This keeps chunks semantically coherent
instead of cutting mid-sentence whenever possible.

**Tokens vs. characters.** Chunk size here is measured in **tokens**
(via `tiktoken`'s `cl100k_base` encoding), not raw characters. This matters
because token count is what actually determines how much of a model's
context window a chunk consumes — and different text (dense financial
jargon vs. plain prose) can pack a very different number of characters into
the same token budget. Sizing chunks in tokens keeps the "how much context
does this cost" answer consistent across documents.

**Why overlap matters.** If chunk boundaries were hard cuts with zero
overlap, a sentence or idea that happens to straddle a boundary would be
split across two chunks, and neither chunk alone would contain the full
context needed to answer a question about it — this is "context bleed."
`chunk_overlap` (100 tokens by default) makes each chunk repeat the tail end
of the previous chunk, so a concept near a boundary is guaranteed to appear
whole in at least one chunk. The tradeoff is redundant storage/embedding
cost — more overlap means more (partially duplicate) chunks to embed and
store.

**Offset metadata.** Every chunk records `chunk_start_offset` /
`chunk_end_offset` — character offsets into the source file's reconstructed
full text. This is what lets `src/store/inspect.py:show_overlap` later prove,
concretely, that consecutive chunks really do share text, rather than just
asserting it.

## In this repo

- `src/ingest/extract.py`
  - `extract_pdf(path) -> list[dict]` — one extraction unit
    `{"text": str, "page_number": int}` per PDF page (via `pypdf.PdfReader`),
    skipping blank pages.
  - `extract_csv(path) -> list[dict]` — groups CSV rows into blocks of
    `ROWS_PER_BLOCK` (20) rows per unit, prefixed with a header summary
    block (columns + row count). Units have `page_number: None`.
  - `extract_file(path)` — dispatches by suffix (`.pdf` / `.csv`).
  - `iter_source_files(cfg) -> list[Path]` — gathers all PDFs under
    `cfg.paths.pdfs` and `cfg.paths.sops`, and all CSVs under
    `cfg.paths.csvs`.

- `src/ingest/chunk.py`
  - `get_splitter(cfg=None)` — builds a `RecursiveCharacterTextSplitter`
    via `.from_tiktoken_encoder(encoding_name=cfg.chunking.encoding,
    chunk_size=cfg.chunking.chunk_size, chunk_overlap=cfg.chunking.chunk_overlap)`.
    Defaults from `config.yaml`: `chunk_size: 500`, `chunk_overlap: 100`,
    `encoding: cl100k_base`.
  - `chunk_document(text, base_meta) -> list[dict]` — splits one page/unit's
    text and attaches metadata (`source_file`, `doc_type`, `chunk_index`,
    `chunk_start_offset`, `chunk_end_offset`, `page_number`, `created_at`).
    The offset logic (documented in detail in the function's own docstring)
    locates each split piece in the file's full concatenated text via
    `full_text.find(chunk_text, cursor)`, then advances the search cursor by
    only **1 character** past the match start (not past the chunk's end) —
    this is precisely what allows the *next* chunk, which overlaps the tail
    of this one, to still be found starting inside this chunk's span,
    producing genuinely overlapping `[start, end)` ranges.
  - `chunk_source_file(path, doc_type, cfg=None) -> list[dict]` — extracts a
    file and chunks it page-by-page (PDF) or block-by-block (CSV), while
    keeping `chunk_index` and offsets tracked globally across the whole file
    via a shared `base_meta` dict passed into repeated `chunk_document`
    calls.
  - `_doc_type_for_path(path, cfg) -> str` — maps a source file's parent
    directory to the coarse `doc_type` used everywhere downstream:
    `sops/` → `"sop"`, `csvs/` → `"csv"`, everything else (`pdfs/`) →
    `"pdf"`.
  - `chunk_all(cfg=None) -> list[dict]` — chunks every source file returned
    by `iter_source_files`, and prints a summary: total files, total chunks,
    chunks per `doc_type`, and min/avg/max chunk token length (via
    `tiktoken.get_encoding(cfg.chunking.encoding).encode(...)`).

## Try it

Chunk the whole corpus and see the summary statistics:

```powershell
python -m src.ingest.chunk
```

This is the centerpiece exercise for this lesson — but `show_overlap`
requires chunks to already be **in Chroma** (it reads from the collection,
not from `chunk_all()` directly), so run the ingest pipeline first, then
inspect overlap for one specific source file:

```powershell
python -m src.store.ingest_to_chroma --reset
python -m src.store.inspect --overlap 01_policy_account_overdraft_policy.pdf
```

(Substitute any filename actually present in `data/raw/pdfs/` or
`data/raw/sops/` — `--overlap` filters by exact `source_file` metadata, so
check `python -m src.store.inspect --sample 5` first if you're not sure
of an exact filename.)

## What to look for / checkpoint

From `python -m src.ingest.chunk`, confirm:

- `Chunks per doc_type:` breaks down into `pdf`, `sop`, `csv` (matching the
  three source directories).
- `Chunk token length: min=... avg=... max=...` — `max` should be at or
  below `chunk_size` (500); `avg` will typically be somewhat below 500
  since the splitter prefers natural boundaries over always filling the
  budget.

From `python -m src.store.inspect --overlap <file>`, for each consecutive
chunk pair you should see **both**:

1. `[offsets]` — an explicit overlap region `[start, end)` with a nonzero
   character count, computed purely from the stored `chunk_start_offset` /
   `chunk_end_offset` metadata.
2. `[text]` — an independent longest-common-substring check
   (`_longest_common_substring_tail_head`) confirming the tail of chunk *i*
   and the head of chunk *i+1* really do share text, with the shared span
   printed between `>>> <<<` markers.

Seeing both confirmations agree is the "aha" moment: the overlap isn't a
metadata bookkeeping artifact, it's real duplicated text you can read.

## Teaching note

Have trainees open the printed `[offsets]` overlap span next to the
`>>> <<<`-highlighted text in the same terminal output and manually count
that the highlighted text length roughly matches the offset arithmetic
(`offset_overlap_end - offset_overlap_start`). Small discrepancies are
expected and fine — token-based splitting doesn't map 1:1 to character
offsets, and the text-based check is char-based tail/head matching, so the
two measurements aren't measuring an identical quantity — but the two
methods should broadly agree on "yes, overlap exists here, and here's
what it says."
