# Cascade Bank RAG — a local-first RAG teaching demo

**Cascade Bank** is a fictional bank invented purely for this project. This
repo is a complete, readable, end-to-end **Retrieval-Augmented Generation
(RAG)** pipeline built over a synthetic corpus of Cascade Bank policy PDFs,
SOPs, and CSVs: generate documents, chunk them, embed them, store them in a
vector database, retrieve, rerank, and generate a cited answer — with a
Chainlit chat UI on top. Every stage is a small, runnable Python module
rather than a framework black box, so it doubles as a course (see
[`docs/course/00-overview.md`](docs/course/00-overview.md)).

## Architecture

```
 ┌──────────────┐   ┌───────────────┐   ┌───────────────┐   ┌────────────────┐
 │ 1. GENERATE  │──▶│ 2. EXTRACT    │──▶│ 3. CHUNK      │──▶│ 4. EMBED       │
 │ synthetic    │   │ raw text from │   │ token-bounded │   │ local          │
 │ PDFs/SOPs/   │   │ PDFs & CSVs   │   │ splits with   │   │ sentence-      │
 │ CSVs (Faker) │   │ (pypdf/pandas)│   │ overlap       │   │ transformers   │
 └──────────────┘   └───────────────┘   └───────────────┘   └───────┬────────┘
                                                                     │
 ┌──────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────▼────────┐
 │ 8. UI        │◀──│ 7. GENERATE   │◀──│ 6. RERANK     │◀──│ 5. CHROMA      │
 │ Chainlit chat│   │ OpenRouter    │   │ local cross-  │   │ persistent     │
 │ (retrieve/   │   │ (gpt-4o-mini),│   │ encoder       │   │ vector store   │
 │ rerank/answer│   │ cited answer  │   │ re-scores     │   │ (unified       │
 │ panels)      │   │ or fallback   │   │ shortlist     │   │ collection)    │
 └──────────────┘   └───────────────┘   └───────▲───────┘   └───────┬────────┘
                                                 │                   │
                                                 └── 6. SEARCH ◀─────┘
                                             query embedding + cosine
                                                 top-k lookup
```

Read left-to-right, top row then bottom row: **generate → extract → chunk →
embed → Chroma → search → rerank → generate (answer) → UI**. The first four
stages ("ingest-time") run once, or whenever the corpus changes; the last
four ("query-time") run on every user question.

### The local-vs-OpenRouter split, and why

This project deliberately runs embeddings and reranking **locally** and only
calls OpenRouter for the final answer:

| Stage | Where it runs | Model | Why |
|---|---|---|---|
| Embeddings | **Local** (CPU, sentence-transformers) | `BAAI/bge-small-en-v1.5` (384-dim) | **OpenRouter has no embeddings endpoint** — it is a chat-completions proxy over hosted LLMs, not an embeddings provider. This is the deliberate correction to an earlier "everything through OpenRouter" spec: embeddings must be local (or another dedicated embeddings API) because there's simply no OpenRouter route to call for them. A small local bi-encoder embeds every chunk in milliseconds on CPU, for free, offline, and deterministically — which also makes the on-disk embedding cache (`src/embed/cache.py`) possible. |
| Reranking | **Local** (CPU, sentence-transformers `CrossEncoder`) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoders score every (query, passage) pair jointly — much more accurate than bi-encoder cosine similarity, but too slow to run over a whole corpus. Running it only over the retrieved shortlist is fast, local, and needs no API key. |
| Answer generation | **OpenRouter** | `openai/gpt-4o-mini` | The one place an LLM is actually required, to turn retrieved chunks into fluent, cited prose. OpenRouter provides an OpenAI-compatible chat endpoint over many hosted models behind a single credential. |

Net effect: **two local models, one API key** (`OPENROUTER_API_KEY`).
Ingestion (generate → extract → chunk → embed → store) is entirely offline
and free to re-run. If the key isn't set, search and rerank still work, and
answer generation falls back to a no-LLM "retrieved-context-only" mode.

## Prerequisites

- Python 3.12
- Windows, macOS, or Linux (paths below assume Windows/PowerShell; Unix
  equivalents are noted)

## Quickstart

```powershell
# 1. Create and activate a virtual environment
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your OpenRouter key (get one at https://openrouter.ai/keys)
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=sk-or-...

# 4. Run the full pipeline: generate corpus -> chunk -> embed -> store -> project
python scripts/run_full_pipeline.py

# 5. Launch the chat UI
chainlit run src/ui/app.py
# then open the localhost URL Chainlit prints (default http://localhost:8000)
```

`scripts/run_full_pipeline.py` generates the synthetic corpus, chunks and
embeds it (with caching), upserts everything into Chroma, prints collection
stats, and writes a 2D embedding scatter plot to
`docs/course/embedding_scatter.html` / `.png`.

If `OPENROUTER_API_KEY` isn't set yet, steps 1-4 (and search/rerank in the
UI) still work fully offline — only the final LLM answer step falls back to
a retrieved-context-only mode.

## Running individual stages

Every stage is independently runnable as a module or script:

```bash
python -m src.data_gen.generate_all                          # regenerate the synthetic corpus
python -m src.ingest.chunk                                   # chunk the corpus, print a summary
python -m src.store.ingest_to_chroma --reset                  # chunk + embed + upsert (clean rebuild)
python -m src.store.inspect --stats                           # collection totals, per doc_type
python -m src.store.inspect --sample 10 --doc-type pdf        # preview stored chunks
python -m src.store.inspect --overlap <source_file.pdf>       # visualize chunk-overlap for one file
python -m src.store.inspect --project                         # write the 2D embedding scatter (UMAP/t-SNE)
python -m src.search.semantic_search --query "..."             # vector search only
python -m src.rerank.cross_encoder_rerank --query "..."        # search + cross-encoder rerank, before/after
python -m src.generate.answer_synthesis --query "..."          # search + rerank + cited LLM answer
```

## Configuration (`config.yaml`)

| Key | Meaning |
|---|---|
| `paths.*` | Data/cache/index locations, resolved to absolute paths relative to the project root |
| `corpus.num_pdfs` / `num_sops` / `num_csvs` / `seed` | Size and reproducibility of the synthetic corpus |
| `chunking.chunk_size` / `chunk_overlap` | Token-bounded splitter settings (tiktoken `cl100k_base`), default 500 / 100 tokens |
| `embedding.model` / `dim` / `batch_size` | Local embedding model (`BAAI/bge-small-en-v1.5`, 384-dim) and encode batch size |
| `vector_store.collection` | Chroma collection name (`cascade_docs`) |
| `search.top_k` | Number of candidates returned by semantic search (default 10) |
| `rerank.model` / `top_n` | Local cross-encoder model and how many results survive reranking (default 4) |
| `generation.provider` / `base_url` / `model` / `temperature` / `max_tokens` | OpenRouter chat settings (`openai/gpt-4o-mini`) |

Secrets (just `OPENROUTER_API_KEY`) live in `.env`, loaded by
`src/config.get_config()` alongside `config.yaml`.

## What's in the box

```
RAG/
├── config.yaml                     # all tunable knobs
├── .env.example                    # OPENROUTER_API_KEY template
├── ingest.py                       # project-root convenience CLI
├── scripts/run_full_pipeline.py    # one-shot generate -> ingest -> inspect -> project
├── src/
│   ├── config.py                   # get_config(), require_openrouter_key()
│   ├── data_gen/                   # synthetic PDF/SOP/CSV corpus generation
│   ├── ingest/                     # extract.py, chunk.py (token-bounded, overlapping)
│   ├── embed/                      # local_embedder.py, cache.py (on-disk embedding cache)
│   ├── store/                      # chroma_client.py, ingest_to_chroma.py, inspect.py
│   ├── search/                     # semantic_search.py
│   ├── rerank/                     # cross_encoder_rerank.py
│   ├── generate/                   # openrouter_client.py, answer_synthesis.py
│   └── ui/app.py                   # Chainlit chat UI
├── docs/course/                    # 10-lesson course, start at 00-overview.md
└── tests/                          # pytest suite (see below)
```

For a guided walkthrough of every stage, start at
[`docs/course/00-overview.md`](docs/course/00-overview.md).

## Testing

A fast pytest suite lives in `tests/`. Tests that load real ML models or
touch the populated vector store are marked `@pytest.mark.slow` (the models
are cached locally, so in practice they still run quickly).

```powershell
./.venv/Scripts/python.exe -m pytest -q            # run everything
./.venv/Scripts/python.exe -m pytest -q -m "not slow"   # skip model-loading tests
```

Coverage:

- `test_config.py` — config loads, paths are absolute, `embedding.dim == 384`, collection name.
- `test_chunk_overlap.py` — the key teaching invariant: a long synthetic document splits into multiple chunks, and consecutive chunks from the same source overlap (verified via both offset ranges and shared tail/head text).
- `test_cache.py` — `chunk_hash` is deterministic and collision-sensitive to text/model; `EmbeddingCache` round-trips a vector via a temp directory; a missing key returns `None`.
- `test_embedder.py` *(slow)* — the local embedder returns 384-dim, L2-normalized vectors.
- `test_search_smoke.py` *(slow)* — `search_and_rerank` against the real populated Chroma store returns non-empty, correctly-shaped results (skips gracefully if the store is empty).

## Verified / validated

This pipeline has been run end-to-end and validated:

- Full pipeline produced **348 chunks** in the Chroma collection `cascade_docs`: **206 pdf / 40 sop / 102 csv**.
- Semantic search + cross-encoder rerank confirmed working against the populated store.
- Live answer generation via OpenRouter (`openai/gpt-4o-mini`) confirmed working with cited output.
- Chainlit UI boots clean (`chainlit run src/ui/app.py`).

## Security

- `.env` (containing `OPENROUTER_API_KEY`) is listed in `.gitignore` and must never be committed.
- Only `.env.example` (a template with no real key) is tracked in git.
- If a key is ever accidentally committed, rotate it immediately at https://openrouter.ai/keys.

## License

MIT — see [`LICENSE`](LICENSE).
