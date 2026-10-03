# Cascade Bank RAG Demo

Welcome! This is a **local, teaching-oriented** Retrieval-Augmented Generation
(RAG) demo built around **Cascade Bank**, a completely fictional bank. All
policy PDFs, SOPs, and CSVs in the corpus are synthetic data generated for
this demo -- nothing here describes a real institution, customer, or
financial record.

## The flow

Every question you ask is shown running through three distinct stages, so
you can see exactly what each stage contributes:

1. **Retrieve** -- a local bi-encoder embedding model searches the Qdrant
   vector store and returns the `top_k` most similar chunks by cosine
   similarity.
2. **Rerank** -- a cross-encoder re-scores that shortlist for finer-grained
   relevance and keeps the `top_n` best (toggle this off in settings to see
   how much it actually changes the ranking).
3. **Generate** -- an LLM (via OpenRouter) synthesizes a grounded, cited
   answer from the reranked context. If no API key is configured, you'll see
   the retrieved context only, clearly labeled as such.

## Commands

- `/inspect` -- show vector-store collection stats (chunk counts by document
  type) and a 2D scatter plot of the embedding space.
- `/help` -- show command help again at any time.

## Settings

Open the gear icon above the chat box to adjust:

- **top_k** -- how many candidates the retrieval step pulls before reranking.
- **top_n** -- how many candidates survive reranking and get passed to the
  LLM.
- **use_rerank** -- turn cross-encoder reranking on/off.
- **doc_type** -- restrict retrieval to `pdf`, `sop`, `csv`, or `all`.

## One-key note

Set `OPENROUTER_API_KEY` in a `.env` file at the project root (see
`.env.example`) to enable LLM-generated answers. Without it, the demo still
works fully for retrieval and reranking -- you'll just see retrieved context
instead of a synthesized answer.
