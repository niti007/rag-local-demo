# 06 — Semantic Search

## Concept

**Semantic search** answers "which stored chunks are most relevant to this
query?" by comparing vectors, not keywords. The query text is embedded with
the exact same model used to embed every stored chunk, and the vector store
returns the *k* nearest stored vectors by the configured distance metric.
Because the embedding space is meant to place semantically similar text
close together, this finds relevant passages even when they don't share
exact words with the query (e.g. a query about "overdraft fees" can surface
a chunk that talks about "insufficient funds charges" without either
phrase appearing in the other).

**Same model, both sides.** This is a hard requirement, not just good
practice: a query vector produced by model A is not comparable to passage
vectors produced by model B — their vector spaces have no shared geometry.
This project guarantees consistency by having exactly one embedding
function, `embed_texts` / `embed_query` in `src/embed/local_embedder.py`
(lesson 03), used both at ingest time (`ingest_all`) and at query time
(`semantic_search`).

**Distance vs. similarity.** Cosine *distance* is `1 - cosine_similarity`:
it decreases as vectors get more alike (0 = identical direction, 1 =
orthogonal, 2 = opposite). Qdrant's `COSINE` metric already returns the
**similarity** as its `score` (higher is better), so this project uses
`score = point.score` directly and derives `distance = 1 - score` to keep the
result shape stable. Results can be sorted and compared the "obvious" way
(bigger number = closer match) rather than requiring everyone to remember
"lower distance is better."

**Top-k.** Semantic search doesn't try to find *the* single best match — it
returns the top `k` candidates (`cfg.search.top_k`, default 10), on the
theory that (a) a broader shortlist gives the reranking step in lesson 07
more material to sift through, and (b) approximate nearest-neighbor search
isn't perfectly precise anyway, so casting a slightly wider net catches
near-misses.

**Metadata filtering.** Sometimes you know in advance which document type
is relevant (e.g. "what does the SOP say," or "look only at transaction
data"). Qdrant supports a payload `Filter` alongside the vector search (made fast
by the keyword index on `doc_type`), so you can restrict the
nearest-neighbor search to only chunks whose `doc_type` payload matches, without touching the embedding or distance computation at
all.

## In this repo

`src/search/semantic_search.py`:

- `semantic_search(query, top_k=None, doc_type=None) -> list[dict]` — the
  main entry point.
  1. `k = top_k or cfg.search.top_k`
  2. `query_vec = embed_query(query)` — embeds the query with the local
     bge-small model (with `is_query=True`, applying `cfg.embedding.query_prefix`
     if set).
  3. `collection = get_or_create_collection()` — the shared `cascade_docs`
     collection name (lesson 04).
  4. Builds a payload `Filter` on `doc_type` if given, else `None` (no
     filter, searches everything).
  5. Calls `client.query_points(collection_name=collection, query=query_vec,
     limit=k, query_filter=query_filter, with_payload=True).points`.
  6. Converts each point into a flat
     `{"id", "text", "metadata", "distance", "score"}` dict: `score` is
     `point.score`, `distance` is `1 - score`, `id` is the payload's
     `chunk_id`, and `metadata` is the payload minus `text`/`chunk_id`.
  7. Sorts by `score` descending (Qdrant already returns results best-first,
     but this makes the "higher score = better" invariant explicit and
     stable regardless of the server's internal ordering guarantees).
- `_print_results(results)` — terminal-friendly renderer: rank, score,
  source file, page, and a 120-character snippet per result.
- CLI (`main()`): `python -m src.search.semantic_search --query "..." [--k N] [--doc-type TYPE]`.

## Try it

Run a plain semantic search over the whole corpus:

```powershell
python -m src.search.semantic_search --query "What is the overdraft fee policy?"
```

Restrict `top_k` to a smaller shortlist:

```powershell
python -m src.search.semantic_search --query "wire transfer authorization steps" --k 3
```

Filter to only one `doc_type` (SOPs, for a procedure-shaped question):

```powershell
python -m src.search.semantic_search --query "wire transfer authorization steps" --doc-type sop
```

Compare that against filtering to `pdf` only, for the same query, and note
how the results (and their scores) differ:

```powershell
python -m src.search.semantic_search --query "wire transfer authorization steps" --doc-type pdf
```

## What to look for / checkpoint

- Each printed result shows `score=X.XXXX` where higher values (closer to
  1.0) indicate closer matches; results should be printed in descending
  score order.
- The top result's snippet should be topically related to the query even
  when it doesn't share exact keywords — this is the semantic-vs-keyword
  distinction in action.
- With `--doc-type sop`, every returned result's underlying metadata is
  restricted to SOP chunks (only visible if you inspect `metadata` directly,
  since `_print_results` doesn't print `doc_type` — cross-check with
  `python -m src.store.inspect --sample` on the same source files, or add a
  quick `print(r["metadata"]["doc_type"])` in a REPL).
- Changing `--k` changes only how many rows are printed — it does not
  change the score of the top result, since scores are a property of query
  vs. stored vector, independent of how many results you asked for.

## Teaching note

Have trainees run the same query with and without `--doc-type`, and check
whether the *unfiltered* top result would have been excluded by the filter
they'd have picked (i.e. "did filtering to SOP docs actually throw away a
better PDF match?"). This is a concrete way to discuss the risk of metadata
filtering: it can improve precision when you're confident about which
document type is relevant, but it can also discard a genuinely more relevant
result that happens to live in a different `doc_type`. There's no filter
that's always correct — it's a precision/recall tradeoff made explicit.
