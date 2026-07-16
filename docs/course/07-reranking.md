# 07 — Reranking

## Concept

**Bi-encoder retrieval vs. cross-encoder reranking** is a two-stage
retrieval pattern used throughout production RAG and search systems:

- A **bi-encoder** (the embedding model from lesson 03/06) encodes the
  query and every passage *independently*, into vectors that live in a
  shared space, and compares them with a cheap operation (cosine
  similarity / dot product). This is fast enough to run against an entire
  corpus of thousands or millions of chunks, because each passage's vector
  can be precomputed once, at ingest time, and reused for every future
  query.
- A **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2` here) takes
  the query *and* a candidate passage **together**, as a single input pair,
  and lets the model's attention mechanism directly compare them
  token-by-token. This produces a much more accurate relevance judgment
  than comparing two independently-computed vectors — but it cannot be
  precomputed, since the model needs to see the specific (query, passage)
  pair at inference time. Running it against an entire corpus for every
  query would be far too slow.

**Why two stages.** The combination gets the best of both: use the cheap,
precomputed bi-encoder to cast a wide net (`cfg.search.top_k`, e.g. 10
candidates) over the whole corpus, cheaply and quickly, then spend the
cross-encoder's more expensive, more accurate scoring only on that small
shortlist, to pick the best few (`cfg.rerank.top_n`, e.g. 4) to actually
send to the LLM. "Retrieve broad, then rerank the shortlist" — this project
implements exactly that pattern and nothing more exotic.

**When top-1 changes.** Because the bi-encoder and cross-encoder are
different models trained differently, their orderings frequently
disagree — a passage that scores highest by cosine similarity is not
guaranteed to be the passage the cross-encoder judges most relevant, and
vice versa. Comparing the "before" (semantic search order) and "after"
(cross-encoder order) rankings side by side is the most direct way to see
reranking add value: when top-1 changes, it usually means the bi-encoder's
cheap approximation missed something the more expensive joint model caught.

## In this repo

`src/rerank/cross_encoder_rerank.py`:

- `get_reranker() -> CrossEncoder` — lazily loads a module-level singleton
  `CrossEncoder(cfg.rerank.model, device="cpu")`
  (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default).
- `rerank(query, results, top_n=None) -> list[dict]` — takes
  `semantic_search`'s output list, builds `(query, r["text"])` pairs for
  every result, scores them all in one batch via `model.predict(pairs)`,
  then deep-copies each result dict and adds two new keys:
  `"rerank_score"` (raw cross-encoder logit, higher = better) and
  `"rerank_rank"` (1-based rank after sorting by `rerank_score` descending,
  truncated to `top_n or cfg.rerank.top_n`). The original `"score"` /
  `"distance"` from semantic search are preserved unchanged, so both
  orderings remain inspectable on the same objects.
- `search_and_rerank(query, top_k=None, top_n=None, doc_type=None,
  use_rerank=True) -> dict` — ties the two stages together in one call:
  runs `semantic_search(query, top_k=top_k, doc_type=doc_type)` for the
  `"retrieved"` list, then either reranks it (`use_rerank=True`, the
  default) or just truncates it to `top_n` unreranked (`use_rerank=False`,
  used for the "toggle rerank off" comparison). Returns
  `{"query", "retrieved", "reranked", "used_rerank"}` — having both lists
  side by side in one dict is exactly what powers the UI's before/after
  panels (lesson 09) and the CLI's before/after printout below.
- `_print_before_after(result)` — prints the `"BEFORE (semantic search
  order)"` and `"AFTER (cross-encoder rerank order)"` sections, then
  explicitly reports whether the top-1 result's `id` changed between the
  two.
- CLI (`main()`): `python -m src.rerank.cross_encoder_rerank --query "..."
  [--k N] [--n N] [--doc-type TYPE] [--no-rerank]`.

## Try it

Run search + rerank together and see the before/after comparison:

```powershell
python -m src.rerank.cross_encoder_rerank --query "What happens if a customer disputes a wire transfer?"
```

Run the same query with reranking disabled, to see the raw semantic-search
order truncated to `top_n` with no cross-encoder pass:

```powershell
python -m src.rerank.cross_encoder_rerank --query "What happens if a customer disputes a wire transfer?" --no-rerank
```

Try a broader `top_k` shortlist so the reranker has more candidates to sift
through:

```powershell
python -m src.rerank.cross_encoder_rerank --query "ATM cash replenishment steps" --k 15 --n 5
```

## What to look for / checkpoint

- With reranking on, the script's final line reads either
  `"Top-1 unchanged: <id>"` or `"Top-1 changed: '<old_id>' -> '<new_id>'"`.
  Both outcomes are valid and expected — try several different queries
  until you find one where it changes, since not every query will show a
  difference.
- Compare the `score=` (semantic/cosine) values shown in the `BEFORE`
  section against the `rerank_score=` values in the `AFTER` section — they
  are on **completely different numeric scales** (cosine similarity is
  bounded in `[-1, 1]`; cross-encoder logits are unbounded raw scores) and
  should never be compared directly to each other, only used to rank
  *within* their own stage.
- With `--no-rerank`, the `AFTER` section should be identical in ordering
  to the top `n` rows of the `BEFORE` section (just truncated), and
  `rerank_score` values should all read `n/a`.

## Teaching note

Ask trainees to run 4-5 different queries through
`cross_encoder_rerank.py` and tally how often top-1 changes vs. stays the
same. There's no universally "correct" ratio to expect — it depends heavily
on how semantically close the top few candidates are for a given query —
but the exercise builds intuition that reranking is a real, query-dependent
correction, not a rubber stamp on the bi-encoder's order. This directly
sets up Exercise 4 in [`10-exercises.md`](10-exercises.md).
