# 10 — Hands-On Exercises

These exercises assume you've completed the quickstart in
[`00-overview.md`](00-overview.md) (venv created, dependencies installed,
corpus generated and ingested at least once). Each exercise references the
lesson that explains the underlying concept in depth.

---

## Exercise 1 — Change `chunk_overlap` and observe the overlap region

**Concept:** [`02-chunking-and-overlap.md`](02-chunking-and-overlap.md)

**Task:** Reduce `chunk_overlap` from its default (100 tokens) to something
much smaller (say, 10), re-chunk and re-ingest, and inspect how the
overlap region shrinks.

**Commands:**

```powershell
# 1. Note the current behavior first, before changing anything:
python -m src.store.inspect --overlap 01_policy_account_overdraft_policy.pdf

# 2. Edit config.yaml: chunking.chunk_overlap: 100 -> 10

# 3. Re-ingest with the new setting (reset=True is required -- chunk_overlap
#    changes every chunk's boundaries, so old chunk ids no longer apply)
python -m src.store.ingest_to_chroma --reset

# 4. Inspect the same file again
python -m src.store.inspect --overlap 01_policy_account_overdraft_policy.pdf
```

**Expected result:** The `[offsets]` overlap span between consecutive
chunks should be noticeably smaller (or occasionally absent for some pairs)
compared to the first run. The `[text]` longest-common-substring section
should show a shorter (or no) shared run.

**Reflection question:** If you set `chunk_overlap` to `0`, is it possible
for two consecutive chunks to still show *some* offset overlap? (Hint: look
at how `chunk_document` finds chunk boundaries via `full_text.find` — the
splitter's chunk boundaries and the offset-finding logic are not perfectly
tied to `chunk_overlap` in every edge case.)

---

## Exercise 2 — Swap the embedding model and compare

**Concept:** [`03-embeddings-and-cache.md`](03-embeddings-and-cache.md)

**Task:** Change `cfg.embedding.model` to a different `sentence-transformers`
model with a **different output dimension** than 384, without updating
`cfg.embedding.dim`, and observe what breaks.

**Commands:**

```powershell
# 1. Edit config.yaml: embedding.model: BAAI/bge-small-en-v1.5 -> sentence-transformers/all-MiniLM-L6-v2
#    (all-MiniLM-L6-v2 outputs 384-dim too, so first try a genuinely
#    different-dim model, e.g. sentence-transformers/all-mpnet-base-v2, which outputs 768-dim)

# 2. Try to re-ingest without touching embedding.dim (still 384)
python -m src.store.ingest_to_chroma --reset
```

**Expected result:** `_verify_dim` in `src/embed/local_embedder.py` raises a
`RuntimeError` reporting a dimension mismatch between the model's actual
output length and `cfg.embedding.dim`. Now fix `embedding.dim` to match
(768 for `all-mpnet-base-v2`) and re-run — ingestion should succeed, but
note the entire embedding cache under `.cache/embeddings/` is now
effectively invalidated for the new model (different model name → different
`chunk_hash` → all cache misses on first re-ingest).

**Reflection question:** Why does `chunk_hash` include the model name in
its input (`f"{model}\n{text}"`) instead of hashing the text alone? What
would go wrong if it didn't?

---

## Exercise 3 — Filter search by `doc_type`

**Concept:** [`06-semantic-search.md`](06-semantic-search.md)

**Task:** Run the same query with no filter, then with each `doc_type`
filter in turn, and compare result sets.

**Commands:**

```powershell
python -m src.search.semantic_search --query "how are exceptions escalated"
python -m src.search.semantic_search --query "how are exceptions escalated" --doc-type pdf
python -m src.search.semantic_search --query "how are exceptions escalated" --doc-type sop
python -m src.search.semantic_search --query "how are exceptions escalated" --doc-type csv
```

**Expected result:** The unfiltered run mixes doc types; each filtered run
returns only chunks from that one source type, and (usually) different
scores than the unfiltered run's corresponding rank, since the filter
changes the *candidate pool* Chroma searches within.

**Reflection question:** For this query, which `doc_type` filter, if any,
would you pick in a real usage scenario, and why? Would your answer change
for a query about "checking account minimum balance" instead?

---

## Exercise 4 — Toggle rerank off and find a query where top-1 differs

**Concept:** [`07-reranking.md`](07-reranking.md)

**Task:** Find at least one query where reranking changes which chunk ranks
first.

**Commands:**

```powershell
python -m src.rerank.cross_encoder_rerank --query "What ID do I need to open an account?"
python -m src.rerank.cross_encoder_rerank --query "What ID do I need to open an account?" --no-rerank
```

Try several different queries if the first doesn't show a change — try
questions that could plausibly match multiple different documents about
similar topics (policy vs. SOP vs. manual all mentioning identity
verification, for instance).

**Expected result:** For at least one query, the final line changes from
`Top-1 unchanged: <id>` (when comparing the reranked run's stated ordering
against itself) to a *different* top-1 chunk when you diff the reranked
run's top-1 against the `--no-rerank` run's top-1 by hand (the tool
reports top-1 changes based on rerank-vs-retrieval within a single run, so
compare the two runs' printed top rows directly).

**Reflection question:** When top-1 changes, is the cross-encoder's pick
always the more obviously relevant chunk to a human reader? Try reading
both candidates directly and judge for yourself.

---

## Exercise 5 — Change `top_k`/`top_n` in the UI

**Concept:** [`09-chainlit-ui.md`](09-chainlit-ui.md)

**Task:** In the Chainlit UI, set `top_k` to its minimum (2) and `top_n` to
1, ask a question, then set `top_k` to its maximum (20) and `top_n` to 10
and ask the same question.

**Commands:**

```powershell
chainlit run src/ui/app.py
# then use the gear icon in the browser
```

**Expected result:** With `top_k=2, top_n=1`, the Retrieved section shows
only 2 candidates and the Reranked section shows exactly 1. With
`top_k=20, top_n=10`, both sections are much longer, and the Generated
answer may read differently since the LLM now has more (or less) context
to work with.

**Reflection question:** Does giving the LLM more context (`top_n=10`)
strictly make the answer better? What could go wrong with a very large
`top_n`?

---

## Exercise 6 — Inspect the scatter and explain clusters

**Concept:** [`05-inspection-and-viz.md`](05-inspection-and-viz.md)

**Task:** Generate the 2D embedding projection and identify which visual
clusters correspond to which `doc_type`.

**Commands:**

```powershell
python -m src.store.inspect --project
# then open docs/course/embedding_scatter.html in a browser
```

**Expected result:** You should be able to point at one or more visually
distinct regions and, by hovering (the HTML version shows `source_file` and
a text preview per point), confirm they're dominated by one `doc_type` —
`csv` chunks in particular, given their tabular `"Row N: col: value"` text
format, often form a visually separate cluster from PDF/SOP prose.

**Reflection question:** Do `pdf` and `sop` chunks separate cleanly, or do
they overlap? What does that tell you about how similar (or different)
policy-document prose and SOP-document prose actually are, at the level of
sentence content?

---

## Exercise 7 — Ask a question the corpus can't answer

**Concept:** [`08-answer-generation.md`](08-answer-generation.md)

**Task:** Ask a question that has no basis anywhere in the synthetic
Cascade Bank corpus, and observe how the system responds.

**Commands:**

```powershell
python -m src.generate.answer_synthesis --query "What was Cascade Bank's stock price in 2019?"
```

(Or via the UI: ask the same question in the Chainlit chat.)

**Expected result:** Since Cascade Bank is entirely fictional and has no
stock price data anywhere in the corpus, a well-grounded answer should
explicitly say the context doesn't contain this information, rather than
inventing a number. If the model instead invents a plausible-sounding
figure, that's a real, teachable grounding failure — not every LLM call
will refuse perfectly every time, and that's worth discussing honestly.

**Reflection question:** What in `SYSTEM_PROMPT` (in
`src/generate/answer_synthesis.py`) is responsible for encouraging this
refusal behavior? What would you change in the prompt to make refusal more
or less likely?

---

## Exercise 8 — Trace one chunk's id from cache to Chroma

**Concept:** [`03-embeddings-and-cache.md`](03-embeddings-and-cache.md),
[`04-vector-store-chromadb.md`](04-vector-store-chromadb.md)

**Task:** Pick one stored chunk, compute its content hash by hand, and
confirm the exact same hash shows up both as an embedding-cache filename
and as its Chroma document id.

**Commands:**

```powershell
python -c "
from src.store.chroma_client import get_or_create_collection
from src.embed.cache import chunk_hash
from src.config import get_config

collection = get_or_create_collection()
result = collection.get(include=['documents'], limit=1)
chunk_id = result['ids'][0]
text = result['documents'][0]

cfg = get_config()
computed_hash = chunk_hash(text, cfg.embedding.model)

print('Chroma id:       ', chunk_id)
print('Computed hash:   ', computed_hash)
print('Match:           ', chunk_id == computed_hash)

import os
cache_file = cfg.paths.embedding_cache / f'{computed_hash}.npy'
print('Cache file exists:', cache_file.exists())
"
```

**Expected result:** `chunk_id == computed_hash` prints `True`, and the
corresponding `.npy` file exists in the embedding cache directory — proof
that the same `chunk_hash(text, model)` function (from
`src/embed/cache.py`) is used as both the cache key at embed time and the
document id at Chroma-upsert time, which is exactly what makes re-ingestion
idempotent (lesson 04).

**Reflection question:** If you edited that one chunk's source PDF slightly
(changing its text) and re-ingested, would the old Chroma entry for this
chunk be deleted, or would it become an orphan? (Revisit the teaching note
at the end of lesson 04 if you're not sure.)
