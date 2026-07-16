# 08 — Answer Generation

## Concept

Retrieval and reranking (lessons 06-07) find the right passages, but they
don't produce a *readable answer* — that's a fluent-language task, which is
where a large language model comes in. **Answer generation** is the step
that turns a handful of retrieved chunks plus the user's question into a
natural-language answer.

**Grounding.** The critical property that distinguishes RAG generation from
just asking an LLM a question directly is **grounding**: the model is
instructed to answer *using only* the provided context, and to say so
plainly when the context doesn't contain the answer, rather than falling
back on its own (potentially wrong, unverifiable, or simply irrelevant to
this fictional corpus) background knowledge. This is enforced entirely
through the system prompt here — there's no additional verification step
that checks the LLM actually complied, which is a real limitation worth
naming explicitly when teaching this lesson.

**The context block + citations.** Retrieved/reranked chunks are formatted
into a numbered list (`[1]`, `[2]`, ...) with each entry showing its source
file, doc type, and page number before the chunk text. The system prompt
then instructs the model to cite the bracketed index inline (e.g. "Rates
vary by loan type [2][3].") right after any claim it draws from that
source. This is what lets a reader trace any generated sentence back to a
specific chunk of the underlying corpus — the same mechanism a research
paper's citations provide.

**The system prompt.** Beyond grounding and citations, the prompt also
explicitly frames the whole exercise as synthetic/fictional (matching the
disclaimers baked into the corpus itself in lesson 01) — the model is told
none of this describes a real institution, so it doesn't need to hedge or
refuse on "this looks like real financial data" grounds.

**The OpenRouter call.** Generation is the one stage in this pipeline that
calls a hosted LLM. OpenRouter exposes an OpenAI-compatible chat
completions API, so this project reuses the official `openai` Python SDK,
just pointed at OpenRouter's `base_url` with an `OPENROUTER_API_KEY`. See
[`00-overview.md`](00-overview.md#the-local-vs-openrouter-split-and-why)
for why this is the *only* networked/paid call in the whole pipeline.

**Retrieved-context-only fallback with no key.** Because
`OPENROUTER_API_KEY` might not be set (a fresh clone of this repo, an
offline demo, a trainee who hasn't gotten a key yet), the generation module
is import-safe without a key, and provides `answer_only_mode` — a pure
Python function with no network call — that just formats and returns the
retrieved context with its citations attached, plus a note that no LLM call
was made. This means the entire pipeline through reranking is always
runnable and demonstrable with zero external dependencies or cost, and only
the very last "make it read like an answer" step needs a key.

## In this repo

- `src/generate/openrouter_client.py`
  - `get_client() -> OpenAI` — lazily builds and caches an `OpenAI` SDK
    client with `base_url=cfg.generation.base_url`
    (`https://openrouter.ai/api/v1`) and `api_key=require_openrouter_key()`.
    The key is fetched **only here**, not at module import time, so
    `import src.generate.openrouter_client` never fails just because a key
    is missing.
  - `_create_completion(client, **kwargs)` — wrapped in `tenacity`'s
    `@retry` decorator: retries `APIConnectionError`, `RateLimitError`, and
    `InternalServerError` up to 4 attempts with exponential backoff
    (`wait_exponential(multiplier=1, min=1, max=20)`).
  - `chat_completion(messages, model=None, temperature=None,
    max_tokens=None, stream=False) -> str | Iterable` — resolves any unset
    arguments from `cfg.generation.*` (`model: openai/gpt-4o-mini`,
    `temperature: 0.2`, `max_tokens: 800`), then calls
    `client.chat.completions.create(...)`. Returns the plain response text
    (`resp.choices[0].message.content`) when `stream=False`, or the raw
    streaming iterator when `stream=True`.

- `src/generate/answer_synthesis.py`
  - `SYSTEM_PROMPT` — the full grounding instructions: answer only from the
    numbered context block, cite sources inline via `[n]`, and explicitly
    say so when the context is insufficient rather than guessing.
  - `build_context(chunks) -> str` — formats a list of reranked chunk dicts
    into the numbered `[1] (source_file, doc_type, page N)` / text block
    described above. Reads `page_number` from metadata (falling back to
    `page` if present, else `"N/A"`), and degrades gracefully on missing
    metadata rather than raising.
  - `synthesize_answer(query, chunks, stream=False) -> str | Iterable` —
    builds the context block, assembles a two-message chat
    (`SYSTEM_PROMPT` + a user message containing the context and question),
    and calls `chat_completion(messages, stream=stream)`.
  - `answer_only_mode(query, chunks) -> str` — the no-API fallback: returns
    the question plus the same `build_context` output, explicitly labeled
    "(retrieved-context-only mode -- no LLM call)".
  - CLI (`main()`): `python -m src.generate.answer_synthesis --query "..."
    [--k N] [--n N] [--doc-type TYPE]` — retrieves + reranks via
    `search_and_rerank`, prints the context block, then either calls
    `synthesize_answer` (if `OPENROUTER_API_KEY` is set) or falls back to
    `answer_only_mode` (if unset, or if a `RuntimeError` is raised when the
    key check fails).

## Try it

With `OPENROUTER_API_KEY` set in `.env`, ask a question the corpus can
actually answer:

```powershell
python -m src.generate.answer_synthesis --query "What is Cascade Bank's overdraft fee policy?"
```

Without a key set (temporarily rename or unset it to see the fallback path):

```powershell
$env:OPENROUTER_API_KEY=""
python -m src.generate.answer_synthesis --query "What is Cascade Bank's overdraft fee policy?"
```

Ask something entirely outside the synthetic corpus's scope, to observe
grounding in action:

```powershell
python -m src.generate.answer_synthesis --query "What is Cascade Bank's stock ticker symbol?"
```

## What to look for / checkpoint

- With a valid key: `=== ANSWER ===` shows fluent prose with inline `[n]`
  citations that correspond exactly to the `[n]` entries printed just above
  in `=== CONTEXT ===`.
- Without a key: the output explicitly states
  `(No OPENROUTER_API_KEY set -- falling back to answer-only mode, no LLM
  call made.)` followed by `=== ANSWER (retrieved-context-only mode) ===`
  showing the raw context block, unsynthesized.
- For the out-of-scope question (no ticker symbol exists in a fictional
  bank's synthetic corpus), a well-grounded answer should say something
  close to "the provided context does not contain this information" rather
  than inventing a plausible-sounding ticker symbol — this is the payoff of
  the grounding instruction in `SYSTEM_PROMPT`. If it doesn't refuse
  cleanly, that's a real, observable limit of prompt-only grounding worth
  discussing rather than hiding.

## Teaching note

Deliberately trigger the fallback (unset the key) in front of trainees
before showing the "real" LLM-generated answer — seeing the honest,
un-synthesized context block first makes it much clearer, by contrast,
exactly what value the generation step adds (fluency, synthesis across
multiple chunks) versus what it doesn't add (new facts not present in
retrieval). This ordering also reinforces that the pipeline's earlier
stages are the ones doing the actual "finding the right information" work;
generation only formats it.
