# 09 — The Chainlit Chat UI

## Concept

All the pipeline stages up through answer generation are individually
runnable as CLIs, but a chat interface makes the whole retrieve → rerank →
generate flow interactive and, importantly, **visible** — a trainee should
be able to see not just the final answer, but exactly what was retrieved
and how reranking changed the order, for every single question they ask.
[Chainlit](https://docs.chainlit.io/) is a Python framework purpose-built
for exactly this kind of "show your work" LLM chat UI, with minimal
boilerplate (decorator-based event handlers, built-in settings widgets,
streaming-friendly message/step primitives).

**Three-section render.** Rather than showing only a final answer, this
UI renders three separate messages per question:

1. **Retrieved (before rerank)** — the raw semantic search shortlist, in
   cosine-similarity order.
2. **Reranked (after)** — the cross-encoder's re-ordering of that
   shortlist, plus an explicit note on whether the top-1 result changed.
3. **Generated answer** — the LLM's grounded, cited answer (or the
   retrieved-context-only fallback banner if no API key is configured).

This mirrors the exact `retrieved` / `reranked` / (LLM call) structure
`search_and_rerank` and `synthesize_answer` already return — the UI is a
thin rendering layer over the same functions the CLIs in lessons 06-08 use,
not a separate reimplementation.

**`ChatSettings` widgets.** Chainlit's `cl.ChatSettings` renders a settings
panel (gear icon) with live-adjustable controls, backed here by a `Slider`
for `top_k`, a `Slider` for `top_n`, a `Switch` for `use_rerank`, and a
`Select` for `doc_type`. Changing any of these live, without restarting the
app or editing `config.yaml`, is what makes several of the hands-on
exercises in lesson 10 possible (e.g. toggling rerank off, comparing
`top_k`/`top_n` values) directly from the chat window.

**The `/inspect` command.** A special chat command (not a real chat
message) that surfaces the same `collection_stats()` and
`embedding_projection()` tooling from lesson 05 directly inside the chat
session — trainees don't have to leave the UI and drop to a terminal to
see collection totals or the 2D embedding scatter.

**Light / dark theme.** Chainlit 2.x ships a theme toggle in the chat
header with Light, Dark and System options, so the UI needs no custom CSS
or code for it. On a first visit the theme follows the operating system's
`prefers-color-scheme`; once a visitor picks an option, the choice is saved
per browser in `localStorage` and survives reloads. Setting `default_theme`
under `[UI]` in `.chainlit/config.toml` would override the OS default for
first-time visitors, so this repo leaves it unset. This is also why
`requirements.txt` pins `chainlit>=2.0,<3`.

## In this repo

`src/ui/app.py`:

- Import-safety design note (from the module docstring): only `src.config`
  is imported at module load time; everything else (search, rerank,
  generate, `store.inspect`) is imported **lazily inside handlers**, so the
  app still starts cleanly even if the Qdrant collection is empty, the
  sentence-transformers models haven't been downloaded yet, or
  `OPENROUTER_API_KEY` is unset. Heavy/blocking calls
  (`search_and_rerank`, `synthesize_answer`, `collection_stats`,
  `embedding_projection`) are all wrapped in `cl.make_async(...)` so they
  run off Chainlit's event loop and don't freeze the UI.
- `DOC_TYPE_OPTIONS = ["all", "pdf", "sop", "csv"]` and
  `_doc_type_for_query(doc_type)` — maps the UI's `"all"` sentinel back to
  the `None` that `semantic_search`/`search_and_rerank` expect for "no
  filter."
- `@cl.on_chat_start` → `on_chat_start()` — builds the `cl.ChatSettings`
  panel: `Slider(id="top_k", initial=cfg.search.top_k, min=2, max=20)`,
  `Slider(id="top_n", initial=cfg.rerank.top_n, min=1, max=10)`,
  `Switch(id="use_rerank", initial=True)`, `Select(id="doc_type",
  values=DOC_TYPE_OPTIONS, initial_index=0)`. Stores each in
  `cl.user_session`. Also probes `require_openrouter_key()` (catching any
  exception) to set a `has_key` session flag, and sends the welcome
  message (`WELCOME_MD`) with a note on whether generation will be
  LLM-synthesized or retrieved-context-only for this session.
- `@cl.on_settings_update` → `on_settings_update(settings)` — copies any
  changed `top_k`/`top_n`/`use_rerank`/`doc_type` values back into
  `cl.user_session`, so adjustments in the gear-icon panel take effect on
  the very next message.
- `_handle_inspect()` — the `/inspect` command's implementation: calls
  `collection_stats()` and `embedding_projection()` (both via
  `cl.make_async`), renders the stats as a markdown list, and attaches the
  freshly regenerated `embedding_scatter.png` inline via `cl.Image`.
- `@cl.on_message` → `on_message(message)` — the main handler:
  - `/inspect` → `_handle_inspect()`.
  - `/help` → sends `HELP_MD`.
  - Empty message → prompts for a question.
  - Otherwise: reads `top_k`/`top_n`/`use_rerank`/`doc_type`/`has_key` from
    `cl.user_session`, runs `search_and_rerank(text, top_k=top_k,
    top_n=top_n, doc_type=doc_type, use_rerank=use_rerank)` inside a
    `cl.Step(name="Retrieve", type="retrieval")` step, renders the
    **Retrieved** section (`_fmt_retrieved_line`) inside a
    `cl.Step(name="Rerank", type="rerank")` step boundary, renders the
    **Reranked** section (`_fmt_reranked_line`) with an explicit
    top-1-changed/unchanged note, then — inside a
    `cl.Step(name="Generate", type="llm")` step — either calls
    `synthesize_answer(text, reranked)` (if `has_key`, falling back to
    `answer_only_mode` on any exception) or calls `answer_only_mode`
    directly (if no key), and sends the final **Generated answer** message.
  - Any unhandled exception in the whole flow is caught and reported back
    to the chat as a `⚠️` warning message rather than crashing the session.

## Try it

Launch the UI from the project root:

```powershell
PYTHONPATH=. chainlit run src/ui/app.py
```

In the browser tab that opens:

1. Click the gear icon and note the four controls: `top_k`, `top_n`,
   `use_rerank`, `doc_type`.
2. Ask: `What is Cascade Bank's overdraft fee policy?`
3. Observe the three response messages: Retrieved, Reranked, Generated
   answer.
4. Type `/inspect` and observe collection stats plus the embedding scatter
   image render inline.
5. Type `/help` to see the command reference again.

## What to look for / checkpoint

- The welcome message on chat start explicitly states whether
  `OPENROUTER_API_KEY` was detected — confirm it matches whether you have
  a `.env` key set.
- Each question produces exactly three chat messages in order: `## 🔎
  Retrieved (before rerank)`, `## 🎯 Reranked (after)`, `## 🧠 Generated
  answer` — plus the three collapsible `Step` entries (Retrieve, Rerank,
  Generate) above them showing timing/status.
- The Reranked section explicitly states either `_Top-1 unchanged after
  reranking._` or `**Top-1 changed after reranking:** ... -> ...` — this
  should visually match whatever you'd get running the same query through
  `python -m src.rerank.cross_encoder_rerank --query "..."` (lesson 07).
- `/inspect` renders a stats list (`Collection`, `Total chunks`, `By
  doc_type`) followed by an inline PNG of the current embedding scatter.
- Moving the `top_k` or `top_n` sliders and re-asking the same question
  should change how many candidates appear in the Retrieved/Reranked
  sections on the next message (settings apply going forward, not
  retroactively to messages already sent).

## Teaching note

Walk trainees through toggling `use_rerank` off in the settings panel,
re-asking the exact same question they asked with it on, and comparing the
Reranked section's contents both times side by side in the chat log
(Chainlit keeps full session history, so both responses stay visible).
With `use_rerank` off, the Reranked section is just a truncated copy of the
Retrieved section (`rerank_score` reads `n/a` for a reason — see
`search_and_rerank`'s `use_rerank=False` branch in lesson 07) — this is a
fast, no-terminal way to run Exercise 4 from lesson 10 for a live audience.
