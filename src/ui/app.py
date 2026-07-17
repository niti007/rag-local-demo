"""
Chainlit UI for the Cascade Bank local RAG teaching demo.

Flow shown to the user on every query:
    1. Retrieve  -- semantic_search() top_k candidates (bi-encoder cosine similarity)
    2. Rerank    -- cross-encoder re-scores the shortlist down to top_n
    3. Generate  -- an LLM (via OpenRouter) synthesizes a cited answer from the
                    reranked context, or -- if no OPENROUTER_API_KEY is set --
                    falls back to a retrieved-context-only rendering.

Commands:
    /inspect  -- show collection stats + a 2D embedding scatter (UMAP/TSNE)
    /help     -- show this help text again

Design notes:
    - Only `src.config` is imported at module load time. Everything else
      (search, rerank, generate, store.inspect) is imported lazily inside the
      handlers, so the app still imports cleanly even if the Chroma
      collection is empty, sentence-transformers models haven't been
      downloaded yet, or OPENROUTER_API_KEY is unset.
    - Heavy / blocking calls (search_and_rerank, synthesize_answer,
      collection_stats, embedding_projection) are all run off the event loop
      via cl.make_async so the UI stays responsive.
"""

from __future__ import annotations

import threading

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch

from src.config import get_config

DOC_TYPE_OPTIONS = ["all", "pdf", "sop", "csv"]

# Start loading the local models (embedder + cross-encoder) in a background
# thread the moment the server process boots, so they're warm by the time the
# user sends their first query. On CPU this cold-load can take a couple of
# minutes; doing it here (not on first message) keeps the chat responsive.
_MODELS_READY = threading.Event()


def _warm_models() -> None:
    """Force the lazy model singletons to load (blocking; run off the event loop)."""
    from src.embed.local_embedder import embed_query
    from src.rerank.cross_encoder_rerank import get_reranker

    embed_query("warmup")
    get_reranker().predict([("warmup", "warmup")])


def _bg_warm() -> None:
    try:
        _warm_models()
    finally:
        _MODELS_READY.set()


threading.Thread(target=_bg_warm, name="model-warmup", daemon=True).start()


def _doc_type_for_query(doc_type: str | None) -> str | None:
    """Map the UI's "all" sentinel to the None the search layer expects."""
    if not doc_type or doc_type == "all":
        return None
    return doc_type


def _snippet(text: str, n: int = 140) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[:n] + "..."


def _fmt_retrieved_line(rank: int, r: dict) -> str:
    meta = r.get("metadata") or {}
    source_file = meta.get("source_file", "?")
    page = meta.get("page_number", meta.get("page", "?"))
    score = r.get("score")
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
    return f"**[{rank}]** score=`{score_str}`  ·  `{source_file}` p.{page}  \n> {_snippet(r.get('text', ''))}"


def _fmt_reranked_line(r: dict) -> str:
    meta = r.get("metadata") or {}
    source_file = meta.get("source_file", "?")
    page = meta.get("page_number", meta.get("page", "?"))
    rr = r.get("rerank_score")
    rr_str = f"{rr:.4f}" if isinstance(rr, (int, float)) else "n/a"
    score = r.get("score")
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
    rank = r.get("rerank_rank", "?")
    return (
        f"**[{rank}]** rerank_score=`{rr_str}` (orig score=`{score_str}`)  ·  "
        f"`{source_file}` p.{page}  \n> {_snippet(r.get('text', ''))}"
    )


WELCOME_MD = """\
# Cascade Bank RAG Demo

This is a **local, teaching-oriented** Retrieval-Augmented Generation (RAG) demo \
built on a fictional bank ("Cascade Bank") with synthetic policy PDFs, SOPs, and \
CSVs. Nothing here reflects a real institution.

Every question you ask runs through three visible stages so you can see exactly \
what retrieval and reranking are doing to the final answer:

1. **Retrieve** -- bi-encoder semantic search over the local Chroma vector store \
(cosine similarity).
2. **Rerank** -- a cross-encoder re-scores the retrieved shortlist for finer-grained \
relevance.
3. **Generate** -- an LLM synthesizes a cited answer from the reranked context \
(or, with no API key, you'll see the retrieved context only).

**Commands**
- `/inspect` -- show vector-store collection stats and a 2D embedding scatter plot.
- `/help` -- show this help again.

Use the settings panel (gear icon) to adjust `top_k`, `top_n`, toggle reranking, \
and filter by document type.
"""

HELP_MD = """\
### Help

- Type any question about Cascade Bank's (fictional) policies, procedures, or data \
to run it through retrieve -> rerank -> generate.
- `/inspect` -- prints Chroma collection stats (total chunks, counts by doc_type) \
and regenerates the 2D embedding scatter plot (UMAP, falls back to TSNE), attaching \
the PNG here and pointing you to the interactive HTML version.
- `/help` -- shows this message.
- Open the settings panel (gear icon, top of the chat) to change:
  - **top_k** -- how many candidates semantic search retrieves before reranking.
  - **top_n** -- how many candidates survive reranking and get sent to the LLM.
  - **use_rerank** -- turn cross-encoder reranking on/off (compare with it off!).
  - **doc_type** -- restrict retrieval to one corpus type (pdf / sop / csv) or "all".
"""


@cl.on_chat_start
async def on_chat_start() -> None:
    cfg = get_config()

    settings = await cl.ChatSettings(
        [
            Slider(
                id="top_k",
                label="top_k (retrieve)",
                initial=cfg.search.top_k,
                min=2,
                max=20,
                step=1,
            ),
            Slider(
                id="top_n",
                label="top_n (rerank)",
                initial=cfg.rerank.top_n,
                min=1,
                max=10,
                step=1,
            ),
            Switch(
                id="use_rerank",
                label="Use cross-encoder rerank",
                initial=True,
            ),
            Select(
                id="doc_type",
                label="Document type filter",
                values=DOC_TYPE_OPTIONS,
                initial_index=0,
            ),
        ]
    ).send()

    cl.user_session.set("top_k", settings["top_k"])
    cl.user_session.set("top_n", settings["top_n"])
    cl.user_session.set("use_rerank", settings["use_rerank"])
    cl.user_session.set("doc_type", settings["doc_type"])

    # Detect key presence without requiring it -- import lazily so a missing
    # key (or missing .env) never breaks module import.
    has_key = False
    try:
        from src.config import require_openrouter_key

        require_openrouter_key()
        has_key = True
    except Exception:
        has_key = False
    cl.user_session.set("has_key", has_key)

    key_note = (
        "**OPENROUTER_API_KEY detected** -- answers will be LLM-generated "
        "(grounded + cited) via OpenRouter."
        if has_key
        else "**No OPENROUTER_API_KEY found** -- running in retrieved-context-only "
        "mode. You'll see the retrieved/reranked chunks but no LLM-generated "
        "answer. Set `OPENROUTER_API_KEY` in `.env` to enable generation."
    )

    await cl.Message(content=WELCOME_MD + "\n---\n" + key_note).send()

    # Models begin loading at server boot (see _bg_warm). Here we just wait for
    # that to finish so the user's first query isn't blocked mid-request.
    if not _MODELS_READY.is_set():
        warm = cl.Message(
            content="⏳ Loading local models (embedder + reranker)… first load can "
            "take a couple of minutes on CPU. Subsequent queries are fast."
        )
        await warm.send()
        await cl.make_async(_MODELS_READY.wait)()
        warm.content = "✅ Models ready. Ask a question, or type `/inspect`."
        await warm.update()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    for key in ("top_k", "top_n", "use_rerank", "doc_type"):
        if key in settings:
            cl.user_session.set(key, settings[key])


async def _handle_inspect() -> None:
    from src.store.chroma_client import collection_stats
    from src.store.inspect import embedding_projection

    stats = await cl.make_async(collection_stats)()

    lines = [
        "### Vector store inspection",
        f"- **Collection:** `{stats['collection']}`",
        f"- **Total chunks:** {stats['total']}",
    ]
    if stats["by_doc_type"]:
        lines.append("- **By doc_type:**")
        for doc_type, count in sorted(stats["by_doc_type"].items()):
            lines.append(f"  - `{doc_type}`: {count}")
    else:
        lines.append("- (collection is empty -- run ingestion first)")

    await cl.Message(content="\n".join(lines)).send()

    try:
        html_path = await cl.make_async(embedding_projection)()
        png_path = html_path.with_suffix(".png")
        elements = []
        if png_path.exists():
            elements.append(
                cl.Image(name="embedding_scatter.png", path=str(png_path), display="inline")
            )
        await cl.Message(
            content=(
                "2D embedding projection (colored by doc_type). "
                f"Interactive version: `{html_path}`"
            ),
            elements=elements,
        ).send()
    except Exception as e:
        await cl.Message(content=f"Could not generate embedding projection: `{e}`").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = (message.content or "").strip()

    if text.startswith("/inspect"):
        await _handle_inspect()
        return

    if text.startswith("/help"):
        await cl.Message(content=HELP_MD).send()
        return

    if not text:
        await cl.Message(content="Please type a question, or `/help` for commands.").send()
        return

    try:
        top_k = cl.user_session.get("top_k")
        top_n = cl.user_session.get("top_n")
        use_rerank = cl.user_session.get("use_rerank")
        doc_type_raw = cl.user_session.get("doc_type")
        has_key = cl.user_session.get("has_key")
        doc_type = _doc_type_for_query(doc_type_raw)

        top_k = int(top_k) if top_k is not None else None
        top_n = int(top_n) if top_n is not None else None
        use_rerank = True if use_rerank is None else bool(use_rerank)

        from src.rerank.cross_encoder_rerank import search_and_rerank
        from src.generate.answer_synthesis import synthesize_answer, answer_only_mode

        async with cl.Step(name="Retrieve", type="retrieval") as retrieve_step:
            retrieve_step.input = text
            result = await cl.make_async(search_and_rerank)(
                text,
                top_k=top_k,
                top_n=top_n,
                doc_type=doc_type,
                use_rerank=use_rerank,
            )
            retrieved = result["retrieved"]
            retrieve_step.output = f"Retrieved {len(retrieved)} candidates."

        reranked = result["reranked"]

        async with cl.Step(name="Rerank", type="rerank") as rerank_step:
            rerank_step.output = (
                f"Reranked down to {len(reranked)} results "
                f"({'cross-encoder ON' if result['used_rerank'] else 'rerank OFF, order unchanged'})."
            )

        # --- (a) Retrieved (before rerank) -----------------------------------
        retrieved_lines = [
            "## 🔎 Retrieved (before rerank)",
            f"_top_k={top_k or get_config().search.top_k}, doc_type={doc_type_raw}_",
            "",
        ]
        preview_n = len(reranked) if reranked else len(retrieved)
        for rank, r in enumerate(retrieved[: max(preview_n, 1)], start=1):
            retrieved_lines.append(_fmt_retrieved_line(rank, r))
            retrieved_lines.append("")
        await cl.Message(content="\n".join(retrieved_lines)).send()

        # --- (b) Reranked (after) --------------------------------------------
        reranked_lines = [
            "## 🎯 Reranked (after)",
            f"_top_n={top_n or get_config().rerank.top_n}, use_rerank={use_rerank}_",
            "",
        ]
        for r in reranked:
            reranked_lines.append(_fmt_reranked_line(r))
            reranked_lines.append("")

        old_top_id = retrieved[0].get("id") if retrieved else None
        new_top_id = reranked[0].get("id") if reranked else None
        if old_top_id is not None and new_top_id is not None:
            if old_top_id == new_top_id:
                reranked_lines.append("_Top-1 unchanged after reranking._")
            else:
                old_meta = (retrieved[0].get("metadata") or {}) if retrieved else {}
                new_meta = (reranked[0].get("metadata") or {}) if reranked else {}
                reranked_lines.append(
                    "**Top-1 changed after reranking:** "
                    f"`{old_meta.get('source_file', '?')}` -> `{new_meta.get('source_file', '?')}`"
                )
        await cl.Message(content="\n".join(reranked_lines)).send()

        # --- (c) Generated answer --------------------------------------------
        async with cl.Step(name="Generate", type="llm") as gen_step:
            if has_key:
                gen_step.input = "synthesize_answer (LLM call via OpenRouter)"
                try:
                    answer = await cl.make_async(synthesize_answer)(text, reranked)
                    banner = "## 🧠 Generated answer\n_LLM-synthesized, grounded in the reranked context above._"
                except Exception as e:
                    answer = await cl.make_async(answer_only_mode)(text, reranked)
                    banner = (
                        "## 🧠 Generated answer\n"
                        f"_LLM call failed ({e}); falling back to retrieved-context-only mode._"
                    )
            else:
                gen_step.input = "answer_only_mode (no API key, no LLM call)"
                answer = await cl.make_async(answer_only_mode)(text, reranked)
                banner = (
                    "## 🧠 Generated answer\n"
                    "_No OPENROUTER_API_KEY set -- **retrieved-context-only mode**, "
                    "no LLM call was made._"
                )
            gen_step.output = "done"

        await cl.Message(content=banner + "\n\n---\n\n" + answer).send()

    except Exception as e:
        await cl.Message(content=f"⚠️ Something went wrong handling your request:\n\n```\n{e}\n```").send()
