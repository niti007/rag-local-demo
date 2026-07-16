"""
Answer synthesis for the Cascade Bank local-first RAG teaching demo.

Takes reranked retrieval chunks (as produced by
`src.rerank.cross_encoder_rerank.search_and_rerank`) and either:
  - synthesizes a grounded, cited answer via OpenRouter (`synthesize_answer`), or
  - falls back to a no-API "answer-only" mode that just echoes the retrieved
    context with citations, for demos/offline use without a key
    (`answer_only_mode`).

Import-safety note: this module does not require OPENROUTER_API_KEY at import
time -- the key is only needed if/when `synthesize_answer` is actually called.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

from src.generate.openrouter_client import chat_completion

SYSTEM_PROMPT = """\
You are a retrieval-grounded assistant for Cascade Bank, a fictional demo \
bank used for a local RAG (retrieval-augmented generation) teaching demo. \
All source documents are synthetic, fabricated data created for this demo -- \
none of it describes a real institution, real customers, or real financial \
information.

Answer the user's question using ONLY the information in the numbered context \
block provided below the question. Do not use outside knowledge and do not \
guess or invent facts that are not present in the context.

When you use information from the context, cite it inline using its bracketed \
index, e.g. [1] or [2][3], right after the claim it supports.

If the provided context does not contain enough information to answer the \
question, say so plainly (e.g. "The provided context does not contain this \
information.") rather than speculating or answering from general knowledge.
"""


def build_context(chunks: list[dict]) -> str:
    """
    Format reranked chunks into a numbered context block for citations, e.g.:

        [1] (policy_manual.pdf, sop, page 3)
        <chunk text>

        [2] (transactions.csv, csv, page N/A)
        <chunk text>

    Each chunk dict is expected to have "text" and "metadata" (with
    source_file / doc_type / page or page_number) keys, as produced by
    semantic_search / cross_encoder_rerank. Missing metadata fields degrade
    gracefully rather than raising.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata") or {}
        source_file = meta.get("source_file", "unknown source")
        doc_type = meta.get("doc_type", "unknown type")
        page = meta.get("page_number", meta.get("page", "N/A"))
        text = (chunk.get("text") or "").strip()

        lines.append(f"[{i}] ({source_file}, {doc_type}, page {page})")
        lines.append(text)
        lines.append("")  # blank line between entries

    return "\n".join(lines).strip() + "\n"


def synthesize_answer(query: str, chunks: list[dict], stream: bool = False) -> str | Iterable:
    """
    Build a context block from `chunks`, prompt the OpenRouter model to answer
    `query` grounded strictly in that context (with [n] citations), and
    return the answer text (or a raw streaming iterator if stream=True).
    """
    context_block = build_context(chunks)

    user_message = (
        f"Context:\n\n{context_block}\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above, citing sources by "
        "their [n] index."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    return chat_completion(messages, stream=stream)


def answer_only_mode(query: str, chunks: list[dict]) -> str:
    """
    No-API fallback: just returns the concatenated top chunks with citations,
    and a note that no LLM call was made. Useful when OPENROUTER_API_KEY is
    not set, for offline demos, or for comparing "raw retrieval" against a
    synthesized answer.
    """
    context_block = build_context(chunks)
    return (
        f"Question: {query}\n\n"
        f"Retrieved context (retrieved-context-only mode -- no LLM call):\n\n"
        f"{context_block}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve, rerank, and synthesize a grounded answer (or fall back to answer-only mode)."
    )
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--k", type=int, default=None, help="Semantic search top_k (default: cfg.search.top_k)")
    parser.add_argument("--n", type=int, default=None, help="Rerank top_n (default: cfg.rerank.top_n)")
    parser.add_argument("--doc-type", default=None, help="Restrict to a specific doc_type metadata value")
    args = parser.parse_args()

    from src.rerank.cross_encoder_rerank import search_and_rerank

    result = search_and_rerank(args.query, top_k=args.k, top_n=args.n, doc_type=args.doc_type)
    chunks = result["reranked"]

    context_block = build_context(chunks)
    print("=== CONTEXT ===")
    print(context_block)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("(No OPENROUTER_API_KEY set -- falling back to answer-only mode, no LLM call made.)")
        print()
        print("=== ANSWER (retrieved-context-only mode) ===")
        print(answer_only_mode(args.query, chunks))
        return

    print("=== ANSWER ===")
    try:
        answer = synthesize_answer(args.query, chunks)
        print(answer)
    except RuntimeError as e:
        print(f"(Falling back to answer-only mode: {e})")
        print(answer_only_mode(args.query, chunks))


if __name__ == "__main__":
    main()
