"""
OpenRouter client wrapper for the Cascade Bank local-first RAG teaching demo.

OpenRouter exposes an OpenAI-compatible chat completions API, so we reuse the
`openai` v1 SDK and just point it at OpenRouter's base_url with an OpenRouter
API key.

Import-safety note: this module must be importable even when OPENROUTER_API_KEY
is not set (e.g. during offline/answer-only-mode demos or plain `import` for
type-checking). `require_openrouter_key()` is therefore only called lazily,
inside `get_client()`, never at module import time.
"""

from __future__ import annotations

from typing import Any, Iterable

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_config, require_openrouter_key

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """
    Build (and cache) an OpenAI SDK client pointed at OpenRouter.

    Reads `cfg.generation.base_url` for the endpoint and requires
    OPENROUTER_API_KEY (via `require_openrouter_key()`) for auth. The key is
    only fetched here -- not at import time -- so this module stays
    import-safe without a key present.
    """
    global _client
    if _client is None:
        cfg = get_config()
        _client = OpenAI(
            base_url=cfg.generation.base_url,
            api_key=require_openrouter_key(),
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Cascade Bank RAG Demo",
            },
        )
    return _client


_RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, InternalServerError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def _create_completion(client: OpenAI, **kwargs: Any):
    return client.chat.completions.create(**kwargs)


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
) -> str | Iterable:
    """
    Call OpenRouter's chat completions endpoint, falling back to
    `cfg.generation.*` for any unset arguments.

    Retries transient errors (connection issues, rate limits, 5xx) up to 4
    attempts total with exponential backoff (via tenacity).

    Returns:
        - stream=False: the full response text (str), i.e.
          `resp.choices[0].message.content`.
        - stream=True: the raw streaming iterator from the SDK -- the caller
          is responsible for iterating chunks (each chunk exposes
          `.choices[0].delta.content`).
    """
    cfg = get_config()
    client = get_client()

    resolved_model = model or cfg.generation.model
    resolved_temperature = cfg.generation.temperature if temperature is None else temperature
    resolved_max_tokens = cfg.generation.max_tokens if max_tokens is None else max_tokens

    if stream:
        # Streaming responses are handed back raw; retrying a stream after
        # it has started yielding chunks isn't safe/meaningful, so we only
        # retry the initial request-creation call.
        return _create_completion(
            client,
            model=resolved_model,
            messages=messages,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            stream=True,
        )

    resp = _create_completion(
        client,
        model=resolved_model,
        messages=messages,
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        stream=False,
    )
    return resp.choices[0].message.content
