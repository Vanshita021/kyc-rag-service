"""Thin wrapper around Gemini API (generation + embeddings) with retries."""

import time
from google import genai
from google.genai import types
from app import config

_client = None


def client() -> genai.Client:
    """Initialize Gemini Developer API client using API key."""
    global _client

    if _client is None:
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is missing. Please configure it in your .env file."
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)

    return _client


def _retry(fn, attempts=5):
    """Retry API requests. Per-minute limits: wait and retry. Per-day limits: fail fast."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg or i == attempts - 1:
                raise  # a daily quota can't be fixed by waiting a few seconds
            limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            time.sleep(40 if limited else 2 ** i)


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    """
    Generate embeddings for text.
    task_type: RETRIEVAL_DOCUMENT for document chunks, RETRIEVAL_QUERY for questions.
    """
    resp = _retry(
        lambda: client().models.embed_content(
            model=config.EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
    )
    return [e.values for e in resp.embeddings]


def generate(prompt: str) -> dict:
    """Generate a Gemini response with token usage, latency and retry count."""
    state = {"attempts": 0, "seconds": 0.0}

    def call():
        state["attempts"] += 1
        t = time.perf_counter()
        r = client().models.generate_content(
            model=config.GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        state["seconds"] = time.perf_counter() - t  # successful attempt only
        return r

    resp = _retry(call)
    u = resp.usage_metadata

    in_tok = (u.prompt_token_count or 0) if u else 0
    # Thinking tokens are billed as output tokens.
    out_tok = (
        (u.candidates_token_count or 0)
        + (getattr(u, "thoughts_token_count", 0) or 0)
    ) if u else 0

    return {
        "text": resp.text or "",
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "gen_seconds": state["seconds"],
        "retries": state["attempts"] - 1,
    }