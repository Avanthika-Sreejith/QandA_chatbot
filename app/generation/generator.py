"""Grounded answer generation using Groq LLM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure .env is loaded regardless of import order
load_dotenv()

_SYSTEM_PROMPT = """\
You are a helpful document assistant. Answer the user's question based ONLY on the
provided context passages. Be concise and accurate. If the context does not contain
enough information to answer, say so clearly.

Do NOT add source citations inside your answer text. They will be appended separately.
"""


def _build_context_block(results: list[dict[str, Any]]) -> str:
    """Format retrieved passages into a numbered context block."""
    parts: list[str] = []
    for i, result in enumerate(results, start=1):
        source_name = Path(result.get("source", "unknown")).name
        text = result.get("text", "").strip()
        parts.append(f"[{i}] {source_name}\n{text}")
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    results: list[dict[str, Any]],
) -> str:
    """
    Synthesise a single chatbot-style answer from retrieved passages using Groq.

    Falls back to a formatted summary when GROQ_API_KEY is not set.
    """
    if not results:
        return "I could not find any relevant passages in the indexed documents to answer your question."

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    context = _build_context_block(results)

    if api_key:
        return _groq_answer(question, context, api_key)
    else:
        return _fallback_answer(question, context)


def _groq_answer(question: str, context: str, api_key: str) -> str:
    """Call Groq chat completion and return the answer text."""
    from groq import Groq  # imported lazily to avoid import-time errors

    client = Groq(api_key=api_key)
    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _fallback_answer(question: str, context: str) -> str:
    """
    Return a best-effort answer without an LLM by stitching together the
    most relevant passage. Used when GROQ_API_KEY is absent.
    """
    first_passage = context.split("\n\n")[0] if context else ""
    return (
        "*(No LLM API key configured — showing the best-matching passage as the answer.)*\n\n"
        + first_passage
    )
