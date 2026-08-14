"""Grounded answer generation using Groq LLM."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure .env is loaded regardless of import order
load_dotenv()

_SYSTEM_PROMPT = """\
You are a helpful document assistant. Answer the user's question based ONLY on the
provided context passages. Be concise and accurate. If the context does not contain
enough information to answer, say so clearly.

Add an inline citation such as [1] or [2] after every factual statement or bullet.
The number must match the supplied context passage number exactly. Never cite a
passage merely because it discusses the same topic: every cited definition, formula,
or claim must appear in that exact numbered passage.

If the question asks for key concepts, definitions, terms, formulas, or a list, give
exactly the items labelled in the passage that defines them (for example a section
titled "Key concepts"). Match that list one-to-one: one bullet per labelled item, in
the same order, with nothing added and nothing dropped. Never pad the list with rules,
steps, formulas, or revision-summary bullets from other passages, even when those
passages are about the same topic and appear in the context.

At the very end of your answer, on its own line, report exactly which context passage
numbers you used, in this exact format:

CITATIONS: [1, 3]

If you did not use any passage, write:

CITATIONS: []
"""

_CITATION_PATTERN = re.compile(r"CITATIONS:\s*\[([0-9,\s]*)\]", re.IGNORECASE)


def _split_citations(content: str) -> tuple[str, list[int]]:
    """Separate the answer text from the passage numbers it cited.

    Returns the cleaned answer and a sorted list of 1-based passage numbers.
    Falls back to showing no citations when the model omits the line.
    """
    matches = list(_CITATION_PATTERN.finditer(content))
    if not matches:
        return content.strip(), []
    last = matches[-1]
    answer = content[: last.start()].strip()
    raw = last.group(1).strip()
    numbers = [int(token) for token in re.split(r"[,\s]+", raw) if token.strip().isdigit()]
    return answer, sorted(set(numbers))


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
) -> tuple[str, list[int]]:
    """
    Synthesise a chatbot-style answer from retrieved passages using Groq.

    Returns the answer text and the 1-based indices of the passages the answer
    actually used. Falls back to a formatted summary when GROQ_API_KEY is not set.
    """
    if not results:
        return (
            "I could not find any relevant passages in the indexed documents to answer your question.",
            [],
        )

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    context = _build_context_block(results)

    if api_key:
        return _groq_answer(question, context, api_key)
    else:
        return _fallback_answer(question, context), []


def _groq_answer(question: str, context: str, api_key: str) -> tuple[str, list[int]]:
    """Call Groq chat completion and return the answer and cited passage numbers."""
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
        max_tokens=2048,
    )
    content = response.choices[0].message.content or ""
    return _split_citations(content)


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
