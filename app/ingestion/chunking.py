"""Create topic-coherent chunks while preserving source metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable

from app.config import (
    SEMANTIC_MAX_CHARACTERS,
    SEMANTIC_MIN_SENTENCES,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from app.embeddings import get_dense_embeddings
from app.parsers import ParsedSegment


@dataclass(frozen=True)
class ChunkedSegment:
    """A retrieval-sized unit with the source information required for citations."""

    text: str
    metadata: dict[str, Any]


# Split after sentence-ending punctuation even when a citation tag such as
# "[web:52]" sits between the period and the next word. Otherwise an entire
# definition list plus its numbered steps becomes one un-splittable sentence.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[.!?])\[web:\d+\]\s*"
    r"|(?<=\n)\s*(?=\S)"
)

# A numbered list item ("1. Start with all activities...") always starts a new
# chunk. Without this, semantic chunking glues numbered procedure steps onto
# the preceding definition list (all same topic), diluting the definition
# chunk's embedding and pushing it down the retrieval ranking.
_NUMBERED_ITEM_START = re.compile(r"^\d+\.\s+\S")

_PAGE_FOOTER = re.compile(r"(?i)^(?:page\s+)?\d+\s*(?:[-–/of]*\s*\d+)?\.?\s*$")
_COURSE_OUTCOME_TAG = re.compile(r"^[cC][oO]?\d+\s*\(\s*\d+\s*\)\s*$")
_DIAGRAM_LABELS = {"START", "END", "STOP"}


def _is_junk(text: str) -> bool:
    """Return True when a chunk is pure extraction noise, not document content.

    Conservative rules: symbol-only diagram borders, page-number footers,
    course-outcome grading tags, and single-token flow-diagram labels. These
    never carry retrievable meaning on their own, so dropping them keeps noise
    out of the index without risking real content.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if not any(char.isalnum() for char in stripped):
        return True
    if len(stripped) <= 12 and _PAGE_FOOTER.fullmatch(stripped):
        return True
    if _COURSE_OUTCOME_TAG.fullmatch(stripped):
        return True
    if stripped in _DIAGRAM_LABELS and not any(char in stripped for char in ".!?;:,"):
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    """Split text into useful semantic candidates without discarding content."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    return [sum(values) / len(vectors) for values in zip(*vectors, strict=True)]


def _embed_candidates(sentences: list[str]) -> list[list[float]]:
    """Bound embedding API request sizes for long documents."""
    vectors: list[list[float]] = []
    for start in range(0, len(sentences), 96):
        vectors.extend(get_dense_embeddings(sentences[start : start + 96]))
    return vectors


def _semantic_chunks(text: str) -> list[str]:
    """Split where sentence meaning changes, with a maximum-size safety guard."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    sentence_vectors = _embed_candidates(sentences)
    chunks: list[str] = []
    current_sentences: list[str] = [sentences[0]]
    current_vectors: list[list[float]] = [sentence_vectors[0]]

    for sentence, vector in zip(sentences[1:], sentence_vectors[1:], strict=True):
        candidate_length = len(" ".join(current_sentences)) + len(sentence) + 1
        similarity = _cosine(_mean_vector(current_vectors), vector)
        hard_split = (
            len(current_sentences) > 0 and bool(_NUMBERED_ITEM_START.match(sentence))
        )
        should_split = (
            len(current_sentences) >= SEMANTIC_MIN_SENTENCES
            and similarity < SEMANTIC_SIMILARITY_THRESHOLD
        )
        must_split_for_size = candidate_length > SEMANTIC_MAX_CHARACTERS

        if hard_split or should_split or must_split_for_size:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_vectors = [vector]
        else:
            current_sentences.append(sentence)
            current_vectors.append(vector)

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def chunk_segments(segments: Iterable[ParsedSegment]) -> list[ChunkedSegment]:
    """Create semantic chunks for prose and preserve table row groups as units.

    Headings are included with their following section text, not independently
    indexed. This prevents a following heading from being returned as an item
    in the preceding list or procedure.
    """
    output: list[ChunkedSegment] = []
    for source_index, segment in enumerate(segments):
        content_kind = segment.metadata.get("content_kind", "prose")
        if content_kind == "heading":
            continue
        if content_kind in ("table", "image"):
            chunks = [segment.text]
            method = content_kind
        else:
            text = segment.text
            section = str(segment.metadata.get("section") or "").strip()
            if section and section != "Document body" and not text.startswith(section):
                text = f"{section}\n{text}"
            chunks = _semantic_chunks(text)
            method = "semantic"

        for chunk_index, text in enumerate(chunks):
            if _is_junk(text):
                continue
            metadata = dict(segment.metadata)
            metadata.update(
                {
                    "source_segment": source_index,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "chunk_characters": len(text),
                    "chunking_method": method,
                }
            )
            output.append(ChunkedSegment(text=text, metadata=metadata))
    return output
