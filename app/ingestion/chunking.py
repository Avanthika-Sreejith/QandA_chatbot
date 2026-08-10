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

# A glossary line such as "Normal time: The standard duration..." A run of
# these (e.g. the five crash-cost definitions) must stay in ONE chunk, or a
# query asking for "key concepts" retrieves only half of the list.
_DEFINITION_ITEM = re.compile(r"^[A-Z][A-Za-z ]+:\s")

_PAGE_FOOTER = re.compile(r"(?i)^(?:page\s+)?\d+\s*(?:[-–/of]*\s*\d+)?\.?\s*$")
_COURSE_OUTCOME_TAG = re.compile(r"^[cC][oO]?\d+\s*\(\s*\d+\s*\)\s*$")
_DIAGRAM_LABELS = {"START", "END", "STOP"}
# A chunk that only contains citation tags such as "[web:52]" carries no
# retrievable content but still pollutes dense/BM25 rankings ("web", "60").
_CITATION_ONLY = re.compile(r"^(\[\s*[wW][eE][bB]\s*:\s*\d+\s*\]\s*)+$")


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
    if _CITATION_ONLY.fullmatch(stripped):
        return True
    if _COURSE_OUTCOME_TAG.fullmatch(stripped):
        return True
    if stripped in _DIAGRAM_LABELS and not any(char in stripped for char in ".!?;:,"):
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    """Split text into useful semantic candidates without discarding content."""
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    sentences: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if i + 1 < len(parts) and re.fullmatch(r"\d+\.", part):
            # "1." was cut from "1. Start with..." by the period rule; re-attach
            # it so the numbered-item hard split sees a complete marker.
            sentences.append(f"{part} {parts[i + 1]}")
            i += 2
        elif _CITATION_ONLY.fullmatch(part):
            # A trailing "[web:53]" carries no meaning on its own.
            i += 1
        else:
            sentences.append(part)
            i += 1
    return sentences


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
        # Two consecutive glossary lines share one topic; forcing a split
        # between "Normal time: ..." and "Crash time: ..." breaks the list so
        # retrieval only ever surfaces half of it.
        hard_join = bool(_DEFINITION_ITEM.match(sentence)) and bool(
            _DEFINITION_ITEM.match(current_sentences[-1])
        )
        should_split = (
            len(current_sentences) >= SEMANTIC_MIN_SENTENCES
            and similarity < SEMANTIC_SIMILARITY_THRESHOLD
            and not hard_join
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
