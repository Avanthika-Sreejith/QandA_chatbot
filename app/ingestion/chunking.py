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


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=\n)\s*(?=\S)")


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
        should_split = (
            len(current_sentences) >= SEMANTIC_MIN_SENTENCES
            and similarity < SEMANTIC_SIMILARITY_THRESHOLD
        )
        must_split_for_size = candidate_length > SEMANTIC_MAX_CHARACTERS

        if should_split or must_split_for_size:
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
    """Create semantic chunks for prose and preserve table row groups as units."""
    output: list[ChunkedSegment] = []
    for source_index, segment in enumerate(segments):
        content_kind = segment.metadata.get("content_kind", "prose")
        if content_kind in ("table", "heading"):
            chunks = [segment.text]
            method = content_kind
        else:
            chunks = _semantic_chunks(segment.text)
            method = "semantic"

        for chunk_index, text in enumerate(chunks):
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
