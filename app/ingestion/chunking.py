"""Split parsed document segments into retrieval-sized chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from chonkie import RecursiveChunker

from app.parsers import ParsedSegment


DEFAULT_CHUNK_SIZE = 800


@dataclass(frozen=True)
class ChunkedSegment:
    """A retrieval-sized text chunk with its original source metadata."""

    text: str
    metadata: dict[str, Any]


def chunk_segments(
    segments: Iterable[ParsedSegment],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[ChunkedSegment]:
    """Chunk parsed segments at natural boundaries while retaining citations.

    ``RecursiveChunker`` attempts the largest meaningful delimiter first
    (such as paragraph breaks), then falls back through sentence and word
    boundaries only when a unit exceeds ``chunk_size``. This keeps ideas and
    sections together more reliably than a fixed character window.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    # Chonkie's default recursive rules preserve increasingly smaller natural
    # boundaries (paragraphs, sentences, then words) before splitting by
    # character count as a last resort.
    #
    # RecursiveChunker does not implement window overlap. Keeping each chunk
    # intact is intentional: sentence/paragraph boundaries preserve context
    # without duplicating text in the vector store.
    chunker = RecursiveChunker(
        tokenizer="character",
        chunk_size=chunk_size,
    )
    output: list[ChunkedSegment] = []
    for source_index, segment in enumerate(segments):
        chunks = chunker.chunk(segment.text)
        for chunk_index, chunk in enumerate(chunks):
            metadata = dict(segment.metadata)
            metadata.update(
                {
                    "source_segment": source_index,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "chunk_characters": len(chunk.text),
                }
            )
            output.append(ChunkedSegment(text=chunk.text, metadata=metadata))
    return output
