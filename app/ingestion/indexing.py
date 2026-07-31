"""Generate OpenAI/BM25 vectors and store document chunks in Qdrant."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import PointStruct, SparseVector

from app.config import DENSE_VECTOR_NAME, DENSE_VECTOR_SIZE, SPARSE_VECTOR_NAME
from app.embeddings import get_dense_embeddings, get_sparse_embeddings
from app.ingestion.chunking import ChunkedSegment, chunk_segments
from app.parsers import SUPPORTED_EXTENSIONS, parse_document
from app.retrieval.collection import QDRANT_COLLECTION, get_client, ensure_collection


def _make_point_id(chunk: ChunkedSegment, document_chat_id: str | None = None) -> str:
    """Create a stable ID scoped to an optional document chat."""
    source = f"{document_chat_id or ''}:{chunk.metadata['file_path']}:{chunk.metadata['source_segment']}:{chunk.metadata['chunk_index']}"
    return str(uuid5(NAMESPACE_URL, source))


ProgressCallback = Callable[[str, int, int], None]


def index_chunks(
    chunks: list[ChunkedSegment],
    batch_size: int = 16,
    progress_callback: ProgressCallback | None = None,
    document_chat_id: str | None = None,
    document_chat_name: str | None = None,
) -> int:
    """Create dense and sparse embeddings, then upsert them into Qdrant."""
    if not chunks:
        return 0

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    if progress_callback:
        progress_callback("Embedding chunks…", 0, total_batches)
    client = get_client()

    for start in range(0, len(chunks), batch_size):
        batch_number = start // batch_size + 1
        if progress_callback:
            progress_callback(f"Embedding and saving batch {batch_number} of {total_batches}…", batch_number - 1, total_batches)
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        dense_vectors = get_dense_embeddings(texts)
        sparse_vectors = get_sparse_embeddings(texts)

        points: list[PointStruct] = []
        for chunk, dense_vector, sparse_vector in zip(batch, dense_vectors, sparse_vectors, strict=True):
            if len(dense_vector) != DENSE_VECTOR_SIZE:
                raise ValueError(
                    f"Embedding API returned {len(dense_vector)} dimensions; expected {DENSE_VECTOR_SIZE}."
                )
            payload = {"text": chunk.text, **chunk.metadata}
            if document_chat_id:
                payload["document_chat_id"] = document_chat_id
                payload["document_chat_name"] = document_chat_name or "Untitled chat"
            points.append(
                PointStruct(
                    id=_make_point_id(chunk, document_chat_id),
                    vector={
                        DENSE_VECTOR_NAME: dense_vector,
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )
        client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        if progress_callback:
            progress_callback(f"Saved batch {batch_number} of {total_batches}", batch_number, total_batches)
    return len(chunks)


def ingest_files(
    file_paths: Iterable[str | Path],
    progress_callback: ProgressCallback | None = None,
    document_chat_id: str | None = None,
    document_chat_name: str | None = None,
) -> tuple[int, int, list[str]]:
    """Parse and index several files together using one embedding batch.

    Returns the number of parsed segments, the number of chunks indexed, and
    the names of any files from which no text could be extracted.
    """
    paths = [Path(file_path) for file_path in file_paths]
    if not paths:
        return 0, 0, []

    ensure_collection()
    parsed_segments = []
    skipped: list[str] = []
    for number, path in enumerate(paths, start=1):
        if progress_callback:
            progress_callback(f"Parsing {path.name} ({number} of {len(paths)})…", number - 1, len(paths))
        before = len(parsed_segments)
        parsed_segments.extend(parse_document(path))
        if len(parsed_segments) == before:
            skipped.append(path.name)
    if progress_callback:
        progress_callback(f"Chunking {len(parsed_segments)} parsed segment(s)…", len(paths), len(paths))
    chunks = chunk_segments(parsed_segments)
    if progress_callback:
        progress_callback(f"Created {len(chunks)} chunk(s). Preparing embeddings…", 0, 1)
    indexed = index_chunks(
        chunks,
        progress_callback=progress_callback,
        document_chat_id=document_chat_id,
        document_chat_name=document_chat_name,
    )
    return len(parsed_segments), indexed, skipped


def ingest_file(file_path: str | Path) -> tuple[int, int, list[str]]:
    """Parse, chunk, embed, and index one supported file."""
    return ingest_files([file_path])


def ingest_folder(
    folder_path: str | Path,
    progress_callback: ProgressCallback | None = None,
    document_chat_id: str | None = None,
    document_chat_name: str | None = None,
) -> tuple[int, int, int, list[str]]:
    """Index all supported files in a folder and its subfolders.

    Returns the number of files, source segments, chunks indexed, and the
    names of any files from which no text could be extracted.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")

    files = sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"No supported files found. Supported types: {supported}")

    segments, chunks, skipped = ingest_files(
        files,
        progress_callback=progress_callback,
        document_chat_id=document_chat_id,
        document_chat_name=document_chat_name,
    )
    return len(files), segments, chunks, skipped
