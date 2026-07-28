"""Generate Qwen3/BM25 vectors and store document chunks in Qdrant."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable
from uuid import NAMESPACE_URL, uuid5

from fastembed import SparseTextEmbedding
from qdrant_client.models import PointStruct, SparseVector
from sentence_transformers import SentenceTransformer

from app.config import (
    DENSE_EMBEDDING_MODEL,
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    SPARSE_EMBEDDING_MODEL,
    SPARSE_VECTOR_NAME,
    EMBEDDING_DEVICE,
)
from app.ingestion.chunking import ChunkedSegment, chunk_segments
from app.parsers import SUPPORTED_EXTENSIONS, parse_document
from app.retrieval.collection import QDRANT_COLLECTION, get_client, ensure_collection


def _make_point_id(chunk: ChunkedSegment, document_chat_id: str | None = None) -> str:
    """Create a stable ID scoped to an optional document chat."""
    source = f"{document_chat_id or ''}:{chunk.metadata['file_path']}:{chunk.metadata['source_segment']}:{chunk.metadata['chunk_index']}"
    return str(uuid5(NAMESPACE_URL, source))


@lru_cache(maxsize=1)
def _load_embedding_models() -> tuple[SentenceTransformer, SparseTextEmbedding]:
    """Load and warm embedding models once for the running app session."""
    dense_model = SentenceTransformer(DENSE_EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL)

    # FastEmbed initializes the BM25 model lazily on its first ``embed`` call.
    # Run one tiny inference at startup so the first document upload does not
    # pay the model-initialization cost.
    dense_model.encode(["warm up"], normalize_embeddings=True, show_progress_bar=False)
    list(sparse_model.embed(["warm up"]))
    return dense_model, sparse_model


_model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedding-model-loader")
_model_future: Future[tuple[SentenceTransformer, SparseTextEmbedding]] | None = None
_model_future_lock = Lock()


def start_model_preload() -> Future[tuple[SentenceTransformer, SparseTextEmbedding]]:
    """Begin model warm-up in the background, returning the shared task."""
    global _model_future
    with _model_future_lock:
        if _model_future is None:
            _model_future = _model_executor.submit(_load_embedding_models)
        return _model_future


def get_embedding_models() -> tuple[SentenceTransformer, SparseTextEmbedding]:
    """Return fully warmed models, waiting for the shared preload if needed."""
    return start_model_preload().result()


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
        models_are_cached = _load_embedding_models.cache_info().currsize > 0
        message = "Using ready embedding models…" if models_are_cached else "Loading embedding models…"
        progress_callback(message, 0, total_batches)
    dense_model, sparse_model = get_embedding_models()
    client = get_client()

    for start in range(0, len(chunks), batch_size):
        batch_number = start // batch_size + 1
        if progress_callback:
            progress_callback(f"Embedding and saving batch {batch_number} of {total_batches}…", batch_number - 1, total_batches)
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        dense_vectors = dense_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sparse_vectors = list(sparse_model.embed(texts))

        points: list[PointStruct] = []
        for chunk, dense_vector, sparse_vector in zip(batch, dense_vectors, sparse_vectors, strict=True):
            if len(dense_vector) != DENSE_VECTOR_SIZE:
                raise ValueError(
                    f"Qwen3 returned {len(dense_vector)} dimensions; expected {DENSE_VECTOR_SIZE}."
                )
            payload = {"text": chunk.text, **chunk.metadata}
            if document_chat_id:
                payload["document_chat_id"] = document_chat_id
                payload["document_chat_name"] = document_chat_name or "Untitled chat"
            points.append(
                PointStruct(
                    id=_make_point_id(chunk, document_chat_id),
                    vector={
                        DENSE_VECTOR_NAME: dense_vector.tolist(),
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
) -> tuple[int, int]:
    """Parse and index several files together using one embedding batch."""
    paths = [Path(file_path) for file_path in file_paths]
    if not paths:
        return 0, 0

    ensure_collection()
    parsed_segments = []
    for number, path in enumerate(paths, start=1):
        if progress_callback:
            progress_callback(f"Parsing {path.name} ({number} of {len(paths)})…", number - 1, len(paths))
        parsed_segments.extend(parse_document(path))
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
    return len(parsed_segments), indexed


def ingest_file(file_path: str | Path) -> tuple[int, int]:
    """Parse, chunk, embed, and index one supported file."""
    return ingest_files([file_path])


def ingest_folder(
    folder_path: str | Path,
    progress_callback: ProgressCallback | None = None,
    document_chat_id: str | None = None,
    document_chat_name: str | None = None,
) -> tuple[int, int, int]:
    """Index all supported files in a folder and its subfolders.

    Returns the number of files, source segments, and chunks indexed.
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

    segments, chunks = ingest_files(
        files,
        progress_callback=progress_callback,
        document_chat_id=document_chat_id,
        document_chat_name=document_chat_name,
    )
    return len(files), segments, chunks
