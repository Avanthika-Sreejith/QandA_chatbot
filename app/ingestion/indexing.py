"""Generate OpenAI/BM25 vectors and store document chunks in Qdrant."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct, SparseVector

from app.config import DENSE_VECTOR_NAME, DENSE_VECTOR_SIZE, SPARSE_VECTOR_NAME
from app.embeddings import get_dense_embeddings, get_sparse_embeddings
from app.ingestion.chunking import ChunkedSegment, chunk_segments
from app.parsers import SUPPORTED_EXTENSIONS, parse_document
from app.retrieval.collection import QDRANT_COLLECTION, get_client, ensure_collection
from app.retrieval.structured import assess_document_structure, build_section_tree
from app.database import upsert_structured_document_index, delete_structured_document_index_by_name


def _make_point_id(chunk: ChunkedSegment, document_chat_id: str | None = None) -> str:
    """Create a stable ID scoped to an optional document chat."""
    source = f"{document_chat_id or ''}:{chunk.metadata['file_path']}:{chunk.metadata['source_segment']}:{chunk.metadata['chunk_index']}"
    return str(uuid5(NAMESPACE_URL, source))


ProgressCallback = Callable[[str, int, int], None]

_DEFAULT_SECTION = "Document body"


def _delete_stale_points(file_name: str, document_chat_id: str) -> None:
    """Remove a previously indexed version of a file before re-ingesting it.

    Uploads are saved under a fresh random path each time, so the point IDs for
    an older upload of the same file never collide with the new ones. Without
    this cleanup, re-uploading a document would leave its old chunks searchable
    alongside the new ones.
    """
    if not document_chat_id:
        return
    try:
        if not get_client().collection_exists(QDRANT_COLLECTION):
            return
        get_client().delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="document_chat_id", match=MatchValue(value=document_chat_id)),
                        FieldCondition(key="file_name", match=MatchValue(value=file_name)),
                    ]
                )
            ),
            wait=True,
        )
    except Exception:
        # A legacy deployment without the payload index should never block ingest.
        return


def _embedding_text(chunk: ChunkedSegment) -> str:
    """Return the text used for embeddings, prefixed with its section heading.

    Contextualising each chunk with its section makes it match queries about
    the section's topic, instead of only the bare body text. The stored text
    stays clean; only the embedding input carries the prefix.
    """
    section = (chunk.metadata.get("section") or "").strip()
    if section and section != _DEFAULT_SECTION and section != chunk.text:
        return f"{section}\n{chunk.text}"
    return chunk.text


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
        embedding_texts = [_embedding_text(chunk) for chunk in batch]
        dense_vectors = get_dense_embeddings(embedding_texts)
        sparse_vectors = get_sparse_embeddings(embedding_texts)

        points: list[PointStruct] = []
        for chunk, dense_vector, sparse_vector in zip(batch, dense_vectors, sparse_vectors, strict=True):
            if len(dense_vector) != DENSE_VECTOR_SIZE:
                raise ValueError(
                    f"Embedding API returned {len(dense_vector)} dimensions; expected {DENSE_VECTOR_SIZE}."
                )
            payload = {"text": chunk.text, "retrieval_mode": "hybrid_vector", **chunk.metadata}
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

    parsed_segments = []
    skipped: list[str] = []
    structured_units = 0
    for number, path in enumerate(paths, start=1):
        if progress_callback:
            progress_callback(f"Parsing {path.name} ({number} of {len(paths)})…", number - 1, len(paths))
        # Re-upload cleanup first: any older hybrid chunks or structured tree
        # for the same file name in this chat are stale once we re-ingest.
        delete_structured_document_index_by_name(document_chat_id, path.name)
        _delete_stale_points(path.name, document_chat_id)
        file_segments = parse_document(path)
        if not file_segments:
            skipped.append(path.name)
            continue

        assessment = assess_document_structure(path, file_segments)
        if assessment.is_structured and document_chat_id:
            tree = build_section_tree(path, file_segments, assessment)
            try:
                upsert_structured_document_index(
                    document_chat_id,
                    str(path.resolve()),
                    path.name,
                    assessment.score,
                    tree,
                )
                structured_units += len(tree["nodes"])
                if progress_callback:
                    progress_callback(
                        f"Built vectorless section tree for {path.name} ({assessment.score}/structure score)",
                        number,
                        len(paths),
                    )
                continue
            except Exception:
                # An older Supabase schema should never stop the working
                # hybrid pipeline; it is a safe fallback until migration.
                pass
        parsed_segments.extend(file_segments)
    if progress_callback:
        progress_callback(f"Chunking {len(parsed_segments)} parsed segment(s)…", len(paths), len(paths))
    chunks = chunk_segments(parsed_segments)
    if progress_callback:
        progress_callback(f"Created {len(chunks)} chunk(s). Preparing embeddings…", 0, 1)
    if chunks:
        ensure_collection()
    indexed = index_chunks(
        chunks,
        progress_callback=progress_callback,
        document_chat_id=document_chat_id,
        document_chat_name=document_chat_name,
    )
    return len(parsed_segments), indexed + structured_units, skipped


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
