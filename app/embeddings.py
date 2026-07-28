"""Shared embedding model loading helpers for ingestion and retrieval."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from threading import Lock

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from app.config import (
    DENSE_EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    SPARSE_EMBEDDING_MODEL,
)


@lru_cache(maxsize=1)
def _load_embedding_models() -> tuple[SentenceTransformer, SparseTextEmbedding]:
    dense_model = SentenceTransformer(DENSE_EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL)

    dense_model.encode(["warm up"], normalize_embeddings=True, show_progress_bar=False)
    list(sparse_model.embed(["warm up"]))
    return dense_model, sparse_model


_model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedding-model-loader")
_model_future: Future[tuple[SentenceTransformer, SparseTextEmbedding]] | None = None
_model_future_lock = Lock()


def start_model_preload() -> Future[tuple[SentenceTransformer, SparseTextEmbedding]]:
    global _model_future
    with _model_future_lock:
        if _model_future is None:
            _model_future = _model_executor.submit(_load_embedding_models)
        return _model_future


def get_embedding_models() -> tuple[SentenceTransformer, SparseTextEmbedding]:
    return start_model_preload().result()
