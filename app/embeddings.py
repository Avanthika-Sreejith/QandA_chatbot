"""Dense embeddings via OpenAI API, sparse embeddings via fastembed."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding
from openai import OpenAI

from app.config import DENSE_VECTOR_SIZE, OPENAI_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL


def get_dense_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI's embedding API."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
        dimensions=DENSE_VECTOR_SIZE,
    )
    return [item.embedding for item in response.data]


@lru_cache(maxsize=1)
def _get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL)


def get_sparse_embeddings(texts: list[str]) -> list[Any]:
    """Embed a list of texts using fastembed's BM25 model."""
    model = _get_sparse_model()
    return list(model.embed(texts))
