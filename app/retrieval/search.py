"""Query the Qdrant collection using semantic search."""

from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.embeddings import get_dense_embeddings
from app.retrieval.collection import DENSE_VECTOR_NAME, QDRANT_COLLECTION, get_client


def search_documents(
    query: str,
    top_k: int = 5,
    file_paths: list[str] | None = None,
    document_chat_id: str | None = None,
) -> list[Any]:
    """Return the top-k semantic search results from Qdrant."""
    query_vector = get_dense_embeddings([query])[0]

    query_filter = None
    if document_chat_id:
        query_filter = Filter(
            must=[FieldCondition(key="document_chat_id", match=MatchValue(value=document_chat_id))]
        )
    elif file_paths:
        if len(file_paths) == 1:
            query_filter = Filter(
                must=[FieldCondition(key="file_path", match=MatchValue(value=file_paths[0]))]
            )
        else:
            query_filter = Filter(
                should=[
                    FieldCondition(key="file_path", match=MatchValue(value=file_path))
                    for file_path in file_paths
                ]
            )

    client = get_client()
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter,
    )

    raw_hits = []
    if hasattr(response, "points"):
        raw_hits = response.points or []
    elif isinstance(response, list):
        raw_hits = response

    seen_texts: set[str] = set()
    deduped_hits: list[Any] = []
    for hit in raw_hits:
        if isinstance(hit, dict):
            payload = hit.get("payload") or {}
        else:
            payload = getattr(hit, "payload", {}) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            deduped_hits.append(hit)
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        deduped_hits.append(hit)

    return deduped_hits
