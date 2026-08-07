"""Query the Qdrant collection using two-phase semantic search with source selection."""

from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app.config import BROAD_SEARCH_TOP_K, SEARCH_SCORE_THRESHOLD
from app.embeddings import get_dense_embeddings
from app.retrieval.collection import DENSE_VECTOR_NAME, QDRANT_COLLECTION, get_client


def _scope_conditions(
    document_chat_id: str | None,
    file_paths: list[str] | None,
) -> list[FieldCondition]:
    """Return the Qdrant must-conditions that scope a search to its target."""
    if document_chat_id:
        return [
            FieldCondition(key="document_chat_id", match=MatchValue(value=document_chat_id))
        ]
    if file_paths:
        if len(file_paths) == 1:
            return [FieldCondition(key="file_path", match=MatchValue(value=file_paths[0]))]
        return [FieldCondition(key="file_path", match=MatchAny(any=file_paths))]
    return []


def _hit_payload(hit: Any) -> dict[str, Any]:
    """Return the payload dict of a Qdrant point result."""
    if isinstance(hit, dict):
        return hit.get("payload") or {}
    return getattr(hit, "payload", None) or {}


def _hit_score(hit: Any) -> float:
    """Return the similarity score of a Qdrant point result."""
    if isinstance(hit, dict):
        return hit.get("score") or 0.0
    return getattr(hit, "score", 0.0) or 0.0


def _query_points(
    query_vector: list[float],
    must_conditions: list[FieldCondition],
    limit: int,
) -> list[Any]:
    """Run a dense-only search with the given must filters."""
    client = get_client()
    query_filter = Filter(must=must_conditions) if must_conditions else None
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter,
    )
    if hasattr(response, "points"):
        return response.points or []
    return response if isinstance(response, list) else []


def search_documents(
    query: str,
    top_k: int = 5,
    file_paths: list[str] | None = None,
    document_chat_id: str | None = None,
    score_threshold: float = SEARCH_SCORE_THRESHOLD,
) -> list[Any]:
    """Return top-k hits after automatic best-source selection.

    Phase 1 scans the whole search scope and keeps only the file(s) whose
    strongest chunk meets the score threshold, so unrelated files never appear
    in the answer. Phase 2 re-searches within those selected files only, then
    rejects individual weak chunks using the same threshold. If more than one
    file has strong evidence, all of them are returned.
    """
    query_vector = get_dense_embeddings([query])[0]
    scope_conditions = _scope_conditions(document_chat_id, file_paths)

    broad_hits = _query_points(query_vector, scope_conditions, BROAD_SEARCH_TOP_K)

    best_scores: dict[str, float] = {}
    for hit in broad_hits:
        payload = _hit_payload(hit)
        file_path = payload.get("file_path") or payload.get("file_name") or ""
        if not file_path:
            continue
        best_scores[file_path] = max(best_scores.get(file_path, 0.0), _hit_score(hit))

    selected_files = [
        file_path for file_path, score in best_scores.items() if score >= score_threshold
    ]
    if not selected_files:
        return []

    hits: list[Any] = []
    for file_path in selected_files:
        file_conditions = scope_conditions + [
            FieldCondition(key="file_path", match=MatchValue(value=file_path))
        ]
        hits.extend(_query_points(query_vector, file_conditions, top_k))

    seen_texts: set[str] = set()
    ranked: list[Any] = []
    for hit in sorted(hits, key=_hit_score, reverse=True):
        if _hit_score(hit) < score_threshold:
            continue
        text = (_hit_payload(hit).get("text") or "").strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        ranked.append(hit)
    return ranked[:top_k]
