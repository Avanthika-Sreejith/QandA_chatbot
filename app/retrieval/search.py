"""Query the Qdrant collection using hybrid dense+sparse search with source selection."""

from __future__ import annotations

import re
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    SparseVector,
)

from app.config import BROAD_SEARCH_TOP_K, RRF_K, SEARCH_RRF_THRESHOLD, SOURCE_SELECTION_RATIO
from app.embeddings import get_dense_embeddings, get_sparse_embeddings
from app.retrieval.collection import DENSE_VECTOR_NAME, QDRANT_COLLECTION, SPARSE_VECTOR_NAME, get_client


_SEQUENCE_QUERY = re.compile(r"\b(steps?|process|procedure|sequence|stages?|how\s+to)\b", re.IGNORECASE)


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


def _hit_id(hit: Any) -> str:
    """Return the point ID of a Qdrant point result."""
    if isinstance(hit, dict):
        return str(hit.get("id"))
    return str(getattr(hit, "id", ""))


def _hit_payload(hit: Any) -> dict[str, Any]:
    """Return the payload dict of a Qdrant point result."""
    if isinstance(hit, dict):
        return hit.get("payload") or {}
    return getattr(hit, "payload", None) or {}


def _query_points(
    query: list[float] | SparseVector,
    vector_name: str,
    must_conditions: list[FieldCondition],
    limit: int,
) -> list[Any]:
    """Run a single-vector search with the given must filters."""
    client = get_client()
    query_filter = Filter(must=must_conditions) if must_conditions else None
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query,
        using=vector_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter,
    )
    if hasattr(response, "points"):
        return response.points or []
    return response if isinstance(response, list) else []


def _with_score(hit: Any, score: float) -> Any:
    """Return a copy of a Qdrant hit carrying the given score."""
    if isinstance(hit, dict):
        return {**hit, "score": score}
    return hit.model_copy(update={"score": score})


def _fuse_rrf(ranked_lists: list[list[Any]]) -> list[tuple[float, Any]]:
    """Fuse ranked hit lists into one reciprocal-rank-fusion score per point.

    Each method's ranking contributes 1/(RRF_K + rank) to the point's score,
    so a point must be ranked highly by the methods to reach a high score.
    Returns (rrf_score, hit) pairs sorted descending by score; each hit's own
    score field is overwritten with the fused value for consistent display.
    """
    scores: dict[str, float] = {}
    hits: dict[str, Any] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            point_id = _hit_id(hit)
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (RRF_K + rank)
            hits.setdefault(point_id, hit)
    fused = [(score, _with_score(hits[point_id], score)) for point_id, score in scores.items()]
    fused.sort(key=lambda item: item[0], reverse=True)
    return fused


def _fused_query(
    query_vector: list[float],
    query_sparse: SparseVector,
    must_conditions: list[FieldCondition],
    limit: int,
) -> list[tuple[float, Any]]:
    """Run dense and sparse searches together and fuse them with RRF."""
    dense_hits = _query_points(query_vector, DENSE_VECTOR_NAME, must_conditions, limit)
    sparse_hits = _query_points(query_sparse, SPARSE_VECTOR_NAME, must_conditions, limit)
    return _fuse_rrf([dense_hits, sparse_hits])


def _sequence_anchor(hit: Any) -> tuple[int, int]:
    """Rank likely procedure chunks above FAQ/reference noise."""
    text = (_hit_payload(hit).get("text") or "").lower()
    numbered_items = len(re.findall(r"(?:^|\s)\d+\.\s", text))
    score = numbered_items * 10 + (5 if "step" in text else 0)
    if text.startswith(("explain ", "what is ", "describe ")):
        score -= 20
    return score, numbered_items


def _expand_sequence_context(
    ranked: list[Any],
    scope_conditions: list[FieldCondition],
) -> list[Any]:
    """Bring the full neighbouring sequence into context for ordered questions.

    A semantic search can rank step 2 above step 1. Once a likely numbered
    procedure chunk is found, scroll its source segment and place all sibling
    chunks first in their original order so the answer model sees step 1..N.
    """
    eligible = [
        hit
        for hit in ranked
        if _hit_payload(hit).get("chunk_count", 1) > 1
        and _hit_payload(hit).get("source_segment") is not None
    ]
    if not eligible:
        return ranked
    anchor = max(eligible, key=_sequence_anchor)
    if _sequence_anchor(anchor)[0] <= 0:
        return ranked

    payload = _hit_payload(anchor)
    file_path = payload.get("file_path")
    source_segment = payload.get("source_segment")
    if not file_path:
        return ranked
    sibling_filter = Filter(
        must=scope_conditions
        + [
            FieldCondition(key="file_path", match=MatchValue(value=file_path)),
            FieldCondition(key="source_segment", match=MatchValue(value=source_segment)),
        ]
    )
    try:
        points, _ = get_client().scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=sibling_filter,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        # Search still works if a legacy collection has not yet created the
        # source_segment payload index.
        return ranked

    siblings = sorted(
        points,
        key=lambda point: (_hit_payload(point).get("chunk_index", 0)),
    )
    seen_ids = {_hit_id(point) for point in siblings}
    return siblings + [hit for hit in ranked if _hit_id(hit) not in seen_ids]


def search_documents(
    query: str,
    top_k: int = 8,
    file_paths: list[str] | None = None,
    document_chat_id: str | None = None,
    rrf_threshold: float = SEARCH_RRF_THRESHOLD,
) -> list[Any]:
    """Return top-k hits after automatic best-source selection.

    Dense (semantic) and sparse (keyword/BM25) rankings are fused with
    reciprocal rank fusion so exact query terms can rescue chunks that a pure
    semantic search ranks too low, and the two together suppress weak noise.

    Phase 1 scans the whole search scope and keeps only the file(s) whose
    strongest chunk meets the RRF threshold and is close to the strongest file
    (SOURCE_SELECTION_RATIO), so unrelated files never appear in the answer.
    Phase 2 re-searches within those selected files only, then rejects weak
    chunks using the RRF threshold. If more than one file has strong evidence,
    all of them are returned.
    """
    query_vector = get_dense_embeddings([query])[0]
    sparse = get_sparse_embeddings([query])[0]
    query_sparse = SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )
    scope_conditions = _scope_conditions(document_chat_id, file_paths)

    broad_fused = _fused_query(query_vector, query_sparse, scope_conditions, BROAD_SEARCH_TOP_K)

    best_scores: dict[str, float] = {}
    for score, hit in broad_fused:
        payload = _hit_payload(hit)
        file_path = payload.get("file_path") or payload.get("file_name") or ""
        if not file_path:
            continue
        best_scores[file_path] = max(best_scores.get(file_path, 0.0), score)

    if not best_scores:
        return []
    top_score = max(best_scores.values())
    selection_floor = max(rrf_threshold, SOURCE_SELECTION_RATIO * top_score)
    selected_files = [
        file_path
        for file_path, score in best_scores.items()
        if score >= rrf_threshold and score >= selection_floor
    ]
    if not selected_files:
        return []

    candidates: list[tuple[float, Any]] = []
    for file_path in selected_files:
        file_conditions = scope_conditions + [
            FieldCondition(key="file_path", match=MatchValue(value=file_path))
        ]
        candidates.extend(_fused_query(query_vector, query_sparse, file_conditions, top_k))

    candidates.sort(key=lambda item: item[0], reverse=True)

    seen_texts: set[str] = set()
    ranked: list[Any] = []
    for score, hit in candidates:
        if score < rrf_threshold:
            continue
        text = (_hit_payload(hit).get("text") or "").strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        ranked.append(hit)
    if _SEQUENCE_QUERY.search(query):
        ranked = _expand_sequence_context(ranked, scope_conditions)
    return ranked[:top_k]
