"""Vectorless, PageIndex-style retrieval for documents with clear hierarchy.

No embedding is created and no Qdrant query is run in this module. At
indexing time it preserves natural sections, headings and page ranges; at
question time Groq navigates that tree. A transparent lexical tree search is
used if Groq is unavailable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.config import ENABLE_STRUCTURED_RETRIEVAL, STRUCTURE_SCORE_THRESHOLD, STRUCTURED_MAX_SECTIONS
from app.database import get_structured_document_indexes
from app.parsers import ParsedSegment


_DEFAULT_SECTION = "Document body"
_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")
_MAX_CATALOG_CHARS = 24_000


@dataclass(frozen=True)
class StructureAssessment:
    """Evidence that a document has a usable heading hierarchy."""

    score: int
    heading_count: int
    labelled_body_count: int
    has_numbered_headings: bool
    is_structured: bool


def _tokens(value: str) -> set[str]:
    return {_stem(token.lower()) for token in _WORD.findall(value) if len(token) > 2}


def _stem(word: str) -> str:
    """Fold a few common plural/verb forms so lexical matching is more forgiving."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _is_heading(segment: ParsedSegment) -> bool:
    return bool(segment.metadata.get("is_heading") or segment.metadata.get("content_kind") == "heading")


def assess_document_structure(path: Path, segments: Iterable[ParsedSegment]) -> StructureAssessment:
    """Assess extracted structure instead of assuming every PDF is structured."""
    material = list(segments)
    headings = [segment for segment in material if _is_heading(segment)]
    heading_titles = {segment.text.strip().casefold() for segment in headings if segment.text.strip()}
    labelled_bodies = [
        segment
        for segment in material
        if not _is_heading(segment)
        and (section := str(segment.metadata.get("section") or "")).strip()
        and section != _DEFAULT_SECTION
    ]
    numbered = any(re.match(r"^\d+(?:\.\d+)*\.?\s+", segment.text.strip()) for segment in headings)
    levels = {segment.metadata.get("heading_level") for segment in headings if segment.metadata.get("heading_level")}

    # A few headings are not enough: body text must actually be labelled by
    # them. This avoids classifying title pages or badly extracted PDFs as a
    # trustworthy hierarchy.
    score = min(len(heading_titles), 6)
    score += 3 if len(labelled_bodies) >= 3 else (1 if labelled_bodies else 0)
    if numbered:
        score += 2
    if len(levels) >= 2:
        score += 2
    if path.suffix.lower() == ".pdf" and len({segment.metadata.get("page") for segment in headings}) >= 2:
        score += 1

    # The official PageIndex document-processing API currently accepts PDFs.
    # Structured DOCX files therefore remain on the proven hybrid route until
    # PageIndex adds DOCX processing or a local converter is deliberately
    # introduced.
    supported_type = path.suffix.lower() == ".pdf"
    return StructureAssessment(
        score=score,
        heading_count=len(headings),
        labelled_body_count=len(labelled_bodies),
        has_numbered_headings=numbered,
        is_structured=ENABLE_STRUCTURED_RETRIEVAL and supported_type and score >= STRUCTURE_SCORE_THRESHOLD,
    )


def _section_level(title: str, metadata: dict[str, Any], previous_level: int) -> int:
    explicit = metadata.get("heading_level")
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    numbered = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", title)
    if numbered:
        return numbered.group(1).count(".") + 1
    # Short labels such as "Key concepts" below a numbered chapter are likely
    # subsection labels when no explicit PDF heading level is available.
    return 2 if previous_level == 1 else max(previous_level, 1)


def build_section_tree(path: Path, segments: Iterable[ParsedSegment], assessment: StructureAssessment) -> dict[str, Any]:
    """Create natural retrieval units with page and section citation data."""
    material = list(segments)
    headings = [segment for segment in material if _is_heading(segment)]
    nodes: list[dict[str, Any]] = []
    title_to_ids: dict[str, list[str]] = {}
    previous_level = 0
    for order, heading in enumerate(headings, start=1):
        title = heading.text.strip()
        if not title:
            continue
        node = {
            "node_id": f"section-{order}",
            "title": title,
            "level": _section_level(title, heading.metadata, previous_level),
            "page_start": heading.metadata.get("page"),
            "page_end": heading.metadata.get("page"),
            "text": "",
            "children": [],
        }
        previous_level = node["level"]
        nodes.append(node)
        title_to_ids.setdefault(title.casefold(), []).append(node["node_id"])

    node_by_id = {node["node_id"]: node for node in nodes}
    stack: list[dict[str, Any]] = []
    for node in nodes:
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node["node_id"])
        stack.append(node)

    latest_node: dict[str, Any] | None = None
    for segment in material:
        if _is_heading(segment):
            ids = title_to_ids.get(segment.text.strip().casefold(), [])
            latest_node = node_by_id.get(ids[-1]) if ids else latest_node
            continue
        section = str(segment.metadata.get("section") or "").strip()
        ids = title_to_ids.get(section.casefold(), [])
        target = node_by_id.get(ids[-1]) if ids else latest_node
        if target is None or not segment.text.strip():
            continue
        text = segment.text.strip()
        if text not in target["text"]:
            target["text"] = (target["text"] + "\n\n" + text).strip()
        page = segment.metadata.get("page")
        if isinstance(page, int):
            target["page_start"] = min(target["page_start"], page) if target["page_start"] else page
            target["page_end"] = max(target["page_end"], page) if target["page_end"] else page

    return {
        "format": "pageindex-style-v1",
        "retrieval_mode": "vectorless_structured_tree",
        "file_path": str(path.resolve()),
        "file_name": path.name,
        "structure_score": assessment.score,
        "nodes": [node for node in nodes if node["text"].strip()],
    }


def _catalog(records: list[dict[str, Any]]) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], str]:
    lookup: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    lines: list[str] = []
    for record in records:
        for node in (record.get("tree") or {}).get("nodes") or []:
            identifier = f"{record.get('file_name', 'document')}::{node.get('node_id')}"
            lookup[identifier] = (record, node)
            preview = " ".join(str(node.get("text") or "").split())[:220]
            lines.append(
                f"ID: {identifier}\nFile: {record.get('file_name')}\n"
                f"Section: {node.get('title')}\nPage: {node.get('page_start') or '?'}\nPreview: {preview}"
            )
    catalog = "\n\n".join(lines)
    # Keep the router prompt bounded: drop the least-relevant trailing sections
    # (highest node ids) when a large catalog would blow the model context.
    while len(catalog) > _MAX_CATALOG_CHARS and len(lines) > 1:
        lines.pop()
        catalog = "\n\n".join(lines)
    return lookup, catalog


def _reason_over_tree(question: str, catalog: str, valid_ids: set[str]) -> list[str]:
    """Use Groq to select exact section IDs, never to answer the question."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or not catalog:
        return []
    try:
        from groq import Groq

        prompt = (
            "You are a document-navigation router. Select only section IDs containing direct evidence "
            "for the question. Do not select merely related sections. If no listed section is clearly "
            "relevant, return an empty list. Return strict JSON only: {\"section_ids\":[...]}\n\n"
            f"Question: {question}\n\nSection tree catalog:\n{catalog}"
        )
        response = Groq(api_key=api_key).chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=250,
        )
        content = response.choices[0].message.content or "{}"
        match = re.search(r"\{.*\}", content, re.DOTALL)
        selected = json.loads(match.group(0) if match else "{}")
        return [item for item in selected.get("section_ids", []) if item in valid_ids]
    except Exception:
        return []


def _lexical_tree_search(question: str, lookup: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> list[str]:
    """Non-vector fallback used when the tree-navigation LLM is unavailable."""
    query_terms = _tokens(question)
    scored: list[tuple[float, str]] = []
    for identifier, (_, node) in lookup.items():
        title_hits = len(query_terms & _tokens(str(node.get("title") or "")))
        text_hits = len(query_terms & _tokens(str(node.get("text") or "")))
        score = title_hits * 5 + text_hits
        # Avoid routing vague questions to a structured PDF only because they
        # share a single common word. A real title match or several text terms
        # is required; otherwise the normal RRF route remains the fallback.
        if title_hits or score >= 3:
            scored.append((score, identifier))
    scored.sort(reverse=True)
    return [identifier for _, identifier in scored[:STRUCTURED_MAX_SECTIONS]]


def search_structured_documents(question: str, document_chat_id: str | None, top_k: int = 8) -> list[dict[str, Any]]:
    """Return exact structured sections without embeddings or vector search."""
    if not ENABLE_STRUCTURED_RETRIEVAL or not document_chat_id:
        return []
    records = get_structured_document_indexes(document_chat_id)
    if not records:
        return []
    lookup, catalog = _catalog(records)
    selected = _reason_over_tree(question, catalog, set(lookup))
    if not selected:
        selected = _lexical_tree_search(question, lookup)

    results: list[dict[str, Any]] = []
    for rank, identifier in enumerate(selected[:top_k], start=1):
        record, node = lookup[identifier]
        results.append(
            {
                "score": 1.0 / rank,
                "payload": {
                    "text": node.get("text", ""),
                    "file_name": record.get("file_name"),
                    "file_path": record.get("file_path"),
                    "source_type": "structured_document",
                    "section": node.get("title"),
                    "page": node.get("page_start"),
                    "page_end": node.get("page_end"),
                    "section_node_id": node.get("node_id"),
                    "structure_score": record.get("structure_score"),
                    "retrieval_mode": "vectorless_structured_tree",
                },
            }
        )
    return results
