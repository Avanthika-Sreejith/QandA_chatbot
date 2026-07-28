"""Extract text and source metadata from supported document types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xlsm", ".txt"}


def _looks_like_numbered_heading(text: str) -> bool:
    """Recognize simple headings whose DOCX style was not set to Heading."""
    return bool(re.fullmatch(r"\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{0,100}", text))


@dataclass(frozen=True)
class ParsedSegment:
    """A source-preserving unit of extracted text before chunking."""

    text: str
    metadata: dict[str, Any]


def _base_metadata(path: Path) -> dict[str, Any]:
    return {"file_name": path.name, "file_path": str(path.resolve())}


def parse_pdf(path: Path) -> list[ParsedSegment]:
    """Return one segment per text-bearing PDF page."""
    import fitz

    segments: list[ParsedSegment] = []
    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                segments.append(
                    ParsedSegment(text, _base_metadata(path) | {"source_type": "pdf", "page": index})
                )
    return segments


def parse_docx(path: Path) -> list[ParsedSegment]:
    """Return paragraphs and tables, labelled by their nearest heading where possible."""
    from docx import Document

    document = Document(path)
    section = "Document body"
    segments: list[ParsedSegment] = []
    for paragraph_number, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style else ""
        if style_name.lower().startswith("heading") or _looks_like_numbered_heading(text):
            section = text
            continue
        segments.append(
            ParsedSegment(
                text,
                _base_metadata(path)
                | {"source_type": "docx", "section": section, "paragraph": paragraph_number},
            )
        )

    # DOCX tables often contain essential structured information (for example,
    # a project technology stack). Keep each table as a searchable segment.
    for table_number, table in enumerate(document.tables, start=1):
        rows: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            segments.append(
                ParsedSegment(
                    "\n".join(rows),
                    _base_metadata(path)
                    | {
                        "source_type": "docx",
                        "section": f"Table {table_number}",
                        "table": table_number,
                    },
                )
            )
    return segments


def parse_excel(path: Path) -> list[ParsedSegment]:
    """Return one text segment for every non-empty worksheet."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    segments: list[ParsedSegment] = []
    try:
        for worksheet in workbook.worksheets:
            rows: list[str] = []
            for row_number, row in enumerate(worksheet.iter_rows(values_only=True), start=1):
                values = [str(value).strip() for value in row if value is not None and str(value).strip()]
                if values:
                    rows.append(f"Row {row_number}: " + " | ".join(values))
            if rows:
                segments.append(
                    ParsedSegment(
                        "\n".join(rows),
                        _base_metadata(path)
                        | {
                            "source_type": "excel",
                            "sheet": worksheet.title,
                            "row_start": 1,
                            "row_end": worksheet.max_row,
                        },
                    )
                )
    finally:
        workbook.close()
    return segments


def parse_text(path: Path) -> list[ParsedSegment]:
    """Return a plain-text file as one source-preserving segment."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [
        ParsedSegment(
            text,
            _base_metadata(path)
            | {"source_type": "txt", "section": "Full document", "line_start": 1, "line_end": len(text.splitlines())},
        )
    ]


def parse_document(file_path: str | Path) -> list[ParsedSegment]:
    """Dispatch parsing based on the file extension."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension}'. Supported types: {supported}")
    if extension == ".pdf":
        return parse_pdf(path)
    if extension == ".docx":
        return parse_docx(path)
    if extension in {".xlsx", ".xlsm"}:
        return parse_excel(path)
    return parse_text(path)
