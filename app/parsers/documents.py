"""Extract text and source metadata from supported document types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
from zipfile import ZipFile


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xlsm", ".txt"}
TABLE_ROWS_PER_SEGMENT = 20


def _looks_like_numbered_heading(text: str) -> bool:
    """Recognize simple headings whose DOCX style was not set to Heading."""
    return bool(re.fullmatch(r"\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{0,100}", text))


def _heading_level(style_name: str) -> int:
    """Extract the heading level from a style name such as 'Heading 2'."""
    match = re.search(r"(\d+)", style_name)
    return int(match.group(1)) if match else 1


@dataclass(frozen=True)
class ParsedSegment:
    """A source-preserving unit of extracted text before chunking."""

    text: str
    metadata: dict[str, Any]


def _base_metadata(path: Path) -> dict[str, Any]:
    return {"file_name": path.name, "file_path": str(path.resolve())}


def _table_row_groups(
    rows: list[list[str]],
    *,
    path: Path,
    source_type: str,
    table_number: int | None = None,
    sheet: str | None = None,
    has_embedded_images: bool = False,
) -> list[ParsedSegment]:
    """Create header-preserving chunks for a native document table."""
    non_empty = [row for row in rows if any(cell.strip() for cell in row)]
    if not non_empty:
        return []

    header_values = non_empty[0]
    headers = [value or f"Column {index}" for index, value in enumerate(header_values, start=1)]
    data_rows = non_empty[1:] or [non_empty[0]]
    output: list[ParsedSegment] = []
    for start in range(0, len(data_rows), TABLE_ROWS_PER_SEGMENT):
        group = data_rows[start : start + TABLE_ROWS_PER_SEGMENT]
        rendered_rows = []
        for row_offset, row in enumerate(group, start=start + 1):
            values = (row + [""] * len(headers))[: len(headers)]
            cells = [f"{header}: {value}" for header, value in zip(headers, values, strict=True) if value]
            if cells:
                rendered_rows.append(f"Row {row_offset}: " + "; ".join(cells))
        if not rendered_rows:
            continue

        label = f"Table {table_number}" if table_number is not None else f"Sheet {sheet}"
        text = f"{label}\nHeaders: " + " | ".join(headers) + "\n" + "\n".join(rendered_rows)
        metadata = _base_metadata(path) | {
            "source_type": source_type,
            "content_kind": "table",
            "table_row_start": start + 1,
            "table_row_end": start + len(group),
            "table_headers": headers,
            "has_embedded_images": has_embedded_images,
        }
        if table_number is not None:
            metadata["table"] = table_number
            metadata["section"] = label
        if sheet is not None:
            metadata["sheet"] = sheet
        output.append(ParsedSegment(text, metadata))
    return output


def _median_font_size(spans: list[dict[str, Any]]) -> float:
    """Return the median span size, used as the body text size."""
    sizes = sorted(span.get("size", 0.0) for span in spans)
    if not sizes:
        return 0.0
    return sizes[len(sizes) // 2]


def _looks_like_pdf_heading(text: str, max_size: float, is_bold: bool, body_size: float) -> bool:
    """Heuristically detect a PDF heading line from its size and weight."""
    if len(text) < 2 or len(text) > 120:
        return False
    if text.strip().isdigit():  # skip page numbers
        return False
    if body_size <= 0:
        return is_bold and len(text) <= 80
    larger_font = max_size >= body_size + 1.5
    return (larger_font and len(text) <= 80) or (is_bold and len(text) <= 60)


def parse_pdf(path: Path) -> list[ParsedSegment]:
    """Return per-page segments split into sections at detected headings."""
    import fitz

    segments: list[ParsedSegment] = []
    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            blocks = page.get_text("dict").get("blocks", [])
            spans = [
                span
                for block in blocks
                if block.get("type") == 0
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ]
            if not spans:
                continue
            body_size = _median_font_size(spans)
            current_section = "Document body"
            pending: list[str] = []
            page_segments: list[ParsedSegment] = []

            def flush() -> None:
                text = " ".join(pending).strip()
                if text:
                    page_segments.append(
                        ParsedSegment(
                            text,
                            _base_metadata(path)
                            | {"source_type": "pdf", "page": index, "section": current_section},
                        )
                    )
                pending.clear()

            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_spans = line.get("spans", [])
                    if not line_spans:
                        continue
                    text = "".join(span.get("text", "") for span in line_spans).strip()
                    if not text:
                        continue
                    max_size = max((span.get("size", 0.0) for span in line_spans), default=0.0)
                    is_bold = any(span.get("flags", 0) & 16 for span in line_spans)
                    if _looks_like_pdf_heading(text, max_size, is_bold, body_size):
                        flush()
                        current_section = text
                        page_segments.append(
                            ParsedSegment(
                                text,
                                _base_metadata(path)
                                | {
                                    "source_type": "pdf_heading",
                                    "content_kind": "heading",
                                    "page": index,
                                    "section": text,
                                    "is_heading": True,
                                },
                            )
                        )
                    else:
                        pending.append(text)
            flush()
            segments.extend(page_segments)
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
        is_heading = style_name.lower().startswith("heading") or _looks_like_numbered_heading(text)
        if is_heading:
            section = text
            segments.append(
                ParsedSegment(
                    text,
                    _base_metadata(path)
                    | {
                        "source_type": "docx_heading",
                        "content_kind": "heading",
                        "section": text,
                        "paragraph": paragraph_number,
                        "heading_level": _heading_level(style_name),
                        "is_heading": True,
                    },
                )
            )
            continue
        segments.append(
            ParsedSegment(
                text,
                _base_metadata(path)
                | {"source_type": "docx", "section": section, "paragraph": paragraph_number},
            )
        )

    # Preserve table headers in each row group. This is more useful for RAG
    # than flattening an entire table into one unstructured block of text.
    for table_number, table in enumerate(document.tables, start=1):
        rows: list[list[str]] = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows.append(cells)
        image_count = len(table._element.xpath(".//a:blip"))
        segments.extend(
            _table_row_groups(
                rows,
                path=path,
                source_type="docx_table",
                table_number=table_number,
                has_embedded_images=bool(image_count),
            )
        )
    return segments


def parse_excel(path: Path) -> list[ParsedSegment]:
    """Return header-aware row groups and retain formulas plus cached values."""
    from openpyxl import load_workbook

    formula_workbook = load_workbook(path, read_only=True, data_only=False)
    value_workbook = load_workbook(path, read_only=True, data_only=True)
    with ZipFile(path) as archive:
        workbook_image_count = sum(name.startswith("xl/media/") for name in archive.namelist())
    segments: list[ParsedSegment] = []
    try:
        for worksheet, value_worksheet in zip(
            formula_workbook.worksheets, value_workbook.worksheets, strict=True
        ):
            rows: list[list[str]] = []
            for formula_row, value_row in zip(
                worksheet.iter_rows(values_only=True),
                value_worksheet.iter_rows(values_only=True),
                strict=True,
            ):
                values: list[str] = []
                for formula, calculated_value in zip(formula_row, value_row, strict=True):
                    if formula is None and calculated_value is None:
                        values.append("")
                    elif isinstance(formula, str) and formula.startswith("="):
                        values.append(f"Formula: {formula}; calculated value: {calculated_value!s}")
                    else:
                        values.append(str(calculated_value if calculated_value is not None else formula).strip())
                if any(values):
                    rows.append(values)
            segments.extend(
                _table_row_groups(
                    rows,
                    path=path,
                    source_type="excel_table",
                    sheet=worksheet.title,
                    has_embedded_images=bool(workbook_image_count),
                )
            )
    finally:
        formula_workbook.close()
        value_workbook.close()
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
