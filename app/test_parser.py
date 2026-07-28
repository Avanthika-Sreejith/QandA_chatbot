"""Print extracted text and metadata for one supported file."""

from __future__ import annotations

import argparse

from app.parsers import parse_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a supported document parser.")
    parser.add_argument("file", help="Path to a PDF, DOCX, XLSX, XLSM, or TXT file")
    args = parser.parse_args()

    segments = parse_document(args.file)
    print(f"Extracted {len(segments)} source segment(s).")
    for number, segment in enumerate(segments[:3], start=1):
        preview = segment.text.replace("\n", " ")[:300]
        print(f"\nSegment {number} metadata: {segment.metadata}")
        print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
