"""Parse a document, chunk it, and show the first few chunks."""

from __future__ import annotations

import argparse

from app.ingestion import chunk_segments
from app.parsers import parse_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Test document chunking.")
    parser.add_argument("file", help="Path to a supported document")
    args = parser.parse_args()

    parsed_segments = parse_document(args.file)
    chunks = chunk_segments(parsed_segments)
    print(f"Parsed {len(parsed_segments)} source segment(s) into {len(chunks)} chunk(s).")
    for number, chunk in enumerate(chunks[:3], start=1):
        preview = chunk.text.replace("\n", " ")[:300]
        print(f"\nChunk {number} metadata: {chunk.metadata}")
        print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
