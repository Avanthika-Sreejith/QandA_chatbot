"""Day 5 command: index a document in the local Qdrant collection."""

from __future__ import annotations

import argparse

from app.ingestion.indexing import ingest_file, ingest_folder
from app.retrieval.collection import collection_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse, embed, and index a supported document or folder.")
    parser.add_argument("path", help="Path to a PDF, DOCX, XLSX, XLSM, TXT file, or a folder")
    args = parser.parse_args()

    print("Indexing document using OpenAI embeddings and BM25…")
    try:
        from pathlib import Path

        path = Path(args.path)
        if path.is_dir():
            files, segments, chunks = ingest_folder(path)
            print(f"Found {files} supported file(s).")
        else:
            segments, chunks = ingest_file(str(path))
    except Exception as error:
        raise SystemExit(f"Indexing failed: {error}") from error

    print(f"Parsed {segments} source segment(s).")
    print(f"Indexed {chunks} chunk(s) in Qdrant.")
    print(collection_summary())


if __name__ == "__main__":
    main()
