"""Small CLI helper to query the indexed documents."""

from __future__ import annotations

import argparse

from app.retrieval.search import search_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Query indexed documents in Qdrant.")
    parser.add_argument("query", help="The natural language query to search.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    args = parser.parse_args()

    results = search_documents(args.query, top_k=args.top_k)
    if not results:
        print("No results found.")
        return

    for index, hit in enumerate(results, start=1):
        payload = hit.payload or {}
        text = payload.get("text", "")
        source = payload.get("file_path", "unknown")
        print(f"Result {index}")
        print(f"Source: {source}")
        print(f"Score: {hit.score}")
        print("---")
        print(text)
        print()


if __name__ == "__main__":
    main()
