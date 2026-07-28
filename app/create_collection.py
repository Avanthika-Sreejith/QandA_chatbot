"""Day 4 command: create the local hybrid-search Qdrant collection."""

from app.retrieval.collection import collection_summary, ensure_collection


def main() -> None:
    try:
        created = ensure_collection()
    except Exception as error:
        raise SystemExit(
            "Could not connect to Qdrant. Check QDRANT_URL and QDRANT_API_KEY in `.env`.\n"
            f"Details: {error}"
        ) from error

    print("Created hybrid Qdrant collection." if created else "Hybrid Qdrant collection already exists.")
    print(collection_summary())


if __name__ == "__main__":
    main()
