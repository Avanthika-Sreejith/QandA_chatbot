"""Small Day 1 health check for the local Qdrant container."""

from qdrant_client import QdrantClient

from app.config import QDRANT_API_KEY, QDRANT_URL


def main() -> None:
    """Connect to Qdrant and print its available collections."""
    try:
        client_kwargs = {"url": QDRANT_URL, "timeout": 5}
        if QDRANT_API_KEY:
            client_kwargs["api_key"] = QDRANT_API_KEY
        client = QdrantClient(**client_kwargs)
        collections = client.get_collections().collections
    except Exception as error:
        raise SystemExit(
            f"Could not reach Qdrant at {QDRANT_URL}. "
            "Run `docker compose up -d` after starting Docker Desktop.\n"
            f"Details: {error}"
        ) from error

    print(f"Qdrant is reachable at {QDRANT_URL}")
    print(f"Existing collections: {len(collections)}")


if __name__ == "__main__":
    main()
