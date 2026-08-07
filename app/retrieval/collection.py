"""Create and inspect the Qdrant collection used by the chatbot."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from app.config import (
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    SPARSE_VECTOR_NAME,
)


def get_client() -> QdrantClient:
    """Connect to the Qdrant server, optionally using a cloud API key."""
    client_kwargs = {"url": QDRANT_URL, "timeout": 10}
    if QDRANT_API_KEY:
        client_kwargs["api_key"] = QDRANT_API_KEY
    return QdrantClient(**client_kwargs)


def ensure_collection() -> bool:
    """Create the hybrid-search collection if it does not already exist.

    Returns True only when a new collection was created. Existing collections
    are never deleted, so running this command is safe during development.
    """
    client = get_client()
    collection_created = False
    if client.collection_exists(QDRANT_COLLECTION):
        info = client.get_collection(QDRANT_COLLECTION)
        vectors = info.config.params.vectors
        existing_dense = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        if existing_dense is not None and existing_dense.size != DENSE_VECTOR_SIZE:
            raise RuntimeError(
                f"The existing collection uses {existing_dense.size} dense dimensions, but "
                f"{DENSE_VECTOR_SIZE} dimensions are required for Jina embeddings. Delete the "
                "`document_chunks` collection in the Qdrant dashboard, then run this command again."
            )
    else:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
            },
        )
        collection_created = True

    # Qdrant Cloud requires explicit payload indexes for filtered searches.
    for field_name in ("document_chat_id", "file_path"):
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
    return collection_created


def collection_summary() -> str:
    """Return a short human-readable summary for the terminal demo."""
    client = get_client()
    info = client.get_collection(QDRANT_COLLECTION)
    return (
        f"Collection: {QDRANT_COLLECTION}\n"
        f"Points indexed: {info.points_count}\n"
        f"Dense vector: {DENSE_VECTOR_NAME} ({DENSE_VECTOR_SIZE} dimensions, cosine distance)\n"
        f"Sparse vector: {SPARSE_VECTOR_NAME} (BM25 / IDF)"
    )
