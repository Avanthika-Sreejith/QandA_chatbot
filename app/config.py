"""Central configuration for the application."""

import os

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "document_chunks")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 1024
JINA_EMBEDDING_MODEL = "jina-embeddings-v3"
JINA_EMBEDDING_URL = "https://api.jina.ai/v1"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
