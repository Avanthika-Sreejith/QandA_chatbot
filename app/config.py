"""Central configuration for the application."""

import os

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "document_chunks")

# Qwen3-Embedding-0.6B is downloaded from Hugging Face in Day 5.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 1024
DENSE_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
# Device to load embedding models on: 'cpu' or 'cuda' (GPU). Set via env EMBEDDING_DEVICE.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
