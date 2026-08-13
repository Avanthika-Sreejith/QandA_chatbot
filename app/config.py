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

# Semantic chunking uses the same Jina embedding API as retrieval to identify
# topic shifts. These are safety limits, not fixed chunk targets.
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.72"))
SEMANTIC_MIN_SENTENCES = int(os.getenv("SEMANTIC_MIN_SENTENCES", "2"))
SEMANTIC_MAX_CHARACTERS = int(os.getenv("SEMANTIC_MAX_CHARACTERS", "3500"))

# Retrieval fuses dense (Jina) and sparse (BM25) rankings with reciprocal rank
# fusion, then runs a two-phase search: detect the best source file(s), then
# re-search within those files only. Chunks whose fused RRF score falls below
# the threshold are rejected as weak evidence.
RRF_K = int(os.getenv("RRF_K", "60"))
SEARCH_RRF_THRESHOLD = float(os.getenv("SEARCH_RRF_THRESHOLD", "0.012"))
BROAD_SEARCH_TOP_K = int(os.getenv("BROAD_SEARCH_TOP_K", "20"))
# Phase-1 source selection only keeps a file when its best chunk scores at least
# this fraction of the strongest file's best chunk. A file whose evidence is
# only half as strong as the leader is likely noise, not a second source.
SOURCE_SELECTION_RATIO = float(os.getenv("SOURCE_SELECTION_RATIO", "0.6"))

# RapidOCR loads ONNX models into process memory. Keep it disabled on the
# 512 MB Render plan; enable it only on a larger instance or when using a
# dedicated OCR deployment.
ENABLE_PDF_OCR = os.getenv("ENABLE_PDF_OCR", "false").strip().lower() in {"1", "true", "yes"}

# Documents with reliable headings use a vectorless, PageIndex-style section
# tree. Other files continue through the established Qdrant hybrid pipeline.
# Defaults to off so a local run stays on the proven hybrid route; production
# opts in explicitly via render.yaml.
ENABLE_STRUCTURED_RETRIEVAL = os.getenv("ENABLE_STRUCTURED_RETRIEVAL", "false").strip().lower() in {"1", "true", "yes"}
STRUCTURE_SCORE_THRESHOLD = int(os.getenv("STRUCTURE_SCORE_THRESHOLD", "8"))
STRUCTURED_MAX_SECTIONS = int(os.getenv("STRUCTURED_MAX_SECTIONS", "6"))
