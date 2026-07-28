from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SPARSE_MODEL = "Qdrant/bm25"

print("Preloading dense model...")
dense = SentenceTransformer(DENSE_MODEL)
dense.encode(["warm up"], normalize_embeddings=True, show_progress_bar=False)

print("Preloading sparse model...")
sparse = SparseTextEmbedding(model_name=SPARSE_MODEL)
list(sparse.embed(["warm up"]))

print("Model preload complete.")
