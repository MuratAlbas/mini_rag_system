import os
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)


# -----------------------------
# create index
# -----------------------------
def create_faiss_index(embeddings: np.ndarray):
    """
    Create FAISS IndexFlatL2 and add embeddings.
    """
    dim = embeddings.shape[1]
    logger.info(f"Creating FAISS index with dim={dim}")

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    logger.info(f"FAISS index size: {index.ntotal}")
    return index


# -----------------------------
# save index
# -----------------------------
def save_faiss_index(index, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    logger.info(f"FAISS index saved to {path}")


# -----------------------------
# load index
# -----------------------------
def load_faiss_index(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index not found at {path}")

    index = faiss.read_index(path)
    logger.info("FAISS index loaded.")
    return index


# -----------------------------
# search
# -----------------------------
def search_index(index, query_embedding: np.ndarray, top_k: int = 5):
    """
    Perform similarity search.
    """
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    distances, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    return distances[0], indices[0]
