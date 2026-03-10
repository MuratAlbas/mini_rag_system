import sys
import logging
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.retrieval import (
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
    search_index,
)

# -----------------------------
# logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# load embedding model (global)
# -----------------------------
logger.info("Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Embedding model loaded.")

# -----------------------------
# PDF -> raw text
# -----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    logger.info(f"Reading PDF: {pdf_path}")

    reader = PdfReader(pdf_path)
    text_parts = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        logger.info(f"Page {i+1} extracted ({len(page_text)} chars)")
        text_parts.append(page_text)

    full_text = "\n".join(text_parts)
    logger.info(f"Total extracted length: {len(full_text)}")
    return full_text

# -----------------------------
# simple text cleaning
# -----------------------------
def clean_text(text: str) -> str:
    cleaned = text.replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned

# -----------------------------
# chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100, page_number: int = 1):
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        chunks.append({
            "text": chunk_str,
            "metadata": {
                "page_number": page_number,
                "chunk_id": chunk_id
            }
        })

        chunk_id += 1
        start += chunk_size - overlap

    return chunks

# -----------------------------
# embedding
# -----------------------------
def embed_chunks(chunks: list):
    texts = [chunk["text"] for chunk in chunks]

    logger.info("Generating embeddings...")

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings

# -----------------------------
# CLI entrypoint
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        logger.error("PDF file not found!")
        sys.exit(1)

    # extract
    raw_text = extract_text_from_pdf(pdf_path)

    # clean
    cleaned_text = clean_text(raw_text)
    logger.info("Text cleaning completed.")

    # chunk
    logger.info("Starting chunking...")
    chunks = chunk_text(cleaned_text)
    logger.info(f"Total chunks created: {len(chunks)}")

    # embed
    embeddings = embed_chunks(chunks)
    logger.info(f"Embedding vector size: {len(embeddings[0])}")

    # FAISS build
    index = create_faiss_index(embeddings)
    save_faiss_index(index, "vector_store/faiss.index")

    # TEST SEARCH
    logger.info("Running test query...")

    query = "What is this document about?"
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    loaded_index = load_faiss_index("vector_store/faiss.index")
    distances, indices = search_index(loaded_index, query_embedding, top_k=3)

    print("\n--- SEARCH RESULTS ---")
    print("indices:", indices)
    print("distances:", distances)


if __name__ == "__main__":
    main()
