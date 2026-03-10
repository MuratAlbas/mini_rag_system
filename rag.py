import logging
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

from app.retrieval import load_faiss_index, search_index

logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_PATH = "vector_store/faiss.index"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

TOP_K = 3

# -----------------------------
# LOAD ONCE (important)
# -----------------------------
logger.info("Loading embedding model...")
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

logger.info("Loading FAISS index...")
index = load_faiss_index(FAISS_PATH)


# -----------------------------
# PROMPT TEMPLATE
# -----------------------------
PROMPT_TEMPLATE = """
Use the context below to answer the question.
If you don't know, say you don't know.

Context:
{context}

Question:
{question}
"""


# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve_chunks(question: str, top_k: int = TOP_K):
    logger.info("Embedding query...")

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    distances, indices = search_index(index, query_embedding, top_k=top_k)

    return distances, indices

import subprocess
import logging

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Use the context below to answer the question.
If you don't know, say you don't know.

Context:
{context}

Question:
{question}
"""


def ask_question(question: str, context: str) -> str:
    """Send prompt to Ollama and return answer."""

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )

        answer = result.stdout.strip()
        return answer

    except subprocess.CalledProcessError as e:
        logger.error(f"Ollama error: {e}")
        return "LLM error occurred."

# -----------------------------
# BUILD CONTEXT
# -----------------------------
def build_context(indices):
    context_parts = []

    # indices shape fix
    if isinstance(indices[0], (list, np.ndarray)):
        iterable_indices = indices[0]
    else:
        iterable_indices = indices

    for idx in iterable_indices:
        context_parts.append(f"[Chunk {idx}] Relevant document content here.")

    return "\n".join(context_parts)


# -----------------------------
# CALL OLLAMA
# -----------------------------
def call_ollama(prompt: str) -> str:
    logger.info("Calling Ollama...")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["response"]


# -----------------------------
# MAIN RAG PIPELINE
# -----------------------------
def answer_question(question: str) -> str:
    logger.info(f"User question: {question}")

    # retrieve
    distances, indices = retrieve_chunks(question)

    # build context
    context = build_context(indices)

    # prompt
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    # LLM
    answer = call_ollama(prompt)

    return answer


# -----------------------------
# CLI TEST
# -----------------------------
if __name__ == "__main__":
    q = "What is this document about?"
    result = answer_question(q)

    print("\n=== ANSWER ===")
    print(result)
