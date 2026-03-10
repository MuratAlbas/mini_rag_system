import requests
import logging

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


def build_prompt(context: str, question: str) -> str:
    """Create RAG prompt."""
    prompt = f"""
    Use the context below to answer the question.
    If you don't know, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """
    return prompt.strip()


def generate_answer(context: str, question: str) -> str:
    """Call Ollama local LLM."""
    prompt = build_prompt(context, question)

    logger.info("Sending request to Ollama...")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    return data["response"]
