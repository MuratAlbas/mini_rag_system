# Mini RAG System (Local LLM + FAISS)

End-to-end Retrieval Augmented Generation (RAG) pipeline that ingests PDF documents, indexes them with FAISS, and answers user questions using a local LLM via Ollama.

This project demonstrates production-minded AI engineering practices including modular architecture, caching, logging, and source attribution.

---

## Architecture

```
User Question
     ↓
Embed Query
     ↓
FAISS Search
     ↓
Top-k Chunks
     ↓
Prompt
     ↓
Local LLM (Ollama)
     ↓
Answer + Sources
```

---

## Tech Stack

* Python
* FastAPI
* FAISS
* Sentence-Transformers
* Ollama (Mistral 7B)
* NumPy
* PyPDF
* Uvicorn

---

# Example Workflow

1. Upload a PDF via `/ingest`
2. Ask a question via `/query`
3. System retrieves relevant chunks
4. Local LLM generates grounded answer

---

## Future Improvements

* Hybrid search (BM25 + vector)
* Docker support
* Evaluation pipeline
