from fastapi import FastAPI, UploadFile, File
import shutil
import os
import numpy as np

from app.ingest import (
    extract_text_from_pdf,
    clean_text,
    chunk_text,
    embed_chunks
)

from app.retrieval import (
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
    search_index
)

from app.rag import ask_question

app = FastAPI(title="Mini RAG API")

chunks_store = []
index = None


# -----------------------------
# INGEST ENDPOINT
# -----------------------------
@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    global chunks_store, index

    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"

    # save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # pipeline
    raw_text = extract_text_from_pdf(file_path)
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)

    embeddings = embed_chunks(chunks)

    index = create_faiss_index(embeddings)
    save_faiss_index(index, "vector_store/faiss.index")

    chunks_store = chunks

    return {
        "status": "ingested",
        "num_chunks": len(chunks)
    }


# -----------------------------
# QUERY ENDPOINT
# -----------------------------
@app.post("/query")
async def query_rag(question: str):
    global chunks_store, index

    if index is None or not chunks_store:
        return {"error": "No document indexed yet."}

    # embed query
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    distances, indices = search_index(index, query_embedding, top_k=3)

    # normalize shape
    indices = np.atleast_2d(indices)

    # build context + sources
    retrieved_chunks = []
    sources = []

    for idx in indices[0]:
        chunk = chunks_store[idx]
        retrieved_chunks.append(chunk["text"])
        sources.append(chunk["metadata"])

    context = "\n\n".join(retrieved_chunks)

    # LLM call
    answer = ask_question(question, context)

    return {
        "answer": answer,
        "sources": sources
    }
