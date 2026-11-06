"""FastAPI server for the LKY chatbot. Local use only; the public demo is a
static page with pre-generated example chats (see web/).

Run:  uvicorn src.app:app --port 8000   (from the project root, venv active)
      .env must contain OPENROUTER_API_KEY (never commit it).
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import retrieve, generate, load_index

app = FastAPI(title="LKY chatbot (99 Group AI Challenge, Section 3)")

class ChatIn(BaseModel):
    question: str
    k: int = 5

@app.get("/health")
def health():
    corpus, vecs = load_index()
    return {"status": "ok", "chunks": len(corpus),
            "embed_dim": int(vecs.shape[1]),
            "model": os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")}

@app.post("/chat")
def chat(body: ChatIn):
    if not body.question.strip():
        raise HTTPException(400, "question is empty")
    chunks = retrieve(body.question, k=body.k)
    try:
        answer = generate(body.question, chunks)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    sources = [{"chunk_id": c["chunk_id"], "title": c["title"], "date": c["date"],
                "source": c["source"], "url": c["url"], "score": round(c["score"], 3),
                "flags": {k: v for k, v in c.items() if k in ("is_translation", "is_draft", "is_qa") and v}}
               for c in chunks]
    return {"question": body.question, "answer": answer, "sources": sources}

@app.get("/")
def root():
    return {"name": "LKY chatbot", "docs": "/docs",
            "note": "RAG over 13 public Lee Kuan Yew speeches (1965-2011). "
                    "Answers only from the retrieved passages, with citations."}
