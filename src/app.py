"""FastAPI server for the LKY chatbot (99 Group AI Challenge, Section 3).

Run:  uvicorn src.app:app --port 8000   (from the project root, venv active)
      .env must contain OPENROUTER_API_KEY (never commit it).
      The public demo is a static page on Cloudflare Pages; this server is
      the live backend it calls.
"""
import logging
import os
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import llm
from .rag import build_messages, get_model, load_index, retrieve

log = logging.getLogger("lky.app")

# One JSON line per answer goes to stdout (spec section 5.5). Uvicorn's
# default logging leaves the root logger at WARNING, which would swallow
# log.info calls, so give the lky loggers their own stdout handler.
_lky_log = logging.getLogger("lky")
_lky_log.setLevel(logging.INFO)
if not _lky_log.handlers:
    _lky_log.addHandler(logging.StreamHandler(sys.stdout))

RATE_LIMIT = 20
RATE_WINDOW = 600  # seconds

_hits: dict[str, deque] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model at startup so /health 200 means truly ready.
    get_model()
    corpus, vecs = load_index()
    app.state.corpus = corpus
    app.state.vecs = vecs
    yield


app = FastAPI(title="LKY chatbot (99 Group AI Challenge, Section 3)",
              lifespan=lifespan)

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
           if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)


class ChatIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=8)


def check_rate_limit(request: Request):
    ip = (request.headers.get("x-forwarded-for")
          or (request.client.host if request.client else "unknown"))
    ip = ip.split(",")[0].strip()
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limited",
                    "retry_after": int(RATE_WINDOW - (now - q[0]))})
    q.append(now)


def _sources(chunks):
    flags = ("is_translation", "is_draft", "is_qa", "is_letter", "is_toast",
             "is_partial_malay")
    return [{"chunk_id": c["chunk_id"], "title": c["title"], "date": c["date"],
             "source": c["source"], "url": c["url"],
             "score": round(c["score"], 4),
             "flags": {k: v for k, v in c.items() if k in flags and v}}
            for c in chunks]


@app.get("/health")
def health():
    corpus = getattr(app.state, "corpus", None)
    vecs = getattr(app.state, "vecs", None)
    if corpus is None or vecs is None:
        corpus, vecs = load_index()
    return {"status": "ok", "chunks": len(corpus),
            "embed_dim": int(vecs.shape[1]),
            "free_model": os.environ.get("OPENROUTER_MODEL",
                                         llm.OR_MODEL_DEFAULT),
            "paid_model": os.environ.get("VIKEY_MODEL",
                                         llm.VIKEY_MODEL_DEFAULT),
            "providers": {
                "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
                "vikey": bool(os.environ.get("VIKEY_API_KEY")
                             and os.environ.get("VIKEY_BASE_URL")),
            }}


@app.post("/chat")
def chat(body: ChatIn, request: Request):
    check_rate_limit(request)
    if not body.question.strip():
        raise HTTPException(400, "question is empty")
    t0 = time.monotonic()
    chunks = retrieve(body.question, k=body.k)
    t_retrieval = int((time.monotonic() - t0) * 1000)
    result = llm.generate(build_messages(body.question, chunks))
    total_ms = int((time.monotonic() - t0) * 1000)
    sources = _sources(chunks)
    meta = {
        "provider": result["provider"], "model": result["model"],
        "tier": result["tier"], "fallback_used": result["fallback_used"],
        "truncated": result["truncated"],
        "attempts": result["attempts"],
        "retrieval": {"k": body.k, "method": "hybrid-rrf",
                      "top_score": round(chunks[0]["score"], 4) if chunks else None},
        "usage": result["usage"],
        "est_cost_idr": result["est_cost_idr"],
        "latency_ms": {"retrieval": t_retrieval,
                       "generation": total_ms - t_retrieval,
                       "total": total_ms},
    }
    if not result["ok"]:
        meta["error"] = "generation_unavailable"
    llm.log_answer(result, meta["retrieval"]["top_score"],
                   [c["chunk_id"] for c in chunks], meta["latency_ms"])
    return {"question": body.question, "answer": result["answer"],
            "sources": sources, "meta": meta}


@app.get("/")
def root():
    return {"name": "LKY chatbot", "docs": "/docs",
            "note": "RAG over 33 public Lee Kuan Yew speeches (1959-2011). "
                    "Answers only from the retrieved passages, with citations."}
