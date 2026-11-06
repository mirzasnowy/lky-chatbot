# What Would Lee Kuan Yew Do? — a RAG chatbot

Section 3 of the 99 Group AI Aptitude Challenge. A chatbot that answers
questions in Lee Kuan Yew's voice, grounded ONLY in his own public speeches.
Built by Muhammad Mirza Kurniawan (mirzasnowy) for the challenge; the code,
prompts, corpus list and evaluation are all in this repo so the evaluator
can open everything.

## Corpus — 13 public speeches, 31,014 words (measured)

Full texts from the National Archives of Singapore (NAS), the Prime Minister's
Office Singapore (PMO), NTUC, and Australian PM Transcripts. 1965-2011.
Every chunk carries title, date, source and URL; see `corpus/corpus.json`
and the corpus list in `lky-corpus.md` (in the workspace, not the repo).

Honest limits, stated upfront:
- 13 speeches only. The 1970s and 1980s are thin (one speech each) because
  that is what exists as free full text; famous material like the 1983
  "graduate mothers" rally exists only as audio/video or excerpts.
- Two speeches carry flags: the 2003 SARS speech is labelled "Draft opening"
  by NAS, and the 1966 Geylang Serai speech is an English translation (likely
  delivered in Malay). The chatbot says so when it relies on them.
- NAS states "Viewing permitted. Use and reproduction require written
  permission from copyright owner." This demo retrieves short cited passages
  and does not republish full texts.
- The chatbot can only answer from these 13 speeches. If a question is not
  covered, it says so instead of inventing an answer.

## How it works

1. **ETL** (`src/etl.py`): the 13 texts are cleaned (press-release headers
   and annex tables dropped, words untouched) and split into 75 chunks of
   ~500 words with ~80-word overlap.
2. **Embeddings** (`src/embed.py`): BAAI/bge-small-en-v1.5, 384-dim, run
   locally with ONNX (fastembed). No database: vectors live in
   `corpus/embeddings.npy`.
3. **Retrieval** (`src/rag.py`): hybrid — dense cosine + BM25 (title indexed
   twice), fused by Reciprocal Rank Fusion (k=60), top-5 chunks. The same
   hybrid design as my thesis system NusaNara (dense + lexical, RRF).
4. **Generation**: the top-5 passages plus a strict system prompt
   (`prompts/system.md`) go to an OpenRouter model. Default:
   `google/gemma-4-26b-a4b-it:free` — verified free on 2026-11-06. There is
   currently NO free DeepSeek model on OpenRouter; the cheapest DeepSeek
   costs ~$0.0002 per answer. The model is an env var (`OPENROUTER_MODEL`),
   re-confirm the free list before a run.

## Retrieval evaluation (`eval/`)

12 questions, each with the expected speech. Run: `python eval/run_eval.py`.

| metric | value |
|---|---|
| Recall@3 | 0.92 |
| Recall@5 | 1.00 |
| MRR | 0.85 |

(n=12. Dense-only scored 2 misses in 5 probe questions before the fix:
the 1966 education speech ranked 4th on its own title question and the
2004 China speech ranked 6th on the China question. BM25 + RRF fixed both.)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # put OPENROUTER_API_KEY in .env, never commit it
```

Environment quirk (this host): httpx crashes on the bracketed-IPv6 entries
in the default NO_PROXY (`InvalidURL: Invalid port: ':1]'`). Run with
`NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1`. Unsetting all
proxy vars is NOT a fix for external downloads (TLS fails).

## Run

```bash
# ask one question (needs OPENROUTER_API_KEY)
python src/rag.py "How should Singapore handle immigration?" --k 5
# retrieval only, no key needed
python src/rag.py "How should Singapore handle immigration?" --retrieve-only
# API server
uvicorn src.app:app --port 8000
# retrieval eval
python eval/run_eval.py
```

## Example chats

`examples/` holds real transcripts produced by running this system
(question, retrieved chunks, model answer). Nothing in there is invented:
each file records the model, the retrieval scores and the sources cited.
The static demo page (`web/index.html`) renders them.

## What went wrong

- Dense-only retrieval missed the obvious speeches (see eval section). Fixed
  with BM25 + RRF and a title boost.
- Two NAS pages are behind an AWS WAF challenge for curl; their text was
  fetched through the browser reader instead.
- The PMO page returns 403 without a browser user-agent.
- `pip install` into the system Python failed on a Debian-owned
  `typing_extensions`; the project uses a venv.
- No free DeepSeek model exists on OpenRouter right now (verified
  2026-11-06 via the public /models list). The design switched to the free
  Gemma 4 26B model; the model id is an env var.

## What I would change

- More speeches, especially 1970s-1980s, if full texts become available.
- PostgreSQL + pgvector as the vector store (as in NusaNara) instead of a
  .npy file, once the corpus grows.
- A larger embedding model and a human-rated answer eval (groundedness,
  persona, citation accuracy) on top of the retrieval eval.
- RAG grounding for facts the speeches do not cover, instead of refusing.
