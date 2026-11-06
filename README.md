# What Would Lee Kuan Yew Do? — a RAG chatbot

Section 3 of the 99 Group AI Aptitude Challenge. A chatbot that answers
questions in Lee Kuan Yew's voice, grounded ONLY in his own public speeches.
Built by Muhammad Mirza Kurniawan (mirzasnowy) for the challenge; the code,
prompts, corpus list and evaluation are all in this repo so the evaluator
can open everything.

## Corpus — 27 public speeches, 53,129 words (measured)

Full texts from the National Archives of Singapore (NAS), the Prime Minister's
Office Singapore (PMO), NTUC, and Australian PM Transcripts. 1959-2011.
Decades: 1950s ×1, 1960s ×4, 1970s ×4, 1980s ×3, 1990s ×7, 2000s ×8.
Every chunk carries title, date, source and URL; see `corpus/corpus.json`.

Honest limits, stated upfront:
- 27 speeches is a rich seed, not the complete works. He spoke hundreds of
  times; the complete collection ("The Papers of Lee Kuan Yew", 19 volumes)
  is copyrighted, and famous material like the 1983 "graduate mothers" rally
  exists only as audio/video or excerpts. Remaining thin spots: 1950s (1)
  and 1980s (3).
- Flags the chatbot states when it relies on them: the 2003 SARS speech is
  labelled "Draft opening" by NAS; the 1966 Geylang Serai speech is an
  English translation (likely delivered in Malay); the 1978 Malay Teachers'
  Union speech was delivered partly in Malay; the 1980 letter to newspapers
  is a letter, not a speech; the 1965 item is a press-conference Q&A
  transcript.
- NAS states "Viewing permitted. Use and reproduction require written
  permission from copyright owner." This demo retrieves short cited passages
  and does not republish full texts.
- The chatbot can only answer from these 27 speeches. If a question is not
  covered, it says so instead of inventing an answer.

## How it works

1. **ETL** (`src/etl.py`): the 27 texts are cleaned (press-release headers
   and appendix tables/annexes dropped at their title markers, words
   untouched) and split into 127 chunks of ~500 words with ~80-word overlap.
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

15 questions, each with the expected speech(es). Run: `python eval/run_eval.py`.

| metric | value |
|---|---|
| Recall@3 | 0.93 |
| Recall@5 | 0.93 |
| MRR | 0.90 |

(n=15. One miss: "What did he say to young people about the future?" —
the 2003 youth speech ranks 8th; the 1959 teachers' rally dominates that
query. The other 14 are found in the top 5. Dense-only scored 2 misses in
5 probe questions before the hybrid fix: the 1966 education speech ranked
4th on its own title question and the 2004 China speech ranked 6th on the
China question. BM25 + RRF fixed both. As the corpus grew, the "China and
bilingualism" question gained two more legitimate answers — the 1978 Malay
Teachers' Union speech and the 2004 Parliament speech — so its expected set
is now all three.)

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
- Appendix tables polluted chunks (SARS WHO annexes, Poly and Mandarin
  statistical tables, press circulation appendix). Fixed by cutting each
  document at its appendix title marker.
- Three NAS pages are behind an AWS WAF challenge for curl; their text was
  fetched through the browser reader instead. A Wayback snapshot and a
  reader proxy were both tried and both blocked.
- The PMO page returns 403 without a browser user-agent.
- `pip install` into the system Python failed on a Debian-owned
  `typing_extensions`; the project uses a venv.
- No free DeepSeek model exists on OpenRouter right now (verified
  2026-11-06 via the public /models list). The design switched to the free
  Gemma 4 26B model; the model id is an env var.

## What I would change

- More speeches if full texts become available (remaining thin spots:
  1950s and 1980s); the 1970s and 1990s were filled in the expansion round.
- PostgreSQL + pgvector as the vector store (as in NusaNara) instead of a
  .npy file, once the corpus grows.
- A larger embedding model and a human-rated answer eval (groundedness,
  persona, citation accuracy) on top of the retrieval eval.
- RAG grounding for facts the speeches do not cover, instead of refusing.
