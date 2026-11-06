"""LKY chatbot: hybrid retrieval (dense bge-small-en-v1.5 + BM25, fused by RRF)
+ generation via the llm fallback chain (src/llm.py). No database: embeddings
live in a .npy file; BM25 runs in pure Python over the same chunks.
Same hybrid design as the NusaNara thesis system (dense + lexical, RRF).

Usage:
  python src/rag.py "your question" [--k 5] [--retrieve-only]
  OPENROUTER_API_KEY=... OPENROUTER_MODEL=... python src/rag.py "your question"
Env: OPENROUTER_API_KEY (required for the free tier, even for :free models),
     OPENROUTER_MODEL (default google/gemma-4-26b-a4b-it:free — verified free
     on 2026-11-06 22:23 WIB via the public /models list; the free list
     rotates, so re-confirm before a run. There is NO free DeepSeek model on
     OpenRouter right now: the cheapest DeepSeek (deepseek-v4-flash) costs
     ~$0.0002 per answer, so a $1 credit would cover thousands of answers,
     but the zero-budget path uses a :free model. Never guess a model ID.)
     See src/llm.py for the full fallback chain (OpenRouter free -> Vikey
     deepseek-v4-pro -> Vikey deepseek-v4-flash) and its env vars.
Proxy note: run with NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
(see TOOLS.md) or httpx crashes on the bracketed-IPv6 NO_PROXY entries.
"""
import argparse, json, math, os, re, sys
import numpy as np

STOP = set("a an the and or of to in on for with by at from as is are was were be been it its this that these those what how why when where who which do does did can could should would will shall may might i you he she we they him her his their our your my me us them not no yes if then than so such very just only also into over under about between through after before say said says lee kuan yew".split())
# Note: "lee kuan yew" and "say/said" are corpus stopwords: every question is
# about him, so the name does not discriminate between speeches, and "say" is
# noise. Dropping them is standard IR practice for a single-author corpus.

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM_PROMPT = open(os.path.join(BASE, "prompts/system.md"), encoding="utf-8").read()

_model = None
def _model_cache_dir():
    return os.environ.get(
        "MODEL_CACHE_DIR", os.path.join(BASE, ".cache", "fastembed"))

def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5",
                               cache_dir=_model_cache_dir())
    return _model

def load_index():
    corpus = json.load(open(os.path.join(BASE, "corpus/corpus.json"), encoding="utf-8"))
    vecs = np.load(os.path.join(BASE, "corpus/embeddings.npy"))
    return corpus, vecs

def _tokens(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]

def _bm25_scores(query: str, corpus):
    """Classic BM25 (k1=1.5, b=0.75) over title + chunk text. The title is
    indexed twice: a speech's title is its strongest topical signal (e.g.
    the query 'education and nation-building' is literally speech 01's title),
    and without the boost the long 500-word chunks lose to short ones."""
    docs = [_tokens(c["title"] + " " + c["title"] + " " + c["text"]) for c in corpus]
    n = len(docs)
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    avgdl = sum(len(d) for d in docs) / n
    q = _tokens(query)
    scores = np.zeros(n)
    for term in set(q):
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(term)
            if tf:
                scores[i] += idf * (tf * 2.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * len(d) / avgdl))
    return scores

def retrieve(query: str, k: int = 5):
    """Hybrid: dense cosine + BM25, fused by Reciprocal Rank Fusion (RRF k=60).
    Dense-only on this 75-chunk corpus bunched scores (0.58-0.67) and missed
    the title-matching speech on 2/5 probe questions; BM25 fixes exactly that."""
    corpus, vecs = load_index()
    q = list(get_model().embed(
        ["Represent this sentence for searching relevant passages: " + query]))[0]
    q = q / np.linalg.norm(q)
    dense = vecs @ q
    bm25 = _bm25_scores(query, corpus)
    dense_rank = {i: r for r, i in enumerate(np.argsort(-dense))}
    bm25_rank = {i: r for r, i in enumerate(np.argsort(-bm25))}
    fused = {i: 1 / (60 + dense_rank[i]) + 1 / (60 + bm25_rank[i]) for i in range(len(corpus))}
    top = sorted(fused, key=lambda i: -fused[i])[:k]
    return [{"score": float(fused[i]), "dense": float(dense[i]), "bm25": float(bm25[i]),
             **corpus[i]} for i in top]

def build_context(chunks):
    parts = []
    for c in chunks:
        tag = c["title"] + f" ({c['date']}, {c['source']})"
        if c.get("is_translation"):
            tag += " [English translation; not his delivered English wording]"
        if c.get("is_draft"):
            tag += " [NAS labels this a draft]"
        if c.get("is_qa"):
            tag += " [press-conference Q&A transcript]"
        if c.get("is_letter"):
            tag += " [letter to newspapers, not a speech]"
        if c.get("is_toast"):
            tag += " [short dinner toast/reply, not a full speech]"
        if c.get("is_partial_malay"):
            tag += " [delivered partly in Malay; this is the English transcript]"
        parts.append(f"--- Source: {tag} ---\n{c['text']}")
    return "\n\n".join(parts)

def build_messages(query: str, chunks):
    """System prompt + retrieved passages labelled [S1]..[S5] + question.
    The labels let the UI turn citations into clickable source cards."""
    parts = []
    for i, c in enumerate(chunks, 1):
        tag = c["title"] + f" ({c['date']}, {c['source']})"
        if c.get("is_translation"):
            tag += " [English translation; not his delivered English wording]"
        if c.get("is_draft"):
            tag += " [NAS labels this a draft]"
        if c.get("is_qa"):
            tag += " [press-conference Q&A transcript]"
        if c.get("is_letter"):
            tag += " [letter to newspapers, not a speech]"
        if c.get("is_toast"):
            tag += " [short dinner toast/reply, not a full speech]"
        if c.get("is_partial_malay"):
            tag += " [delivered partly in Malay; this is the English transcript]"
        parts.append(f"[S{i}] {tag}\n{c['text']}")
    user = ("Retrieved passages from Lee Kuan Yew's speeches:\n\n"
            + "\n\n".join(parts)
            + f"\n\nQUESTION\n{query}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--retrieve-only", action="store_true")
    args = ap.parse_args()
    chunks = retrieve(args.question, k=args.k)
    print(f"== retrieved top-{args.k} ==")
    for c in chunks:
        print(f"[{c['score']:.3f}] {c['chunk_id']} {c['title']} ({c['date']})")
    if args.retrieve_only:
        for c in chunks:
            print("\n---", c["chunk_id"], c["title"], f"({c['date']})", "---")
            print(c["text"][:600])
        return
    print("\n== answer ==\n")
    try:
        from . import llm as _llm
    except ImportError:
        import llm as _llm
    result = _llm.generate(build_messages(args.question, chunks))
    if result["ok"]:
        print(result["answer"])
        print(f"\n-- via {result['provider']} / {result['model']} "
              f"(fallback: {result['fallback_used']}) --")
    else:
        print("generation unavailable:",
              [a.get("error") for a in result["attempts"]])

if __name__ == "__main__":
    main()
