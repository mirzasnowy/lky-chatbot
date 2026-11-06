#!/usr/bin/env python3
"""Generate web/corpus.json from the real project data.

Reads corpus/corpus.json, corpus/corpus_stats.json and eval/results.json,
and writes the single data file the Corpus/About panels render from.
No number is typed by hand: every figure below is computed from those
files, except the model price table, which comes from Vikey's published
price list (2026-11-06) and is labelled as such.

Word counts, three honest numbers (they differ, and the About panel says
why): raw words in the original speech texts, clean words after ETL, and
words in the 142 chunk texts (higher, because chunks overlap ~80 words).

Usage: python3 scripts/build_info.py   (from the project root)
"""
import collections
import datetime
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model chain prices, Rp per 1M tokens, from Vikey's published price list
# (2026-11-06) and OpenRouter's /models list. Per-answer estimates assume
# ~2,000 input tokens (system prompt + 5 passages + question) and ~300
# output tokens. Re-check the lists before any run that depends on a price.
MODEL_CHAIN = [
    {"tier": "free", "provider": "openrouter",
     "model": "google/gemma-4-26b-a4b-it:free",
     "price_in_idr_per_million": 0, "price_out_idr_per_million": 0,
     "est_cost_idr_per_answer": 0.0,
     "note": "Free tier. The free list rotates and free models are rate "
             "limited per model; if it fails, the chain moves on."},
    {"tier": "paid", "provider": "vikey",
     "model": "deepseek/deepseek-v4-pro",
     "price_in_idr_per_million": 1800, "price_out_idr_per_million": 3600,
     "est_cost_idr_per_answer": 5.0,
     "note": "First paid fallback. About Rp 5 per answer."},
    {"tier": "paid", "provider": "vikey",
     "model": "deepseek/deepseek-v4-flash",
     "price_in_idr_per_million": 1080, "price_out_idr_per_million": 1800,
     "est_cost_idr_per_answer": 3.0,
     "note": "Second paid fallback. About Rp 3 per answer."},
]

FLAG_LABELS = {
    "is_draft": "NAS draft",
    "is_translation": "English translation (likely delivered in Malay)",
    "is_partial_malay": "Partly in Malay",
    "is_letter": "Letter, not a speech",
    "is_toast": "Short toast, not a full speech",
    "is_qa": "Press-conference Q&A transcript",
}

# Short source names as printed in the corpus. The raw values in
# corpus.json are abbreviations; the About panel shows these.
SOURCE_LABELS = {
    "NAS": "National Archives of Singapore",
    "PMO Singapore": "PMO Singapore",
    "NTUC": "NTUC",
    "Australian Government PM Transcripts": "Australian PM Transcripts",
}


def main():
    corpus = json.load(open(os.path.join(BASE, "corpus", "corpus.json"),
                            encoding="utf-8"))
    stats = json.load(open(os.path.join(BASE, "corpus", "corpus_stats.json"),
                           encoding="utf-8"))
    results = json.load(open(os.path.join(BASE, "eval", "results.json"),
                             encoding="utf-8"))

    speeches = stats["speeches"]
    total_raw = sum(s["raw_words"] for s in speeches)
    total_clean = sum(s["clean_words"] for s in speeches)
    total_chunk_words = sum(c["words"] for c in corpus)
    dates = [s["date"] for s in speeches]
    decades = collections.Counter()
    for d in dates:
        decades[f"{d[:3]}0s"] += 1

    # Flag counts are per SPEECH (a flagged speech has several chunks).
    flag_counts = collections.Counter()
    for s in speeches:
        sflags = {k for c in corpus
                  if c["speech_id"] == s["speech_id"]
                  for k in FLAG_LABELS if c.get(k)}
        for k in sflags:
            flag_counts[k] += 1

    by_speech = {}
    for c in corpus:
        by_speech.setdefault(c["speech_id"], []).append(c)
    speech_rows = []
    for s in speeches:
        flags = sorted({k for c in by_speech.get(s["speech_id"], [])
                        for k in FLAG_LABELS if c.get(k)})
        speech_rows.append({
            "speech_id": s["speech_id"],
            "title": s["title"],
            "date": s["date"],
            "source": by_speech[s["speech_id"]][0]["source"],
            "url": by_speech[s["speech_id"]][0]["url"],
            "words_raw": s["raw_words"],
            "words_clean": s["clean_words"],
            "n_chunks": s["n_chunks"],
            "flags": flags,
        })

    summary = results["summary"]
    weak_spots = []
    for row in results["rows"]:
        if not row["recall@3"]:
            weak_spots.append({
                "question": row["q"],
                "expected": row["expect"],
                "first_hit_rank": row["first_hit_rank"],
            })

    sources = sorted({SOURCE_LABELS.get(c["source"], c["source"])
                      for c in corpus})

    out = {
        "generated_by": "scripts/build_info.py",
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                        .isoformat(timespec="seconds"),
        "stats": {
            "speeches": len(speeches),
            "chunks": stats["total_chunks"],
            "total_words_raw": total_raw,
            "total_words_clean": total_clean,
            "total_chunk_words": total_chunk_words,
            "date_min": min(dates),
            "date_max": max(dates),
        },
        "decades": [{"decade": d, "count": decades[d]}
                    for d in sorted(decades)],
        "sources": sources,
        "flag_counts": {k: flag_counts.get(k, 0) for k in FLAG_LABELS},
        "flag_labels": FLAG_LABELS,
        "eval": {
            "n": summary["n"],
            "recall_at_3": round(summary["recall@3"], 2),
            "recall_at_5": round(summary["recall@5"], 2),
            "mrr": round(summary["mrr"], 2),
            "method": summary["method"],
            "weak_spots": weak_spots,
        },
        "model_chain": MODEL_CHAIN,
        "speeches": speech_rows,
    }
    path = os.path.join(BASE, "web", "corpus.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {path}")
    print(f"speeches={len(speeches)} chunks={stats['total_chunks']} "
          f"raw_words={total_raw} clean_words={total_clean} "
          f"chunk_words={total_chunk_words}")
    print(f"decades={dict(sorted(decades.items()))} "
          f"eval n={summary['n']} R@3={summary['recall@3']:.2f} "
          f"R@5={summary['recall@5']:.2f} MRR={summary['mrr']:.2f}")


if __name__ == "__main__":
    main()
