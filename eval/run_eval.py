"""Retrieval eval: 12 questions, each with the expected speech id(s).
Metrics: Recall@3, Recall@5, MRR (rank of first expected chunk).
Run: python eval/run_eval.py  (needs the model cache; no API key needed)
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rag import retrieve

BASE = os.path.expanduser("~/workspace/aic-section1/lky-chatbot")

def main():
    qs = json.load(open(os.path.join(BASE, "eval/questions.json")))
    rows, r3 = [], 0
    r5, mrr = 0, 0.0
    for item in qs:
        hits = retrieve(item["q"], k=5)
        got = [h["speech_id"] for h in hits]
        rank = next((i + 1 for i, s in enumerate(got) if s in item["expect"]), None)
        ok3, ok5 = bool(rank and rank <= 3), bool(rank)
        r3 += ok3; r5 += ok5; mrr += (1 / rank) if rank else 0
        rows.append({"q": item["q"], "expect": item["expect"], "top5": got,
                     "first_hit_rank": rank, "recall@3": ok3, "recall@5": ok5})
        print(f"[{'OK ' if ok5 else 'MISS'}] rank={rank} expect={item['expect']} top5={got} :: {item['q'][:70]}")
    n = len(qs)
    summary = {"n": n, "recall@3": r3 / n, "recall@5": r5 / n, "mrr": mrr / n,
               "method": "hybrid dense (bge-small-en-v1.5) + BM25, RRF k=60, top-5"}
    print(f"\nRecall@3={summary['recall@3']:.2f}  Recall@5={summary['recall@5']:.2f}  MRR={summary['mrr']:.2f}  (n={n})")
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(BASE, "eval/results.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
