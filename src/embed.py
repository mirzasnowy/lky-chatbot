"""Embed the 75 corpus chunks with BAAI/bge-small-en-v1.5 (384-dim, local ONNX).
Output: corpus/embeddings.npy (75 x 384 float32, L2-normalised) + corpus/embed_meta.json
Model downloads once (~130MB) into ~/.cache/fastembed.
"""
import json, os, numpy as np
from fastembed import TextEmbedding

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "BAAI/bge-small-en-v1.5"

def _model_cache_dir():
    return os.environ.get(
        "MODEL_CACHE_DIR", os.path.join(BASE, ".cache", "fastembed"))

def main():
    corpus = json.load(open(os.path.join(BASE, "corpus/corpus.json"), encoding="utf-8"))
    texts = [c["text"] for c in corpus]
    print(f"embedding {len(texts)} chunks with {MODEL} ...")
    model = TextEmbedding(MODEL, cache_dir=_model_cache_dir())
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    np.save(os.path.join(BASE, "corpus/embeddings.npy"), vecs)
    json.dump({"model": MODEL, "dim": int(vecs.shape[1]), "n": int(vecs.shape[0]),
               "chunk_ids": [c["chunk_id"] for c in corpus]},
              open(os.path.join(BASE, "corpus/embed_meta.json"), "w"), indent=1)
    print(f"saved embeddings {vecs.shape}, dtype {vecs.dtype}")

if __name__ == "__main__":
    main()
