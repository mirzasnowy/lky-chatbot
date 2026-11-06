"""Download/cache the embedding model at build time (Render buildCommand).

The model lives in MODEL_CACHE_DIR (default <project>/.cache/fastembed) so
the runtime does not re-download it on every cold start. Files outside the
project folder may not survive between build and runtime.
"""
import os
from pathlib import Path

from fastembed import TextEmbedding

BASE = Path(__file__).resolve().parent.parent
cache = os.getenv("MODEL_CACHE_DIR", str(BASE / ".cache" / "fastembed"))
list(TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=cache).embed(["warmup"]))
print("model cached at", cache)
