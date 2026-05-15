"""
Re-embed contextualized chunks for the contextual retrieval experiment.

Differs from the standard embedder:
  - Reads chunks_with_context.json (contains LLM-generated context per chunk)
  - Embeds the 'text_for_embedding' field (context + original chunk)
  - Saves to a separate file so we never overwrite baseline embeddings
"""

import json
import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings


INPUT_PATH = settings.PAPERS_PROCESSED_DIR / "chunks_with_context.json"
OUTPUT_PATH = settings.PAPERS_PROCESSED_DIR / "chunks_with_context_embeddings.json"


def embed_contextualized() -> None:
    if not INPUT_PATH.exists():
        print(f"⚠ {INPUT_PATH} not found. Run contextualizer first.")
        return

    print(f"📂 Loading contextualized chunks from {INPUT_PATH.name}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Loaded {len(chunks)} chunks")

    # Filter to only chunks that actually have context
    chunks = [c for c in chunks if c.get("text_for_embedding")]
    print(f"   {len(chunks)} chunks have contextualized text\n")

    print(f"📥 Loading embedding model: {settings.EMBEDDING_MODEL}")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    print(f"✅ Model loaded\n")

    texts = [c["text_for_embedding"] for c in chunks]

    print(f"🧮 Generating embeddings...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - start
    print(f"✅ {len(embeddings)} embeddings in {elapsed:.1f}s ({len(texts)/elapsed:.1f}/s)")

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    print(f"\n💾 Saving to {OUTPUT_PATH.name}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Saved {len(chunks)} embedded chunks ({size_mb:.1f} MB)")

    # Sanity check
    norm = np.linalg.norm(chunks[0]["embedding"])
    print(f"\n🔍 Sanity: L2 norm of chunk 0 = {norm:.4f} (should be ~1.0)")


if __name__ == "__main__":
    embed_contextualized()