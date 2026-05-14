"""
Generates embeddings for all chunked text using BGE-small model.

The model runs locally on CPU — no API calls, no costs.
First run downloads the model (~130 MB) and caches it locally.

Usage:
    python -m backend.app.ingestion.embedder
"""

import json
import time
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from backend.app.core.config import settings


def load_model() -> SentenceTransformer:
    """Load the BGE-small embedding model (downloads on first run)."""
    print(f"📥 Loading embedding model: {settings.EMBEDDING_MODEL}")
    print(f"   (First run downloads ~130 MB to local cache)")
    start = time.time()
    
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    elapsed = time.time() - start
    print(f"✅ Model loaded in {elapsed:.1f}s")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(chunks: List[dict], model: SentenceTransformer, batch_size: int = 32) -> List[dict]:
    """
    Generate embeddings for a list of chunks.

    Adds an 'embedding' field (list of floats) to each chunk dict.
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"\n🧮 Generating embeddings for {len(texts)} chunks (batch size {batch_size})...")
    start = time.time()

    # BGE models work better with the recommended prompt prefix for retrieval
    # See: https://huggingface.co/BAAI/bge-small-en-v1.5
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # critical: enables cosine similarity via dot product
        convert_to_numpy=True,
    )

    elapsed = time.time() - start
    print(f"✅ Embeddings generated in {elapsed:.1f}s ({len(texts)/elapsed:.1f} chunks/sec)")
    print(f"   Shape: {embeddings.shape}")

    # Attach embeddings back to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()  # JSON-serializable

    return chunks


def embed_all_chunks() -> None:
    """Load chunks, generate embeddings, save to disk."""
    chunks_path = settings.PAPERS_PROCESSED_DIR / "all_chunks.json"

    if not chunks_path.exists():
        print(f"⚠ Chunks file not found: {chunks_path}")
        print(f"   Run chunker first: python -m backend.app.ingestion.chunker")
        return

    print(f"📂 Loading chunks from {chunks_path.name}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Loaded {len(chunks)} chunks")

    model = load_model()
    chunks_with_embeddings = embed_chunks(chunks, model)

    # Save as a separate file (embeddings are large — keep raw chunks pristine)
    output_path = settings.PAPERS_PROCESSED_DIR / "chunks_with_embeddings.json"
    print(f"\n💾 Saving to {output_path.name}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_with_embeddings, f, ensure_ascii=False)

    # Size check
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved {len(chunks_with_embeddings)} embedded chunks ({file_size_mb:.1f} MB)")

    # Sanity check the first embedding
    first_emb = chunks_with_embeddings[0]["embedding"]
    norm = np.linalg.norm(first_emb)
    print(f"\n🔍 Sanity check (chunk 0):")
    print(f"   Vector length: {len(first_emb)}")
    print(f"   L2 norm: {norm:.4f} (should be ~1.0 since normalized)")


if __name__ == "__main__":
    embed_all_chunks()