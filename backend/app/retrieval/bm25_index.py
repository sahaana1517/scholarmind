"""
Sparse retrieval using BM25 over the same chunk corpus.

BM25 is a classic keyword-based ranking function that complements
dense (semantic) retrieval — it catches exact term matches that
embeddings can miss (acronyms, rare terms, IDs).

Index is persisted to disk and reloaded for queries.
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple

from rank_bm25 import BM25Okapi

from backend.app.core.config import settings


# Where to persist the BM25 index
BM25_INDEX_PATH = settings.PROJECT_ROOT / "data" / "bm25_index.pkl"


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    Lowercases, splits on whitespace/punctuation, filters short tokens.
    """
    # Lowercase + split on non-word characters
    tokens = re.findall(r"\b\w+\b", text.lower())
    # Drop very short tokens (single letters, mostly noise)
    return [t for t in tokens if len(t) > 1]


def build_bm25_index() -> None:
    """
    Build a BM25 index over all chunks and persist to disk.

    Stores: (bm25_model, chunks_metadata) so queries can return full chunk info.
    """
    chunks_path = settings.PAPERS_PROCESSED_DIR / "all_chunks.json"

    if not chunks_path.exists():
        print(f"⚠ Chunks file not found: {chunks_path}")
        print(f"   Run chunker first.")
        return

    print(f"📂 Loading chunks from {chunks_path.name}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Loaded {len(chunks)} chunks")

    print(f"\n🔨 Tokenizing chunks...")
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

    # Compute average tokens per chunk for sanity check
    avg_tokens = sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus)
    print(f"   Avg tokens per chunk: {avg_tokens:.0f}")

    print(f"\n🔨 Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Store both the model and the chunk metadata side-by-side
    # (so retrieval can return chunk_id, paper_id, page, text)
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "paper_id": c["paper_id"],
            "page": c["page"],
            "text": c["text"],
        }
        for c in chunks
    ]

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "metadata": metadata}, f)

    size_mb = BM25_INDEX_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ BM25 index saved: {BM25_INDEX_PATH.name} ({size_mb:.1f} MB)")


def load_bm25_index() -> Tuple[BM25Okapi, List[Dict]]:
    """Load the persisted BM25 index from disk."""
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}. "
            f"Run `python -m backend.app.retrieval.bm25_index` first."
        )

    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    return data["bm25"], data["metadata"]


def search_bm25(query: str, top_k: int = 10) -> List[Dict]:
    """
    Sparse keyword search using BM25.

    Returns list of dicts with: chunk_id, paper_id, page, text, score
    """
    bm25, metadata = load_bm25_index()

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices (descending by score)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "chunk_id": metadata[idx]["chunk_id"],
            "paper_id": metadata[idx]["paper_id"],
            "page": metadata[idx]["page"],
            "text": metadata[idx]["text"],
        })

    return results


if __name__ == "__main__":
    build_bm25_index()