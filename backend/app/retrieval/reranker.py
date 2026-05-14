"""
Cross-encoder re-ranking layer.

Takes a list of candidate chunks (from hybrid retrieval) and re-orders
them using a cross-encoder model that scores (query, chunk) pairs directly.

This is a precision-focused stage:
- First stage (hybrid): fast, high recall, returns ~20 candidates
- This stage (reranker): slower but more accurate, returns top 5

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking
  - 22M parameters, runs fast on CPU
  - State-of-the-art quality for its size
"""

import time
from typing import List, Dict, Optional

from sentence_transformers import CrossEncoder

from backend.app.core.config import settings


# Lazy-loaded model (downloaded on first use, ~90 MB)
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder model."""
    global _reranker
    if _reranker is None:
        print(f"📥 Loading reranker model: {settings.RERANKER_MODEL}")
        _reranker = CrossEncoder(settings.RERANKER_MODEL)
        print(f"✅ Reranker ready")
    return _reranker


def rerank(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Re-rank candidates using a cross-encoder.

    Args:
        query: The user's question
        candidates: List of dicts (from hybrid_search), each must have 'text' field
        top_k: Number of top results to return after reranking

    Returns:
        Reranked list (length = top_k), with 'rerank_score' added to each entry
    """
    if not candidates:
        return []

    reranker = get_reranker()

    # Build (query, chunk_text) pairs for the cross-encoder
    pairs = [(query, c["text"]) for c in candidates]

    # Score all pairs in one batch (efficient)
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Attach scores and sort
    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    # Sort by rerank score descending
    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    # Add new ranks
    for new_rank, cand in enumerate(reranked, start=1):
        cand["rerank_position"] = new_rank

    return reranked[:top_k]


def pretty_print_reranked(query: str, results: List[Dict]) -> None:
    """Display reranked results with before/after rank info."""
    print(f"\n{'='*72}")
    print(f"🔍 Query: {query}")
    print(f"{'='*72}\n")

    for r in results:
        rerank_pos = r["rerank_position"]
        rerank_score = r["rerank_score"]
        original_rank = r.get("final_rank", "?")

        # Show rank movement
        if original_rank != "?":
            if original_rank > rerank_pos:
                movement = f"↑ {original_rank} → {rerank_pos}"
            elif original_rank < rerank_pos:
                movement = f"↓ {original_rank} → {rerank_pos}"
            else:
                movement = f"= {rerank_pos}"
        else:
            movement = f"#{rerank_pos}"

        print(f"--- Rerank #{rerank_pos} | Score: {rerank_score:.3f} | Movement: {movement} ---")
        print(f"📄 Paper: {r['paper_id']} | Page: {r['page']}")
        preview = r["text"][:300].replace("\n", " ")
        print(f"💬 {preview}...\n")


def interactive_reranked_search() -> None:
    """REPL: hybrid retrieval + reranking."""
    from backend.app.retrieval.hybrid_search import hybrid_search
    from backend.app.retrieval.search import get_model

    print("\n🎓 ScholarMind Reranked Hybrid Search")
    print("   Pipeline: hybrid (20 candidates) → cross-encoder → top 5")
    print("   Type a question (or 'quit' to exit)\n")

    # Warm up both models (first query is slow otherwise)
    print("🔥 Warming up models...")
    get_model()
    get_reranker()
    print()

    while True:
        try:
            query = input("Your question › ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("👋 Bye!")
            break

        start = time.time()

        # Stage 1: hybrid retrieval (broader candidate pool)
        candidates = hybrid_search(query, top_k=20, fetch_k=30)
        stage1_time = time.time() - start

        # Stage 2: rerank to top 5
        rerank_start = time.time()
        final = rerank(query, candidates, top_k=5)
        stage2_time = time.time() - rerank_start

        total = time.time() - start

        pretty_print_reranked(query, final)
        print(f"⏱  Total: {total*1000:.0f}ms "
              f"(hybrid: {stage1_time*1000:.0f}ms, rerank: {stage2_time*1000:.0f}ms)\n")


if __name__ == "__main__":
    interactive_reranked_search()