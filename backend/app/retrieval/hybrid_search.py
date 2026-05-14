"""
Hybrid retrieval combining dense (semantic) and sparse (BM25) search
via Reciprocal Rank Fusion (RRF).

Why hybrid?
- Dense retrieval: captures semantic meaning, paraphrases, related concepts
- Sparse retrieval (BM25): captures exact keyword matches, acronyms, technical terms
- Together they outperform either alone on most benchmarks

RRF reference: Cormack et al. (2009),
"Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods"
"""

import time
from typing import List, Dict

from backend.app.core.config import settings
from backend.app.retrieval.search import search as dense_search
from backend.app.retrieval.bm25_index import search_bm25


# Standard RRF constant from the original paper (Cormack et al., 2009)
RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = RRF_K,
) -> List[Dict]:
    """
    Fuse two ranked lists into one using Reciprocal Rank Fusion.

    For each document appearing in either list, compute:
        rrf_score = sum over rankers of  1 / (k + rank)

    where rank is 1-indexed (best document has rank 1).
    """
    # We identify chunks by chunk_id (UUID) since both retrievers return it
    fused_scores: Dict[str, float] = {}
    chunk_lookup: Dict[str, Dict] = {}  # chunk_id -> metadata for output

    # Dense results
    for rank, hit in enumerate(dense_results, start=1):
        # Dense search returns paper_id, page, score, text — we need a stable key
        # Use (paper_id, page, text_hash) as a fallback since dense doesn't return chunk_id
        # Easier: re-fetch chunk_id from text content matching, OR use text as key
        key = (hit["paper_id"], hit["page"], hit["text"][:100])  # composite key
        key_str = f"{key[0]}|{key[1]}|{key[2]}"

        fused_scores[key_str] = fused_scores.get(key_str, 0.0) + 1.0 / (k + rank)
        chunk_lookup[key_str] = {
            "paper_id": hit["paper_id"],
            "page": hit["page"],
            "text": hit["text"],
            "dense_score": hit["score"],
            "dense_rank": rank,
            "sparse_score": None,
            "sparse_rank": None,
        }

    # Sparse results
    for rank, hit in enumerate(sparse_results, start=1):
        key_str = f"{hit['paper_id']}|{hit['page']}|{hit['text'][:100]}"

        fused_scores[key_str] = fused_scores.get(key_str, 0.0) + 1.0 / (k + rank)

        if key_str in chunk_lookup:
            # Chunk appeared in both — add sparse info
            chunk_lookup[key_str]["sparse_score"] = hit["score"]
            chunk_lookup[key_str]["sparse_rank"] = rank
        else:
            chunk_lookup[key_str] = {
                "paper_id": hit["paper_id"],
                "page": hit["page"],
                "text": hit["text"],
                "dense_score": None,
                "dense_rank": None,
                "sparse_score": hit["score"],
                "sparse_rank": rank,
            }

    # Sort by fused score (descending)
    ranked_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

    # Build final result list with RRF score attached
    final_results = []
    for rank, key in enumerate(ranked_keys, start=1):
        entry = chunk_lookup[key].copy()
        entry["rrf_score"] = fused_scores[key]
        entry["final_rank"] = rank
        final_results.append(entry)

    return final_results


def hybrid_search(query: str, top_k: int = 10, fetch_k: int = 20) -> List[Dict]:
    """
    Run both dense and sparse retrievers, fuse with RRF, return top_k.

    Args:
        query: Natural language question
        top_k: Number of final results to return
        fetch_k: How many to fetch from each retriever before fusion (larger = more
                 candidates considered; typically fetch_k > top_k)
    """
    # Run both retrievers
    dense_results = dense_search(query, top_k=fetch_k)
    sparse_results = search_bm25(query, top_k=fetch_k)

    # Fuse
    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    return fused[:top_k]


def pretty_print_hybrid(query: str, results: List[Dict]) -> None:
    """Display hybrid search results with ranker provenance."""
    print(f"\n{'='*72}")
    print(f"🔍 Query: {query}")
    print(f"{'='*72}\n")

    for r in results:
        rrf = r["rrf_score"]
        dense_rank = r["dense_rank"] if r["dense_rank"] else "—"
        sparse_rank = r["sparse_rank"] if r["sparse_rank"] else "—"
        dense_score = f"{r['dense_score']:.3f}" if r["dense_score"] is not None else "—"
        sparse_score = f"{r['sparse_score']:.2f}" if r["sparse_score"] is not None else "—"

        print(f"--- Rank {r['final_rank']} | RRF: {rrf:.4f} ---")
        print(f"📄 Paper: {r['paper_id']} | Page: {r['page']}")
        print(f"   Dense rank #{dense_rank} (score {dense_score})  |  "
              f"BM25 rank #{sparse_rank} (score {sparse_score})")
        preview = r["text"][:300].replace("\n", " ")
        print(f"💬 {preview}...\n")


def interactive_hybrid_search() -> None:
    """REPL for testing hybrid search interactively."""
    print("\n🎓 ScholarMind Hybrid Search (Dense + BM25 + RRF)")
    print("   Type a question (or 'quit' to exit)\n")

    # Warm up dense model (avoids cold-start delay on first query)
    from backend.app.retrieval.search import get_model
    get_model()

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
        results = hybrid_search(query, top_k=5, fetch_k=20)
        elapsed = time.time() - start

        pretty_print_hybrid(query, results)
        print(f"⏱  Retrieved in {elapsed*1000:.0f}ms\n")


if __name__ == "__main__":
    interactive_hybrid_search()