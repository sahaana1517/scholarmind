import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
"""
Compares retrieval methods on a golden eval set.

Methods compared:
  1. Dense only (BGE-small + Qdrant)
  2. Hybrid (dense + BM25 + RRF)
  3. Hybrid + Cross-encoder rerank

Metrics:
  - MRR (Mean Reciprocal Rank): how high is the first relevant paper?
  - Recall@K: fraction of relevant papers found in top K
  - Hit@1: did the #1 result contain a relevant paper?
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Set, Callable

from backend.app.retrieval.search import search as dense_search
from backend.app.retrieval.hybrid_search import hybrid_search
from backend.app.retrieval.reranker import rerank


EVAL_SET_PATH = Path("experiments/eval_set.json")
RESULTS_PATH = Path("experiments/eval_results.json")


def load_eval_set() -> List[Dict]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_retrieved_paper_ids(results: List[Dict]) -> List[str]:
    """
    Extract the ordered list of unique paper IDs from chunk-level results.

    A chunk-level result may have multiple chunks from the same paper;
    we count the paper once at the position of its first appearance.
    """
    seen: Set[str] = set()
    ordered: List[str] = []
    for r in results:
        pid = r.get("paper_id")
        if pid and pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """RR for a single query: 1/rank of first relevant item, or 0 if none."""
    for i, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant papers found in top-k retrieved."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def hit_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Did any relevant paper appear in top-k? Binary 0/1."""
    top_k = set(retrieved[:k])
    return 1.0 if top_k & relevant else 0.0


# ── Retriever wrappers (uniform interface) ────────────────────────────

def retrieve_dense(query: str, top_k_chunks: int = 20) -> List[Dict]:
    return dense_search(query, top_k=top_k_chunks)


def retrieve_hybrid(query: str, top_k_chunks: int = 20) -> List[Dict]:
    return hybrid_search(query, top_k=top_k_chunks, fetch_k=30)


def retrieve_hybrid_reranked(query: str, top_k_chunks: int = 20) -> List[Dict]:
    candidates = hybrid_search(query, top_k=top_k_chunks, fetch_k=30)
    return rerank(query, candidates, top_k=top_k_chunks)


# ── Evaluation loop ───────────────────────────────────────────────────

def evaluate_method(
    method_name: str,
    retrieve_fn: Callable[[str, int], List[Dict]],
    eval_set: List[Dict],
    top_k_chunks: int = 20,
) -> Dict:
    """Run one retrieval method over the entire eval set and aggregate metrics."""
    print(f"\n▶ Evaluating: {method_name}")

    per_query_results = []
    total_latency_ms = 0.0

    for item in eval_set:
        query = item["query"]
        relevant = set(item["relevant_papers"])

        start = time.time()
        chunks = retrieve_fn(query, top_k_chunks)
        latency_ms = (time.time() - start) * 1000
        total_latency_ms += latency_ms

        retrieved_papers = get_retrieved_paper_ids(chunks)

        per_query_results.append({
            "id": item["id"],
            "query": query,
            "relevant": list(relevant),
            "retrieved_top10": retrieved_papers[:10],
            "rr": reciprocal_rank(retrieved_papers, relevant),
            "recall@5": recall_at_k(retrieved_papers, relevant, 5),
            "recall@10": recall_at_k(retrieved_papers, relevant, 10),
            "hit@1": hit_at_k(retrieved_papers, relevant, 1),
            "hit@3": hit_at_k(retrieved_papers, relevant, 3),
            "latency_ms": latency_ms,
        })

    # Aggregate
    n = len(per_query_results)
    aggregate = {
        "method": method_name,
        "num_queries": n,
        "MRR": sum(r["rr"] for r in per_query_results) / n,
        "Recall@5": sum(r["recall@5"] for r in per_query_results) / n,
        "Recall@10": sum(r["recall@10"] for r in per_query_results) / n,
        "Hit@1": sum(r["hit@1"] for r in per_query_results) / n,
        "Hit@3": sum(r["hit@3"] for r in per_query_results) / n,
        "avg_latency_ms": total_latency_ms / n,
        "per_query": per_query_results,
    }

    print(f"  MRR:       {aggregate['MRR']:.3f}")
    print(f"  Recall@5:  {aggregate['Recall@5']:.3f}")
    print(f"  Recall@10: {aggregate['Recall@10']:.3f}")
    print(f"  Hit@1:     {aggregate['Hit@1']:.3f}")
    print(f"  Hit@3:     {aggregate['Hit@3']:.3f}")
    print(f"  Avg latency: {aggregate['avg_latency_ms']:.0f}ms")

    return aggregate


def main():
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} eval queries\n")

    # Warm up models so first-query latency doesn't skew results
    print("🔥 Warming up models...")
    from backend.app.retrieval.search import get_model
    from backend.app.retrieval.reranker import get_reranker
    get_model()
    get_reranker()

    methods = [
        ("dense_only", retrieve_dense),
        ("hybrid", retrieve_hybrid),
        ("hybrid_reranked", retrieve_hybrid_reranked),
    ]

    all_results = {}
    for name, fn in methods:
        all_results[name] = evaluate_method(name, fn, eval_set)

    # Comparative summary
    print(f"\n{'='*72}")
    print(f"{'Method':<22} {'MRR':>8} {'Recall@5':>10} {'Recall@10':>11} {'Hit@1':>8} {'Hit@3':>8} {'Lat(ms)':>10}")
    print(f"{'-'*72}")
    for name, r in all_results.items():
        print(f"{name:<22} {r['MRR']:>8.3f} {r['Recall@5']:>10.3f} {r['Recall@10']:>11.3f} "
              f"{r['Hit@1']:>8.3f} {r['Hit@3']:>8.3f} {r['avg_latency_ms']:>10.0f}")

    # Persist results for the README/blog
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved detailed results to {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()