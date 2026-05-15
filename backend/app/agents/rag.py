"""
End-to-end Retrieval-Augmented Generation pipeline.

Combines:
  - Hybrid retrieval (dense + BM25 + RRF) for high-recall candidate fetching
  - Cross-encoder reranking (optional, off by default based on eval findings)
  - LLM answer generation with grounded citations

Single entry point: `answer_question(query)` returns the answer + sources + diagnostics.
"""

import time
from typing import Dict, List

from backend.app.retrieval.hybrid_search import hybrid_search
from backend.app.retrieval.reranker import rerank
from backend.app.agents.generator import generate_answer


def answer_question(
    query: str,
    top_k_retrieval: int = 5,
    use_reranker: bool = False,
    fetch_k: int = 30,
) -> Dict:
    """
    End-to-end RAG: retrieve relevant chunks, generate cited answer.

    Args:
        query: Natural language question
        top_k_retrieval: How many chunks to feed into the generator (default 5).
            More chunks = more context but slower + more tokens.
        use_reranker: Whether to apply cross-encoder reranking.
            Default False based on eval findings (reranker degrades MRR on our corpus).
        fetch_k: Candidates to fetch from hybrid before optional rerank.

    Returns:
        Dict with:
            answer: str          — generated answer text
            sources: list[dict]  — citation list (paper_id, page, preview)
            chunks_used: list    — full chunks fed to the generator
            timings: dict        — latency breakdown
            metadata: dict       — LLM call stats (tokens, model)
    """
    timings = {}

    # Stage 1: hybrid retrieval
    t0 = time.time()
    if use_reranker:
        candidates = hybrid_search(query, top_k=fetch_k, fetch_k=fetch_k)
        timings["retrieval_ms"] = (time.time() - t0) * 1000

        # Stage 2: rerank to top_k
        t1 = time.time()
        chunks = rerank(query, candidates, top_k=top_k_retrieval)
        timings["rerank_ms"] = (time.time() - t1) * 1000
    else:
        chunks = hybrid_search(query, top_k=top_k_retrieval, fetch_k=fetch_k)
        timings["retrieval_ms"] = (time.time() - t0) * 1000
        timings["rerank_ms"] = 0.0

    # Stage 3: answer generation
    t2 = time.time()
    answer, sources, gen_metadata = generate_answer(query, chunks)
    timings["generation_ms"] = (time.time() - t2) * 1000
    timings["total_ms"] = sum(timings.values())

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": chunks,
        "timings": timings,
        "metadata": gen_metadata,
    }


def pretty_print_result(result: Dict) -> None:
    """Display a RAG result in human-readable form."""
    print("\n" + "=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(result["answer"])

    print("\n" + "-" * 72)
    print("SOURCES")
    print("-" * 72)
    for src in result["sources"]:
        print(f"  [{src['index']}] Paper {src['paper_id']}, p.{src['page']}")
        print(f"      {src['preview']}...")

    print("\n" + "-" * 72)
    print("PIPELINE TIMINGS")
    print("-" * 72)
    t = result["timings"]
    print(f"  Retrieval:  {t['retrieval_ms']:>7.0f}ms")
    if t["rerank_ms"]:
        print(f"  Reranking:  {t['rerank_ms']:>7.0f}ms")
    print(f"  Generation: {t['generation_ms']:>7.0f}ms")
    print(f"  TOTAL:      {t['total_ms']:>7.0f}ms")

    m = result["metadata"]
    print(f"\n  Model: {m['model']}")
    print(f"  Tokens: {m['total_tokens']} (prompt {m['prompt_tokens']}, completion {m['completion_tokens']})")


def interactive_rag() -> None:
    """REPL for end-to-end RAG."""
    print("\n🎓 ScholarMind RAG")
    print("   Hybrid retrieval → LLM answer with citations")
    print("   Type a question (or 'quit' to exit)\n")

    # Warm up dense embedding model (avoid cold-start on first query)
    from backend.app.retrieval.search import get_model
    get_model()

    while True:
        try:
            query = input("\n❓ Your question › ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("👋 Bye!")
            break

        try:
            result = answer_question(query, top_k_retrieval=5)
            pretty_print_result(result)
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    interactive_rag()