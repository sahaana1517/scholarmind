"""
Side-by-side qualitative comparison: baseline retrieval vs contextual retrieval.

Runs identical queries against both Qdrant collections and prints results
in a paired format for visual inspection.

We use dense-only search for both (apples-to-apples — same retriever,
different embeddings).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from backend.app.core.config import settings


BASELINE_COLLECTION = settings.QDRANT_COLLECTION_NAME  # scholarmind_papers
CONTEXTUAL_COLLECTION = "scholarmind_papers_contextual"


def make_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=60,
    )


def search_collection(client, model, collection_name, query, top_k=5):
    """Dense-only search against a specific collection."""
    qv = model.encode(query, normalize_embeddings=True, convert_to_numpy=True).tolist()
    response = client.query_points(
        collection_name=collection_name,
        query=qv,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "score": float(hit.score),
            "paper_id": hit.payload.get("paper_id"),
            "page": hit.payload.get("page"),
            "text": hit.payload.get("text"),
            "context": hit.payload.get("context"),  # only contextual collection has this
        }
        for hit in response.points
    ]


def print_side_by_side(query: str, baseline, contextual):
    print(f"\n{'='*78}")
    print(f"QUERY: {query}")
    print(f"{'='*78}\n")

    max_results = max(len(baseline), len(contextual))

    for i in range(max_results):
        print(f"--- Rank {i+1} ---")

        if i < len(baseline):
            b = baseline[i]
            print(f"  BASELINE      ({b['score']:.3f}): "
                  f"{b['paper_id']} p.{b['page']}")
            print(f"    {b['text'][:160].replace(chr(10), ' ')}...")
        else:
            print(f"  BASELINE      : (no result)")

        if i < len(contextual):
            c = contextual[i]
            print(f"  CONTEXTUAL    ({c['score']:.3f}): "
                  f"{c['paper_id']} p.{c['page']}")
            print(f"    {c['text'][:160].replace(chr(10), ' ')}...")
        else:
            print(f"  CONTEXTUAL    : (no result)")

        print()


def main():
    print("📥 Loading model...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    client = make_client()
    print("✅ Ready\n")

    # Three queries chosen to test where context matters most
    # Each targets a paper we have in the contextual subset
    queries = [
        # Query 1: highly paraphrased, tests semantic match
        "How do you efficiently isolate vector indices when many customers share infrastructure?",
        # Query 2: tests whether context helps surface non-obvious chunks
        "Can autonomous AI systems exploit web application vulnerabilities without human guidance?",
        # Query 3: ambiguous reference to a figure / table
        "What is the scalability of Curator with respect to dataset size?",
    ]

    for q in queries:
        baseline = search_collection(client, model, BASELINE_COLLECTION, q, top_k=5)
        contextual = search_collection(client, model, CONTEXTUAL_COLLECTION, q, top_k=5)
        print_side_by_side(q, baseline, contextual)

    print(f"\n{'='*78}")
    print(f"Note: contextual collection has only 200 chunks (10 papers).")
    print(f"Baseline has all 1103 chunks (30 papers). Comparisons valid only")
    print(f"when relevant paper is among the 10 contextualized papers.")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()