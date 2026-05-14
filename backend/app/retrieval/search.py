"""
Semantic search over the indexed paper corpus.

Given a natural language query, returns the top-K most relevant
chunks based on vector similarity (cosine).

Usage (interactive):
    python -m backend.app.retrieval.search
"""

import time
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from backend.app.core.config import settings


# Global model + client (loaded once)
_model: SentenceTransformer | None = None
_client: QdrantClient | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        print(f"📥 Loading embedding model...")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print(f"✅ Model ready")
    return _model


def get_client() -> QdrantClient:
    """Lazy-load the Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    return _client


def search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Find the top-K most relevant chunks for a natural language query.

    Returns list of dicts with: paper_id, page, score, text
    """
    model = get_model()
    client = get_client()

    # Embed the query (normalized for cosine)
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    # Search Qdrant (new query_points API in qdrant-client v1.10+)
    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    # query_points returns a QueryResponse object; hits are in .points
    formatted = []
    for hit in response.points:
        formatted.append({
            "score": float(hit.score),
            "paper_id": hit.payload.get("paper_id"),
            "page": hit.payload.get("page"),
            "text": hit.payload.get("text"),
        })

    return formatted


def pretty_print_results(query: str, results: List[Dict]) -> None:
    """Display search results in a readable format."""
    print(f"\n{'='*70}")
    print(f"🔍 Query: {query}")
    print(f"{'='*70}\n")

    for i, hit in enumerate(results, 1):
        print(f"--- Result {i} | Score: {hit['score']:.4f} ---")
        print(f"📄 Paper: {hit['paper_id']} | Page: {hit['page']}")
        # Show first 350 chars of the chunk
        preview = hit['text'][:350].replace('\n', ' ')
        print(f"💬 {preview}...")
        print()


def interactive_search() -> None:
    """REPL for testing search interactively."""
    print("\n🎓 ScholarMind Semantic Search")
    print("   Type a question (or 'quit' to exit)\n")

    # Warm up the model
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
        results = search(query, top_k=5)
        elapsed = time.time() - start

        pretty_print_results(query, results)
        print(f"⏱  Retrieved in {elapsed*1000:.0f}ms\n")


if __name__ == "__main__":
    interactive_search()