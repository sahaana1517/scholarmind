"""
Tools available to the ScholarMind agent.

Each tool is a focused operation the agent can choose to call.
Docstrings serve as tool descriptions — the planner LLM reads them to
decide which tool fits the query.

Design principle: tools return *structured data*, not free text. The agent
synthesizes the final answer from these structured returns.
"""

from typing import List, Dict

from backend.app.retrieval.hybrid_search import hybrid_search
from backend.app.retrieval.graph_query import graph_query as _graph_query_dispatch


def search_papers(query: str, top_k: int = 5) -> List[Dict]:
    """
    General-purpose semantic search over the research paper corpus.

    Use this for any straightforward question that asks about a topic,
    concept, technique, or paper. This is the default tool for most queries.

    Args:
        query: Natural language search query.
        top_k: Number of chunks to return (default 5).

    Returns:
        List of chunks, each with: paper_id, page, text, dense_score, sparse_score.
    """
    results = hybrid_search(query, top_k=top_k, fetch_k=30)
    cleaned = [
        {
            "paper_id": r["paper_id"],
            "page": r["page"],
            "text": r["text"],
        }
        for r in results
    ]
    return cleaned


def compare_papers(topic_a: str, topic_b: str, top_k_each: int = 3) -> Dict:
    """
    Compare two topics, concepts, or methods by retrieving evidence for each separately.

    Use this when the user wants a comparison, contrast, or side-by-side analysis.
    Example queries: "How does X differ from Y?", "Compare A and B", "X vs Y".

    Args:
        topic_a: The first topic / method / paper / concept to research.
        topic_b: The second topic / method / paper / concept to research.
        top_k_each: Chunks to retrieve per topic (default 3).

    Returns:
        Dict with keys 'topic_a' and 'topic_b', each containing a list of chunks.
    """
    chunks_a = search_papers(topic_a, top_k=top_k_each)
    chunks_b = search_papers(topic_b, top_k=top_k_each)
    return {
        "topic_a": {"query": topic_a, "chunks": chunks_a},
        "topic_b": {"query": topic_b, "chunks": chunks_b},
    }


def extract_methodology(method_or_paper: str, top_k: int = 6) -> List[Dict]:
    """
    Retrieve methodology / approach / technique descriptions for a specific
    method, system, or paper.

    Use this when the user asks specifically "HOW does X work?", "What is the
    methodology of Y?", or wants implementation/algorithmic details rather than
    background or results.

    Args:
        method_or_paper: Name or short description of the method/paper.
        top_k: Number of chunks to return (default 6).

    Returns:
        List of chunks biased toward methods/approach content.
    """
    method_biased_query = (
        f"methodology approach algorithm architecture implementation "
        f"of {method_or_paper}"
    )
    results = hybrid_search(method_biased_query, top_k=top_k, fetch_k=30)
    return [
        {
            "paper_id": r["paper_id"],
            "page": r["page"],
            "text": r["text"],
        }
        for r in results
    ]


def graph_query(
    intent: str,
    method: str = "",
    concept: str = "",
    arxiv_id: str = "",
) -> Dict:
    """
    Query the knowledge graph for relationship-based information about papers.

    Use this tool when the user wants to find papers based on RELATIONSHIPS
    rather than content — questions about which papers use a method, which
    study a concept, what methods a specific paper uses, or which papers are
    similar to another paper by shared methods/concepts.

    Args:
        intent: One of the following (REQUIRED):
            - "papers_using_method"        (also need: method)
            - "papers_studying_concept"    (also need: concept)
            - "methods_of_paper"           (also need: arxiv_id)
            - "concepts_of_paper"          (also need: arxiv_id)
            - "papers_similar_by_methods"  (also need: arxiv_id)
            - "papers_similar_by_concepts" (also need: arxiv_id)
        method: Method/technique name (e.g. "BM25", "transformer attention").
        concept: Concept/topic name (e.g. "retrieval", "knowledge graph").
        arxiv_id: arXiv ID like "2406.13249".

    Returns:
        Dict with 'intent' and 'results' (a list of papers or attributes).
    """
    # Pass only the relevant arg based on intent
    kwargs = {"intent": intent}
    if method:
        kwargs["method"] = method
    if concept:
        kwargs["concept"] = concept
    if arxiv_id:
        kwargs["arxiv_id"] = arxiv_id
    return _graph_query_dispatch(**kwargs)


# Registry — the planner uses this to look up tools by name
TOOL_REGISTRY = {
    "search_papers": search_papers,
    "compare_papers": compare_papers,
    "extract_methodology": extract_methodology,
    "graph_query": graph_query,
}


# Tool descriptions for the planner prompt
TOOL_DESCRIPTIONS = """
Available tools:

1. search_papers(query: str) -> list[chunks]
   - General-purpose semantic search over the corpus.
   - USE FOR: Any straightforward question about a topic, concept, or paper.
   - This is the DEFAULT tool. When unsure, use this.

2. compare_papers(topic_a: str, topic_b: str) -> dict
   - Retrieves evidence for two topics separately so they can be compared.
   - USE FOR: Comparison/contrast questions. ("How does X differ from Y?",
     "X vs Y", "compare A and B")

3. extract_methodology(method_or_paper: str) -> list[chunks]
   - Retrieves methodology/approach sections specifically.
   - USE FOR: Questions asking HOW something works, the algorithm, the
     architecture, the implementation details.

4. graph_query(intent: str, method?: str, concept?: str, arxiv_id?: str) -> dict
   - Queries the knowledge graph of papers, methods, concepts, authors.
   - USE FOR: RELATIONSHIP questions — finding papers BY criteria, not by content.
     Examples:
       "Which papers use BM25?"           -> intent=papers_using_method, method="BM25"
       "Which papers study knowledge graphs?" -> intent=papers_studying_concept, concept="knowledge graph"
       "What methods does paper 2406.13249 use?" -> intent=methods_of_paper, arxiv_id="2406.13249"
       "Find papers similar to 2502.01113"  -> intent=papers_similar_by_concepts, arxiv_id="2502.01113"
   - Available intents:
       papers_using_method        (method=...)
       papers_studying_concept    (concept=...)
       methods_of_paper           (arxiv_id=...)
       concepts_of_paper          (arxiv_id=...)
       papers_similar_by_methods  (arxiv_id=...)
       papers_similar_by_concepts (arxiv_id=...)
""".strip()


if __name__ == "__main__":
    # Quick sanity check of each tool
    print("Testing search_papers...")
    r = search_papers("What is RAG?", top_k=2)
    print(f"  ✅ returned {len(r)} chunks; first: {r[0]['paper_id']} p.{r[0]['page']}")

    print("\nTesting compare_papers...")
    r = compare_papers("R2AG", "GFM-RAG", top_k_each=2)
    print(f"  ✅ topic_a returned {len(r['topic_a']['chunks'])} chunks")
    print(f"  ✅ topic_b returned {len(r['topic_b']['chunks'])} chunks")

    print("\nTesting extract_methodology...")
    r = extract_methodology("Curator multi-tenant indexing", top_k=2)
    print(f"  ✅ returned {len(r)} chunks; first: {r[0]['paper_id']} p.{r[0]['page']}")

    print("\nTesting graph_query...")
    r = graph_query(intent="papers_using_method", method="transformer")
    print(f"  ✅ returned {len(r['results'])} graph results")

    print("\n🎉 All tools work")