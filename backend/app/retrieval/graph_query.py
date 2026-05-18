"""
Knowledge graph queries over the paper corpus.

Exposes a small set of Cypher-backed queries that the agent can use for
relationship-style questions. Each query returns plain dicts so the
generator can synthesize answers from them.
"""

import re
from typing import List, Dict, Optional

from neo4j import GraphDatabase

from backend.app.core.config import settings


_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _driver


def _normalize(s: str) -> str:
    """Match how graph_loader normalized names (lowercase, strip parens)."""
    n = s.strip()
    n = re.sub(r"\s*\([^)]*\)\s*", " ", n).strip()
    n = n.lower()
    n = re.sub(r"\s+", " ", n)
    return n


# ── Individual query functions ──────────────────────────────────────────────

def papers_using_method(method_name: str, limit: int = 10) -> List[Dict]:
    """Find papers that use a given method/technique."""
    normalized = _normalize(method_name)
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (m:Method)
            WHERE toLower(m.name) CONTAINS $method
            MATCH (p:Paper)-[:USES_METHOD]->(m)
            RETURN p.arxiv_id AS arxiv_id, p.title AS title, m.name AS matched_method
            LIMIT $limit
            """,
            method=normalized, limit=limit,
        )
        return [dict(r) for r in result]


def papers_studying_concept(concept_name: str, limit: int = 10) -> List[Dict]:
    """Find papers that study a given concept."""
    normalized = _normalize(concept_name)
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS $concept
            MATCH (p:Paper)-[:STUDIES]->(c)
            RETURN p.arxiv_id AS arxiv_id, p.title AS title, c.name AS matched_concept
            LIMIT $limit
            """,
            concept=normalized, limit=limit,
        )
        return [dict(r) for r in result]


def methods_used_by_paper(arxiv_id: str) -> List[str]:
    """List all methods a specific paper uses."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Paper {arxiv_id: $arxiv_id})-[:USES_METHOD]->(m:Method)
            RETURN m.name AS method
            """,
            arxiv_id=arxiv_id,
        )
        return [r["method"] for r in result]


def concepts_studied_by_paper(arxiv_id: str) -> List[str]:
    """List all concepts a specific paper studies."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Paper {arxiv_id: $arxiv_id})-[:STUDIES]->(c:Concept)
            RETURN c.name AS concept
            """,
            arxiv_id=arxiv_id,
        )
        return [r["concept"] for r in result]


def papers_similar_by_methods(arxiv_id: str, limit: int = 5) -> List[Dict]:
    """
    Find papers most similar to the given paper by shared methods.

    Uses Cypher COUNT to rank by # of shared method nodes.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Paper {arxiv_id: $arxiv_id})-[:USES_METHOD]->(m:Method)
                <-[:USES_METHOD]-(other:Paper)
            WHERE other.arxiv_id <> $arxiv_id
            WITH other, count(m) AS shared_count,
                 collect(m.name) AS shared_methods
            RETURN other.arxiv_id AS arxiv_id,
                   other.title AS title,
                   shared_count,
                   shared_methods
            ORDER BY shared_count DESC
            LIMIT $limit
            """,
            arxiv_id=arxiv_id, limit=limit,
        )
        return [dict(r) for r in result]


def papers_similar_by_concepts(arxiv_id: str, limit: int = 5) -> List[Dict]:
    """Find papers most similar by shared concepts."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Paper {arxiv_id: $arxiv_id})-[:STUDIES]->(c:Concept)
                <-[:STUDIES]-(other:Paper)
            WHERE other.arxiv_id <> $arxiv_id
            WITH other, count(c) AS shared_count,
                 collect(c.name) AS shared_concepts
            RETURN other.arxiv_id AS arxiv_id,
                   other.title AS title,
                   shared_count,
                   shared_concepts
            ORDER BY shared_count DESC
            LIMIT $limit
            """,
            arxiv_id=arxiv_id, limit=limit,
        )
        return [dict(r) for r in result]


# ── Tool-facing dispatcher ─────────────────────────────────────────────────

def graph_query(
    intent: str,
    method: Optional[str] = None,
    concept: Optional[str] = None,
    arxiv_id: Optional[str] = None,
) -> Dict:
    """
    Single entry point the agent calls. Dispatches based on `intent`.

    Args:
        intent: One of:
            "papers_using_method"     — requires `method`
            "papers_studying_concept" — requires `concept`
            "methods_of_paper"         — requires `arxiv_id`
            "concepts_of_paper"        — requires `arxiv_id`
            "papers_similar_by_methods"  — requires `arxiv_id`
            "papers_similar_by_concepts" — requires `arxiv_id`
        method: Method name (for method-based queries).
        concept: Concept name (for concept-based queries).
        arxiv_id: arXiv ID like "2406.13249" (for paper-anchored queries).

    Returns:
        Dict with 'intent' echoed back, and 'results' (list of dicts).
    """
    if intent == "papers_using_method":
        if not method:
            raise ValueError("`method` is required for papers_using_method")
        return {"intent": intent, "method": method,
                "results": papers_using_method(method)}

    elif intent == "papers_studying_concept":
        if not concept:
            raise ValueError("`concept` is required for papers_studying_concept")
        return {"intent": intent, "concept": concept,
                "results": papers_studying_concept(concept)}

    elif intent == "methods_of_paper":
        if not arxiv_id:
            raise ValueError("`arxiv_id` is required for methods_of_paper")
        return {"intent": intent, "arxiv_id": arxiv_id,
                "results": methods_used_by_paper(arxiv_id)}

    elif intent == "concepts_of_paper":
        if not arxiv_id:
            raise ValueError("`arxiv_id` is required for concepts_of_paper")
        return {"intent": intent, "arxiv_id": arxiv_id,
                "results": concepts_studied_by_paper(arxiv_id)}

    elif intent == "papers_similar_by_methods":
        if not arxiv_id:
            raise ValueError("`arxiv_id` is required for papers_similar_by_methods")
        return {"intent": intent, "arxiv_id": arxiv_id,
                "results": papers_similar_by_methods(arxiv_id)}

    elif intent == "papers_similar_by_concepts":
        if not arxiv_id:
            raise ValueError("`arxiv_id` is required for papers_similar_by_concepts")
        return {"intent": intent, "arxiv_id": arxiv_id,
                "results": papers_similar_by_concepts(arxiv_id)}

    else:
        raise ValueError(f"Unknown intent: {intent}")


if __name__ == "__main__":
    # Smoke test each query type
    print("=== papers_using_method('transformer') ===")
    r = graph_query(intent="papers_using_method", method="transformer")
    for hit in r["results"][:5]:
        print(f"  - {hit['arxiv_id']}: {hit['title'][:60]}")
        print(f"    matched method: {hit['matched_method']}")

    print("\n=== papers_studying_concept('retrieval') ===")
    r = graph_query(intent="papers_studying_concept", concept="retrieval")
    for hit in r["results"][:5]:
        print(f"  - {hit['arxiv_id']}: {hit['title'][:60]}")

    print("\n=== methods_of_paper('2406.13249') (R2AG) ===")
    r = graph_query(intent="methods_of_paper", arxiv_id="2406.13249")
    print(f"  Methods: {r['results']}")

    print("\n=== papers_similar_by_concepts('2406.13249') ===")
    r = graph_query(intent="papers_similar_by_concepts", arxiv_id="2406.13249")
    for hit in r["results"]:
        print(f"  - {hit['arxiv_id']}: {hit['title'][:50]}")
        print(f"    shared ({hit['shared_count']}): {hit['shared_concepts']}")

    print("\n🎉 All graph queries work")