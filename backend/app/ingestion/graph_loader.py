"""
Load extracted entities into Neo4j as a knowledge graph.

Schema:
    (:Paper {arxiv_id, title})
       -[:USES_METHOD]->   (:Method {name})
       -[:STUDIES]->       (:Concept {name})
       -[:AUTHORED_BY]->   (:Author {name})
       -[:CITES]->         (:Paper {arxiv_id})

Name normalization:
    Methods and Concepts get lowercased and have parentheticals stripped,
    so "LLMs" and "Large Language Models (LLMs)" merge to "large language models".

Idempotent: uses MERGE everywhere, so re-running won't create duplicates.
"""

import json
import re
import time
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm

from backend.app.core.config import settings


ENTITIES_PATH = settings.PAPERS_PROCESSED_DIR / "entities.json"


def normalize_name(name: str) -> str:
    """Standardize an entity name for de-duplication."""
    n = name.strip()
    # Strip parenthetical aliases like "Large Language Models (LLMs)" -> "Large Language Models"
    n = re.sub(r"\s*\([^)]*\)\s*", " ", n).strip()
    # Lowercase everything for consistent matching
    n = n.lower()
    # Collapse repeated whitespace
    n = re.sub(r"\s+", " ", n)
    return n


def get_driver():
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def setup_constraints(driver):
    """Create uniqueness constraints (idempotent)."""
    with driver.session() as session:
        constraints = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
            "CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
        ]
        for c in constraints:
            session.run(c)
    print("✅ Constraints created")


def wipe_graph(driver):
    """Delete all nodes and relationships (called at the start for clean re-load)."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("🧹 Wiped existing graph")


def load_paper(session, entity: dict):
    """Load one paper and all its relationships."""
    arxiv_id = entity["paper_id"]
    title = entity.get("title", "")

    # Paper node
    session.run(
        "MERGE (p:Paper {arxiv_id: $arxiv_id}) SET p.title = $title",
        arxiv_id=arxiv_id, title=title,
    )

    # Authors
    for author in entity.get("authors", []):
        name = author.strip()
        if not name:
            continue
        session.run(
            """
            MERGE (a:Author {name: $name})
            WITH a
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:AUTHORED_BY]->(a)
            """,
            name=name, arxiv_id=arxiv_id,
        )

    # Methods
    for method in entity.get("methods_used", []):
        n = normalize_name(method)
        if not n:
            continue
        session.run(
            """
            MERGE (m:Method {name: $name})
            WITH m
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:USES_METHOD]->(m)
            """,
            name=n, arxiv_id=arxiv_id,
        )

    # Concepts
    for concept in entity.get("concepts_studied", []):
        n = normalize_name(concept)
        if not n:
            continue
        session.run(
            """
            MERGE (c:Concept {name: $name})
            WITH c
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:STUDIES]->(c)
            """,
            name=n, arxiv_id=arxiv_id,
        )

    # Citations
    for cited_id in entity.get("papers_cited", []):
        session.run(
            """
            MERGE (cited:Paper {arxiv_id: $cited_id})
            WITH cited
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:CITES]->(cited)
            """,
            cited_id=cited_id, arxiv_id=arxiv_id,
        )


def load_all_entities():
    if not ENTITIES_PATH.exists():
        print(f"⚠ {ENTITIES_PATH} not found. Run entity_extractor first.")
        return

    print(f"📂 Loading entities from {ENTITIES_PATH.name}")
    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        entities = json.load(f)
    print(f"   {len(entities)} papers to load")

    driver = get_driver()

    print("\n🔌 Verifying Neo4j connection...")
    driver.verify_connectivity()
    print("✅ Connected")

    wipe_graph(driver)
    setup_constraints(driver)

    print(f"\n📤 Loading {len(entities)} papers into graph...")
    start = time.time()
    with driver.session() as session:
        for entity in tqdm(entities, desc="Loading papers"):
            load_paper(session, entity)
    elapsed = time.time() - start
    print(f"✅ Load complete in {elapsed:.1f}s")

    # Quick stats
    print(f"\n📊 Graph statistics:")
    with driver.session() as session:
        for label in ["Paper", "Method", "Concept", "Author"]:
            r = session.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()
            print(f"   {label}s:  {r['n']}")

        r = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()
        print(f"   Relationships: {r['n']}")

    # Sample query: most-studied concepts
    print(f"\n🔍 Top 5 most-studied concepts:")
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (c:Concept)<-[:STUDIES]-(p:Paper)
            RETURN c.name AS concept, count(p) AS num_papers
            ORDER BY num_papers DESC LIMIT 5
            """
        )
        for row in rows:
            print(f"   {row['num_papers']:>2}x  {row['concept']}")

    driver.close()


if __name__ == "__main__":
    load_all_entities()