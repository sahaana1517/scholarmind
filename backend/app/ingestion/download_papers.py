"""
Downloads research papers from arXiv for the ScholarMind corpus.

Usage:
    python -m backend.app.ingestion.download_papers
"""

import arxiv
from pathlib import Path
import time

OUTPUT_DIR = Path("data/papers_raw")

# More specific queries with arXiv's category filter for CS papers only
QUERIES = [
    'all:"retrieval augmented generation" AND cat:cs.CL',
    'all:"LLM agents" AND cat:cs.AI',
    'all:"vector database" AND cat:cs.IR',
    'all:"transformer attention" AND cat:cs.LG',
    'all:"knowledge graph" AND cat:cs.AI',
]

PAPERS_PER_QUERY = 6


def download_papers():
    """Fetch papers from arXiv across multiple AI/ML topics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for query in QUERIES:
        print(f"\n=== Searching: {query} ===")

        search = arxiv.Search(
            query=query,
            max_results=PAPERS_PER_QUERY,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for paper in search.results():
            paper_id = paper.entry_id.split("/")[-1].split("v")[0]
            filename = f"{paper_id}.pdf"
            filepath = OUTPUT_DIR / filename

            if filepath.exists():
                print(f"  [SKIP] {paper.title[:60]}...")
                continue

            try:
                print(f"  [DOWNLOAD] {paper.title[:70]}...")
                paper.download_pdf(dirpath=str(OUTPUT_DIR), filename=filename)
                total_downloaded += 1
                time.sleep(1)
            except Exception as e:
                print(f"  [ERROR] {e}")

    print(f"\n✅ Downloaded {total_downloaded} new papers")
    print(f"📁 Saved to: {OUTPUT_DIR.resolve()}")

    all_pdfs = list(OUTPUT_DIR.glob("*.pdf"))
    print(f"📚 Total corpus: {len(all_pdfs)} papers")


if __name__ == "__main__":
    download_papers()