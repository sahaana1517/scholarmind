"""
Extracts clean text from research paper PDFs.

Handles common PDF issues:
- Hyphenated words split across lines
- Excessive whitespace and page artifacts
- Page-level extraction with metadata preservation

Usage:
    python -m backend.app.ingestion.pdf_extractor
"""

import json
import re
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from tqdm import tqdm

from backend.app.core.config import settings


def clean_extracted_text(text: str) -> str:
    """Clean up common PDF extraction artifacts."""
    # Fix hyphenated words split across lines: "infor-\nmation" -> "information"
    text = re.sub(r"-\n", "", text)

    # Collapse multiple newlines into double newlines (paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Strip whitespace from each line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Remove leading/trailing whitespace
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> List[Dict]:
    """
    Extract text from a single PDF, one entry per page.

    Returns list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
            cleaned = clean_extracted_text(raw_text)

            # Skip pages with almost no text (likely figures/blank)
            if len(cleaned) < 100:
                continue

            pages.append({
                "page": page_num,
                "text": cleaned,
                "char_count": len(cleaned),
            })
        except Exception as e:
            print(f"  [WARN] Failed to extract page {page_num} of {pdf_path.name}: {e}")
            continue

    return pages


def extract_all_papers() -> None:
    """Extract text from every PDF in papers_raw, save JSON to papers_processed."""
    settings.PAPERS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(settings.PAPERS_RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠ No PDFs found in {settings.PAPERS_RAW_DIR}")
        return

    print(f"Found {len(pdf_files)} PDFs to process\n")

    successful = 0
    failed = 0
    total_pages = 0

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        paper_id = pdf_path.stem  # e.g., "2403.12345"
        output_path = settings.PAPERS_PROCESSED_DIR / f"{paper_id}.json"

        # Skip if already processed
        if output_path.exists():
            successful += 1
            continue

        try:
            pages = extract_pdf_pages(pdf_path)

            if not pages:
                print(f"\n  [SKIP] {pdf_path.name}: no extractable text")
                failed += 1
                continue

            paper_data = {
                "paper_id": paper_id,
                "filename": pdf_path.name,
                "num_pages": len(pages),
                "pages": pages,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(paper_data, f, ensure_ascii=False, indent=2)

            successful += 1
            total_pages += len(pages)

        except Exception as e:
            print(f"\n  [ERROR] {pdf_path.name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"✅ Successfully processed: {successful}/{len(pdf_files)} papers")
    print(f"📄 Total pages extracted:  {total_pages}")
    if failed:
        print(f"⚠ Failed: {failed}")
    print(f"📁 Output: {settings.PAPERS_PROCESSED_DIR}")


if __name__ == "__main__":
    extract_all_papers()