"""
Splits extracted paper text into semantically coherent chunks for embedding.

Strategy:
- Token-aware splitting (not character-based)
- Respects paragraph boundaries when possible
- Overlapping chunks to preserve context across boundaries
- Each chunk includes rich metadata for filtering and citation
"""

import json
import re
import uuid
from pathlib import Path
from typing import List, Dict

import tiktoken
from tqdm import tqdm

from backend.app.core.config import settings


TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a string using cl100k_base encoding."""
    return len(TOKENIZER.encode(text))


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs using blank lines as separators."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    target_tokens: int = settings.CHUNK_SIZE,
    overlap_tokens: int = settings.CHUNK_OVERLAP,
) -> List[str]:
    """Split text into chunks of ~target_tokens each, with overlap."""
    paragraphs = split_into_paragraphs(text)
    chunks: List[str] = []
    current_chunk_parts: List[str] = []
    current_token_count = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > target_tokens:
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_token_count = 0

            tokens = TOKENIZER.encode(para)
            for i in range(0, len(tokens), target_tokens - overlap_tokens):
                sub_tokens = tokens[i : i + target_tokens]
                chunks.append(TOKENIZER.decode(sub_tokens))
            continue

        if current_token_count + para_tokens > target_tokens and current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))

            overlap_parts = []
            overlap_count = 0
            for part in reversed(current_chunk_parts):
                part_tokens = count_tokens(part)
                if overlap_count + part_tokens > overlap_tokens:
                    break
                overlap_parts.insert(0, part)
                overlap_count += part_tokens

            current_chunk_parts = overlap_parts
            current_token_count = overlap_count

        current_chunk_parts.append(para)
        current_token_count += para_tokens

    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    return chunks


def chunk_paper(paper_data: Dict) -> List[Dict]:
    """Convert one extracted paper into a list of chunks with metadata."""
    paper_id = paper_data["paper_id"]
    all_chunks = []

    for page_data in paper_data["pages"]:
        page_num = page_data["page"]
        page_text = page_data["text"]

        page_chunks = chunk_text(page_text)

        for chunk_idx, chunk_text_content in enumerate(page_chunks):
            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "paper_id": paper_id,
                "page": page_num,
                "chunk_index_on_page": chunk_idx,
                "text": chunk_text_content,
                "token_count": count_tokens(chunk_text_content),
                "char_count": len(chunk_text_content),
            }
            all_chunks.append(chunk)

    return all_chunks


def chunk_all_papers() -> None:
    """Process every extracted paper, save chunks to a single JSON file."""
    settings.PAPERS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(settings.PAPERS_PROCESSED_DIR.glob("*.json"))
    json_files = [f for f in json_files if f.name != "all_chunks.json"]

    if not json_files:
        print(f"⚠ No processed papers found in {settings.PAPERS_PROCESSED_DIR}")
        return

    print(f"Found {len(json_files)} processed papers to chunk\n")

    all_chunks: List[Dict] = []
    papers_processed = 0

    for json_path in tqdm(json_files, desc="Chunking papers"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                paper_data = json.load(f)

            paper_chunks = chunk_paper(paper_data)
            all_chunks.extend(paper_chunks)
            papers_processed += 1

        except Exception as e:
            print(f"\n  [ERROR] {json_path.name}: {e}")

    output_path = settings.PAPERS_PROCESSED_DIR / "all_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    token_counts = [c["token_count"] for c in all_chunks]

    print(f"\n{'='*60}")
    print(f"✅ Chunked {papers_processed} papers into {len(all_chunks)} chunks")
    print(f"📊 Token stats: min={min(token_counts)}, max={max(token_counts)}, avg={sum(token_counts)//len(token_counts)}")
    print(f"📁 Output: {output_path}")


if __name__ == "__main__":
    chunk_all_papers()