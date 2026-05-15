"""
Contextual Retrieval (Anthropic's technique).

For each chunk, generate a short summary explaining how it fits into
the broader paper. This context is prepended to the chunk text before
embedding, dramatically improving retrieval on standalone-ambiguous chunks.

Reference: https://www.anthropic.com/news/contextual-retrieval

Process:
  1. Group all chunks by paper_id
  2. For each chunk, ask Groq: "given this paper and this chunk, write
     a 50-100 word context explaining what the chunk discusses"
  3. Save context alongside the chunk
  4. Progress is checkpointed every 25 chunks so a crash doesn't lose progress

Rate limiting: Groq free tier is 30k tokens/min on llama-3.1-8b-instant.
We use 8b instead of 70b for context generation — much cheaper, plenty good.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

from groq import Groq
from tqdm import tqdm

from backend.app.core.config import settings


CHECKPOINT_PATH = settings.PAPERS_PROCESSED_DIR / "chunks_with_context.json"


# Lazy Groq client
_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


CONTEXT_PROMPT = """<document>
{paper_summary}
</document>

Here is a chunk from this paper:
<chunk>
{chunk_text}
</chunk>

Write a short (50-100 words) context that situates this chunk within the paper. Explain what topic the chunk discusses and how it relates to the paper's main contribution. Do NOT include any preamble like "this chunk discusses" - just write the context directly. Do NOT quote the chunk. Be specific."""


def generate_context_for_chunk(
    chunk_text: str,
    paper_summary: str,
    model: str = "llama-3.1-8b-instant",
) -> str:
    """Generate a 50-100 word context for a single chunk via Groq."""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": CONTEXT_PROMPT.format(
                    paper_summary=paper_summary,
                    chunk_text=chunk_text,
                ),
            }
        ],
        temperature=0.2,
        max_tokens=200,  # context is short, hard cap protects us
    )
    return response.choices[0].message.content.strip()


def build_paper_summaries(chunks: List[Dict]) -> Dict[str, str]:
    """
    Group chunks by paper and build a short summary of each paper.

    The 'summary' is just the first ~1500 chars of the paper — enough to
    capture the title, abstract, and introduction. This becomes the
    'document context' the LLM uses to situate each chunk.
    """
    by_paper: Dict[str, List[Dict]] = {}
    for c in chunks:
        by_paper.setdefault(c["paper_id"], []).append(c)

    summaries = {}
    for paper_id, paper_chunks in by_paper.items():
        # Sort chunks by (page, chunk_index_on_page) for natural reading order
        paper_chunks.sort(key=lambda c: (c["page"], c.get("chunk_index_on_page", 0)))
        # Concatenate first few chunks until we hit ~1500 chars
        accumulated = ""
        for c in paper_chunks:
            if len(accumulated) >= 1500:
                break
            accumulated += c["text"] + "\n\n"
        summaries[paper_id] = accumulated[:1500]

    return summaries


def load_checkpoint() -> List[Dict]:
    """Load chunks_with_context.json if it exists (resume support)."""
    if not CHECKPOINT_PATH.exists():
        return []
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(chunks: List[Dict]) -> None:
    """Write progress to disk."""
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)


def contextualize_all_chunks(
    batch_size: int = 25,
    sleep_between_batches: float = 1.5,
) -> None:
    """
    Generate context for every chunk. Saves progress every batch.

    Args:
        batch_size: How many chunks to process before saving checkpoint.
        sleep_between_batches: Seconds to wait between batches (rate limiting).
    """
    chunks_path = settings.PAPERS_PROCESSED_DIR / "all_chunks.json"
    if not chunks_path.exists():
        print(f"⚠ {chunks_path} not found. Run chunker first.")
        return

    print(f"📂 Loading chunks from {chunks_path.name}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   {len(chunks)} chunks total")

    # Resume support: load any prior progress
    existing = load_checkpoint()
    done_chunk_ids = {c["chunk_id"] for c in existing if c.get("context")}
    print(f"   {len(done_chunk_ids)} chunks already contextualized (resuming)")

    # Build paper-level summaries once (used as document context for every chunk)
    print(f"\n📚 Building per-paper summaries...")
    paper_summaries = build_paper_summaries(chunks)
    print(f"   Built summaries for {len(paper_summaries)} papers")

    # Build the output list, starting from any existing work
    output: List[Dict] = list(existing)
    existing_by_id = {c["chunk_id"]: c for c in existing}

    # Process remaining chunks
    todo = [c for c in chunks if c["chunk_id"] not in done_chunk_ids]
    print(f"\n🧮 Processing {len(todo)} chunks (~{len(todo) * 1.5 / 60:.1f} min estimated)\n")

    processed_in_batch = 0
    for i, chunk in enumerate(tqdm(todo, desc="Contextualizing")):
        try:
            context = generate_context_for_chunk(
                chunk_text=chunk["text"],
                paper_summary=paper_summaries[chunk["paper_id"]],
            )
        except Exception as e:
            print(f"\n  ⚠ Failed on chunk {chunk['chunk_id']}: {e}")
            print(f"     Skipping (will be retried on next run)")
            time.sleep(5)  # back off briefly
            continue

        # Build the enriched chunk
        enriched = dict(chunk)
        enriched["context"] = context
        # The "contextualized text" we'll embed combines context + original
        enriched["text_for_embedding"] = f"{context}\n\n{chunk['text']}"

        output.append(enriched)
        processed_in_batch += 1

        # Checkpoint every batch_size processed
        if processed_in_batch >= batch_size:
            save_checkpoint(output)
            processed_in_batch = 0
            time.sleep(sleep_between_batches)

    # Final save
    save_checkpoint(output)

    print(f"\n{'='*60}")
    print(f"✅ Contextualized {len(output)} / {len(chunks)} chunks")
    print(f"📁 Saved to: {CHECKPOINT_PATH}")

    # Sample inspection
    if output:
        sample = output[len(output) // 2]  # middle chunk
        print(f"\n🔍 Sample (chunk from {sample['paper_id']}, p.{sample['page']}):")
        print(f"   CONTEXT: {sample['context'][:200]}...")


if __name__ == "__main__":
    contextualize_all_chunks()