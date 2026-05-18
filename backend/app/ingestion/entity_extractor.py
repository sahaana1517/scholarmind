"""
LLM-based entity extraction from research papers.

For each paper, extracts structured entities:
  - title              (string)
  - authors            (list of strings)
  - methods_used       (list of strings — concrete techniques the paper uses)
  - concepts_studied   (list of strings — research areas/problems the paper addresses)
  - papers_cited       (list of arXiv IDs found in the text — best effort)

Output is checkpointed to data/papers_processed/entities.json for resume support.

Notes:
- Sends the first 3 pages (title + abstract + intro) — enough context, low cost.
- Uses Llama 3.3 70B (not 8B) for extraction. 8B leaks prompt examples into outputs
  on weakly-understood papers; 70B is more discerning. Worth the small extra latency.
- Prompt deliberately avoids naming concrete concepts as examples (which the model
  would otherwise reuse as defaults).
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List

from groq import Groq
from tqdm import tqdm

from backend.app.core.config import settings


ENTITIES_PATH = settings.PAPERS_PROCESSED_DIR / "entities.json"

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


# Stricter prompt — no leaky example concepts.
EXTRACTION_PROMPT = """You extract structured entities from an academic paper.

Read the paper excerpt below and output ONE JSON object with this exact schema:

{{
  "title": "<the paper's actual title>",
  "authors": ["<author 1>", "<author 2>", ...],
  "methods_used": ["<method 1>", "<method 2>", ...],
  "concepts_studied": ["<concept 1>", "<concept 2>", ...],
  "papers_cited": ["<arxiv id 1>", ...]
}}

CRITICAL RULES — read carefully:

1. EVERY field must be grounded in the actual paper text below. Do NOT invent,
   guess, or pattern-match. If something isn't explicitly in the excerpt, do
   NOT include it.

2. "methods_used" = concrete, named techniques the paper USES or BUILDS UPON.
   Must be specific enough that someone could search for them. Examples of the
   right level of specificity: a named algorithm, a named neural architecture,
   a named library or framework, a named statistical method. Examples of the
   wrong level: generic terms like "machine learning", "AI", "deep learning".
   Aim for 3-6 items unless the paper genuinely uses fewer.

3. "concepts_studied" = the research areas or problems this paper addresses.
   These should be 2-5 items that ACCURATELY describe what THIS paper is about.
   They must come from the paper itself, not from related work you happen to
   know about.

4. "papers_cited" = arXiv IDs in format NNNN.NNNNN that appear in the excerpt.
   Best effort. If you don't see any, return [].

5. "authors" = primary authors listed on the title (up to 5).

6. NEVER copy phrases from these rules into your output.

OUTPUT FORMAT: a single JSON object. No preamble, no markdown fences, no
explanation. Just the JSON.

Paper excerpt:
---
{paper_text}
---
""".strip()


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def extract_entities_from_paper(
    paper_id: str,
    paper_text: str,
    model: str = "llama-3.3-70b-versatile",
) -> Dict:
    """Send one paper's first-pages text to Groq and parse the entity JSON."""
    client = get_client()
    prompt = EXTRACTION_PROMPT.format(paper_text=paper_text[:6000])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic — extraction is not creative writing
        max_tokens=800,
    )

    raw = response.choices[0].message.content
    cleaned = _strip_code_fences(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed for {paper_id}. Raw:\n{raw}\nError: {e}")

    return {
        "paper_id": paper_id,
        "title": parsed.get("title", "").strip(),
        "authors": [a.strip() for a in parsed.get("authors", []) if a.strip()],
        "methods_used": [m.strip() for m in parsed.get("methods_used", []) if m.strip()],
        "concepts_studied": [c.strip() for c in parsed.get("concepts_studied", []) if c.strip()],
        "papers_cited": [
            p.strip() for p in parsed.get("papers_cited", [])
            if isinstance(p, str) and re.match(r"^\d{4}\.\d{4,5}$", p.strip())
        ],
    }


def load_checkpoint() -> List[Dict]:
    if not ENTITIES_PATH.exists():
        return []
    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(entities: List[Dict]) -> None:
    with open(ENTITIES_PATH, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)


def extract_all_papers() -> None:
    """Extract entities for every paper. Skips ones already processed."""
    json_files = sorted(
        p for p in settings.PAPERS_PROCESSED_DIR.glob("*.json")
        if p.name not in {
            "all_chunks.json", "chunks_with_embeddings.json",
            "chunks_with_context.json", "chunks_with_context_embeddings.json",
            "entities.json",
        }
    )

    if not json_files:
        print(f"⚠ No processed paper JSONs found.")
        return

    print(f"📂 Found {len(json_files)} paper files")

    existing = load_checkpoint()
    done_ids = {e["paper_id"] for e in existing}
    print(f"   {len(done_ids)} already extracted (resuming)")

    output = list(existing)

    for json_path in tqdm(json_files, desc="Extracting entities"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        paper_id = data["paper_id"]
        if paper_id in done_ids:
            continue

        # First 3 pages — title, abstract, intro
        first_pages_text = "\n\n".join(p["text"] for p in data["pages"][:3])

        try:
            entity = extract_entities_from_paper(paper_id, first_pages_text)
        except Exception as e:
            print(f"\n  ⚠ Failed on {paper_id}: {e}")
            time.sleep(3)
            continue

        output.append(entity)
        save_checkpoint(output)

    print(f"\n{'='*60}")
    print(f"✅ Extracted entities for {len(output)} papers")
    print(f"📁 Saved to: {ENTITIES_PATH}")

    if output:
        sample = output[len(output) // 2]
        print(f"\n🔍 Sample (paper {sample['paper_id']}):")
        print(f"   Title:    {sample['title']}")
        print(f"   Authors:  {sample['authors'][:3]}")
        print(f"   Methods:  {sample['methods_used'][:6]}")
        print(f"   Concepts: {sample['concepts_studied']}")


if __name__ == "__main__":
    extract_all_papers()