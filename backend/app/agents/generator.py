"""
Answer generation module.

Given a user query and retrieved chunks, calls an LLM (Groq Llama 3.3)
to synthesize a grounded, citation-backed answer.

Key principles:
- The LLM is instructed to use ONLY provided sources (no hallucination)
- Citations are inline as [1], [2], etc.
- A source list maps citations back to paper_id + page
- If the chunks don't contain the answer, the model says so honestly
"""

from typing import List, Dict, Tuple
import time

from groq import Groq

from backend.app.core.config import settings


# Lazy-loaded Groq client
_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = """You are ScholarMind, an AI assistant that answers questions about academic papers.

You will receive a user's question and a numbered list of relevant excerpts from research papers.

Your job:
1. Answer the question using ONLY the information in the provided excerpts.
2. Cite sources inline using bracketed numbers like [1], [2], [3] — these refer to the numbered excerpts.
3. If multiple excerpts support a claim, cite all of them: [1][3].
4. If the excerpts don't contain enough information to answer, say so clearly. Do NOT make up information.
5. Be concise but complete. Prefer 2-4 paragraphs over long-winded responses.
6. Do not include a sources list at the end — the system handles that separately.

Always remain grounded in the provided text. Quoting short phrases is fine; paraphrasing is preferred."""


def format_chunks_for_prompt(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks as a numbered list the LLM can reference.

    Returns text like:
        [1] (Paper 2406.13249, p.1)
        R2AG: Incorporating Retrieval Information into ...

        [2] (Paper 2502.01113, p.3)
        Graph Foundation Models offer a unified approach to ...
    """
    formatted_parts = []
    for i, chunk in enumerate(chunks, start=1):
        paper_id = chunk.get("paper_id", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "").strip()
        formatted_parts.append(
            f"[{i}] (Paper {paper_id}, p.{page})\n{text}"
        )
    return "\n\n".join(formatted_parts)


def build_source_list(chunks: List[Dict]) -> List[Dict]:
    """
    Build a citation list parallel to the numbered chunks.

    Returns: [{"index": 1, "paper_id": "...", "page": ..., "preview": "..."}, ...]
    """
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.get("text", "")[:120].replace("\n", " ").strip()
        sources.append({
            "index": i,
            "paper_id": chunk.get("paper_id"),
            "page": chunk.get("page"),
            "preview": preview,
        })
    return sources


def generate_answer(
    query: str,
    chunks: List[Dict],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> Tuple[str, List[Dict], Dict]:
    """
    Generate a cited answer from the provided chunks.

    Returns:
        (answer_text, sources_list, metadata_dict)
    """
    if not chunks:
        return (
            "I couldn't find any relevant information in the available papers to answer this question.",
            [],
            {"model": model, "latency_ms": 0, "tokens": 0},
        )

    client = get_client()
    sources = build_source_list(chunks)

    # Build the user prompt
    user_prompt = (
        f"Question: {query}\n\n"
        f"Relevant excerpts from research papers:\n\n"
        f"{format_chunks_for_prompt(chunks)}\n\n"
        f"Answer the question based only on the excerpts above. Use [n] citations to reference them."
    )

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = (time.time() - start) * 1000

    answer = response.choices[0].message.content.strip()
    metadata = {
        "model": model,
        "latency_ms": latency_ms,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return answer, sources, metadata


if __name__ == "__main__":
    # Smoke test: hardcoded chunks, see if generation works
    fake_chunks = [
        {
            "paper_id": "2406.13249",
            "page": 1,
            "text": "Retrieval Augmented Generation (RAG) is a paradigm that combines retrieval of external documents with text generation by language models. By grounding outputs in retrieved evidence, RAG reduces hallucinations.",
        },
        {
            "paper_id": "2502.01113",
            "page": 2,
            "text": "Traditional RAG uses a dense retriever to find relevant passages. Graph-based RAG extends this by using knowledge graph structure to find documents connected through entity relationships.",
        },
    ]

    answer, sources, meta = generate_answer(
        "What is retrieval augmented generation?",
        fake_chunks,
    )

    print("=== ANSWER ===")
    print(answer)
    print("\n=== SOURCES ===")
    for s in sources:
        print(f"  [{s['index']}] Paper {s['paper_id']} p.{s['page']} — {s['preview']}")
    print(f"\n=== METADATA ===")
    print(f"  Model: {meta['model']}")
    print(f"  Latency: {meta['latency_ms']:.0f}ms")
    print(f"  Tokens: {meta['total_tokens']} (prompt: {meta['prompt_tokens']}, completion: {meta['completion_tokens']})")