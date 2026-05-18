"""
Planner node for the ScholarMind agent.

Given a user query, asks an LLM (Groq Llama 3.3) to choose a tool and
arguments. Returns a structured plan that the graph executes.

Why structured JSON output?
  - Parseable. Free-text "use search_papers" outputs break reliably.
  - Auditable. Every plan is logged with reasoning, tool, and arguments.
  - Testable. We can write unit tests against the planner's decisions.
"""

import json
from typing import Dict

from groq import Groq

from backend.app.core.config import settings
from backend.app.agents.tools import TOOL_DESCRIPTIONS, TOOL_REGISTRY


_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


PLANNER_PROMPT = """You are the planner for an academic research assistant.

Given a user question, decide which ONE tool to call to gather evidence.
You must output ONLY valid JSON — no preamble, no explanation outside the JSON.

{tool_descriptions}

Decision rules:
- For most questions, use search_papers.
- Use compare_papers ONLY if the user explicitly asks to compare/contrast TWO things
  ("X vs Y", "differences between A and B", "compare X and Y").
- Use extract_methodology ONLY if the user asks HOW something works, the algorithm,
  or implementation details of a specific named method/system.
- If a question is ambiguous, default to search_papers.

Output format (JSON only):
{{
  "reasoning": "<1-sentence justification for tool choice>",
  "tool": "<one of: search_papers | compare_papers | extract_methodology>",
  "arguments": {{ <kwargs the chosen tool takes> }}
}}

Argument shapes by tool:
- search_papers:        {{"query": "<rephrased search query>"}}
- compare_papers:       {{"topic_a": "<first topic>", "topic_b": "<second topic>"}}
- extract_methodology:  {{"method_or_paper": "<name>"}}

Examples:

User question: "What is retrieval augmented generation?"
{{
  "reasoning": "Straightforward topic question — general search.",
  "tool": "search_papers",
  "arguments": {{"query": "retrieval augmented generation overview"}}
}}

User question: "How does R2AG differ from GFM-RAG?"
{{
  "reasoning": "Explicit comparison of two named methods.",
  "tool": "compare_papers",
  "arguments": {{"topic_a": "R2AG retrieval augmented generation", "topic_b": "GFM-RAG graph foundation model"}}
}}

User question: "How does Curator's multi-tenant index work?"
{{
  "reasoning": "Asks HOW the method works — methodology query.",
  "tool": "extract_methodology",
  "arguments": {{"method_or_paper": "Curator multi-tenant vector index"}}
}}
""".strip()


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json fences despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        # remove leading fence (with optional language tag)
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        # remove trailing fence
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def plan(
    query: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
) -> Dict:
    """
    Ask the planner LLM which tool to call.

    Returns:
        {"reasoning": str, "tool": str, "arguments": dict}

    Raises:
        ValueError if the LLM output can't be parsed as a valid plan.
    """
    client = get_client()
    system = PLANNER_PROMPT.format(tool_descriptions=TOOL_DESCRIPTIONS)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"User question: {query}"},
        ],
        temperature=temperature,
        max_tokens=300,
    )

    raw = response.choices[0].message.content
    cleaned = _strip_code_fences(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Planner produced invalid JSON. Raw output:\n{raw}\nError: {e}"
        )

    # Validate the schema
    if "tool" not in parsed or "arguments" not in parsed:
        raise ValueError(f"Planner missing required fields: {parsed}")

    tool_name = parsed["tool"]
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Planner chose unknown tool '{tool_name}'. "
            f"Valid tools: {list(TOOL_REGISTRY.keys())}"
        )

    return {
        "reasoning": parsed.get("reasoning", ""),
        "tool": tool_name,
        "arguments": parsed["arguments"],
    }


if __name__ == "__main__":
    # Test the planner on 4 diverse queries
    test_queries = [
        "What is retrieval augmented generation?",
        "How does R2AG differ from GFM-RAG?",
        "How does Curator's multi-tenant index work?",
        "What are the main security risks of LLM agents?",
    ]

    print("=" * 72)
    print("PLANNER SMOKE TEST")
    print("=" * 72)
    for q in test_queries:
        print(f"\n❓ Query: {q}")
        try:
            p = plan(q)
            print(f"   🧠 Reasoning: {p['reasoning']}")
            print(f"   🛠  Tool:      {p['tool']}")
            print(f"   📥 Arguments: {p['arguments']}")
        except Exception as e:
            print(f"   ❌ Planner failed: {e}")

    print("\n" + "=" * 72)
    print("All plans complete")