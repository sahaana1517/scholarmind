"""
ScholarMind agent — a LangGraph state machine that plans + executes tools + synthesizes.

Flow:
    START → planner → execute_tool → synthesize → END
"""

import time
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from backend.app.agents.planner import plan as planner_plan
from backend.app.agents.tools import TOOL_REGISTRY
from backend.app.agents.generator import generate_answer
from backend.app.core.observability import get_langfuse, flush


class AgentState(TypedDict, total=False):
    query: str
    plan: Dict[str, Any]
    tool_result: Any
    chunks_for_llm: List[Dict]
    answer: str
    sources: List[Dict]
    metadata: Dict[str, Any]
    timings: Dict[str, float]
    trace: Any


def planner_node(state: AgentState) -> dict:
    """Step 1: ask the LLM which tool to call."""
    t0 = time.time()
    plan_dict = planner_plan(state["query"])
    elapsed_ms = (time.time() - t0) * 1000

    print(f"\n🧠 Planner decided: {plan_dict['tool']}")
    print(f"   Reasoning: {plan_dict['reasoning']}")
    print(f"   Arguments: {plan_dict['arguments']}")

    trace = state.get("trace")
    if trace is not None:
        trace.span(
            name="planner",
            input={"query": state["query"]},
            output=plan_dict,
            metadata={"latency_ms": elapsed_ms},
        )

    timings = state.get("timings", {})
    timings["planner_ms"] = elapsed_ms
    return {"plan": plan_dict, "timings": timings}


def execute_tool_node(state: AgentState) -> dict:
    """Step 2: run the chosen tool with the planner's arguments."""
    plan_dict = state["plan"]
    tool_name = plan_dict["tool"]
    tool_args = plan_dict["arguments"]

    print(f"\n🛠  Executing tool: {tool_name}({tool_args})")

    tool_fn = TOOL_REGISTRY[tool_name]
    t0 = time.time()
    result = tool_fn(**tool_args)
    elapsed_ms = (time.time() - t0) * 1000

    chunks: List[Dict] = []

    if tool_name == "search_papers":
        chunks = result

    elif tool_name == "extract_methodology":
        chunks = result

    elif tool_name == "compare_papers":
        a_chunks = result["topic_a"]["chunks"]
        b_chunks = result["topic_b"]["chunks"]
        for c in a_chunks:
            c2 = dict(c)
            c2["text"] = f"[TOPIC A: {result['topic_a']['query']}]\n{c['text']}"
            chunks.append(c2)
        for c in b_chunks:
            c2 = dict(c)
            c2["text"] = f"[TOPIC B: {result['topic_b']['query']}]\n{c['text']}"
            chunks.append(c2)

    elif tool_name == "graph_query":
        # Convert graph results into chunk-shaped dicts so the generator
        # treats them like retrieved text. Each "chunk" describes one
        # graph fact attributed to the relevant paper.
        results = result.get("results", [])
        intent = result.get("intent", "")
        for item in results:
            if isinstance(item, dict):
                if "title" in item and "arxiv_id" in item:
                    text_parts = [f"Paper {item['arxiv_id']}: {item['title']}"]
                    if "shared_methods" in item:
                        text_parts.append(
                            f"Shared methods: {', '.join(item['shared_methods'])}"
                        )
                    if "shared_concepts" in item:
                        text_parts.append(
                            f"Shared concepts: {', '.join(item['shared_concepts'])}"
                        )
                    if "matched_method" in item:
                        text_parts.append(
                            f"Matched on method: {item['matched_method']}"
                        )
                    if "matched_concept" in item:
                        text_parts.append(
                            f"Matched on concept: {item['matched_concept']}"
                        )
                    chunks.append({
                        "paper_id": item["arxiv_id"],
                        "page": "—",
                        "text": "\n".join(text_parts),
                    })
            elif isinstance(item, str):
                # methods_of_paper / concepts_of_paper return list[str]
                chunks.append({
                    "paper_id": result.get("arxiv_id", "graph"),
                    "page": "—",
                    "text": f"{intent}: {item}",
                })

    print(f"   → Retrieved {len(chunks)} chunks")

    trace = state.get("trace")
    if trace is not None:
        trace.span(
            name=f"tool:{tool_name}",
            input=tool_args,
            output={"num_chunks": len(chunks)},
            metadata={"latency_ms": elapsed_ms},
        )

    timings = state.get("timings", {})
    timings["tool_ms"] = elapsed_ms
    return {"tool_result": result, "chunks_for_llm": chunks, "timings": timings}


def synthesize_node(state: AgentState) -> dict:
    """Step 3: ask the generator to produce a cited answer from the chunks."""
    print(f"\n✍️  Synthesizing answer...")

    t0 = time.time()
    answer, sources, gen_metadata = generate_answer(
        state["query"], state["chunks_for_llm"]
    )
    elapsed_ms = (time.time() - t0) * 1000

    trace = state.get("trace")
    if trace is not None:
        trace.generation(
            name="synthesis",
            model=gen_metadata.get("model"),
            input={
                "query": state["query"],
                "num_chunks": len(state["chunks_for_llm"]),
            },
            output=answer,
            usage={
                "input": gen_metadata.get("prompt_tokens", 0),
                "output": gen_metadata.get("completion_tokens", 0),
                "total": gen_metadata.get("total_tokens", 0),
            },
            metadata={"latency_ms": elapsed_ms},
        )

    timings = state.get("timings", {})
    timings["synthesis_ms"] = elapsed_ms
    timings["total_ms"] = sum(
        v for k, v in timings.items() if k.endswith("_ms") and k != "total_ms"
    )

    return {
        "answer": answer,
        "sources": sources,
        "metadata": gen_metadata,
        "timings": timings,
    }


def build_agent_graph():
    """Compile the LangGraph state machine."""
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("execute_tool", execute_tool_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "execute_tool")
    builder.add_edge("execute_tool", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent_graph()
    return _agent


def run_agent(query: str) -> Dict:
    """Execute the agent on a query."""
    agent = get_agent()

    lf = get_langfuse()
    trace = None
    trace_id = None
    if lf is not None:
        trace = lf.trace(name="agent-run", input={"query": query})
        trace_id = trace.id

    initial_state: AgentState = {
        "query": query,
        "timings": {},
        "trace": trace,
    }

    final = agent.invoke(initial_state)

    if trace is not None:
        trace.update(
            output={
                "answer": final.get("answer"),
                "tool_used": final["plan"]["tool"],
                "num_sources": len(final.get("sources", [])),
            },
            metadata={"timings": final.get("timings", {})},
        )

    return {
        "answer": final.get("answer", ""),
        "sources": final.get("sources", []),
        "chunks_used": final.get("chunks_for_llm", []),
        "timings": final.get("timings", {}),
        "metadata": final.get("metadata", {}),
        "trace_id": trace_id,
        "plan": final.get("plan", {}),
    }


def pretty_print_agent_result(result: Dict) -> None:
    """Format an agent run for terminal display."""
    print("\n" + "=" * 72)
    print("AGENT PLAN")
    print("=" * 72)
    p = result["plan"]
    print(f"  Reasoning: {p.get('reasoning', '')}")
    print(f"  Tool:      {p.get('tool', '')}")
    print(f"  Arguments: {p.get('arguments', {})}")

    print("\n" + "=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(result["answer"])

    print("\n" + "-" * 72)
    print("SOURCES")
    print("-" * 72)
    for src in result["sources"]:
        print(f"  [{src['index']}] Paper {src['paper_id']}, p.{src['page']}")
        print(f"      {src['preview']}...")

    print("\n" + "-" * 72)
    print("TIMINGS")
    print("-" * 72)
    t = result["timings"]
    print(f"  Planner:    {t.get('planner_ms', 0):>7.0f}ms")
    print(f"  Tool:       {t.get('tool_ms', 0):>7.0f}ms")
    print(f"  Synthesis:  {t.get('synthesis_ms', 0):>7.0f}ms")
    print(f"  TOTAL:      {t.get('total_ms', 0):>7.0f}ms")

    m = result["metadata"]
    if m:
        print(f"\n  Synth model:  {m.get('model', '?')}")
        print(f"  Tokens:       {m.get('total_tokens', 0)} "
              f"(prompt {m.get('prompt_tokens', 0)}, "
              f"completion {m.get('completion_tokens', 0)})")

    if result.get("trace_id"):
        print(f"\n  🔗 Langfuse trace: {result['trace_id']}")


def interactive_agent() -> None:
    """REPL for the agent."""
    print("\n🤖 ScholarMind Agent (LangGraph)")
    print("   Planner → tool execution → cited synthesis")
    print("   Type a question (or 'quit' to exit)\n")

    from backend.app.retrieval.search import get_model
    get_model()

    try:
        while True:
            try:
                query = input("\n❓ Your question › ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Bye!")
                break

            if not query:
                continue
            if query.lower() in {"quit", "exit", "q"}:
                print("👋 Bye!")
                break

            try:
                result = run_agent(query)
                pretty_print_agent_result(result)
            except Exception as e:
                import traceback
                print(f"\n❌ Agent error: {e}")
                traceback.print_exc()
    finally:
        flush()


if __name__ == "__main__":
    interactive_agent()