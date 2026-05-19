"""
ScholarMind FastAPI backend.

Exposes the agent over HTTP for frontend consumption and external API calls.

Endpoints:
  GET  /health        — liveness check
  GET  /info          — corpus / model metadata
  POST /chat          — agentic answer to a question
  POST /search        — raw hybrid retrieval (bypasses the agent)
  GET  /papers        — list all papers
  GET  /papers/{id}   — detail on one paper (from the graph)

Run with:
  uvicorn backend.app.api.main:app --reload --port 8000
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from backend.app.core.config import settings
from backend.app.api.schemas import (
    HealthResponse, InfoResponse,
    ChatRequest, ChatResponse, Source, Plan, Timings,
    SearchRequest, SearchResponse, SearchResult,
    PaperSummary, PaperDetail,
)


# ─── Lifespan: warm up heavy things at startup ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models so the first request isn't slow."""
    print("🔥 Warming up embedding model...")
    from backend.app.retrieval.search import get_model
    get_model()
    print("✅ ScholarMind API ready")
    yield
    # Graceful shutdown — flush Langfuse traces
    print("👋 Shutting down — flushing observability...")
    try:
        from backend.app.core.observability import flush
        flush()
    except Exception:
        pass


# ─── App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ScholarMind",
    description="Agentic RAG over research papers",
    version="0.7.0",
    lifespan=lifespan,
)

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scholarmind")

if settings.FORCE_HTTPS and settings.ENVIRONMENT.lower() == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

trusted_hosts = [
    host.strip()
    for host in settings.BACKEND_TRUSTED_HOSTS.split(",")
    if host.strip()
]
if trusted_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# CORS — open in dev, explicit origins required in production.
allowed_origins = [
    origin.strip()
    for origin in settings.BACKEND_CORS_ORIGINS.split(",")
    if origin.strip()
]
if settings.ENVIRONMENT.lower() == "production" and not allowed_origins:
    raise RuntimeError(
        "Production deployment requires BACKEND_CORS_ORIGINS to be set "
        "to a comma-separated list of trusted frontend origins."
    )
if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.ENABLE_GZIP:
    app.add_middleware(GZipMiddleware, minimum_size=500)


# ─── Health & Info ─────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check — confirms the service is up and key components are reachable."""
    components = {}

    # Qdrant
    try:
        from backend.app.retrieval.search import get_client as get_qdrant
        get_qdrant().get_collections()
        components["qdrant"] = "ok"
    except Exception as e:
        components["qdrant"] = f"error: {type(e).__name__}"

    # Neo4j
    try:
        from backend.app.retrieval.graph_query import get_driver
        get_driver().verify_connectivity()
        components["neo4j"] = "ok"
    except Exception as e:
        components["neo4j"] = f"error: {type(e).__name__}"

    return HealthResponse(
        status="ok" if all(v == "ok" for v in components.values()) else "degraded",
        version="0.7.0",
        components=components,
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    """Metadata about the running deployment."""
    from backend.app.agents.tools import TOOL_REGISTRY

    # Count papers from the graph
    num_papers = 0
    try:
        from backend.app.retrieval.graph_query import get_driver
        with get_driver().session() as s:
            r = s.run("MATCH (p:Paper) RETURN count(p) AS n").single()
            num_papers = r["n"]
    except Exception:
        pass

    return InfoResponse(
        num_papers=num_papers,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model="llama-3.3-70b-versatile",
        tools_available=list(TOOL_REGISTRY.keys()),
    )


# ─── Chat (agentic) ────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Ask the agent. The planner picks a tool, the tool fetches evidence,
    the generator synthesizes a cited answer.
    """
    from backend.app.agents.agent import run_agent

    # run_agent is sync (contains blocking I/O). Run in a thread so the
    # FastAPI event loop stays free for other requests.
    try:
        result = await asyncio.to_thread(run_agent, req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return ChatResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        plan=Plan(**result["plan"]),
        timings=Timings(**result["timings"]),
        metadata=result.get("metadata", {}),
        trace_id=result.get("trace_id"),
    )


# ─── Raw Search ────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Raw hybrid retrieval — bypasses the agent / generator. Use this to
    inspect what the retriever returns without LLM synthesis.
    """
    from backend.app.retrieval.hybrid_search import hybrid_search
    from backend.app.retrieval.reranker import rerank

    t0 = time.time()
    try:
        chunks = await asyncio.to_thread(
            hybrid_search, req.query, req.top_k, 30
        )
        if req.use_reranker:
            chunks = await asyncio.to_thread(rerank, req.query, chunks, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    latency_ms = (time.time() - t0) * 1000

    results = [
        SearchResult(
            paper_id=c["paper_id"],
            page=c["page"],
            text=c["text"],
            dense_score=c.get("dense_score"),
            sparse_score=c.get("sparse_score"),
            rrf_score=c.get("rrf_score"),
            rerank_score=c.get("rerank_score"),
        )
        for c in chunks
    ]
    return SearchResponse(query=req.query, results=results, latency_ms=latency_ms)


# ─── Papers ────────────────────────────────────────────────────────────

@app.get("/papers", response_model=List[PaperSummary])
async def list_papers():
    """List every paper in the corpus."""
    from backend.app.retrieval.graph_query import get_driver

    try:
        with get_driver().session() as session:
            rows = session.run(
                "MATCH (p:Paper) RETURN p.arxiv_id AS arxiv_id, p.title AS title "
                "ORDER BY p.arxiv_id"
            )
            return [
                PaperSummary(arxiv_id=r["arxiv_id"], title=r["title"] or "")
                for r in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph error: {e}")


@app.get("/papers/{arxiv_id}", response_model=PaperDetail)
async def paper_detail(arxiv_id: str):
    """Detail of one paper — methods, concepts, authors via the graph."""
    from backend.app.retrieval.graph_query import get_driver

    try:
        with get_driver().session() as session:
            paper = session.run(
                "MATCH (p:Paper {arxiv_id: $id}) RETURN p.title AS title",
                id=arxiv_id,
            ).single()

            if paper is None:
                raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")

            methods = [r["m"] for r in session.run(
                "MATCH (p:Paper {arxiv_id: $id})-[:USES_METHOD]->(m:Method) RETURN m.name AS m",
                id=arxiv_id,
            )]
            concepts = [r["c"] for r in session.run(
                "MATCH (p:Paper {arxiv_id: $id})-[:STUDIES]->(c:Concept) RETURN c.name AS c",
                id=arxiv_id,
            )]
            authors = [r["a"] for r in session.run(
                "MATCH (p:Paper {arxiv_id: $id})-[:AUTHORED_BY]->(a:Author) RETURN a.name AS a",
                id=arxiv_id,
            )]

            return PaperDetail(
                arxiv_id=arxiv_id,
                title=paper["title"] or "",
                authors=authors,
                methods=methods,
                concepts=concepts,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph error: {e}")