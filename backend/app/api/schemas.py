"""
Pydantic schemas for API request and response models.

Defines the contract between API clients and the backend.
Validation, OpenAPI docs, and serialization all come from these definitions.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ─── Health & Info ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    components: Dict[str, str]


class InfoResponse(BaseModel):
    name: str = "ScholarMind"
    description: str = "Agentic RAG over research papers"
    num_papers: int
    embedding_model: str
    llm_model: str
    tools_available: List[str]


# ─── Chat / Agent ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)


class Source(BaseModel):
    index: int
    paper_id: str
    page: Any   # can be int OR "—" for graph sources
    preview: str


class Plan(BaseModel):
    reasoning: str
    tool: str
    arguments: Dict[str, Any]


class Timings(BaseModel):
    planner_ms: float = 0
    tool_ms: float = 0
    synthesis_ms: float = 0
    total_ms: float = 0


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    plan: Plan
    timings: Timings
    metadata: Dict[str, Any]
    trace_id: Optional[str] = None


# ─── Raw Search ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    use_reranker: bool = False


class SearchResult(BaseModel):
    paper_id: str
    page: Any
    text: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    latency_ms: float


# ─── Papers ──────────────────────────────────────────────────────────────

class PaperSummary(BaseModel):
    arxiv_id: str
    title: str


class PaperDetail(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    methods: List[str]
    concepts: List[str]