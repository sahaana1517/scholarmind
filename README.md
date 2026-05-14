# ScholarMind

> An agentic RAG platform for research paper intelligence — semantic search and reasoning over academic literature, built with production-grade engineering practices.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## The Problem

Researchers spend hours digging through dozens of papers to answer questions like:
- "Which papers propose improvements to the original RAG architecture?"
- "What are the security implications of LLM agents?"
- "How do modern vector databases handle multi-tenancy?"

Existing tools either drown you in keyword-matched results (Google Scholar) or hallucinate confidently without citations (ChatGPT). ScholarMind aims to do both better — return faithful, citation-grounded answers from a curated corpus, using agentic reasoning over hybrid retrieval and a knowledge graph layer.

---

## Current Capabilities

| Capability | Status | Notes |
|---|---|---|
| Automated paper ingestion (arXiv) | ✅ Implemented | 30-paper test corpus auto-downloaded |
| PDF text extraction with cleanup | ✅ Implemented | Hyphenation, multi-page artifacts handled |
| Token-aware paragraph chunking | ✅ Implemented | 800-token chunks with 100-token overlap |
| Local CPU embedding generation | ✅ Implemented | BGE-small-en-v1.5 (384-dim, normalized) |
| Cloud vector indexing | ✅ Implemented | Qdrant Cloud, cosine similarity |
| Semantic search CLI | ✅ Implemented | Sub-second retrieval over 1,100+ chunks |
| Hybrid retrieval (BM25 + dense) | ✅ Implemented | Reciprocal Rank Fusion (RRF) |
| Cross-encoder re-ranking | 🚧 Planned | ms-marco-MiniLM |
| Contextual retrieval | 🚧 Planned | Anthropic's contextual chunking technique |
| Agentic reasoning layer | 🚧 Planned | LangGraph multi-step agent |
| Knowledge graph layer | 🚧 Planned | Neo4j AuraDB for GraphRAG |
| Evaluation framework | 🚧 Planned | RAGAS + golden eval set |
| FastAPI backend | 🚧 Planned | Async endpoints with streaming |
| Next.js frontend | 🚧 Planned | Tailwind + shadcn/ui |
| Production deployment | 🚧 Planned | Railway + Vercel |

---

## Architecture (Current)
                       ┌─────────────────────┐
                       │   arXiv API         │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  PDF Ingestion      │
                       │  (pypdf)            │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Chunking           │
                       │  (tiktoken, regex)  │
                       └──────────┬──────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
   ┌────────────────────┐                  ┌────────────────────┐
   │  BGE-small         │                  │  BM25Okapi         │
   │  embeddings (CPU)  │                  │  (local pickle)    │
   └─────────┬──────────┘                  └─────────┬──────────┘
             │                                       │
             ▼                                       ▼
   ┌────────────────────┐                  ┌────────────────────┐
   │  Qdrant Cloud      │                  │  Sparse Index      │
   │  (dense vectors)   │                  │  (keyword search)  │
   └─────────┬──────────┘                  └─────────┬──────────┘
             │                                       │
             └───────────────────┬───────────────────┘
                                 ▼
                       ┌─────────────────────┐
                       │  Reciprocal Rank    │
                       │  Fusion (RRF)       │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Hybrid Search CLI  │
                       └─────────────────────┘
## Tech Stack

**Languages:** Python 3.12

**LLM & Embeddings:**
- Groq (Llama 3.3 70B, Llama 3.1 8B) — inference
- BGE-small-en-v1.5 — local embeddings via `sentence-transformers`
- Cross-encoder ms-marco-MiniLM (planned) — re-ranking

**Data & Storage:**
- Qdrant Cloud — vector database
- Neo4j AuraDB (planned) — knowledge graph
- Neon Postgres (planned) — relational data
- Upstash Redis (planned) — caching

**Observability:**
- Langfuse Cloud — LLM tracing and evaluation

**Tooling:**
- `pypdf` — PDF extraction
- `tiktoken` — token counting
- `pydantic-settings` — type-safe configuration
- `tqdm` — progress visualization

---

## Project Structure

---

## Getting Started

### Prerequisites

- Python 3.12+
- Free-tier accounts on: Groq, Qdrant Cloud, Neo4j AuraDB, Neon, Upstash, Langfuse, Hugging Face

### Installation

```bash
# Clone
git clone https://github.com/sahaana1517/scholarmind.git
cd scholarmind

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt    # (planned — will add)

# Configure environment
cp .env.example .env
# Fill in your API keys in .env
```

### Run the Pipeline

```bash
# 1. Download papers from arXiv
python -m backend.app.ingestion.download_papers

# 2. Extract text from PDFs
python -m backend.app.ingestion.pdf_extractor

# 3. Chunk extracted text
python -m backend.app.ingestion.chunker

# 4. Generate embeddings
python -m backend.app.ingestion.embedder

# 5. Index in Qdrant (dense vectors)
python -m backend.app.retrieval.indexer

# 6. Build BM25 sparse index
python -m backend.app.retrieval.bm25_index

# 7. Try semantic search (dense only)
python -m backend.app.retrieval.search

# 8. Try hybrid search (dense + BM25 + RRF)
python -m backend.app.retrieval.hybrid_search
```

---

## Sample Query
🔍 Query: What is retrieval augmented generation?
--- Result 1 | Score: 0.8503 ---
📄 Paper: 2406.13249 | Page: 1
💬 R2AG: Incorporating Retrieval Information into Retrieval
Augmented Generation...
--- Result 2 | Score: 0.8182 ---
📄 Paper: 2502.01113 | Page: 12
💬 Retrieval-augmented generation for large language models:
A survey...
⏱  Retrieved in 920ms
---

## Engineering Notes

**Why local CPU embeddings?**
BGE-small-en-v1.5 produces 384-dim embeddings competitive with OpenAI's `text-embedding-3-small` while running entirely on CPU at ~5 chunks/sec. This eliminates per-token costs and works behind corporate firewalls.

**Why normalized cosine similarity?**
Embeddings are L2-normalized at generation time, which means cosine similarity reduces to a dot product — faster computation, cleaner geometry for retrieval ranking.

**Why hybrid retrieval (planned)?**
Dense retrieval captures semantic meaning but can miss exact-term matches — acronyms, paper IDs, rare technical jargon. BM25 catches those. The two are combined using Reciprocal Rank Fusion (Cormack et al., 2009), which sidesteps the problem of incompatible score scales (cosine ranges 0-1, BM25 ranges 0-50+) by fusing on rank position rather than absolute scores. In practice this surfaces chunks that one method alone would have buried, demonstrably improving recall on technical queries.

**Why Neo4j for GraphRAG?**
Some research questions are inherently relational — "papers that cite X and also propose Y." Pure vector search can't express graph traversal; a knowledge graph layer can.

---

## Roadmap

- [ ] Hybrid retrieval (BM25 + dense + RRF)
- [ ] Cross-encoder re-ranking
- [ ] Contextual retrieval (Anthropic technique)
- [ ] Evaluation framework (RAGAS metrics, golden set)
- [ ] LangGraph agent with multi-tool routing
- [ ] Neo4j knowledge graph + GraphRAG queries
- [ ] FastAPI backend with streaming
- [ ] Next.js frontend
- [ ] Production deployment (Railway + Vercel)
- [ ] Observability dashboard (Langfuse + Grafana)

---

## License

MIT

---

*Built as an independent project to explore production-grade RAG architectures.*
