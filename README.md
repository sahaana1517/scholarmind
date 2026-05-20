# ScholarMind

An agentic RAG (Retrieval-Augmented Generation) platform for research papers. Ask questions about machine learning research and get cited answers from a corpus of 30 arXiv papers.

🌐 **Live Demo:** https://scholarmind-ui.vercel.app

---

## What It Does

ScholarMind is a multi-tool AI research assistant that:

- Answers natural-language questions about ML research papers with **inline citations**
- **Plans** which tool to use (semantic search, paper comparison, methodology lookup, or graph query) based on the question
- Retrieves evidence using **hybrid search** (BM25 + dense embeddings + Reciprocal Rank Fusion)
- Queries a **Neo4j knowledge graph** for relationship-based questions
- Synthesizes cited answers using **Llama 3.3 70B** via Groq
- Refuses out-of-corpus questions honestly (won't hallucinate)

Example queries it handles well:
- *"What is retrieval augmented generation?"* → semantic search
- *"How does R²AG differ from GFM-RAG?"* → comparison search
- *"How does Curator's multi-tenant index work?"* → methodology retrieval
- *"Which papers study approximate nearest neighbor search?"* → graph query
- *"How to cook pasta?"* → honest refusal (out of corpus)

---

## Architecture

┌──────────────┐
│  Next.js UI  │  (Vercel)
└──────┬───────┘
│  HTTPS
▼
┌──────────────┐
│   FastAPI    │  (Railway)
│   /chat      │
└──────┬───────┘
│
▼
┌─────────────────────────────────────┐
│         LangGraph Agent             │
│                                     │
│  planner → tool routing → synthesis │
└──┬────────┬─────────┬────────┬──────┘
│        │         │        │
▼        ▼         ▼        ▼
search   compare  methodology graph
papers   papers   extraction   query
│        │         │         │
▼        ▼         ▼         ▼
[Qdrant + BM25]            [Neo4j]
hybrid retrieval     knowledge graph
30 papers              30 papers
1103 chunks         105 methods
80 concepts
319 relationships


---

## Key Results

Evaluated retrieval performance against a 25-query benchmark covering single-paper, multi-paper, and methodology questions:

| Method | MRR | Recall@5 | Recall@10 | Hit@1 | Latency |
|---|---|---|---|---|---|
| Dense only | 0.840 | 0.887 | 0.900 | 0.760 | 455 ms |
| **Hybrid (BM25 + Dense + RRF)** | **0.893** | **0.900** | **1.000** | **0.840** | 598 ms |
| Hybrid + Reranked | 0.847 | 0.879 | 1.000 | 0.760 | 2704 ms |

**Hybrid retrieval wins +6.3% MRR and +10.5% Hit@1 over dense-only baseline.**

The cross-encoder reranker degraded performance due to domain mismatch (trained on MS-MARCO, tested on academic prose).

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16, TypeScript, Tailwind CSS |
| **Backend** | FastAPI, Pydantic v2, async/await |
| **Agent** | LangGraph (state machine), Groq (Llama 3.3 70B planner & generator) |
| **Embeddings** | BGE-small-en-v1.5 (local CPU, 384-dim) |
| **Vector DB** | Qdrant Cloud |
| **Knowledge Graph** | Neo4j AuraDB |
| **Sparse Retrieval** | rank_bm25 |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Observability** | Langfuse |
| **Hosting** | Vercel (frontend), Railway (backend) |

---

## Features

### Hybrid Retrieval
Combines BM25 (sparse) and dense embeddings via Reciprocal Rank Fusion. Captures both exact keyword matches (paper titles, technical terms) and semantic similarity.

### LangGraph Agent
Multi-step state machine: planner → tool execution → synthesis. The planner LLM picks one of 4 tools based on query intent, with JSON-structured output for reliable routing.

### Knowledge Graph
LLM-extracted entities (methods, concepts, authors, citations) stored in Neo4j. Enables relationship queries like *"Find papers similar to X by shared methods"* that vector search struggles with.

### Citation-Grounded Generation
Every claim in the answer is tagged with inline `[1][2]` citations referencing the retrieved chunks. The generator refuses to answer when retrieval returns no relevant chunks.

### Contextual Retrieval
Implemented Anthropic's contextual retrieval technique — prepending paper-level context to each chunk before embedding. Improves recall on chunks that lack standalone context.

### Production Observability
Every LLM call, retrieval, and tool execution is traced in Langfuse with latency, token usage, and inputs/outputs.

---

## Repository Structure

scholarmind/
├── backend/
│   ├── app/
│   │   ├── core/             # config, observability
│   │   ├── ingestion/        # PDF extraction, chunking, embeddings, entity extraction
│   │   ├── retrieval/        # BM25, Qdrant, hybrid search, reranker, graph queries
│   │   ├── agents/           # LangGraph agent, planner, tools, generator
│   │   └── api/              # FastAPI app, endpoints, schemas
├── scholarmind-ui/           # Next.js frontend
│   ├── app/                  # pages, layout
│   └── lib/                  # API client
├── experiments/              # Evaluation framework + benchmark
├── data/
│   └── papers_processed/     # extracted chunks, BM25 index, entities
├── docs/                     # design notes
├── requirements.txt
├── Procfile
└── runtime.txt

---

## Running Locally

### Prerequisites
- Python 3.12+
- Node.js 18+
- A Groq API key (free)
- A Qdrant Cloud account (free)
- A Neo4j AuraDB instance (free)
- A Langfuse account (free, optional)

### 1. Clone and install backend
```bash
git clone https://github.com/sahaana1517/scholarmind.git
cd scholarmind

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Create `.env` at project root
```env
GROQ_API_KEY=gsk_...
QDRANT_URL=https://....cloud.qdrant.io
QDRANT_API_KEY=...
NEO4J_URI=neo4j+ssc://....databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Ingest data (one-time, ~10 minutes)
```bash
python -m backend.app.ingestion.download_papers
python -m backend.app.ingestion.pdf_extractor
python -m backend.app.ingestion.chunker
python -m backend.app.ingestion.embedder
python -m backend.app.retrieval.indexer
python -m backend.app.retrieval.bm25_index
python -m backend.app.ingestion.entity_extractor
python -m backend.app.ingestion.graph_loader
```

### 4. Start the backend
```bash
uvicorn backend.app.api.main:app --reload --port 8000
```

API docs at http://localhost:8000/docs

### 5. Start the frontend (separate terminal)
```bash
cd scholarmind-ui
npm install
npm run dev
```

UI at http://localhost:3000

---

## Evaluation Framework

The `experiments/` directory contains a 25-query benchmark spanning:
- Single-paper questions (e.g., "What is R²AG?")
- Multi-paper questions (e.g., "Compare GFM-RAG and R²AG")
- Methodology questions (e.g., "How does Curator's index work?")

Run the evaluation:
```bash
python -m experiments.evaluate_retrieval
```

Metrics computed: MRR, Recall@5, Recall@10, Hit@1, Hit@3, mean latency.

---

## Known Limitations

- **Graph data sparsity:** The entity extractor was precise rather than exhaustive — methods like BM25 don't appear in the graph because papers reference them as baselines rather than as their own techniques.
- **Reranker domain mismatch:** The cross-encoder is trained on MS-MARCO (web Q&A) and degrades performance on academic prose. A domain-specific reranker would help.
- **Concept normalization:** "Retrieval-Augmented Generation" and "Retrieval-augmented generation" don't perfectly merge in the graph due to punctuation differences.
- **Cold-start latency:** The first request after the backend has been idle takes ~30 seconds while the embedding model loads.

---

## What I Learned

- **Agentic systems are mostly about planning, not retrieval.** Getting the planner LLM to choose the right tool reliably was harder than building the tools themselves.
- **Hybrid retrieval > rerankers (for academic prose).** A simple BM25+dense RRF combination beat a fine-tuned cross-encoder reranker in our corpus.
- **Knowledge graphs complement vector search.** Relationship questions ("find similar by shared methods") are 10-20x faster on the graph than via embedding lookup.
- **Prompt engineering bugs surface in surprising ways.** Our entity extractor was leaking example concepts into outputs until we redesigned the prompt with category-only guidance.
- **Observability matters from day one.** Langfuse traces helped diagnose multiple silent failures (prompt leak, retriever drift, planner misrouting).

---

## License

MIT

---

Built by [Sahaana](https://github.com/sahaana1517).