# Retrieval Method Evaluation

Evaluation set: 25 queries
- 23 single-paper paraphrased queries (no paper names in query)
- 4 cross-paper synthesis queries

Corpus: 30 arXiv papers (1103 chunks)

## Results

| Method | MRR | Recall@5 | Recall@10 | Hit@1 | Hit@3 | Avg Latency |
|---|---|---|---|---|---|---|
| Dense only (BGE-small) | 0.840 | 0.887 | 0.900 | 0.760 | 0.920 | 455ms |
| **Hybrid (Dense + BM25 + RRF)** | **0.893** | **0.900** | **1.000** | **0.840** | 0.920 | 598ms |
| Hybrid + Cross-encoder rerank | 0.847 | 0.879 | 1.000 | 0.760 | 0.920 | 2704ms |

## Key Findings

**Hybrid retrieval is the clear winner on this corpus.**
- +6.3% MRR over dense alone
- +10.5% Hit@1 over dense alone
- Achieves perfect recall@10 (no relevant paper ever missed in top 10)
- 30% latency overhead over dense — acceptable tradeoff

**Cross-encoder reranking degraded results slightly.**
Hypothesis: the ms-marco-MiniLM reranker is trained on web Q&A, which over-weights surface keyword overlap. On paraphrased academic queries with semantic distance from the source text, RRF's rank-level fusion is more robust than the reranker's score-level rescoring.

**Future work:** Try `BAAI/bge-reranker-base` (trained on more diverse data), or fine-tune a reranker on a small academic-paper-specific dataset.