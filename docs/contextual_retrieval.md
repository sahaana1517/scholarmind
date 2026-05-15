# Contextual Retrieval Experiment

## Background

Standard retrieval embeds each chunk in isolation. A chunk that reads
*"This achieves 89% on the benchmark"* gives the embedding model nothing
about *which* benchmark or *which* method — context that the rest of the
paper provides but the chunk loses when extracted.

[Anthropic's contextual retrieval technique](https://www.anthropic.com/news/contextual-retrieval)
addresses this by prepending each chunk with a short LLM-generated summary
that situates the chunk within its parent document. Published results report
30-50% reduction in retrieval failures.

## Implementation

The full pipeline is implemented in this repo:

- `backend/app/ingestion/contextualizer.py` — generates per-chunk context via
  Groq (Llama 3.1 8B), checkpointed every 25 chunks for resumability
- `backend/app/ingestion/contextual_embedder.py` — re-embeds `context + chunk`
  using BGE-small
- `backend/app/retrieval/contextual_indexer.py` — uploads to a separate Qdrant
  collection (`scholarmind_papers_contextual`) so baseline is untouched
- `experiments/compare_contextual.py` — side-by-side qualitative comparison

The technique was validated end-to-end on a 200-chunk subset spanning 10 papers.

## Scope Note

The contextualization step requires one LLM API call per chunk. During
processing, network latency made full-corpus contextualization (1103 chunks)
impractical within the available development window. The pipeline supports
incremental resumption, so the remaining ~900 chunks can be processed in a
future session.

For methodological rigor, the qualitative comparison was conducted only on
queries whose relevant papers are within the 200-chunk subset. A formal
quantitative A/B (MRR / Recall@K comparison) was deferred because the eligible
query count (n=7) is too small for statistically meaningful claims.

## Qualitative Findings

Three queries were compared side-by-side against the same Qdrant deployment,
using identical dense retrieval against baseline vs contextual collections.

**Pattern 1: Contextual scoring is consistently higher.**
On the LLM-hacking query, contextual scores were 0.03-0.04 higher across all
top 5 results vs baseline. This is consistent with Anthropic's claim that
prepending context yields more semantically-rich embeddings.

**Pattern 2: Contextual surfaces deeper technical chunks first.**
For the multi-tenant indices query, baseline returned the paper's title page
at rank 1 (keyword-matched but shallow). Contextual returned page 3, which
actually discusses the memory-isolation mechanism being asked about.

**Pattern 3: For specific keyword queries, both methods converge.**
When the query directly mentions paper-specific entities ("Curator scalability"),
both methods return the same top chunks at near-identical scores. Context
helps most when the query is paraphrased away from the source language.

## Why We Kept the Baseline

The previous evaluation (see `experiments/eval_results.json`) showed that
**hybrid retrieval (BM25 + dense + RRF)** was the strongest single technique
on the full corpus. Hybrid retrieval was kept as the production default.

Contextual retrieval is implemented but not deployed as the default — both
because of the partial corpus coverage and because the qualitative gains,
while real, were modest on a corpus this small. The technique is most
impactful at scale (Anthropic's benchmarks used corpora 100-1000x ours).

## Reproduce

```bash
# Generate context for all chunks (long-running)
python -m backend.app.ingestion.contextualizer

# Embed with context prepended
python -m backend.app.ingestion.contextual_embedder

# Upload to dedicated Qdrant collection
python -m backend.app.retrieval.contextual_indexer

# Compare side-by-side
python -m experiments.compare_contextual
```

## Future Work

- Complete contextualization on the remaining ~900 chunks
- Run formal A/B with statistically sufficient query count (target n≥50)
- Test whether a smaller, distilled context (20-30 words) gives most of the
  retrieval benefit at a fraction of the LLM cost