# Multi-Vector Search Experiment

Validating whether Multi-Vector (ColBERT) search outperforms Single-Vector search for Decide's content, with analysis of performance trade-offs.

## TL;DR - End-to-End Workflow

See [RESULTS.md](RESULTS.md) for more detail on the experiment setup and results of the comparisons. Also available via the task [internal app task, not public]


```bash
make install              # Install dependencies
make index                # Index content with ColBERT (one-time, ~10min)
make generate-queries     # Generate test queries with Claude (~2min)
make pool-results         # Run all systems, capture latency (~15min)
make annotate             # Annotate relevance (http://localhost:8000)
make evaluate-quick       # Compare systems with your annotations
```

**Output**: Quality metrics (MRR, Recall@K, NDCG@10) + latency stats + statistical significance tests comparing 4 retrieval systems.

## Background

An earlier internal project explored multi-vector representations (writeup is internal and not public). This experiment builds a production-grade ColBERT implementation with:
- Hand-rolled MaxSim scorer (no black-box dependencies)
- Postgres storage using pgvector arrays (`vector(1024)[]`)
- Battle-tested dependencies only (torch, transformers)
- Comparison against production baseline + API-based reranking

## Architecture

```
src/
├── embedders/       # ColBERT + OpenAI embedding generation
├── search/          # 4 retrieval systems (see below)
├── data/            # Content extraction from postgres
├── storage/         # Postgres operations for token embeddings
├── evaluation/      # Query generation, pooling, metrics, annotation UI
└── models/          # Pydantic data models
```

**4 Retrieval Systems Compared:**
1. **colbert_local** - Local ColBERT with MaxSim scoring
2. **openai_embedding** - Single-vector cosine similarity baseline
3. **production_hybrid** - 70% vector + 30% keyword (current production)
4. **production_reranked** - Production top-50 → Jina API ColBERT rerank → top-10

## Quick Start

```bash
# 1. Setup
make install

# 2. Index content with ColBERT (generates token embeddings)
make index-small  # 50 records for testing
make index        # all records

# 3. View statistics
make stats

# 4. Ad-hoc search testing
make search QUERY="email integration decisions"
make compare QUERY="budget approvals"
```

## Full Evaluation Pipeline

```bash
# 1. Generate test queries (uses Claude Sonnet 4.5 with extended thinking)
#    - 50 docs/batch, 25 queries/batch
#    - ~375 total queries with ground truth
make generate-queries

# 2. Pool search results from all 4 systems + capture latency
#    - Runs 375 queries × 4 systems = 1,500 searches
#    - Saves latency data for all queries
make pool-results

# 3. Annotate relevance (opens UI at http://localhost:8000)
#    - Keyboard shortcuts: 0-3 for rating
#    - Auto-saves progress incrementally
#    - Resume anytime
make annotate

# 4. Run evaluation and generate comparison report
make evaluate-quick  # Uses pre-computed latency from pooling
```

**Output:**
- Terminal summary: Quality metrics for all 4 systems
- 6 pairwise comparisons with statistical significance
- Latency statistics from 375 queries (not sample)
- `results/evaluation_report.json` with detailed metrics

## Implementation Details

**ColBERT Model**: `jinaai/jina-colbert-v2` (1024-dim embeddings, 8K token context)

**MaxSim Scoring**: For each query token, compute max similarity across all document tokens, then sum.

**Storage**: Token embeddings stored as `vector(1024)[]` in postgres content table

**Metrics**: MRR, Recall@1/5/10, NDCG@10, per-query latency (mean, median, P95, P99)

**Query Generation**: Claude Sonnet 4.5 with extended thinking, 50-doc batches, multi-doc relevance

**Evaluation Methodology**: TREC-style pooling with 4-point relevance scale (0=not relevant, 3=highly relevant)

## Previous Work

- Project spec from the earlier internal project [internal doc, not public]
- Draft PR from the earlier internal project [internal PR, not public]
- [Final presentation (Sofia)](https://drive.google.com/file/d/1-RtOkljtW1w_JJDxhv3w6HwqjM-5ztsr/view?usp=sharing)
