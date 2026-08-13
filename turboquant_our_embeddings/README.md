# TurboQuant Our Embeddings

**Author:** Adam McCabe

## Hypothesis

[TurboQuant](https://arxiv.org/pdf/2504.19874) (arXiv:2504.19874) can compress our 1536-dim `text-embedding-3-small` embeddings to 3-bit precision (~6x compression) while preserving cosine similarity ranking with near-zero accuracy loss.

## Conclusion

Confirmed. 3-bit TurboQuant gives 10.7x compression with 0.998 pairwise Spearman and 94% top-10 overlap against 130 real queries. The code-level inner product path (operating directly on integer codes per paper Algorithm 2) gives nearly identical ranking to the dequantized path. Scan speed is identical once decompressed; the value is storage compression, not compute.

Full run summary can be found in [RUN_SUMMARY.md](RUN_SUMMARY.py).

## Background

We store OpenAI `text-embedding-3-small` embeddings as `vector(1536)` in PostgreSQL (pgvector), indexed with HNSW (`vector_cosine_ops`). Each float32 vector costs 6,144 bytes.

TurboQuant is a data-oblivious quantization technique from Google Research that compresses vectors to 1-4 bits per coordinate. It works by:
1. Applying a random orthogonal rotation (makes coordinates follow a predictable Gaussian-like distribution)
2. Scalar quantization with a precomputed Lloyd-Max codebook (optimal for the rotated distribution)
3. No calibration data needed -- purely mathematical

The paper proves inner product preservation is unbiased with bounded variance, which directly applies to cosine similarity (a normalized inner product).

## Method

1. Sample ~1000 content embeddings from the local dev database
2. For each bit width (2, 3, 4):
   - Quantize all vectors using TurboQuant
   - Dequantize back to float32
   - Measure pairwise cosine similarity preservation (Spearman rho, MSE, MAE)
   - Measure query ranking preservation using real search queries (top-10 overlap, RBO, displacement)
   - Compare code-level inner product (paper Algorithm 2) vs float32 cosine search
   - Benchmark retrieval latency (float32 scan vs decompress-then-scan)

## How to Run

```bash
cd experiments

# Smoke test (fast)
make run_experiment ARGS="turboquant_our_embeddings --limit 50 --pairs 100"

# Full run
make run_experiment ARGS="turboquant_our_embeddings --limit 1000 --pairs 5000"

# Custom bit widths
make run_experiment ARGS="turboquant_our_embeddings --bit-widths 3 4"
```

## Findings

| Bit Width | Compression | Bytes/Vec | Pairwise Spearman | Top-10 Overlap | RBO |
|-----------|-------------|-----------|-------------------|----------------|-----|
| 2-bit | 16.0x | 384 B | 0.992 | 88.8% | 89.8% |
| 3-bit | 10.7x | 576 B | 0.998 | 94.3% | 94.1% |
| 4-bit | 8.0x | 768 B | 0.999 | 96.0% | 96.6% |

3-bit is the sweet spot. Scan speed is identical once decompressed (~3.9ms for 130 queries x 1k docs). Decompression is a one-time cost of ~12ms for 1k vectors.
