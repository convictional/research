# TurboQuant compresses our embeddings 10x with near-zero retrieval loss

## Setup

Tested TurboQuant (arXiv:2504.19874) on 1,000 production `text-embedding-3-small` embeddings (1536-dim, float32) sampled from the content table. Self-contained implementation: random orthogonal rotation + Lloyd-Max scalar quantization at 2/3/4-bit widths. Evaluated against 130 real search queries from the researchquery table, measuring pairwise similarity preservation and query ranking agreement.

## Results

Compression ratios include amortized per-batch overhead (scale/shift vectors); at 1k vectors the overhead is ~12 bytes/vector.

| Bit Width | Compression | Bytes/Vec | Pairwise Spearman | Sim MAE  | Top-10 Overlap | RBO   |
|-----------|------------|-----------|-------------------|----------|----------------|-------|
| 2-bit     | 15.5x      | 396 B     | 0.9919            | 0.0285   | 88.8%          | 0.898 |
| 3-bit     | 10.4x      | 588 B     | 0.9978            | 0.0083   | 94.3%          | 0.941 |
| 4-bit     | 7.9x       | 780 B     | 0.9994            | 0.0029   | 96.0%          | 0.966 |
| Original  | 1.0x       | 6,144 B   | 1.0000            | 0.0000   | 100.0%         | 1.000 |

Code-level inner product search (computing scores directly from integer codes per paper Algorithm 2, grouping dimensions by centroid assignment without materializing float vectors) tracks the dequantized path closely -- no meaningful gap between the two evaluation modes.

## Key observations

**3-bit is the sweet spot.** 10.4x compression, 94.3% top-10 overlap, Spearman 0.9978. The jump from 2-bit to 3-bit is large (MSE drops 10x, top-10 overlap +5.5pp), while 3-bit to 4-bit is marginal (+1.7pp overlap for 2.5x less compression).

**Displacement is bounded.** When a document drops out of the top-10 under compression, it lands at rank ~11 on average across all bit widths. Worst-case displacement is modest -- the errors are near-misses at the ranking boundary, not catastrophic re-orderings.

**The technique is fast and data-oblivious.** Quantizing 1,000 vectors takes ~20ms. No calibration data, no training, no per-dataset tuning. The rotation matrix and codebook are derived purely from mathematical properties (Gaussian marginals after random rotation in high dimensions).

**Two retrieval paths, very different latency profiles.** We benchmarked decompress-then-scan (Option 2 in production: decompress on ingest, search float32 in pgvector) vs. code-level inner product (Option 3: score directly from integer codes, no decompression). At 1,000 docs x 130 queries:

| Path | 2-bit | 3-bit | 4-bit |
|------|-------|-------|-------|
| Float32 scan (baseline) | 4.1 ms | 4.1 ms | 4.1 ms |
| Decompress (one-time) + scan | 15.6 ms | 15.3 ms | 15.3 ms |
| Code-level IP (on integer codes) | 261 ms | 507 ms | 995 ms |

Decompress-then-scan pays a one-time ~11ms decompression cost, after which scan speed is identical to float32. Code-level IP (Algorithm 2 -- scoring from integer codes by looping over 2^b centroids) is 60-240x slower in numpy. A C/CUDA implementation would close this gap, but in pure Python the decompress path dominates.

## Implications

At 3-bit, our ~6 KB/vector drops to ~588 B/vector. For a 100K-document corpus that's ~585 MB down to ~56 MB in embedding storage alone. The retrieval quality loss (5.7% of top-10 results shift by ~1 rank position) is likely imperceptible in a hybrid search system where vector similarity carries 70% weight blended with text rank.

The dequantized vectors are just regular float32 `vector(1536)` -- pgvector needs no patching. The production path would be: store compressed codes in a table that FK's back to content, decompress into the existing embedding column on ingest/backfill. The HNSW index operates on the dequantized float32 as normal.


## Appendix
### TurboQuant: why code-level IP is slow and how decompression scales

#### Context

After running the TurboQuant experiment on our `text-embedding-3-small` embeddings, a few things weren't obvious from the paper alone. This note captures the answers to avoid re-deriving them later.

### How does decompression cost scale with corpus size?

Linearly. The bottleneck in `dequantize()` is the inverse rotation: an `(N, 1536) @ (1536, 1536)` matmul. At 1k vectors it's ~11ms, so at 100k it'd be ~1.1s, at 1M ~11s.

But it's a one-time cost. In a deep research job, you'd load the compressed corpus (10x smaller on disk/wire), decompress once into float32 in memory, and then every retrieval call is a normal matmul against the decompressed matrix. You pay the decompression tax once per session, not per query.

### Why is code-level inner product (Algorithm 2) so much slower than float32 search?

Our benchmark showed code-level IP is 60-240x slower than float32 matmul in numpy:

| Path | 2-bit | 3-bit | 4-bit |
|------|-------|-------|-------|
| Float32 scan (130 queries x 1k docs) | 4.1 ms | 4.1 ms | 4.1 ms |
| Code-level IP | 261 ms | 507 ms | 995 ms |

The cost scales as `2^b` (4:8:16 for 2:3:4-bit), and the reason is the loop structure of Algorithm 2.

#### How Algorithm 2 actually works

The query is **not** compressed. It stays float32, gets rotated into the same domain as the codes, but is never quantized. The algorithm then exploits the fact that there are only `2^b` distinct centroid values across all document dimensions.

Concrete example — 3 docs, 4 dims, 2-bit (4 centroids: c0, c1, c2, c3):

```
         dim0  dim1  dim2  dim3
doc0:  [  1,    3,    0,    2  ]   ← centroid indices, not floats
doc1:  [  2,    1,    1,    3  ]
doc2:  [  0,    2,    3,    1  ]
```

Query (rotated, still float32): `q = [q0, q1, q2, q3]`.

The **naive** way: dequantize each doc to floats, then dot product. For doc0: `c1*q0 + c3*q1 + c0*q2 + c2*q3`.

**Algorithm 2** flips the loop. Instead of iterating over documents, it iterates over centroid levels. For each level, it asks: "which (doc, dim) entries were assigned to this centroid?"

```
Level 0 (c0):
  mask = (codes == 0)  →  [[0,0,1,0],    ← doc0 used c0 at dim2
                            [0,0,0,0],    ← doc1 never used c0
                            [1,0,0,0]]    ← doc2 used c0 at dim0

  mask @ q = [q2, 0, q0]   ← for each doc, sum query components where it used c0
  contribution: c0 * [q2, 0, q0]

Level 1 (c1):
  mask = (codes == 1)  →  [[1,0,0,0],
                            [0,1,1,0],
                            [0,0,0,1]]

  mask @ q = [q0, q1+q2, q3]
  contribution: c1 * [q0, q1+q2, q3]
```

...and so on for levels 2 and 3. Sum all contributions = same answer as the naive dot product, but you never materialized a float array from the codes.

The trick: since there are only `2^b` distinct centroid values, you factor them out. Instead of `N*d` float multiplications, you do `2^b` passes where each pass multiplies one scalar (the centroid value) by a sum of query components.

#### Why it's slow in numpy but fast in theory

In C/CUDA you'd fuse this into a single pass over the integer codes with a lookup table — very cache-friendly on the small integer array.

In numpy, each `mask @ q` allocates and traverses an `(N, 1536)` boolean array, so you get `2^b` full passes over data the same size as the original matrix. The float32 path does it in one pass with BLAS-optimized vectorization.

#### Could precomputing the masks help?

The masks are query-independent (they depend only on document codes), so you can precompute them. But the expensive step isn't building the masks — it's `mask @ q_scaled`, a matrix-vector product on `(N, 1536)` done `2^b` times per query. Precomputing saves the `codes == level` comparison but not the data traversal.

A better restructuring: instead of an `(N, d)` codes matrix, store `2^b` sparse index lists — for each centroid level, which `(doc, dim)` pairs belong to it. Then `mask @ q` becomes a sparse gather-and-sum. This is what the paper means by "group dimensions by centroid assignment."

But even with a perfect sparse implementation, the total arithmetic is identical: at 3-bit, each of 8 levels covers ~192 dimensions on average (1536/8), totaling 1536 components per doc — same as one matmul. The question is memory access pattern. Float32 matmul hits a single contiguous array with BLAS; the sparse approach hits `2^b` scattered index lists. BLAS wins on modern hardware unless float32 doesn't fit in cache but codes do.

#### The bottom line

Code-level IP only wins when:
1. The corpus is too large for float32 to fit in RAM/cache (the 10x smaller codes *do* fit), AND
2. You have a low-level implementation (C/CUDA) that can exploit the compact representation

For our scale and stack (pgvector + numpy), decompress-then-scan dominates.
