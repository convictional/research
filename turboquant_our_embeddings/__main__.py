import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from turboquant_our_embeddings.compare import (
    code_level_ip_ranking,
    pairwise_comparison,
    print_compression_summary,
    print_error_result,
    print_query_ranking_result,
    print_spearman_result,
    query_ranking_comparison,
    save_bit_width_summary_plot,
    save_error_distribution_plot,
    save_overlay_histogram,
    save_query_ranking_histogram,
    similarity_error_analysis,
)
from turboquant_our_embeddings.db import fetch_content_with_embeddings, fetch_search_queries
from turboquant_our_embeddings.embed import embed_with_openai
from turboquant_our_embeddings.quantizer import TurboQuantCompressor

load_dotenv(Path(__file__).parent.parent / ".env.secrets")

DB_CONNECTION_STRING = "postgresql://localhost/decide_development"


async def main(limit: int, num_pairs: int, bit_widths: list[int], seed: int) -> None:
    # --- Load data ---
    print(f"\nFetching up to {limit} content records with OpenAI embeddings...")
    records = await fetch_content_with_embeddings(DB_CONNECTION_STRING, limit=limit)
    print(f"Fetched {len(records)} records")

    if len(records) < 2:
        print("Need at least 2 records for comparison. Exiting.")
        return

    original_embeddings = [r.openai_embedding for r in records]
    original_matrix = np.array(original_embeddings, dtype=np.float32)
    print(f"Embedding matrix shape: {original_matrix.shape}, dtype: {original_matrix.dtype}")

    # Precompute normalized docs for float32 retrieval baseline
    norms = np.linalg.norm(original_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_normed = original_matrix / norms

    # --- Fetch and embed queries ---
    queries = await fetch_search_queries(DB_CONNECTION_STRING)
    print(f"\nFetched {len(queries)} search queries")

    query_embeddings: list[list[float]] = []
    if queries:
        print(f"Embedding {len(queries)} queries with OpenAI text-embedding-3-small...")
        query_result = embed_with_openai(queries)
        query_embeddings = query_result.embeddings
        print(f"Embedded {len(query_embeddings)} queries ({query_result.texts_embedded} API calls)")

    # --- Float32 retrieval latency baseline ---
    float32_retrieval_ms = 0.0
    if query_embeddings:
        query_matrix = np.array(query_embeddings, dtype=np.float32)
        q_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        query_normed = query_matrix / q_norms

        warmup_rounds = 5
        bench_rounds = 50
        for _ in range(warmup_rounds):
            doc_normed @ query_normed[0]

        float32_times: list[float] = []
        for _ in range(bench_rounds):
            t0 = time.monotonic()
            for q in query_normed:
                scores = doc_normed @ q
                scores.argpartition(-10)[-10:]
            float32_times.append(time.monotonic() - t0)

        float32_retrieval_ms = float(np.median(float32_times) * 1000)
        per_query_ms = float32_retrieval_ms / len(query_embeddings)
        print(f"\n  Float32 retrieval baseline ({len(query_embeddings)} queries x {len(records)} docs):")
        print(f"    Total: {float32_retrieval_ms:.3f} ms (median over {bench_rounds} rounds)")
        print(f"    Per query: {per_query_ms:.3f} ms")

    # --- Run experiment for each bit width ---
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    all_results: dict[int, dict] = {}
    summary_spearman: list[float] = []
    summary_mses: list[float] = []
    summary_top_k: list[float] = []
    summary_rbos: list[float] = []
    summary_ratios: list[float] = []
    summary_memory: list[float] = []
    summary_retrieval_ms: list[float] = []

    for bw in bit_widths:
        print(f"\n{'#' * 70}")
        print(f"  {bw}-BIT TURBOQUANT")
        print(f"{'#' * 70}")

        compressor = TurboQuantCompressor(dim=1536, bit_width=bw, seed=seed)

        # Quantize
        t0 = time.monotonic()
        compressed = compressor.quantize(original_matrix)
        quantize_time = time.monotonic() - t0
        print(f"\n  Quantize:   {quantize_time:.3f}s  ({len(records)} vectors)")

        # Dequantize
        t0 = time.monotonic()
        dequantized_matrix = compressor.dequantize(compressed)
        dequantize_time = time.monotonic() - t0
        print(f"  Dequantize: {dequantize_time:.3f}s")

        dequantized_embeddings = dequantized_matrix.tolist()

        # Compression metrics
        ratio = compressor.compression_ratio(num_vectors=len(records))
        mem_per_vec = compressor.memory_per_vector_bytes(num_vectors=len(records))
        print(f"  Compression ratio: {ratio:.1f}x  ({mem_per_vec:.0f} bytes/vector vs 6144 bytes original)")

        # Reconstruction MSE (vector-level)
        vec_mse = float(np.mean((original_matrix - dequantized_matrix) ** 2))
        print(f"  Vector reconstruction MSE: {vec_mse:.6f}")

        # --- Pairwise similarity preservation ---
        print(f"\n  Computing pairwise similarity comparison ({num_pairs} pairs)...")
        pairwise = pairwise_comparison(original_embeddings, dequantized_embeddings, num_pairs=num_pairs, seed=seed)
        print_spearman_result(pairwise.spearman_rho, pairwise.p_value, f"Pairwise Spearman: Original vs {bw}-bit")

        # Similarity error analysis
        errors = similarity_error_analysis(original_embeddings, dequantized_embeddings, num_pairs=num_pairs, seed=seed)
        print_error_result(errors, f"Similarity Error: {bw}-bit TurboQuant")

        # Save plots
        save_overlay_histogram(
            pairwise.sims_a,
            pairwise.sims_b,
            bit_width=bw,
            spearman_rho=pairwise.spearman_rho,
            output_path=output_dir / f"pairwise_{bw}bit.png",
        )
        save_error_distribution_plot(
            errors.per_pair_errors,
            bit_width=bw,
            output_path=output_dir / f"error_distribution_{bw}bit.png",
        )

        # --- Query ranking evaluation ---
        qr_deq = None
        qr_native = None
        if query_embeddings:
            print(f"\n  Computing query ranking comparison ({len(query_embeddings)} queries, {len(records)} docs)...")

            # Dequantized path: same queries, original docs vs dequantized docs
            qr_deq = query_ranking_comparison(
                query_embeddings,
                query_embeddings,
                original_embeddings,
                dequantized_embeddings,
            )
            print_query_ranking_result(qr_deq, f"Dequantized {bw}-bit vs Original")

            save_query_ranking_histogram(
                qr_deq,
                label=f"Dequantized {bw}-bit vs Original",
                output_path=output_dir / f"query_ranking_deq_{bw}bit.png",
            )

            # Code-level inner product path (paper Algorithm 2)
            print(f"  Computing code-level IP ranking comparison...")
            qr_native = code_level_ip_ranking(
                query_embeddings,
                original_matrix,
                compressed,
                compressor,
            )
            print_query_ranking_result(qr_native, f"Code-level IP {bw}-bit vs Float32 Cosine")

            save_query_ranking_histogram(
                qr_native,
                label=f"Code-level IP {bw}-bit vs Float32",
                output_path=output_dir / f"query_ranking_code_ip_{bw}bit.png",
            )

        # --- Retrieval latency benchmark ---
        compressed_retrieval_ms = 0.0
        decompress_ms = 0.0
        deq_scan_ms = 0.0
        code_ip_ms = 0.0
        if query_embeddings:
            bench_rounds = 50

            # Path A: decompress once, then scan (same as float32 after one-time cost)
            decompress_times: list[float] = []
            for _ in range(bench_rounds):
                t0 = time.monotonic()
                deq = compressor.dequantize(compressed)
                decompress_times.append(time.monotonic() - t0)
            decompress_ms = float(np.median(decompress_times) * 1000)

            deq_norms_vec = np.linalg.norm(deq, axis=1, keepdims=True)
            deq_norms_vec[deq_norms_vec == 0] = 1.0
            deq_normed = deq / deq_norms_vec

            for _ in range(5):
                deq_normed @ query_normed[0]

            deq_scan_times: list[float] = []
            for _ in range(bench_rounds):
                t0 = time.monotonic()
                for q in query_normed:
                    scores = deq_normed @ q
                    scores.argpartition(-10)[-10:]
                deq_scan_times.append(time.monotonic() - t0)
            deq_scan_ms = float(np.median(deq_scan_times) * 1000)
            compressed_retrieval_ms = decompress_ms + deq_scan_ms

            # Path B: code-level inner product directly on integer codes (no decompression)
            query_matrix = np.array(query_embeddings, dtype=np.float32)
            for _ in range(5):
                compressor.code_inner_product(query_matrix[0], compressed)

            code_ip_times: list[float] = []
            for _ in range(bench_rounds):
                t0 = time.monotonic()
                for q in query_matrix:
                    scores = compressor.code_inner_product(q, compressed)
                    scores.argpartition(-10)[-10:]
                code_ip_times.append(time.monotonic() - t0)
            code_ip_ms = float(np.median(code_ip_times) * 1000)

            print(f"\n  Retrieval latency ({len(query_embeddings)} queries x {len(records)} docs):")
            print(f"    Float32 scan:              {float32_retrieval_ms:.3f} ms")
            print(f"    Decompress ({bw}-bit):        {decompress_ms:.3f} ms  (one-time)")
            print(f"    Decompressed scan:         {deq_scan_ms:.3f} ms")
            print(f"    Total (decomp + scan):     {compressed_retrieval_ms:.3f} ms")
            print(f"    Code-level IP ({bw}-bit):     {code_ip_ms:.3f} ms")
            per_query_code_ip = code_ip_ms / len(query_embeddings)
            print(f"    Code-level IP per query:   {per_query_code_ip:.3f} ms")

        # Collect results
        bw_result: dict = {
            "bit_width": bw,
            "compression_ratio": ratio,
            "memory_per_vector_bytes": mem_per_vec,
            "quantize_seconds": quantize_time,
            "dequantize_seconds": dequantize_time,
            "vector_reconstruction_mse": vec_mse,
            "pairwise_spearman_rho": pairwise.spearman_rho,
            "pairwise_spearman_p": pairwise.p_value,
            "similarity_mse": errors.mse,
            "similarity_mae": errors.mae,
            "similarity_max_abs_error": errors.max_abs_error,
            "retrieval_float32_scan_ms": float32_retrieval_ms,
            "retrieval_decompress_ms": decompress_ms,
            "retrieval_decompressed_scan_ms": deq_scan_ms,
            "retrieval_decompress_total_ms": compressed_retrieval_ms,
            "retrieval_code_ip_ms": code_ip_ms,
        }
        if qr_deq:
            bw_result["query_ranking_deq"] = {
                "mean_rho": qr_deq.mean_rho,
                "median_rho": qr_deq.median_rho,
                "mean_top_k_overlap": qr_deq.mean_top_k_overlap,
                "mean_rbo": qr_deq.mean_rbo,
                "mean_displacement": qr_deq.mean_displacement,
                "num_queries": qr_deq.num_queries,
            }
        if qr_native:
            bw_result["query_ranking_code_ip"] = {
                "mean_rho": qr_native.mean_rho,
                "median_rho": qr_native.median_rho,
                "mean_top_k_overlap": qr_native.mean_top_k_overlap,
                "mean_rbo": qr_native.mean_rbo,
                "mean_displacement": qr_native.mean_displacement,
                "num_queries": qr_native.num_queries,
            }

        all_results[bw] = bw_result
        summary_spearman.append(pairwise.spearman_rho)
        summary_mses.append(errors.mse)
        summary_top_k.append(qr_deq.mean_top_k_overlap if qr_deq else 0.0)
        summary_rbos.append(qr_deq.mean_rbo if qr_deq else 0.0)
        summary_ratios.append(ratio)
        summary_memory.append(mem_per_vec)
        summary_retrieval_ms.append(compressed_retrieval_ms)

    # --- Summary ---
    print_compression_summary(
        bit_widths=bit_widths,
        compression_ratios=summary_ratios,
        memory_per_vector=summary_memory,
        spearman_rhos=summary_spearman,
        mses=summary_mses,
        top_k_overlaps=summary_top_k,
        rbos=summary_rbos,
        float32_retrieval_ms=float32_retrieval_ms,
        decompress_ms=summary_retrieval_ms,
    )

    save_bit_width_summary_plot(
        bit_widths=bit_widths,
        spearman_rhos=summary_spearman,
        mses=summary_mses,
        top_k_overlaps=summary_top_k,
        output_path=output_dir / "bit_width_comparison.png",
    )

    # Save full results as JSON
    report = {
        "experiment": "turboquant_our_embeddings",
        "num_documents": len(records),
        "num_queries": len(query_embeddings),
        "num_pairs": num_pairs,
        "seed": seed,
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
        "float32_retrieval_ms": float32_retrieval_ms,
        "results_by_bit_width": {str(k): v for k, v in all_results.items()},
    }
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {report_path}")

    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Documents: {len(records)}, Queries: {len(query_embeddings)}, Pairs: {num_pairs}")
    print(f"  Results in: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant compression experiment for OpenAI embeddings")
    parser.add_argument("--limit", type=int, default=1000, help="Number of content records to sample")
    parser.add_argument("--pairs", type=int, default=5000, help="Number of random document pairs for Spearman")
    parser.add_argument("--bit-widths", type=int, nargs="+", default=[2, 3, 4], help="Bit widths to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    asyncio.run(main(args.limit, args.pairs, args.bit_widths, args.seed))
