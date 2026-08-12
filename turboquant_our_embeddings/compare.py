from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as mpl_plt
import numpy as np
import plotext as plt
from scipy.stats import spearmanr

from turboquant_our_embeddings.quantizer import QuantizedVectors, TurboQuantCompressor


def _normalize_matrix(embeddings: list[list[float]]) -> np.ndarray:
    mat = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _sample_pairs(n: int, num_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total_possible = n * (n - 1) // 2
    if total_possible <= num_pairs:
        idx = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    else:
        idx = rng.choice(n, size=(num_pairs * 2, 2), replace=True)
        idx = idx[idx[:, 0] != idx[:, 1]][:num_pairs]
    return idx


@dataclass
class PairwiseResult:
    sims_a: list[float]
    sims_b: list[float]
    spearman_rho: float
    p_value: float
    num_pairs: int


def pairwise_comparison(
    embeddings_a: list[list[float]],
    embeddings_b: list[list[float]],
    num_pairs: int = 5000,
    seed: int = 42,
) -> PairwiseResult:
    """Compute pairwise similarities under both representations and Spearman rank correlation."""
    n = len(embeddings_a)
    normed_a = _normalize_matrix(embeddings_a)
    normed_b = _normalize_matrix(embeddings_b)

    pairs = _sample_pairs(n, num_pairs, seed)

    sims_a = [float(normed_a[i] @ normed_a[j]) for i, j in pairs]
    sims_b = [float(normed_b[i] @ normed_b[j]) for i, j in pairs]

    corr, p_value = spearmanr(sims_a, sims_b)
    return PairwiseResult(
        sims_a=sims_a,
        sims_b=sims_b,
        spearman_rho=float(corr),
        p_value=float(p_value),
        num_pairs=len(pairs),
    )


@dataclass
class SimilarityErrorResult:
    mse: float
    mae: float
    max_abs_error: float
    per_pair_errors: list[float]
    original_sims: list[float]
    dequantized_sims: list[float]


def similarity_error_analysis(
    embeddings_original: list[list[float]],
    embeddings_dequantized: list[list[float]],
    num_pairs: int = 5000,
    seed: int = 42,
) -> SimilarityErrorResult:
    """Compute absolute error of cosine similarities between original and dequantized embeddings."""
    n = len(embeddings_original)
    normed_orig = _normalize_matrix(embeddings_original)
    normed_deq = _normalize_matrix(embeddings_dequantized)

    pairs = _sample_pairs(n, num_pairs, seed)

    original_sims = [float(normed_orig[i] @ normed_orig[j]) for i, j in pairs]
    deq_sims = [float(normed_deq[i] @ normed_deq[j]) for i, j in pairs]

    errors = [abs(o - d) for o, d in zip(original_sims, deq_sims)]
    errors_arr = np.array(errors)

    return SimilarityErrorResult(
        mse=float(np.mean(errors_arr**2)),
        mae=float(np.mean(errors_arr)),
        max_abs_error=float(np.max(errors_arr)),
        per_pair_errors=errors,
        original_sims=original_sims,
        dequantized_sims=deq_sims,
    )


@dataclass
class QueryRankingResult:
    per_query_rhos: list[float]
    mean_rho: float
    median_rho: float
    top_k_overlaps: list[float]
    mean_top_k_overlap: float
    per_query_rbos: list[float]
    mean_rbo: float
    per_query_top_k_rhos: list[float]
    mean_top_k_rho: float
    mean_displacement: float
    max_displacement: float
    displacement_percentiles: dict[str, float]
    num_queries: int
    num_docs: int
    k: int


def _rbo(ranked_a: np.ndarray, ranked_b: np.ndarray, p: float = 0.9) -> float:
    """Rank Biased Overlap between two ranked lists."""
    k = len(ranked_a)
    rbo_sum = 0.0
    set_a: set[int] = set()
    set_b: set[int] = set()
    for d in range(1, k + 1):
        set_a.add(int(ranked_a[d - 1]))
        set_b.add(int(ranked_b[d - 1]))
        overlap_at_d = len(set_a & set_b) / d
        rbo_sum += p ** (d - 1) * overlap_at_d
    return (1 - p) * rbo_sum


def query_ranking_comparison(
    query_embeddings_a: list[list[float]],
    query_embeddings_b: list[list[float]],
    doc_embeddings_a: list[list[float]],
    doc_embeddings_b: list[list[float]],
    k: int = 10,
) -> QueryRankingResult:
    """Compare search rankings between original and dequantized embeddings across queries."""
    doc_matrix_a = np.array(doc_embeddings_a)
    doc_matrix_b = np.array(doc_embeddings_b)

    norms_a = np.linalg.norm(doc_matrix_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(doc_matrix_b, axis=1, keepdims=True)
    norms_a[norms_a == 0] = 1.0
    norms_b[norms_b == 0] = 1.0
    doc_normed_a = doc_matrix_a / norms_a
    doc_normed_b = doc_matrix_b / norms_b

    per_query_rhos: list[float] = []
    top_k_overlaps: list[float] = []
    per_query_rbos: list[float] = []
    per_query_top_k_rhos: list[float] = []
    all_displacements: list[int] = []

    num_docs = len(doc_embeddings_a)

    for q_emb_a, q_emb_b in zip(query_embeddings_a, query_embeddings_b):
        q_vec_a = np.array(q_emb_a)
        q_vec_b = np.array(q_emb_b)

        norm_qa = np.linalg.norm(q_vec_a)
        norm_qb = np.linalg.norm(q_vec_b)
        if norm_qa == 0 or norm_qb == 0:
            continue

        sims_a = doc_normed_a @ (q_vec_a / norm_qa)
        sims_b = doc_normed_b @ (q_vec_b / norm_qb)

        corr, _ = spearmanr(sims_a, sims_b)
        per_query_rhos.append(float(corr))

        ranked_a = np.argsort(sims_a)[::-1]
        ranked_b = np.argsort(sims_b)[::-1]

        effective_k = min(k, num_docs)

        top_k_a = set(ranked_a[:effective_k].tolist())
        top_k_b = set(ranked_b[:effective_k].tolist())
        overlap = len(top_k_a & top_k_b) / effective_k
        top_k_overlaps.append(overlap)

        per_query_rbos.append(_rbo(ranked_a, ranked_b, p=0.9))

        union_top_k = list(top_k_a | top_k_b)
        if len(union_top_k) >= 3:
            union_sims_a = sims_a[union_top_k]
            union_sims_b = sims_b[union_top_k]
            top_k_corr, _ = spearmanr(union_sims_a, union_sims_b)
            per_query_top_k_rhos.append(float(top_k_corr))

        rank_in_b = np.empty(num_docs, dtype=int)
        rank_in_b[ranked_b] = np.arange(num_docs)
        for doc_idx in top_k_a - top_k_b:
            all_displacements.append(int(rank_in_b[doc_idx]))

    rho_arr = np.array(per_query_rhos)
    overlap_arr = np.array(top_k_overlaps)
    rbo_arr = np.array(per_query_rbos)
    top_k_rho_arr = np.array(per_query_top_k_rhos) if per_query_top_k_rhos else np.array([0.0])
    disp_arr = np.array(all_displacements) if all_displacements else np.array([0])

    displacement_percentiles = {
        "p50": float(np.median(disp_arr)),
        "p75": float(np.percentile(disp_arr, 75)),
        "p90": float(np.percentile(disp_arr, 90)),
        "p95": float(np.percentile(disp_arr, 95)),
    }

    return QueryRankingResult(
        per_query_rhos=per_query_rhos,
        mean_rho=float(rho_arr.mean()),
        median_rho=float(np.median(rho_arr)),
        top_k_overlaps=top_k_overlaps,
        mean_top_k_overlap=float(overlap_arr.mean()),
        per_query_rbos=per_query_rbos,
        mean_rbo=float(rbo_arr.mean()),
        per_query_top_k_rhos=per_query_top_k_rhos,
        mean_top_k_rho=float(top_k_rho_arr.mean()),
        mean_displacement=float(disp_arr.mean()),
        max_displacement=int(disp_arr.max()),
        displacement_percentiles=displacement_percentiles,
        num_queries=len(per_query_rhos),
        num_docs=num_docs,
        k=k,
    )


def code_level_ip_ranking(
    query_embeddings: list[list[float]],
    doc_matrix: np.ndarray,
    compressed: QuantizedVectors,
    compressor: TurboQuantCompressor,
    k: int = 10,
) -> QueryRankingResult:
    """Compare float32 cosine rankings vs code-level inner product rankings.

    The code-level IP computes scores directly from integer codes by grouping
    dimensions by centroid assignment (paper Algorithm 2), without materializing
    a full (N, dim) float array from codes.
    """
    norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_normed = doc_matrix / norms

    per_query_rhos: list[float] = []
    top_k_overlaps: list[float] = []
    per_query_rbos: list[float] = []
    per_query_top_k_rhos: list[float] = []
    all_displacements: list[int] = []
    num_docs = doc_matrix.shape[0]

    for q_emb in query_embeddings:
        q_vec = np.array(q_emb, dtype=np.float32)
        norm_q = np.linalg.norm(q_vec)
        if norm_q == 0:
            continue

        float32_sims = doc_normed @ (q_vec / norm_q)
        native_scores = compressor.code_inner_product(q_vec, compressed)

        corr, _ = spearmanr(float32_sims, native_scores)
        per_query_rhos.append(float(corr))

        ranked_float32 = np.argsort(float32_sims)[::-1]
        ranked_native = np.argsort(native_scores)[::-1]

        effective_k = min(k, num_docs)
        top_k_f32 = set(ranked_float32[:effective_k].tolist())
        top_k_nat = set(ranked_native[:effective_k].tolist())
        overlap = len(top_k_f32 & top_k_nat) / effective_k
        top_k_overlaps.append(overlap)

        per_query_rbos.append(_rbo(ranked_float32, ranked_native, p=0.9))

        union_top_k = list(top_k_f32 | top_k_nat)
        if len(union_top_k) >= 3:
            top_k_corr, _ = spearmanr(float32_sims[union_top_k], native_scores[union_top_k])
            per_query_top_k_rhos.append(float(top_k_corr))

        rank_in_native = np.empty(num_docs, dtype=int)
        rank_in_native[ranked_native] = np.arange(num_docs)
        for doc_idx in top_k_f32 - top_k_nat:
            all_displacements.append(int(rank_in_native[doc_idx]))

    rho_arr = np.array(per_query_rhos)
    overlap_arr = np.array(top_k_overlaps)
    rbo_arr = np.array(per_query_rbos)
    top_k_rho_arr = np.array(per_query_top_k_rhos) if per_query_top_k_rhos else np.array([0.0])
    disp_arr = np.array(all_displacements) if all_displacements else np.array([0])

    displacement_percentiles = {
        "p50": float(np.median(disp_arr)),
        "p75": float(np.percentile(disp_arr, 75)),
        "p90": float(np.percentile(disp_arr, 90)),
        "p95": float(np.percentile(disp_arr, 95)),
    }

    return QueryRankingResult(
        per_query_rhos=per_query_rhos,
        mean_rho=float(rho_arr.mean()),
        median_rho=float(np.median(rho_arr)),
        top_k_overlaps=top_k_overlaps,
        mean_top_k_overlap=float(overlap_arr.mean()),
        per_query_rbos=per_query_rbos,
        mean_rbo=float(rbo_arr.mean()),
        per_query_top_k_rhos=per_query_top_k_rhos,
        mean_top_k_rho=float(top_k_rho_arr.mean()),
        mean_displacement=float(disp_arr.mean()),
        max_displacement=int(disp_arr.max()),
        displacement_percentiles=displacement_percentiles,
        num_queries=len(per_query_rhos),
        num_docs=num_docs,
        k=k,
    )


# --- Printing ---


def print_distribution_stats(similarities: list[float], label: str) -> None:
    arr = np.array(similarities)
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Count:   {len(arr)}")
    print(f"  Mean:    {arr.mean():.4f}")
    print(f"  Median:  {np.median(arr):.4f}")
    print(f"  Std:     {arr.std():.4f}")
    print(f"  Min:     {arr.min():.4f}")
    print(f"  Max:     {arr.max():.4f}")
    print(f"  P25:     {np.percentile(arr, 25):.4f}")
    print(f"  P75:     {np.percentile(arr, 75):.4f}")


def print_histogram(similarities: list[float], label: str, bins: int = 30) -> None:
    plt.clear_figure()
    plt.hist(similarities, bins=bins)
    plt.title(label)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.show()
    print()


def print_spearman_result(corr: float, p_value: float, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Spearman rho:  {corr:.4f}")
    print(f"  p-value:       {p_value:.2e}")
    if corr > 0.95:
        interpretation = "Excellent -- nearly identical rankings"
    elif corr > 0.90:
        interpretation = "Very strong -- rankings highly correlated"
    elif corr > 0.80:
        interpretation = "Strong -- rankings well correlated"
    elif corr > 0.60:
        interpretation = "Moderate -- noticeable ranking differences"
    else:
        interpretation = "Weak -- substantial ranking differences"
    print(f"  Interpretation: {interpretation}")


def print_error_result(result: SimilarityErrorResult, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  MSE:             {result.mse:.8f}")
    print(f"  MAE:             {result.mae:.6f}")
    print(f"  Max abs error:   {result.max_abs_error:.6f}")
    errors_arr = np.array(result.per_pair_errors)
    print(f"  P50 error:       {np.percentile(errors_arr, 50):.6f}")
    print(f"  P95 error:       {np.percentile(errors_arr, 95):.6f}")
    print(f"  P99 error:       {np.percentile(errors_arr, 99):.6f}")


def print_query_ranking_result(result: QueryRankingResult, label: str) -> None:
    rho_arr = np.array(result.per_query_rhos)
    print(f"\n{'=' * 60}")
    print(f"  Query Ranking: {label}")
    print(f"{'=' * 60}")
    print(f"  Queries:              {result.num_queries}")
    print(f"  Documents:            {result.num_docs}")
    print(f"")
    print(f"  --- Ranking Agreement ---")
    print(f"  Full Spearman rho:    {result.mean_rho:.4f}  (median {result.median_rho:.4f}, std {rho_arr.std():.4f})")
    print(f"  Top-{result.k} Spearman rho:  {result.mean_top_k_rho:.4f}  (over union of both top-{result.k})")
    print(f"  RBO (p=0.9):          {result.mean_rbo:.4f}  (top-weighted overlap)")
    print(
        f"  Top-{result.k} overlap:       {result.mean_top_k_overlap:.1%}"
        f"  ({result.mean_top_k_overlap * result.k:.1f}/{result.k} results shared)"
    )
    print(f"")
    print(f"  --- Displacement of \"Lost\" Results ---")
    print(f"  Mean rank:            {result.mean_displacement:.0f}")
    print(f"  Median rank:          {result.displacement_percentiles['p50']:.0f}")
    print(f"  P90 rank:             {result.displacement_percentiles['p90']:.0f}")
    print(f"  Worst case:           {result.max_displacement:.0f}")


def print_compression_summary(
    bit_widths: list[int],
    compression_ratios: list[float],
    memory_per_vector: list[float],
    spearman_rhos: list[float],
    mses: list[float],
    top_k_overlaps: list[float],
    rbos: list[float],
    float32_retrieval_ms: float = 0.0,
    decompress_ms: list[float] | None = None,
) -> None:
    has_latency = float32_retrieval_ms > 0 and decompress_ms is not None
    width = 115 if has_latency else 100

    print(f"\n{'=' * width}")
    print(f"  COMPRESSION SUMMARY")
    if has_latency:
        print(f"  (scan speed is identical once decompressed -- decompression is the one-time cost)")
    print(f"{'=' * width}")
    header = f"  {'Bit Width':<12} {'Compress':<12} {'Bytes/Vec':<12} {'Spearman':<12} {'Top-10':<12} {'RBO':<10}"
    if has_latency:
        header += f" {'Decompress':<14}"
    print(header)
    print(f"  {'─' * (width - 4)}")
    for i, (bw, cr, mem, rho, mse, topk, rbo) in enumerate(
        zip(bit_widths, compression_ratios, memory_per_vector, spearman_rhos, mses, top_k_overlaps, rbos)
    ):
        line = f"  {bw}-bit{'':<7} {cr:.1f}x{'':<8} {mem:.0f} B{'':<7} {rho:.4f}{'':<6} {topk:.1%}{'':<6} {rbo:.4f}"
        if has_latency:
            line += f"    {decompress_ms[i]:.1f} ms"
        print(line)
    baseline = f"  Original{'':<3} 1.0x{'':<8} 6144 B{'':<5} 1.0000{'':<6} 100.0%{'':<5} 1.0000"
    if has_latency:
        baseline += f"    {float32_retrieval_ms:.1f} ms (scan)"
    print(baseline)
    print(f"{'=' * width}")


# --- Plotting ---


def save_overlay_histogram(
    sims_original: list[float],
    sims_dequantized: list[float],
    bit_width: int,
    spearman_rho: float,
    output_path: Path,
    bins: int = 40,
) -> None:
    fig, ax = mpl_plt.subplots(figsize=(10, 6))

    ax.hist(sims_original, bins=bins, alpha=0.55, label="Original (float32)", color="#4A90D9", edgecolor="white", linewidth=0.5)
    ax.hist(sims_dequantized, bins=bins, alpha=0.55, label=f"Dequantized ({bit_width}-bit)", color="#E85D4A", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Pairwise Similarity: Original vs {bit_width}-bit TurboQuant", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    ax.annotate(
        f"Spearman rho = {spearman_rho:.4f}",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    mpl_plt.close(fig)
    print(f"  Saved: {output_path}")


def save_error_distribution_plot(
    errors: list[float],
    bit_width: int,
    output_path: Path,
    bins: int = 40,
) -> None:
    fig, ax = mpl_plt.subplots(figsize=(10, 6))
    errors_arr = np.array(errors)

    ax.hist(errors_arr, bins=bins, alpha=0.7, color="#4A90D9", edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(errors_arr), color="#E85D4A", linestyle="--", linewidth=2, label=f"MAE = {np.mean(errors_arr):.6f}")

    ax.set_xlabel("Absolute Cosine Similarity Error", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Similarity Error Distribution: {bit_width}-bit TurboQuant", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    ax.annotate(
        f"MSE = {np.mean(errors_arr**2):.8f}\nMax = {np.max(errors_arr):.6f}\nP95 = {np.percentile(errors_arr, 95):.6f}",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    mpl_plt.close(fig)
    print(f"  Saved: {output_path}")


def save_query_ranking_histogram(
    result: QueryRankingResult,
    label: str,
    output_path: Path,
    bins: int = 20,
) -> None:
    fig, ax = mpl_plt.subplots(figsize=(10, 6))

    ax.hist(result.per_query_rhos, bins=bins, alpha=0.7, color="#4A90D9", edgecolor="white", linewidth=0.5)
    ax.axvline(result.mean_rho, color="#E85D4A", linestyle="--", linewidth=2, label=f"Mean = {result.mean_rho:.4f}")
    ax.axvline(
        result.median_rho, color="#2ECC71", linestyle="--", linewidth=2, label=f"Median = {result.median_rho:.4f}"
    )

    ax.set_xlabel("Spearman rho (per query)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Query Ranking Correlation: {label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    ax.annotate(
        f"Queries: {result.num_queries}\nDocs: {result.num_docs}\n"
        f"Top-{result.k} overlap: {result.mean_top_k_overlap:.1%}\n"
        f"RBO: {result.mean_rbo:.4f}\nMean displacement: {result.mean_displacement:.0f}",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    mpl_plt.close(fig)
    print(f"  Saved: {output_path}")


def save_bit_width_summary_plot(
    bit_widths: list[int],
    spearman_rhos: list[float],
    mses: list[float],
    top_k_overlaps: list[float],
    output_path: Path,
) -> None:
    fig, axes = mpl_plt.subplots(1, 3, figsize=(15, 5))

    bw_labels = [f"{bw}-bit" for bw in bit_widths]

    axes[0].bar(bw_labels, spearman_rhos, color="#4A90D9", edgecolor="white")
    axes[0].set_ylim(min(0.8, min(spearman_rhos) - 0.05), 1.0)
    axes[0].set_title("Pairwise Spearman rho", fontweight="bold")
    axes[0].set_ylabel("Spearman rho")
    for i, v in enumerate(spearman_rhos):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)

    axes[1].bar(bw_labels, mses, color="#E85D4A", edgecolor="white")
    axes[1].set_title("Similarity MSE", fontweight="bold")
    axes[1].set_ylabel("MSE")
    for i, v in enumerate(mses):
        axes[1].text(i, v + max(mses) * 0.02, f"{v:.6f}", ha="center", fontsize=10)

    top_k_pcts = [v * 100 for v in top_k_overlaps]
    axes[2].bar(bw_labels, top_k_pcts, color="#2ECC71", edgecolor="white")
    axes[2].set_ylim(min(50, min(top_k_pcts) - 5), 105)
    axes[2].set_title("Top-10 Overlap (%)", fontweight="bold")
    axes[2].set_ylabel("Overlap %")
    for i, v in enumerate(top_k_pcts):
        axes[2].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("TurboQuant Compression Quality by Bit Width", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    mpl_plt.close(fig)
    print(f"  Saved: {output_path}")
