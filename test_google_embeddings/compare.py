import itertools
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as mpl_plt
import numpy as np
import plotext as plt
from scipy.stats import spearmanr


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


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
    """Compute pairwise similarities under both models and Spearman rank correlation.

    For a sample of document pairs, compute cosine similarity under both embedding
    models, then correlate the two lists of similarities.
    """
    n = len(embeddings_a)
    all_pairs = list(itertools.combinations(range(n), 2))
    rng = random.Random(seed)
    if len(all_pairs) > num_pairs:
        pairs = rng.sample(all_pairs, num_pairs)
    else:
        pairs = all_pairs

    sims_a = [cosine_similarity(embeddings_a[i], embeddings_a[j]) for i, j in pairs]
    sims_b = [cosine_similarity(embeddings_b[i], embeddings_b[j]) for i, j in pairs]

    corr, p_value = spearmanr(sims_a, sims_b)
    return PairwiseResult(
        sims_a=sims_a,
        sims_b=sims_b,
        spearman_rho=float(corr),
        p_value=float(p_value),
        num_pairs=len(pairs),
    )


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


def save_overlay_histogram(
    sims_openai: list[float],
    sims_google: list[float],
    google_label: str,
    spearman_rho: float,
    output_path: Path,
    bins: int = 40,
) -> None:
    fig, ax = mpl_plt.subplots(figsize=(10, 6))

    ax.hist(sims_openai, bins=bins, alpha=0.55, label="OpenAI text-embedding-3-small", color="#4A90D9", edgecolor="white", linewidth=0.5)
    ax.hist(sims_google, bins=bins, alpha=0.55, label=google_label, color="#E85D4A", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Pairwise Similarity Distribution: OpenAI vs {google_label}", fontsize=13, fontweight="bold")
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


def percentile_mapping(
    sims_openai: list[float],
    sims_google: list[float],
    google_label: str,
) -> None:
    """Print a table mapping OpenAI similarity scores to equivalent Google scores via percentile alignment."""
    arr_oai = np.array(sims_openai)
    arr_goog = np.array(sims_google)

    # Use percentiles as the common axis: a score at the 95th percentile in OpenAI
    # "means the same thing" as a score at the 95th percentile in Google.
    percentiles = [50, 75, 90, 95, 97.5, 99]

    oai_values = np.percentile(arr_oai, percentiles)
    goog_values = np.percentile(arr_goog, percentiles)

    print(f"\n{'=' * 52}")
    print(f"  Score Mapping: OpenAI vs {google_label}")
    print(f"  (Percentile-aligned — same percentile = same relative meaning)")
    print(f"{'=' * 52}")
    print(f"  {'Percentile':<14} {'OpenAI':<14} {'Google':<14}")
    print(f"  {'─' * 42}")
    for pct, oai_val, goog_val in zip(percentiles, oai_values, goog_values):
        print(f"  {pct:<14.1f} {oai_val:<14.4f} {goog_val:<14.4f}")
    print()


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
    """Rank Biased Overlap between two ranked lists.

    Weights top positions exponentially more than lower positions. With p=0.9,
    the top ~10 positions receive ~86% of the total weight. Returns a value
    in [0, 1] where 1 means identical rankings.
    """
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
    """Compare search rankings between two embedding models across queries.

    For each query, ranks all documents by cosine similarity under each model,
    then computes multiple agreement metrics:
    - Spearman rho over the full ranking
    - Top-k overlap (fraction of top-k shared between models)
    - RBO (Rank Biased Overlap) — weighted overlap favoring top positions
    - Top-k Spearman — Spearman only over the union of both models' top-k
    - Displacement — how far top-k docs from model A fall in model B's ranking
    """
    doc_matrix_a = np.array(doc_embeddings_a)
    doc_matrix_b = np.array(doc_embeddings_b)

    # Normalize document matrices for fast cosine similarity via dot product
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

        # Full-ranking Spearman
        corr, _ = spearmanr(sims_a, sims_b)
        per_query_rhos.append(float(corr))

        # Ranked doc indices (highest similarity first)
        ranked_a = np.argsort(sims_a)[::-1]
        ranked_b = np.argsort(sims_b)[::-1]

        effective_k = min(k, num_docs)

        # Top-k overlap
        top_k_a = set(ranked_a[:effective_k].tolist())
        top_k_b = set(ranked_b[:effective_k].tolist())
        overlap = len(top_k_a & top_k_b) / effective_k
        top_k_overlaps.append(overlap)

        # RBO (computed over the full ranking, but top-weighted)
        per_query_rbos.append(_rbo(ranked_a, ranked_b, p=0.9))

        # Top-k Spearman: rank correlation over the union of both top-k sets
        union_top_k = list(top_k_a | top_k_b)
        if len(union_top_k) >= 3:
            union_sims_a = sims_a[union_top_k]
            union_sims_b = sims_b[union_top_k]
            top_k_corr, _ = spearmanr(union_sims_a, union_sims_b)
            per_query_top_k_rhos.append(float(top_k_corr))

        # Displacement: for docs in A's top-k but not B's, where do they rank in B?
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
        f"Queries: {result.num_queries}\nDocs: {result.num_docs}\nTop-{result.k} overlap: {result.mean_top_k_overlap:.1%}\nRBO: {result.mean_rbo:.4f}\nMean displacement: {result.mean_displacement:.0f}",
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
    print(f"  Top-{result.k} Spearman rho:  {result.mean_top_k_rho:.4f}  (over union of both models' top-{result.k})")
    print(f"  RBO (p=0.9):          {result.mean_rbo:.4f}  (top-weighted overlap)")
    print(f"  Top-{result.k} overlap:       {result.mean_top_k_overlap:.1%}  ({result.mean_top_k_overlap * result.k:.1f}/{result.k} results shared)")
    print(f"")
    print(f"  --- Displacement of \"Lost\" Results ---")
    print(f"  (Where do model A's top-{result.k} docs rank in model B when they fall out of top-{result.k}?)")
    print(f"  Mean rank in B:       {result.mean_displacement:.0f}")
    print(f"  Median rank in B:     {result.displacement_percentiles['p50']:.0f}")
    print(f"  P90 rank in B:        {result.displacement_percentiles['p90']:.0f}")
    print(f"  Worst case:           {result.max_displacement:.0f}")
    if result.mean_displacement < result.k * 3:
        interpretation = "Near-misses — most displaced docs are close to the cutoff"
    elif result.mean_displacement < result.num_docs * 0.25:
        interpretation = "Moderate displacement — some docs ranked meaningfully differently"
    else:
        interpretation = "Large displacement — models genuinely disagree on relevance"
    print(f"  Interpretation:       {interpretation}")


def print_spearman_result(corr: float, p_value: float, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Spearman rho:  {corr:.4f}")
    print(f"  p-value:       {p_value:.2e}")
    if corr > 0.95:
        interpretation = "Excellent — nearly identical rankings"
    elif corr > 0.90:
        interpretation = "Very strong — rankings highly correlated"
    elif corr > 0.80:
        interpretation = "Strong — rankings well correlated"
    elif corr > 0.60:
        interpretation = "Moderate — noticeable ranking differences"
    else:
        interpretation = "Weak — substantial ranking differences"
    print(f"  Interpretation: {interpretation}")
