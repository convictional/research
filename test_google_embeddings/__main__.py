import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from test_google_embeddings.compare import (
    pairwise_comparison,
    percentile_mapping,
    print_distribution_stats,
    print_histogram,
    print_query_ranking_result,
    print_spearman_result,
    query_ranking_comparison,
    save_overlay_histogram,
    save_query_ranking_histogram,
)
from test_google_embeddings.db import fetch_content_with_embeddings, fetch_search_queries
from test_google_embeddings.embed import (
    EmbeddingResult,
    embed_with_gemini_embedding_001,
    embed_with_openai,
    embed_with_text_embedding_005,
)

load_dotenv(Path(__file__).parent.parent / ".env.secrets")

DB_CONNECTION_STRING = "postgresql://localhost/decide_development"


async def main(limit: int, num_pairs: int) -> None:
    print(f"\nFetching up to {limit} content records with OpenAI embeddings...")
    records = await fetch_content_with_embeddings(DB_CONNECTION_STRING, limit=limit)
    print(f"Fetched {len(records)} records")

    if len(records) < 2:
        print("Need at least 2 records for comparison. Exiting.")
        return

    texts = [r.index_content for r in records]
    openai_embeddings = [r.openai_embedding for r in records]

    # --- text-embedding-005 (768 dims) ---
    print(f"\nGenerating text-embedding-005 embeddings for {len(texts)} texts...")
    te005_embeddings: list[list[float]] = []
    te005_api_seconds = 0.0
    te005_texts_embedded = 0
    for i in tqdm(range(0, len(texts), 50), desc="text-embedding-005"):
        batch = texts[i : i + 50]
        result: EmbeddingResult = await embed_with_text_embedding_005(batch)
        te005_embeddings.extend(result.embeddings)
        te005_api_seconds += result.api_seconds
        te005_texts_embedded += result.texts_embedded

    # --- gemini-embedding-001 (1536 dims to match OpenAI) ---
    print(f"\nGenerating gemini-embedding-001 embeddings for {len(texts)} texts...")
    gemini_embeddings: list[list[float]] = []
    gemini_api_seconds = 0.0
    gemini_texts_embedded = 0
    for i in tqdm(range(0, len(texts), 50), desc="gemini-embedding-001"):
        batch = texts[i : i + 50]
        result = await embed_with_gemini_embedding_001(batch, dimensions=1536)
        gemini_embeddings.extend(result.embeddings)
        gemini_api_seconds += result.api_seconds
        gemini_texts_embedded += result.texts_embedded

    # === Latency ===
    print(f"\n{'=' * 60}")
    print("  Embedding Latency (API calls only, excludes cache hits)")
    print(f"{'=' * 60}")
    if te005_texts_embedded > 0:
        te005_avg_ms = (te005_api_seconds / te005_texts_embedded) * 1000
        print(f"  text-embedding-005:    {te005_api_seconds:.1f}s total, {te005_avg_ms:.1f}ms avg/text  ({te005_texts_embedded} texts)")
    else:
        print(f"  text-embedding-005:    all cached (delete cache/ to re-measure)")
    if gemini_texts_embedded > 0:
        gemini_avg_ms = (gemini_api_seconds / gemini_texts_embedded) * 1000
        print(f"  gemini-embedding-001:  {gemini_api_seconds:.1f}s total, {gemini_avg_ms:.1f}ms avg/text  ({gemini_texts_embedded} texts)")
    else:
        print(f"  gemini-embedding-001:  all cached (delete cache/ to re-measure)")

    # === Analysis ===

    # 1. OpenAI vs text-embedding-005
    result_te005 = pairwise_comparison(openai_embeddings, te005_embeddings, num_pairs=num_pairs)
    print_distribution_stats(result_te005.sims_a, "Pairwise Similarity Distribution: OpenAI text-embedding-3-small")
    print_distribution_stats(result_te005.sims_b, "Pairwise Similarity Distribution: text-embedding-005 (768d)")
    print_histogram(result_te005.sims_a, "OpenAI pairwise similarities")
    print_histogram(result_te005.sims_b, "text-embedding-005 pairwise similarities")
    print_spearman_result(result_te005.spearman_rho, result_te005.p_value, "Pairwise Spearman: OpenAI vs text-embedding-005 (768d)")

    # 2. OpenAI vs gemini-embedding-001
    result_gemini = pairwise_comparison(openai_embeddings, gemini_embeddings, num_pairs=num_pairs)
    print_distribution_stats(result_gemini.sims_b, "Pairwise Similarity Distribution: gemini-embedding-001 (1536d)")
    print_histogram(result_gemini.sims_b, "gemini-embedding-001 pairwise similarities")
    print_spearman_result(result_gemini.spearman_rho, result_gemini.p_value, "Pairwise Spearman: OpenAI vs gemini-embedding-001 (1536d)")

    # 3. text-embedding-005 vs gemini-embedding-001
    result_g2g = pairwise_comparison(te005_embeddings, gemini_embeddings, num_pairs=num_pairs)
    print_spearman_result(result_g2g.spearman_rho, result_g2g.p_value, "Pairwise Spearman: text-embedding-005 vs gemini-embedding-001")

    # Percentile-based score mapping
    percentile_mapping(result_te005.sims_a, result_te005.sims_b, "text-embedding-005 (768d)")
    percentile_mapping(result_gemini.sims_a, result_gemini.sims_b, "gemini-embedding-001 (1536d)")

    # Save overlay histograms as PNGs
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    save_overlay_histogram(
        result_te005.sims_a,
        result_te005.sims_b,
        google_label="text-embedding-005 (768d)",
        spearman_rho=result_te005.spearman_rho,
        output_path=output_dir / "openai_vs_text_embedding_005.png",
    )
    save_overlay_histogram(
        result_gemini.sims_a,
        result_gemini.sims_b,
        google_label="gemini-embedding-001 (1536d)",
        spearman_rho=result_gemini.spearman_rho,
        output_path=output_dir / "openai_vs_gemini_embedding_001.png",
    )

    # === Query-Based Ranking Evaluation ===

    print(f"\n{'=' * 60}")
    print("  QUERY-BASED RANKING EVALUATION")
    print(f"{'=' * 60}")

    queries = await fetch_search_queries(DB_CONNECTION_STRING)
    print(f"\n  Fetched {len(queries)} search queries from researchquery table")

    if queries:
        # Embed queries with all 3 models
        print(f"\n  Embedding {len(queries)} queries with OpenAI...")
        openai_query_result = embed_with_openai(queries)
        openai_query_embeddings = openai_query_result.embeddings

        print(f"  Embedding {len(queries)} queries with text-embedding-005...")
        te005_query_result: EmbeddingResult = await embed_with_text_embedding_005(queries, dimensions=768)
        te005_query_embeddings = te005_query_result.embeddings

        print(f"  Embedding {len(queries)} queries with gemini-embedding-001...")
        gemini_query_result: EmbeddingResult = await embed_with_gemini_embedding_001(queries, dimensions=1536)
        gemini_query_embeddings = gemini_query_result.embeddings

        # OpenAI vs gemini-embedding-001
        qr_gemini = query_ranking_comparison(
            openai_query_embeddings,
            gemini_query_embeddings,
            openai_embeddings,
            gemini_embeddings,
        )
        print_query_ranking_result(qr_gemini, "OpenAI vs gemini-embedding-001")

        # OpenAI vs text-embedding-005
        qr_te005 = query_ranking_comparison(
            openai_query_embeddings,
            te005_query_embeddings,
            openai_embeddings,
            te005_embeddings,
        )
        print_query_ranking_result(qr_te005, "OpenAI vs text-embedding-005")

        # Save histograms
        save_query_ranking_histogram(
            qr_gemini,
            label="OpenAI vs gemini-embedding-001",
            output_path=output_dir / "query_ranking_openai_vs_gemini.png",
        )
        save_query_ranking_histogram(
            qr_te005,
            label="OpenAI vs text-embedding-005",
            output_path=output_dir / "query_ranking_openai_vs_te005.png",
        )
    else:
        print("  No search queries found — skipping query-based evaluation")

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Documents sampled:          {len(records)}")
    print(f"  Document pairs compared:    {result_te005.num_pairs}")
    print(f"")
    print(f"  Pairwise Spearman rho:")
    print(f"    OpenAI vs text-embedding-005:      {result_te005.spearman_rho:.4f}  (p={result_te005.p_value:.2e})")
    print(f"    OpenAI vs gemini-embedding-001:    {result_gemini.spearman_rho:.4f}  (p={result_gemini.p_value:.2e})")
    print(f"    te-005 vs gemini-001:              {result_g2g.spearman_rho:.4f}  (p={result_g2g.p_value:.2e})")
    if queries:
        print(f"")
        print(f"  Query Ranking ({len(queries)} queries, {len(records)} docs):")
        print(f"                                       {'Spearman':>10} {'Top-k rho':>10} {'RBO':>10} {'Top-10':>10} {'Displace':>10}")
        print(f"    OpenAI vs gemini-embedding-001:    {qr_gemini.mean_rho:>10.4f} {qr_gemini.mean_top_k_rho:>10.4f} {qr_gemini.mean_rbo:>10.4f} {qr_gemini.mean_top_k_overlap:>9.1%} {qr_gemini.mean_displacement:>9.0f}")
        print(f"    OpenAI vs text-embedding-005:      {qr_te005.mean_rho:>10.4f} {qr_te005.mean_top_k_rho:>10.4f} {qr_te005.mean_rbo:>10.4f} {qr_te005.mean_top_k_overlap:>9.1%} {qr_te005.mean_displacement:>9.0f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Google vs OpenAI embeddings")
    parser.add_argument("--limit", type=int, default=200, help="Number of content records to sample")
    parser.add_argument("--pairs", type=int, default=5000, help="Number of random document pairs for Spearman")
    args = parser.parse_args()

    asyncio.run(main(args.limit, args.pairs))
