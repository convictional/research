import asyncio
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from src.embedders.colbert import ColBERTEmbedder
from src.embedders.openai_embedder import OpenAIEmbedder
from src.data.extractor import ContentExtractor
from src.storage.postgres import PostgresStorage
from src.search.multi_vector import ColBERTLocalSearch
from src.search.single_vector import OpenAIEmbeddingSearch
from src.search.production_hybrid import ProductionHybridSearch
from src.search.production_reranked import ProductionRerankedSearch
from src.evaluation.query_generator import QueryGenerator
from src.evaluation.query_classifier import QueryClassifier
from src.evaluation.pooling import ResultPooler
from src.evaluation.evaluator import SystemEvaluator
from src.evaluation.benchmark_latency import LatencyBenchmarker
from src.models.content import SearchQuery

load_dotenv(Path(__file__).parent.parent / ".env.secrets")

DB_CONNECTION_STRING = "postgresql://localhost/decide_development"
DATA_DIR = Path("data")


async def index_content(limit: int | None = None, device: str | None = None):
    """Generate and store ColBERT token embeddings for content."""
    print("Initializing ColBERT embedder...")
    embedder = ColBERTEmbedder(device=device)

    print("Connecting to database...")
    extractor = ContentExtractor(DB_CONNECTION_STRING)
    storage = PostgresStorage(DB_CONNECTION_STRING)
    await extractor.connect()
    await storage.connect()

    try:
        print("Fetching content without token embeddings...")
        content_records = await extractor.get_content_without_token_embeddings(limit=limit)
        print(f"Found {len(content_records)} records to index")

        updates = []
        for record in tqdm(content_records, desc="Generating embeddings"):
            token_embeddings = embedder.embed_single(record.index_content)
            updates.append((record.id, token_embeddings))

            if len(updates) >= 10:
                await storage.bulk_update_token_embeddings(updates)
                updates = []

        if updates:
            await storage.bulk_update_token_embeddings(updates)

        print(f"Successfully indexed {len(content_records)} records")

    finally:
        await extractor.close()
        await storage.close()


async def search_content(query_text: str, top_k: int = 10, method: str = "multi"):
    """Search content using specified method."""
    print(f"Initializing {method}-vector search...")

    extractor = ContentExtractor(DB_CONNECTION_STRING)
    await extractor.connect()

    try:
        if method == "multi":
            embedder = ColBERTEmbedder()
            search_engine = MultiVectorSearch(embedder, extractor)
        else:
            print("Single-vector search not yet implemented in CLI")
            return

        query = SearchQuery(text=query_text, top_k=top_k)

        print(f"\nSearching for: '{query_text}'")
        results = await search_engine.search(query)

        print(f"\nTop {len(results)} results:")
        for result in results:
            print(f"\n{result.rank}. {result.title}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Type: {result.content_type}")
            print(f"   Preview: {result.preview[:150]}...")

    finally:
        await extractor.close()


async def compare_methods(query_text: str, top_k: int = 10):
    """Compare multi-vector and single-vector search side-by-side."""
    print("Comparing search methods...")

    extractor = ContentExtractor(DB_CONNECTION_STRING)
    await extractor.connect()

    try:
        embedder = ColBERTEmbedder()
        colbert_search = ColBERTLocalSearch(embedder, extractor)

        query = SearchQuery(text=query_text, top_k=top_k)

        print(f"\n=== Multi-Vector Search ===")
        print(f"Query: '{query_text}'")
        mv_results = await colbert_search.search(query)

        for result in mv_results:
            print(f"{result.rank}. {result.title} (score: {result.score:.4f})")

        print(f"\n=== Single-Vector Search ===")
        print("(Not yet implemented)")

    finally:
        await extractor.close()


async def test_reranker(query_text: str, top_k: int = 10):
    """Test production reranked search with verbose output."""
    openai_key = os.getenv("OPENAI_API_KEY")
    jina_key = os.getenv("JINA_API_KEY")

    if not openai_key:
        print("Error: OPENAI_API_KEY not found in ../experiments/.env.secrets")
        return

    if not jina_key:
        print("Error: JINA_API_KEY not found in ../experiments/.env.secrets")
        return

    print("Initializing production reranked search...")
    extractor = ContentExtractor(DB_CONNECTION_STRING)
    await extractor.connect()

    try:
        openai_embedder = OpenAIEmbedder(openai_key)
        production_search = ProductionHybridSearch(extractor, openai_embedder)
        reranked_search = ProductionRerankedSearch(production_search, extractor, jina_key)
        reranked_search.verbose = True

        query = SearchQuery(text=query_text, top_k=top_k)

        print(f"\n=== Testing Production Reranked Search ===")
        print(f"Query: '{query_text}'")
        print(f"\nStep 1: Running production hybrid search (top-50)...")

        results = await reranked_search.search(query)

        print(f"\n=== Final Reranked Results (top-{top_k}) ===")
        for result in results:
            print(f"{result.rank}. {result.title}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Type: {result.content_type}")
            print(f"   Preview: {result.preview[:150]}...")
            print()

    finally:
        await extractor.close()


async def generate_queries(batch_size: int = 50, queries_per_batch: int = 25):
    """Generate test queries using Claude Sonnet 4.5."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("Error: ANTHROPIC_API_KEY not found in ../experiments/.env.secrets")
        return

    print("Connecting to database...")
    extractor = ContentExtractor(DB_CONNECTION_STRING)
    await extractor.connect()

    try:
        generator = QueryGenerator(extractor, anthropic_key)

        print(f"Generating queries in batches of {batch_size} docs, {queries_per_batch} queries per batch...")
        queries = await generator.generate_queries(batch_size, queries_per_batch)

        output_path = DATA_DIR / "queries.json"
        await generator.save_queries(queries, output_path)

        print(f"\n=== Query Generation Complete ===")
        print(f"Total queries generated: {len(queries)}")
        single_doc = sum(1 for q in queries if len(q.source_doc_ids) == 1)
        multi_doc = sum(1 for q in queries if len(q.source_doc_ids) > 1)
        print(f"Single-doc queries: {single_doc}")
        print(f"Multi-doc queries: {multi_doc}")

    finally:
        await extractor.close()


async def classify_queries(batch_size: int = 25):
    """Classify queries into factoid/navigational/exploratory categories."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("Error: ANTHROPIC_API_KEY not found in ../experiments/.env.secrets")
        return

    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        print(f"Error: {queries_path} not found. Run 'make generate-queries' first.")
        return

    annotations_path = DATA_DIR / "annotations.json"
    if annotations_path.exists():
        backup_path = DATA_DIR / "annotations.backup.json"
        if not backup_path.exists():
            import shutil

            shutil.copy(annotations_path, backup_path)
            print(f"Backed up annotations to {backup_path}")

    print("Loading queries...")
    queries = QueryGenerator.load_queries(queries_path)
    print(f"Loaded {len(queries)} queries")

    print("\nClassifying queries with Claude Sonnet 4.5...")
    classifier = QueryClassifier(anthropic_key)
    classifications = await classifier.classify_queries(queries, batch_size)

    output_path = DATA_DIR / "query_classifications.json"
    await classifier.save_classifications(classifications, output_path)

    factoid = sum(1 for c in classifications if c.classification == "factoid")
    navigational = sum(1 for c in classifications if c.classification == "navigational")
    exploratory = sum(1 for c in classifications if c.classification == "exploratory")

    print(f"\n=== Classification Complete ===")
    print(f"Total queries classified: {len(classifications)}")
    print(f"Distribution:")
    print(f"  Factoid:       {factoid:3d} queries ({factoid / len(classifications) * 100:.1f}%)")
    print(f"  Navigational:  {navigational:3d} queries ({navigational / len(classifications) * 100:.1f}%)")
    print(f"  Exploratory:   {exploratory:3d} queries ({exploratory / len(classifications) * 100:.1f}%)")


async def pool_results(top_k: int = 10, device: str | None = None):
    """Create result pools from all search systems."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY not found in ../experiments/.env.secrets")
        return

    jina_key = os.getenv("JINA_API_KEY")
    if not jina_key:
        print("Warning: JINA_API_KEY not found - skipping production_reranked system")

    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        print(f"Error: {queries_path} not found. Run 'make generate-queries' first.")
        return

    print("Loading queries...")
    queries = QueryGenerator.load_queries(queries_path)
    print(f"Loaded {len(queries)} queries")

    print("Initializing search systems...")
    extractor = ContentExtractor(DB_CONNECTION_STRING)
    await extractor.connect()

    try:
        colbert_embedder = ColBERTEmbedder(device=device)
        openai_embedder = OpenAIEmbedder(openai_key)

        colbert_search = ColBERTLocalSearch(colbert_embedder, extractor)
        openai_search = OpenAIEmbeddingSearch(extractor, openai_embedder)
        production_search = ProductionHybridSearch(extractor, openai_embedder)

        production_reranked = ProductionRerankedSearch(production_search, extractor, jina_key) if jina_key else None

        pooler = ResultPooler(extractor, colbert_search, openai_search, production_search, production_reranked)

        output_path = DATA_DIR / "query_pools.json"
        latency_path = DATA_DIR / "latency_data.json"

        if output_path.exists() and not output_path.with_suffix(".partial.json").exists():
            backup_path = output_path.with_suffix(".old.json")
            output_path.rename(backup_path)
            print(f"Backed up existing file to {backup_path}")

        if latency_path.exists() and not latency_path.with_suffix(".partial.json").exists():
            backup_path = latency_path.with_suffix(".old.json")
            latency_path.rename(backup_path)
            print(f"Backed up existing file to {backup_path}")

        print(f"Pooling results (top-{top_k} from each system)...")
        pools, latencies = await pooler.create_pools(
            queries, top_k, checkpoint_interval=25, pools_path=output_path, latencies_path=latency_path
        )

        await pooler.save_pools(pools, output_path)
        await pooler.save_latencies(latencies, latency_path)

        print(f"\n=== Pooling Complete ===")
        print(f"Total pools created: {len(pools)}")
        avg_pool_size = sum(len(p.pooled_docs) for p in pools) / len(pools)
        print(f"Average pool size: {avg_pool_size:.1f} documents")

        system_latencies = defaultdict(list)
        for lat in latencies:
            system_latencies[lat.system_name].append(lat.latency_ms)

        print(f"\nLatency Statistics ({len(queries)} queries):")
        print(f"{'System':<25} {'Mean (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12}")
        print("-" * 65)
        for system_name in sorted(system_latencies.keys()):
            lats = np.array(system_latencies[system_name])
            print(f"{system_name:<25} {np.mean(lats):<12.1f} {np.median(lats):<12.1f} {np.percentile(lats, 95):<12.1f}")

    finally:
        await extractor.close()


async def evaluate(skip_latency: bool = False):
    """Run full evaluation comparing MV and SV search."""
    pools_path = DATA_DIR / "query_pools.json"
    annotations_path = DATA_DIR / "annotations.json"
    queries_path = DATA_DIR / "queries.json"

    if not pools_path.exists():
        print(f"Error: {pools_path} not found. Run 'make pool-results' first.")
        return

    if not annotations_path.exists():
        print(f"Error: {annotations_path} not found. Run 'make annotate' first.")
        return

    print("Loading data...")
    evaluator = SystemEvaluator(pools_path, annotations_path)

    print("Calculating quality metrics...")
    report = evaluator.evaluate()

    print("\n" + "=" * 80)
    print("Search System Comparison")
    print("=" * 80)

    print("\nQuality Metrics:")

    system_names = sorted(report.systems.keys())
    header = f"{'Metric':<20}"
    for name in system_names:
        display_name = name.replace("_", " ").title()[:13]
        header += f" {display_name:<15}"
    print(header)
    print("-" * (20 + 15 * len(system_names)))

    metrics = ["mrr", "recall_at_1", "recall_at_5", "recall_at_10", "precision_at_1", "precision_at_5", "precision_at_10", "ndcg_at_10", "coverage"]
    metric_labels = ["MRR", "Recall@1", "Recall@5", "Recall@10", "Precision@1", "Precision@5", "Precision@10", "NDCG@10", "Coverage"]

    for metric, label in zip(metrics, metric_labels):
        row = f"{label:<20}"
        for name in system_names:
            value = getattr(report.systems[name], metric)
            row += f" {value:<15.3f}"
        print(row)

    print("\nDataset Statistics:")
    first_system = report.systems[system_names[0]]
    print(f"  Total queries evaluated: {first_system.total_queries}")

    print("\nPairwise Comparisons (Paired t-test on NDCG@10):")
    print(f"{'Comparison':<40} {'Δ NDCG':<12} {'p-value':<12} {'Significant':<12}")
    print("-" * 80)

    for comp in report.pairwise_comparisons:
        if comp.get("test") == "paired_t_test":
            sys1_display = comp['system1'].replace("_", " ").title()
            sys2_display = comp['system2'].replace("_", " ").title()
            comparison_name = f"{sys1_display} vs {sys2_display}"
            delta = f"{comp['improvement_pct']:+.1f}%"
            p_val = f"{comp['p_value']:.4f}"
            sig = "Yes" if comp['significant'] else "No"
            print(f"{comparison_name:<40} {delta:<12} {p_val:<12} {sig:<12}")

    latency_path = DATA_DIR / "latency_data.json"
    if latency_path.exists():
        print("\nLatency Statistics (from pooling run):")
        with open(latency_path) as f:
            latency_data = json.load(f)

        system_latencies = defaultdict(list)
        for item in latency_data:
            system_latencies[item["system_name"]].append(item["latency_ms"])

        print(f"{'System':<25} {'Mean (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12}")
        print("-" * 65)
        for system_name in sorted(system_latencies.keys()):
            lats = np.array(system_latencies[system_name])
            print(f"{system_name:<25} {np.mean(lats):<12.1f} {np.median(lats):<12.1f} {np.percentile(lats, 95):<12.1f}")
    elif not skip_latency:
        print("\nRunning latency benchmark...")
        await run_latency_benchmark()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    report_path = results_dir / "evaluation_report.json"

    with open(report_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    print(f"\nFull report saved to: {report_path}")

    plot_path = results_dir / "precision_recall.png"
    _create_precision_recall_plot(report, plot_path)
    print(f"Precision-Recall plot saved to: {plot_path}")


def _create_precision_recall_plot(report, output_path: Path):
    earthtones = {
        "colbert_local": "#8B4513",
        "openai_embedding": "#D2691E",
        "production_hybrid": "#CD853F",
        "production_reranked": "#DEB887",
    }

    system_labels = {
        "colbert_local": "ColBERT Local",
        "openai_embedding": "OpenAI Embed",
        "production_hybrid": "Production Hybrid",
        "production_reranked": "Production Reranked",
    }

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'sans-serif'

    for system_name, metrics in report.systems.items():
        recall = metrics.recall_at_5
        precision = metrics.precision_at_5
        color = earthtones.get(system_name, "#8B4513")
        label = system_labels.get(system_name, system_name)

        plt.scatter(recall, precision, s=150, color=color, alpha=0.7, edgecolors='black', linewidth=1.5)
        plt.annotate(
            label,
            (recall, precision),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )

    plt.xlabel("Recall@5", fontsize=11, color="#333333")
    plt.ylabel("Precision@5", fontsize=11, color="#333333")
    plt.title("Precision vs Recall Trade-off", fontsize=12, color="#333333", pad=15)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.xlim(0.4, 0.6)
    plt.ylim(0.4, 0.55)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


async def evaluate_stratified():
    """Run evaluation stratified by query classification."""
    pools_path = DATA_DIR / "query_pools.json"
    annotations_path = DATA_DIR / "annotations.json"
    classifications_path = DATA_DIR / "query_classifications.json"

    if not pools_path.exists():
        print(f"Error: {pools_path} not found. Run 'make pool-results' first.")
        return

    if not annotations_path.exists():
        print(f"Error: {annotations_path} not found. Run 'make annotate' first.")
        return

    if not classifications_path.exists():
        print(f"Error: {classifications_path} not found. Run 'make classify-queries' first.")
        return

    print("Loading data...")
    evaluator = SystemEvaluator(pools_path, annotations_path, classifications_path)

    categories = ["factoid", "navigational", "exploratory"]
    category_reports = {}

    print("\n" + "=" * 80)
    print("Stratified Evaluation by Query Type")
    print("=" * 80)

    for category in categories:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {category.upper()} queries")
        print(f"{'=' * 80}")

        report = evaluator.evaluate(classification_filter=category)
        category_reports[category] = report

        first_system = list(report.systems.values())[0]
        if first_system.total_queries == 0:
            print(f"No annotated queries for {category} category")
            continue

        print(f"\nDataset: {first_system.total_queries} annotated {category} queries")

        system_names = sorted(report.systems.keys())
        header = f"{'Metric':<20}"
        for name in system_names:
            display_name = name.replace("_", " ").title()[:13]
            header += f" {display_name:<15}"
        print(header)
        print("-" * (20 + 15 * len(system_names)))

        metrics = ["mrr", "recall_at_1", "recall_at_5", "recall_at_10", "precision_at_1", "precision_at_5", "precision_at_10", "ndcg_at_10"]
        metric_labels = ["MRR", "Recall@1", "Recall@5", "Recall@10", "Precision@1", "Precision@5", "Precision@10", "NDCG@10"]

        for metric, label in zip(metrics, metric_labels):
            row = f"{label:<20}"
            for name in system_names:
                value = getattr(report.systems[name], metric)
                row += f" {value:<15.3f}"
            print(row)

    print("\n" + "=" * 80)
    print("Cross-Category Comparison")
    print("=" * 80)

    all_systems = set()
    for report in category_reports.values():
        all_systems.update(report.systems.keys())
    system_names = sorted(list(all_systems))

    for system_name in system_names:
        print(f"\n{system_name.replace('_', ' ').title()}:")
        print(f"{'Metric':<20} {'Factoid':<15} {'Navigational':<15} {'Exploratory':<15}")
        print("-" * 65)

        metrics = ["mrr", "recall_at_1", "precision_at_5", "ndcg_at_10"]
        metric_labels = ["MRR", "Recall@1", "Precision@5", "NDCG@10"]

        for metric, label in zip(metrics, metric_labels):
            row = f"{label:<20}"
            for category in categories:
                if category in category_reports and system_name in category_reports[category].systems:
                    value = getattr(category_reports[category].systems[system_name], metric)
                    row += f" {value:<15.3f}"
                else:
                    row += f" {'N/A':<15}"
            print(row)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    report_path = results_dir / "evaluation_stratified.json"

    stratified_data = {category: report.model_dump() for category, report in category_reports.items()}

    with open(report_path, "w") as f:
        json.dump(stratified_data, f, indent=2)

    print(f"\nStratified report saved to: {report_path}")


async def run_latency_benchmark():
    """Benchmark search latency."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY not found")
        return

    queries_path = DATA_DIR / "queries.json"
    queries = QueryGenerator.load_queries(queries_path)

    extractor = ContentExtractor(DB_CONNECTION_STRING)
    await extractor.connect()

    try:
        colbert_embedder = ColBERTEmbedder()
        openai_embedder = OpenAIEmbedder(openai_key)

        colbert_search = ColBERTLocalSearch(colbert_embedder, extractor)
        openai_search = OpenAIEmbeddingSearch(extractor, openai_embedder)
        production_search = ProductionHybridSearch(extractor, openai_embedder)

        benchmarker = LatencyBenchmarker(extractor, colbert_search, openai_search, production_search)
        results = await benchmarker.benchmark(queries, sample_size=50)

        print("\nLatency Benchmark (50 queries):")
        print(f"{'Metric':<20} {'ColBERT (ms)':<15} {'OpenAI (ms)':<15} {'Production (ms)':<15}")
        print("-" * 80)

        colbert = results.systems["colbert_local"]
        openai = results.systems["openai_embedding"]
        prod = results.systems["production_hybrid"]

        print(f"{'Mean':<20} {colbert.mean_ms:<15.1f} {openai.mean_ms:<15.1f} {prod.mean_ms:<15.1f}")
        print(f"{'Median':<20} {colbert.median_ms:<15.1f} {openai.median_ms:<15.1f} {prod.median_ms:<15.1f}")
        print(f"{'P95':<20} {colbert.p95_ms:<15.1f} {openai.p95_ms:<15.1f} {prod.p95_ms:<15.1f}")
        print(f"{'P99':<20} {colbert.p99_ms:<15.1f} {openai.p99_ms:<15.1f} {prod.p99_ms:<15.1f}")

        print("\nLatency Ratios (vs Production):")
        print(f"  ColBERT: {colbert.mean_ms / prod.mean_ms:.1f}x slower")
        print(f"  OpenAI: {openai.mean_ms / prod.mean_ms:.1f}x {'slower' if openai.mean_ms > prod.mean_ms else 'faster'}")

    finally:
        await extractor.close()


async def show_stats():
    """Display database statistics."""
    extractor = ContentExtractor(DB_CONNECTION_STRING)
    await extractor.connect()

    try:
        all_content = await extractor.get_all_content()
        with_token_embeddings = [c for c in all_content if c.token_embeddings is not None]

        print(f"\n=== Database Statistics ===")
        print(f"Total content records: {len(all_content)}")
        print(f"Records with token embeddings: {len(with_token_embeddings)}")
        print(f"Records without token embeddings: {len(all_content) - len(with_token_embeddings)}")

        if with_token_embeddings:
            avg_tokens = sum(len(c.token_embeddings) for c in with_token_embeddings) / len(
                with_token_embeddings
            )
            print(f"Average tokens per document: {avg_tokens:.1f}")

    finally:
        await extractor.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-Vector Search Experiment")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    index_parser = subparsers.add_parser("index", help="Index content with ColBERT embeddings")
    index_parser.add_argument("--limit", type=int, help="Limit number of records to index")
    index_parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], help="Force specific device")

    search_parser = subparsers.add_parser("search", help="Search content")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    search_parser.add_argument(
        "--method", choices=["multi", "single"], default="multi", help="Search method"
    )

    compare_parser = subparsers.add_parser("compare", help="Compare search methods")
    compare_parser.add_argument("query", type=str, help="Search query")
    compare_parser.add_argument("--top-k", type=int, default=10, help="Number of results")

    test_reranker_parser = subparsers.add_parser("test-reranker", help="Test production reranker with verbose output")
    test_reranker_parser.add_argument("query", type=str, help="Search query")
    test_reranker_parser.add_argument("--top-k", type=int, default=10, help="Number of results")

    subparsers.add_parser("stats", help="Show database statistics")

    generate_parser = subparsers.add_parser("generate-queries", help="Generate test queries")
    generate_parser.add_argument("--batch-size", type=int, default=50, help="Docs per batch")
    generate_parser.add_argument("--queries-per-batch", type=int, default=25, help="Queries per batch")

    classify_parser = subparsers.add_parser("classify-queries", help="Classify queries into categories")
    classify_parser.add_argument("--batch-size", type=int, default=25, help="Queries per batch")

    pool_parser = subparsers.add_parser("pool-results", help="Pool results from SV and MV search")
    pool_parser.add_argument("--top-k", type=int, default=10, help="Top-K results per system")
    pool_parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], help="Force specific device")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate MV vs SV with annotations")
    eval_parser.add_argument("--skip-latency", action="store_true", help="Skip latency benchmark")

    subparsers.add_parser("evaluate-stratified", help="Evaluate by query classification")

    args = parser.parse_args()

    if args.command == "index":
        asyncio.run(index_content(limit=args.limit, device=args.device))
    elif args.command == "search":
        asyncio.run(search_content(args.query, top_k=args.top_k, method=args.method))
    elif args.command == "compare":
        asyncio.run(compare_methods(args.query, top_k=args.top_k))
    elif args.command == "test-reranker":
        asyncio.run(test_reranker(args.query, top_k=args.top_k))
    elif args.command == "stats":
        asyncio.run(show_stats())
    elif args.command == "generate-queries":
        asyncio.run(generate_queries(batch_size=args.batch_size, queries_per_batch=args.queries_per_batch))
    elif args.command == "classify-queries":
        asyncio.run(classify_queries(batch_size=args.batch_size))
    elif args.command == "pool-results":
        asyncio.run(pool_results(top_k=args.top_k, device=args.device))
    elif args.command == "evaluate":
        asyncio.run(evaluate(skip_latency=args.skip_latency))
    elif args.command == "evaluate-stratified":
        asyncio.run(evaluate_stratified())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
