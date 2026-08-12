"""Command-line interface for running GPT-OSS benchmarks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import os

import dotenv

from src.benchmark.eval_prompts import fetch_eval_request_bodies
from src.benchmark.models import BenchmarkConfig
from src.benchmark.output import CSVExporter, ConsoleFormatter
from src.benchmark.prompts import sample_request_bodies_from_db
from src.benchmark.runner import BenchmarkRunner

dotenv.load_dotenv(Path(__file__).parent.parent / ".env.secrets")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark tool for GPT-OSS Vertex AI endpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        help="Number of requests to execute",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a one-liner about convictional.com's mission.",
        help="Prompt to send in each request",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b-maas",
        help="Model identifier",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save CSV results",
    )

    parser.add_argument(
        "--use-db-prompts",
        action="store_true",
        help="Sample prompts from database (stratified by length)",
    )

    parser.add_argument(
        "--compare-claude",
        action="store_true",
        help="Run parallel comparison with Claude Sonnet 4.5",
    )

    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Use cherry-picked evaluation prompts (implies --use-db-prompts --compare-claude)",
    )

    parser.add_argument(
        "--structured-test",
        action="store_true",
        help="Run structured response testing with Instructor (tests both models)",
    )

    return parser.parse_args()


async def run_benchmark_cli() -> None:
    """Execute benchmark with CLI arguments."""
    args = parse_args()

    if args.structured_test:
        from datetime import datetime
        from pathlib import Path

        from src.benchmark.structured_test import StructuredBenchmark
        from src.llm.instructor_client import InstructorClientFactory, StructuredResponseClient

        print("Running structured response tests...")

        gpt_oss_client = None
        claude_client = None

        gpt_token = os.getenv("GOOGLE_ACCESS_TOKEN")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if gpt_token and project_id:
            instructor_gpt = InstructorClientFactory.create_gpt_oss_client(gpt_token, project_id)
            gpt_oss_client = StructuredResponseClient(instructor_gpt, "openai/gpt-oss-120b-maas")
        else:
            print("Warning: GOOGLE_ACCESS_TOKEN or GOOGLE_CLOUD_PROJECT not set, skipping GPT-OSS")

        claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        if claude_api_key:
            instructor_claude = InstructorClientFactory.create_claude_client(claude_api_key)
            claude_client = StructuredResponseClient(instructor_claude, "claude-sonnet-4-5-20250929")
        else:
            print("Warning: ANTHROPIC_API_KEY not set, skipping Claude")

        if not gpt_oss_client and not claude_client:
            print("Error: No models configured. Set required environment variables.")
            return

        benchmark = StructuredBenchmark(gpt_oss_client, claude_client)
        results = benchmark.run_all_tests()

        benchmark.print_summary()

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = Path(args.output_dir) / f"structured_test_{timestamp_str}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark.export_results(output_path)

        print(f"Results exported to: {output_path}")
        return

    request_bodies = None
    if args.eval_mode:
        print("Loading cherry-picked evaluation prompts...")
        request_bodies = await fetch_eval_request_bodies("local_research_db")
        print(f"Loaded {len(request_bodies)} evaluation prompts")
        args.requests = len(request_bodies)
        args.compare_claude = True
    elif args.use_db_prompts:
        print(f"Sampling {args.requests} request bodies from database (stratified by prompt length)...")
        request_bodies = await sample_request_bodies_from_db("local_research_db", args.requests)
        print(f"Sampled {len(request_bodies)} request bodies")

    config = BenchmarkConfig(
        num_requests=args.requests,
        concurrency=args.concurrency,
        model=args.model,
        prompt=args.prompt if not (args.use_db_prompts or args.eval_mode) else None,
        request_bodies=request_bodies,
        timeout_seconds=args.timeout,
        output_dir=args.output_dir,
        use_db_prompts=args.use_db_prompts or args.eval_mode,
        db_name="local_research_db" if (args.use_db_prompts or args.eval_mode) else None,
    )

    if args.compare_claude:
        from src.benchmark.comparison import ComparisonRunner
        from src.benchmark.output import ComparisonFormatter, CSVExporter

        claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not claude_api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            return

        gpt_token = os.getenv("GOOGLE_ACCESS_TOKEN")
        if not gpt_token:
            print("Error: GOOGLE_ACCESS_TOKEN environment variable not set")
            return

        gpt_headers = {
            "Authorization": f"Bearer {gpt_token}",
            "Content-Type": "application/json",
        }

        save_responses = args.eval_mode
        runner = ComparisonRunner(
            config=config, gpt_headers=gpt_headers, claude_api_key=claude_api_key, save_responses=save_responses
        )
        comparison_metrics = await runner.run_comparison_benchmark()

        ComparisonFormatter.print_comparison_summary(comparison_metrics, config)

        csv_path = CSVExporter.export_comparison_to_csv(
            comparison_metrics=comparison_metrics, config=config, output_dir=config.output_dir
        )

        print(f"Comparison results: {csv_path}")

    else:
        token = os.getenv("GOOGLE_ACCESS_TOKEN")
        if not token:
            print("Error: GOOGLE_ACCESS_TOKEN environment variable not set")
            return

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        runner = BenchmarkRunner(config=config, headers=headers)

        results = await runner.run_benchmark()

        ConsoleFormatter.print_summary(results)

        details_path, summary_path = CSVExporter.export_to_csv(results, runner.collector, config.output_dir)

        print(f"Detailed results: {details_path}")
        print(f"Summary results:  {summary_path}")


def main() -> None:
    """Entry point for CLI."""
    asyncio.run(run_benchmark_cli())


if __name__ == "__main__":
    main()
