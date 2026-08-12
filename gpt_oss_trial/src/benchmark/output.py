"""Output formatting and export for benchmark results."""

from pathlib import Path

import pandas as pd

from src.benchmark.models import BenchmarkResults


class ConsoleFormatter:
    """Formats benchmark results for console display."""

    @staticmethod
    def print_summary(results: BenchmarkResults) -> None:
        """Print formatted summary of benchmark results to console."""
        print("\n" + "=" * 70)
        print("GPT-OSS Benchmark Results")
        print("=" * 70)
        print("\nConfiguration:")
        print(f"  Model:            {results.config.model}")
        print(f"  Total Requests:   {results.total_requests}")
        print(f"  Concurrency:      {results.config.concurrency}")
        if results.config.use_db_prompts:
            print(f"  Prompts:          Sampled from database ({results.config.db_name})")
        elif results.config.prompts:
            print(f"  Prompts:          {len(results.config.prompts)} unique prompts")
        else:
            print(f"  Prompts:          Single prompt")

        if results.mean_prompt_length is not None:
            print("\nPrompt Length Statistics (characters):")
            print(f"  Mean:             {results.mean_prompt_length:.0f}")
            print(f"  Median:           {results.median_prompt_length:.0f}")
            print(f"  Min:              {results.min_prompt_length}")
            print(f"  Max:              {results.max_prompt_length}")

        if results.config.use_db_prompts:
            print("\nRequest Complexity:")
            print(f"  With Tools:       {results.requests_with_tools} ({results.requests_with_tools / results.total_requests * 100:.1f}%)")
            print(f"  With System:      {results.requests_with_system} ({results.requests_with_system / results.total_requests * 100:.1f}%)")

        print("\nLatency Metrics (seconds):")
        if results.mean_latency is not None:
            print(f"  Mean:             {results.mean_latency:.3f}")
            print(f"  Median (p50):     {results.median_latency:.3f}")
            print(f"  p95:              {results.p95_latency:.3f}")
            print(f"  p99:              {results.p99_latency:.3f}")
            print(f"  Min:              {results.min_latency:.3f}")
            print(f"  Max:              {results.max_latency:.3f}")
        else:
            print("  No successful requests")

        print("\nTime to First Token (TTFT):")
        if results.mean_ttft is not None:
            print(f"  Mean:             {results.mean_ttft:.3f}")
            print(f"  Median (p50):     {results.median_ttft:.3f}")
            print(f"  p95:              {results.p95_ttft:.3f}")
            print(f"  p99:              {results.p99_ttft:.3f}")
        else:
            print("  No TTFT data available")

        print("\nThroughput:")
        print(f"  Total Tokens:        {results.total_tokens:,}")
        if results.tokens_per_second is not None:
            print(f"  Tokens per Second:   {results.tokens_per_second:.2f}")
        if results.requests_per_second is not None:
            print(f"  Requests per Second: {results.requests_per_second:.2f}")

        print("\nReliability:")
        print(f"  Success Rate:     {results.success_rate:.1f}%")
        print(f"  Total Errors:     {results.failed_requests}")
        if results.error_breakdown:
            print("  Error Breakdown:")
            for error_type, count in sorted(results.error_breakdown.items()):
                percentage = (count / results.failed_requests) * 100
                print(f"    {error_type:20s} {count:3d} ({percentage:.1f}%)")

        print(f"\nDuration: {results.total_duration_seconds:.2f} seconds")
        print("=" * 70 + "\n")


class CSVExporter:
    """Exports benchmark results to CSV files."""

    @staticmethod
    def export_to_csv(results: BenchmarkResults, collector, output_dir: str) -> tuple[Path, Path]:
        """
        Export detailed request metrics and summary to CSV files.

        Args:
            results: Aggregated benchmark results
            collector: MetricsCollector instance with individual request metrics
            output_dir: Directory to save CSV files

        Returns:
            Tuple of (details_path, summary_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = results.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        details_filename = f"benchmark_{timestamp_str}.csv"
        summary_filename = f"benchmark_{timestamp_str}_summary.csv"

        details_path = output_path / details_filename
        summary_path = output_path / summary_filename

        details_data = []
        for metric in collector.metrics:
            details_data.append(
                {
                    "request_id": metric.request_id,
                    "start_time": metric.start_time.isoformat(),
                    "end_time": metric.end_time.isoformat() if metric.end_time else None,
                    "duration_seconds": metric.duration_seconds,
                    "ttft_seconds": metric.ttft_seconds,
                    "total_tokens": metric.total_tokens,
                    "tokens_per_second": metric.tokens_per_second,
                    "success": metric.success,
                    "error_type": metric.error_type,
                    "error_message": metric.error_message,
                    "prompt_length": metric.prompt_length,
                    "has_tools": metric.has_tools,
                    "has_system_prompt": metric.has_system_prompt,
                }
            )

        details_df = pd.DataFrame(details_data)
        details_df.to_csv(details_path, index=False)

        summary_data = {
            "timestamp": [results.timestamp.isoformat()],
            "model": [results.config.model],
            "total_requests": [results.total_requests],
            "concurrency": [results.config.concurrency],
            "successful_requests": [results.successful_requests],
            "failed_requests": [results.failed_requests],
            "success_rate": [results.success_rate],
            "total_duration_seconds": [results.total_duration_seconds],
            "mean_latency": [results.mean_latency],
            "median_latency": [results.median_latency],
            "p95_latency": [results.p95_latency],
            "p99_latency": [results.p99_latency],
            "min_latency": [results.min_latency],
            "max_latency": [results.max_latency],
            "mean_ttft": [results.mean_ttft],
            "median_ttft": [results.median_ttft],
            "p95_ttft": [results.p95_ttft],
            "p99_ttft": [results.p99_ttft],
            "total_tokens": [results.total_tokens],
            "tokens_per_second": [results.tokens_per_second],
            "requests_per_second": [results.requests_per_second],
            "mean_prompt_length": [results.mean_prompt_length],
            "median_prompt_length": [results.median_prompt_length],
            "min_prompt_length": [results.min_prompt_length],
            "max_prompt_length": [results.max_prompt_length],
            "requests_with_tools": [results.requests_with_tools],
            "requests_with_system": [results.requests_with_system],
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)

        return details_path, summary_path

    @staticmethod
    def export_comparison_to_csv(comparison_metrics: list, config, output_dir: str) -> Path:
        """
        Export per-request comparison metrics to CSV.

        Args:
            comparison_metrics: List of ComparisonRequestMetrics
            config: Benchmark configuration
            output_dir: Directory to save CSV file

        Returns:
            Path to the comparison CSV file
        """
        from datetime import datetime

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"comparison_claude_vs_gpt_{timestamp_str}.csv"
        csv_path = output_path / filename

        comparison_data = []
        for metric in comparison_metrics:
            comparison_data.append(
                {
                    "request_id": metric.request_id,
                    "prompt_length": metric.prompt_length,
                    "has_tools": metric.has_tools,
                    "has_system_prompt": metric.has_system_prompt,
                    "claude_start_time": metric.claude_start_time.isoformat(),
                    "claude_end_time": metric.claude_end_time.isoformat() if metric.claude_end_time else None,
                    "claude_duration_seconds": metric.claude_duration_seconds,
                    "claude_ttft_seconds": metric.claude_ttft_seconds,
                    "claude_total_tokens": metric.claude_total_tokens,
                    "claude_tokens_per_second": metric.claude_tokens_per_second,
                    "claude_success": metric.claude_success,
                    "claude_error_type": metric.claude_error_type,
                    "claude_error_message": metric.claude_error_message,
                    "gpt_start_time": metric.gpt_start_time.isoformat(),
                    "gpt_end_time": metric.gpt_end_time.isoformat() if metric.gpt_end_time else None,
                    "gpt_duration_seconds": metric.gpt_duration_seconds,
                    "gpt_ttft_seconds": metric.gpt_ttft_seconds,
                    "gpt_total_tokens": metric.gpt_total_tokens,
                    "gpt_tokens_per_second": metric.gpt_tokens_per_second,
                    "gpt_success": metric.gpt_success,
                    "gpt_error_type": metric.gpt_error_type,
                    "gpt_error_message": metric.gpt_error_message,
                    "claude_response_text": metric.claude_response_text,
                    "gpt_response_text": metric.gpt_response_text,
                    "duration_delta_seconds": (
                        metric.claude_duration_seconds - metric.gpt_duration_seconds
                        if metric.claude_duration_seconds and metric.gpt_duration_seconds
                        else None
                    ),
                    "ttft_delta_seconds": (
                        metric.claude_ttft_seconds - metric.gpt_ttft_seconds
                        if metric.claude_ttft_seconds and metric.gpt_ttft_seconds
                        else None
                    ),
                    "throughput_delta_tps": (
                        metric.claude_tokens_per_second - metric.gpt_tokens_per_second
                        if metric.claude_tokens_per_second and metric.gpt_tokens_per_second
                        else None
                    ),
                }
            )

        df = pd.DataFrame(comparison_data)
        df.to_csv(csv_path, index=False)

        return csv_path


class ComparisonFormatter:
    """Formats comparison results for console display."""

    @staticmethod
    def print_comparison_summary(comparison_metrics: list, config) -> None:
        """Print formatted comparison summary to console."""
        import numpy as np

        both_success = [m for m in comparison_metrics if m.claude_success and m.gpt_success]

        if not both_success:
            print("\nNo requests succeeded for both models - cannot generate comparison statistics")
            return

        claude_latencies = [m.claude_duration_seconds for m in both_success if m.claude_duration_seconds]
        gpt_latencies = [m.gpt_duration_seconds for m in both_success if m.gpt_duration_seconds]

        claude_ttfts = [m.claude_ttft_seconds for m in both_success if m.claude_ttft_seconds]
        gpt_ttfts = [m.gpt_ttft_seconds for m in both_success if m.gpt_ttft_seconds]

        claude_throughputs = [m.claude_tokens_per_second for m in both_success if m.claude_tokens_per_second]
        gpt_throughputs = [m.gpt_tokens_per_second for m in both_success if m.gpt_tokens_per_second]

        print("\n" + "=" * 70)
        print("Claude Sonnet 4.5 vs GPT-OSS Comparison")
        print("=" * 70)

        print("\nConfiguration:")
        print(f"  Total Requests:   {len(comparison_metrics)}")
        print(f"  Both Successful:  {len(both_success)}")
        print(f"  Concurrency:      {config.concurrency}")

        print("\nLatency Comparison (seconds):")
        print(f"  {'':20} {'Claude':>12} {'GPT-OSS':>12} {'Δ (C-G)':>12}")
        print(f"  {'-'*56}")
        if claude_latencies and gpt_latencies:
            print(
                f"  {'Mean:':20} {np.mean(claude_latencies):>12.3f} {np.mean(gpt_latencies):>12.3f} "
                f"{np.mean(claude_latencies) - np.mean(gpt_latencies):>12.3f}"
            )
            print(
                f"  {'p95:':20} {np.percentile(claude_latencies, 95):>12.3f} "
                f"{np.percentile(gpt_latencies, 95):>12.3f} "
                f"{np.percentile(claude_latencies, 95) - np.percentile(gpt_latencies, 95):>12.3f}"
            )
            print(
                f"  {'p99:':20} {np.percentile(claude_latencies, 99):>12.3f} "
                f"{np.percentile(gpt_latencies, 99):>12.3f} "
                f"{np.percentile(claude_latencies, 99) - np.percentile(gpt_latencies, 99):>12.3f}"
            )

        print("\nTTFT Comparison (seconds):")
        print(f"  {'':20} {'Claude':>12} {'GPT-OSS':>12} {'Δ (C-G)':>12}")
        print(f"  {'-'*56}")
        if claude_ttfts and gpt_ttfts:
            print(
                f"  {'Mean:':20} {np.mean(claude_ttfts):>12.3f} {np.mean(gpt_ttfts):>12.3f} "
                f"{np.mean(claude_ttfts) - np.mean(gpt_ttfts):>12.3f}"
            )
            print(
                f"  {'p95:':20} {np.percentile(claude_ttfts, 95):>12.3f} "
                f"{np.percentile(gpt_ttfts, 95):>12.3f} "
                f"{np.percentile(claude_ttfts, 95) - np.percentile(gpt_ttfts, 95):>12.3f}"
            )
            print(
                f"  {'p99:':20} {np.percentile(claude_ttfts, 99):>12.3f} "
                f"{np.percentile(gpt_ttfts, 99):>12.3f} "
                f"{np.percentile(claude_ttfts, 99) - np.percentile(gpt_ttfts, 99):>12.3f}"
            )

        print("\nThroughput Comparison (tokens/second):")
        print(f"  {'':20} {'Claude':>12} {'GPT-OSS':>12} {'Δ (C-G)':>12}")
        print(f"  {'-'*56}")
        if claude_throughputs and gpt_throughputs:
            print(
                f"  {'Mean:':20} {np.mean(claude_throughputs):>12.2f} "
                f"{np.mean(gpt_throughputs):>12.2f} "
                f"{np.mean(claude_throughputs) - np.mean(gpt_throughputs):>12.2f}"
            )

        print("\nSuccess Rates:")
        claude_success_count = sum(1 for m in comparison_metrics if m.claude_success)
        gpt_success_count = sum(1 for m in comparison_metrics if m.gpt_success)
        print(
            f"  Claude:           {claude_success_count}/{len(comparison_metrics)} "
            f"({claude_success_count/len(comparison_metrics)*100:.1f}%)"
        )
        print(
            f"  GPT-OSS:          {gpt_success_count}/{len(comparison_metrics)} "
            f"({gpt_success_count/len(comparison_metrics)*100:.1f}%)"
        )

        print("=" * 70 + "\n")
