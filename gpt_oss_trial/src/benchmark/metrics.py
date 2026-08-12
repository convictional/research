"""Metrics collection and statistical calculations for benchmark results."""

from datetime import datetime

import numpy as np

from src.benchmark.models import BenchmarkConfig, BenchmarkResults, RequestMetrics


class MetricsCollector:
    """Collects and aggregates metrics from individual requests."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.metrics: list[RequestMetrics] = []
        self.benchmark_start_time: datetime | None = None
        self.benchmark_end_time: datetime | None = None

    def record_request(self, metrics: RequestMetrics) -> None:
        """Add a request's metrics to the collection."""
        self.metrics.append(metrics)

    def set_benchmark_start_time(self, start_time: datetime) -> None:
        """Record when the benchmark started."""
        self.benchmark_start_time = start_time

    def set_benchmark_end_time(self, end_time: datetime) -> None:
        """Record when the benchmark ended."""
        self.benchmark_end_time = end_time

    def calculate_results(self) -> BenchmarkResults:
        """Calculate aggregate statistics from collected metrics."""
        successful_metrics = [m for m in self.metrics if m.success]
        failed_metrics = [m for m in self.metrics if not m.success]

        latencies = [m.duration_seconds for m in successful_metrics if m.duration_seconds is not None]
        ttfts = [m.ttft_seconds for m in successful_metrics if m.ttft_seconds is not None]

        total_duration = 0.0
        if self.benchmark_start_time and self.benchmark_end_time:
            total_duration = (self.benchmark_end_time - self.benchmark_start_time).total_seconds()

        total_tokens = sum(m.total_tokens for m in successful_metrics)

        error_breakdown: dict[str, int] = {}
        for m in failed_metrics:
            if m.error_type:
                error_breakdown[m.error_type] = error_breakdown.get(m.error_type, 0) + 1

        prompt_lengths = [m.prompt_length for m in self.metrics if m.prompt_length is not None]

        requests_with_tools = sum(1 for m in self.metrics if m.has_tools)
        requests_with_system = sum(1 for m in self.metrics if m.has_system_prompt)

        return BenchmarkResults(
            config=self.config,
            total_requests=len(self.metrics),
            successful_requests=len(successful_metrics),
            failed_requests=len(failed_metrics),
            total_duration_seconds=total_duration,
            mean_latency=float(np.mean(latencies)) if latencies else None,
            median_latency=float(np.median(latencies)) if latencies else None,
            p95_latency=float(np.percentile(latencies, 95)) if latencies else None,
            p99_latency=float(np.percentile(latencies, 99)) if latencies else None,
            min_latency=float(np.min(latencies)) if latencies else None,
            max_latency=float(np.max(latencies)) if latencies else None,
            mean_ttft=float(np.mean(ttfts)) if ttfts else None,
            median_ttft=float(np.median(ttfts)) if ttfts else None,
            p95_ttft=float(np.percentile(ttfts, 95)) if ttfts else None,
            p99_ttft=float(np.percentile(ttfts, 99)) if ttfts else None,
            total_tokens=total_tokens,
            tokens_per_second=total_tokens / total_duration if total_duration > 0 else None,
            requests_per_second=len(successful_metrics) / total_duration if total_duration > 0 else None,
            success_rate=len(successful_metrics) / len(self.metrics) * 100 if self.metrics else 0.0,
            error_breakdown=error_breakdown,
            mean_prompt_length=float(np.mean(prompt_lengths)) if prompt_lengths else None,
            median_prompt_length=float(np.median(prompt_lengths)) if prompt_lengths else None,
            min_prompt_length=int(np.min(prompt_lengths)) if prompt_lengths else None,
            max_prompt_length=int(np.max(prompt_lengths)) if prompt_lengths else None,
            requests_with_tools=requests_with_tools,
            requests_with_system=requests_with_system,
        )
