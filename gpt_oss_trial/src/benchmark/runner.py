"""Benchmark runner for orchestrating concurrent API requests."""

import asyncio
from datetime import UTC, datetime

from src.benchmark.format_converter import (
    calculate_prompt_length,
    convert_anthropic_to_openai,
    has_system_prompt,
    has_tools,
)
from src.benchmark.metrics import MetricsCollector
from src.benchmark.models import BenchmarkConfig, BenchmarkResults, RequestMetrics
from src.llm.gpt_oss import stream_gpt_oss_response_with_metrics


class BenchmarkRunner:
    """Orchestrates benchmark execution with concurrency control."""

    def __init__(self, config: BenchmarkConfig, headers: dict) -> None:
        self.config = config
        self.headers = headers
        self.collector = MetricsCollector(config)

    async def run_single_request(self, request_id: int, semaphore: asyncio.Semaphore) -> RequestMetrics:
        """Execute a single API request and capture metrics."""
        async with semaphore:
            request_body = self.config.get_request_body_for_request(request_id)
            has_tools_flag = False
            has_system_flag = False
            prompt_length = 0

            if request_body:
                has_tools_flag = has_tools(request_body)
                has_system_flag = has_system_prompt(request_body)
                prompt_length = calculate_prompt_length(request_body)

                openai_payload = convert_anthropic_to_openai(request_body, target_model=self.config.model)
                openai_payload["stream"] = True
                payload = openai_payload
            else:
                prompt = self.config.get_prompt_for_request(request_id)
                prompt_length = len(prompt)
                payload = {
                    "model": self.config.model,
                    "stream": True,
                    "messages": [{"role": "user", "content": prompt}],
                }

            response_text, raw_metrics = await stream_gpt_oss_response_with_metrics(
                headers=self.headers, payload=payload, timeout_seconds=self.config.timeout_seconds
            )

            start_time = raw_metrics["start_time"]
            end_time = raw_metrics["end_time"]
            duration_seconds = None
            tokens_per_second = None

            if end_time and start_time:
                duration_seconds = (end_time - start_time).total_seconds()
                if duration_seconds > 0 and raw_metrics["total_tokens"] > 0:
                    tokens_per_second = raw_metrics["total_tokens"] / duration_seconds

            success = raw_metrics["error_type"] is None

            return RequestMetrics(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_seconds,
                ttft_seconds=raw_metrics["ttft"],
                total_tokens=raw_metrics["total_tokens"],
                tokens_per_second=tokens_per_second,
                success=success,
                error_type=raw_metrics["error_type"],
                error_message=raw_metrics["error_message"],
                prompt_length=prompt_length,
                has_tools=has_tools_flag,
                has_system_prompt=has_system_flag,
            )

    async def run_benchmark(self) -> BenchmarkResults:
        """Execute the full benchmark with configured concurrency."""
        semaphore = asyncio.Semaphore(self.config.concurrency)

        self.collector.set_benchmark_start_time(datetime.now(UTC))

        tasks = [self.run_single_request(i, semaphore) for i in range(self.config.num_requests)]

        print(f"Running {self.config.num_requests} requests with concurrency {self.config.concurrency}...")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                print(f"Unexpected error in request: {result}")
            else:
                self.collector.record_request(result)
                if not result.success:
                    print(f"Request {result.request_id} failed: {result.error_type}")
                    if result.error_type == "HTTPError_400" and result.error_message:
                        print(f"  Error details: {result.error_message[:500]}")

        self.collector.set_benchmark_end_time(datetime.now(UTC))

        return self.collector.calculate_results()
