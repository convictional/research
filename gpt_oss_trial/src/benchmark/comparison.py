"""Parallel execution and comparison of Claude vs GPT-OSS benchmarks."""

import asyncio
from datetime import UTC, datetime

from src.benchmark.format_converter import (
    calculate_prompt_length,
    convert_anthropic_to_openai,
    has_system_prompt,
    has_tools,
)
from src.benchmark.models import BenchmarkConfig, ComparisonRequestMetrics
from src.llm.claude import stream_claude_response_with_metrics
from src.llm.gpt_oss import stream_gpt_oss_response_with_metrics


class ComparisonRunner:
    """Orchestrates parallel benchmark execution comparing Claude and GPT-OSS."""

    def __init__(self, config: BenchmarkConfig, gpt_headers: dict, claude_api_key: str, save_responses: bool = False) -> None:
        self.config = config
        self.gpt_headers = gpt_headers
        self.claude_api_key = claude_api_key
        self.comparison_metrics: list[ComparisonRequestMetrics] = []
        self.save_responses = save_responses

    async def run_parallel_request(
        self, request_id: int, semaphore: asyncio.Semaphore
    ) -> ComparisonRequestMetrics:
        """
        Execute the same request to both Claude and GPT-OSS in parallel.

        Args:
            request_id: Request identifier
            semaphore: Concurrency control semaphore

        Returns:
            ComparisonRequestMetrics containing metrics from both models
        """
        async with semaphore:
            request_body = self.config.get_request_body_for_request(request_id)
            has_tools_flag = False
            has_system_flag = False
            prompt_length = 0

            if request_body:
                has_tools_flag = has_tools(request_body)
                has_system_flag = has_system_prompt(request_body)
                prompt_length = calculate_prompt_length(request_body)

                claude_payload = request_body.copy()
                if "stream" in claude_payload:
                    del claude_payload["stream"]
                if "max_tokens" not in claude_payload:
                    claude_payload["max_tokens"] = 4096
                claude_payload["model"] = "claude-sonnet-4-5-20250929"

                gpt_payload = convert_anthropic_to_openai(request_body, target_model=self.config.model)
                gpt_payload["stream"] = True
            else:
                prompt = self.config.get_prompt_for_request(request_id)
                prompt_length = len(prompt)

                claude_payload = {
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                }

                gpt_payload = {
                    "model": self.config.model,
                    "stream": True,
                    "messages": [{"role": "user", "content": prompt}],
                }

            claude_task = stream_claude_response_with_metrics(
                api_key=self.claude_api_key,
                request_body=claude_payload,
                timeout_seconds=self.config.timeout_seconds,
            )

            gpt_task = stream_gpt_oss_response_with_metrics(
                headers=self.gpt_headers,
                payload=gpt_payload,
                timeout_seconds=self.config.timeout_seconds,
            )

            results = await asyncio.gather(claude_task, gpt_task, return_exceptions=True)

            if isinstance(results[0], Exception):
                claude_text = ""
                claude_metrics = {
                    "start_time": datetime.now(UTC),
                    "end_time": datetime.now(UTC),
                    "ttft": None,
                    "total_tokens": 0,
                    "error_type": f"UnknownError_{type(results[0]).__name__}",
                    "error_message": str(results[0]),
                }
            else:
                claude_text, claude_metrics = results[0]

            if isinstance(results[1], Exception):
                gpt_text = ""
                gpt_metrics = {
                    "start_time": datetime.now(UTC),
                    "end_time": datetime.now(UTC),
                    "ttft": None,
                    "total_tokens": 0,
                    "error_type": f"UnknownError_{type(results[1]).__name__}",
                    "error_message": str(results[1]),
                }
            else:
                gpt_text, gpt_metrics = results[1]

            def process_metrics(raw_metrics: dict) -> dict:
                start_time = raw_metrics["start_time"]
                end_time = raw_metrics["end_time"]
                duration_seconds = None
                tokens_per_second = None

                if end_time and start_time:
                    duration_seconds = (end_time - start_time).total_seconds()
                    if duration_seconds > 0 and raw_metrics["total_tokens"] > 0:
                        tokens_per_second = raw_metrics["total_tokens"] / duration_seconds

                return {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration_seconds,
                    "ttft_seconds": raw_metrics["ttft"],
                    "total_tokens": raw_metrics["total_tokens"],
                    "tokens_per_second": tokens_per_second,
                    "success": raw_metrics["error_type"] is None,
                    "error_type": raw_metrics["error_type"],
                    "error_message": raw_metrics["error_message"],
                }

            claude_processed = process_metrics(claude_metrics)
            gpt_processed = process_metrics(gpt_metrics)

            return ComparisonRequestMetrics(
                request_id=request_id,
                prompt_length=prompt_length,
                has_tools=has_tools_flag,
                has_system_prompt=has_system_flag,
                claude_start_time=claude_processed["start_time"],
                claude_end_time=claude_processed["end_time"],
                claude_duration_seconds=claude_processed["duration_seconds"],
                claude_ttft_seconds=claude_processed["ttft_seconds"],
                claude_total_tokens=claude_processed["total_tokens"],
                claude_tokens_per_second=claude_processed["tokens_per_second"],
                claude_success=claude_processed["success"],
                claude_error_type=claude_processed["error_type"],
                claude_error_message=claude_processed["error_message"],
                claude_response_text=claude_text if self.save_responses else None,
                gpt_start_time=gpt_processed["start_time"],
                gpt_end_time=gpt_processed["end_time"],
                gpt_duration_seconds=gpt_processed["duration_seconds"],
                gpt_ttft_seconds=gpt_processed["ttft_seconds"],
                gpt_total_tokens=gpt_processed["total_tokens"],
                gpt_tokens_per_second=gpt_processed["tokens_per_second"],
                gpt_success=gpt_processed["success"],
                gpt_error_type=gpt_processed["error_type"],
                gpt_error_message=gpt_processed["error_message"],
                gpt_response_text=gpt_text if self.save_responses else None,
            )

    async def run_comparison_benchmark(self) -> list[ComparisonRequestMetrics]:
        """Execute the full comparison benchmark with configured concurrency."""
        semaphore = asyncio.Semaphore(self.config.concurrency)

        tasks = [self.run_parallel_request(i, semaphore) for i in range(self.config.num_requests)]

        print(
            f"Running {self.config.num_requests} parallel comparisons (Claude vs GPT-OSS) "
            f"with concurrency {self.config.concurrency}..."
        )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                print(f"Unexpected error in comparison request: {result}")
            else:
                self.comparison_metrics.append(result)
                if not result.claude_success:
                    print(f"Request {result.request_id} - Claude failed: {result.claude_error_type}")
                if not result.gpt_success:
                    print(f"Request {result.request_id} - GPT-OSS failed: {result.gpt_error_type}")

        return self.comparison_metrics
