"""Structured response benchmark framework."""

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.benchmark.structured_prompts import STRUCTURED_PROMPTS
from src.llm.instructor_client import StructuredResponseClient
from src.models.structured_responses import ComplexResponse, ReasonedResponse, SimpleResponse


@dataclass
class StructuredTestResult:
    """Results from a single structured response test."""

    model_name: str
    prompt_description: str
    schema_name: str
    success: bool
    validation_error: str | None
    latency: float
    retry_count: int
    response_content: str | None
    timestamp: float


class StructuredBenchmark:
    """Benchmark for structured response testing."""

    SCHEMA_CLASSES = {
        "SimpleResponse": SimpleResponse,
        "ReasonedResponse": ReasonedResponse,
        "ComplexResponse": ComplexResponse,
    }

    def __init__(
        self, gpt_oss_client: StructuredResponseClient | None = None, claude_client: StructuredResponseClient | None = None
    ) -> None:
        self.gpt_oss_client = gpt_oss_client
        self.claude_client = claude_client
        self.results: list[StructuredTestResult] = []

    def run_all_tests(self) -> list[StructuredTestResult]:
        """Run all structured prompts against both models."""
        for prompt_config in STRUCTURED_PROMPTS:
            schema_class = self.SCHEMA_CLASSES[prompt_config.model_class_name]

            if self.gpt_oss_client:
                self._run_single_test(
                    client=self.gpt_oss_client,
                    model_name="gpt-oss-120b-maas",
                    prompt_config=prompt_config,
                    schema_class=schema_class,
                )

            if self.claude_client:
                self._run_single_test(
                    client=self.claude_client,
                    model_name="claude-sonnet-4-5-20250929",
                    prompt_config=prompt_config,
                    schema_class=schema_class,
                )

        return self.results

    def _run_single_test(self, client: StructuredResponseClient, model_name: str, prompt_config, schema_class) -> None:
        """Run a single test and record results."""
        print(f"Testing {model_name} with {prompt_config.model_class_name}: {prompt_config.description}")

        response, metrics = client.generate(
            prompt=prompt_config.prompt,
            response_model=schema_class,
            system_prompt="You are a helpful assistant. Provide structured responses.",
        )

        result = StructuredTestResult(
            model_name=model_name,
            prompt_description=prompt_config.description,
            schema_name=prompt_config.model_class_name,
            success=metrics["success"],
            validation_error=metrics["validation_error"],
            latency=metrics["latency"],
            retry_count=metrics["retry_count"],
            response_content=response.model_dump_json() if response else None,
            timestamp=time.time(),
        )

        self.results.append(result)

        if not result.success:
            print(f"  FAILED: {result.validation_error}")
        else:
            print(f"  SUCCESS (latency: {result.latency:.2f}s, retries: {result.retry_count})")

    def export_results(self, output_path: Path) -> None:
        """Export results to CSV."""
        data = []
        for result in self.results:
            data.append(
                {
                    "model_name": result.model_name,
                    "prompt_description": result.prompt_description,
                    "schema_name": result.schema_name,
                    "success": result.success,
                    "validation_error": result.validation_error or "",
                    "latency_seconds": f"{result.latency:.3f}",
                    "retry_count": result.retry_count,
                    "response_content": result.response_content or "",
                    "timestamp": result.timestamp,
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    def print_summary(self) -> None:
        """Print summary statistics."""
        if not self.results:
            print("No results to summarize")
            return

        by_model = {}
        for result in self.results:
            if result.model_name not in by_model:
                by_model[result.model_name] = []
            by_model[result.model_name].append(result)

        print("\n" + "=" * 70)
        print("Structured Response Benchmark Summary")
        print("=" * 70)

        for model_name, results in by_model.items():
            total = len(results)
            successes = sum(1 for r in results if r.success)
            failures = total - successes
            avg_latency = sum(r.latency for r in results) / total
            avg_retries = sum(r.retry_count for r in results) / total

            print(f"\nModel: {model_name}")
            print(f"  Total tests:      {total}")
            print(f"  Successes:        {successes} ({successes/total*100:.1f}%)")
            print(f"  Failures:         {failures} ({failures/total*100:.1f}%)")
            print(f"  Avg latency:      {avg_latency:.3f}s")
            print(f"  Avg retries:      {avg_retries:.2f}")

            by_schema = {}
            for result in results:
                if result.schema_name not in by_schema:
                    by_schema[result.schema_name] = []
                by_schema[result.schema_name].append(result)

            print(f"\n  By schema complexity:")
            for schema_name, schema_results in by_schema.items():
                schema_successes = sum(1 for r in schema_results if r.success)
                schema_total = len(schema_results)
                schema_avg_latency = sum(r.latency for r in schema_results) / schema_total
                print(
                    f"    {schema_name:20s} {schema_successes}/{schema_total} "
                    f"({schema_successes/schema_total*100:.1f}%) - avg: {schema_avg_latency:.2f}s"
                )

        print("\n" + "=" * 70 + "\n")
