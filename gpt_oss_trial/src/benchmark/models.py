"""Data models for benchmark configuration and results."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RequestMetrics(BaseModel):
    """Metrics for a single API request."""

    request_id: int
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float | None = None
    ttft_seconds: float | None = None
    total_tokens: int = 0
    tokens_per_second: float | None = None
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    prompt_length: int | None = None
    has_tools: bool = False
    has_system_prompt: bool = False


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution."""

    num_requests: int = Field(default=10, ge=1)
    concurrency: int = Field(default=1, ge=1)
    model: str = "openai/gpt-oss-120b-maas"
    prompt: str | None = "Write a one-liner about convictional.com's mission."
    prompts: list[str] | None = None
    request_bodies: list[dict] | None = None
    timeout_seconds: float = 60.0
    output_dir: str = "./results"
    use_db_prompts: bool = False
    db_name: str | None = None

    def get_prompt_for_request(self, request_id: int) -> str:
        """Get the prompt to use for a specific request (simple prompts only)."""
        if self.prompts:
            return self.prompts[request_id % len(self.prompts)]
        return self.prompt or ""

    def get_request_body_for_request(self, request_id: int) -> dict | None:
        """Get the full request body for a specific request (for database-sampled requests)."""
        if self.request_bodies:
            return self.request_bodies[request_id % len(self.request_bodies)]
        return None


class BenchmarkResults(BaseModel):
    """Aggregated benchmark results and statistics."""

    config: BenchmarkConfig
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_seconds: float

    mean_latency: float | None = None
    median_latency: float | None = None
    p95_latency: float | None = None
    p99_latency: float | None = None
    min_latency: float | None = None
    max_latency: float | None = None

    mean_ttft: float | None = None
    median_ttft: float | None = None
    p95_ttft: float | None = None
    p99_ttft: float | None = None

    total_tokens: int = 0
    tokens_per_second: float | None = None
    requests_per_second: float | None = None

    success_rate: float
    error_breakdown: dict[str, int] = Field(default_factory=dict)

    mean_prompt_length: float | None = None
    median_prompt_length: float | None = None
    min_prompt_length: int | None = None
    max_prompt_length: int | None = None

    requests_with_tools: int = 0
    requests_with_system: int = 0

    timestamp: datetime = Field(default_factory=datetime.now)


class ComparisonRequestMetrics(BaseModel):
    """Per-request metrics comparing Claude and GPT-OSS on the same prompt."""

    request_id: int
    prompt_length: int | None = None
    has_tools: bool = False
    has_system_prompt: bool = False

    claude_start_time: datetime
    claude_end_time: datetime | None = None
    claude_duration_seconds: float | None = None
    claude_ttft_seconds: float | None = None
    claude_total_tokens: int = 0
    claude_tokens_per_second: float | None = None
    claude_success: bool = True
    claude_error_type: str | None = None
    claude_error_message: str | None = None
    claude_response_text: str | None = None

    gpt_start_time: datetime
    gpt_end_time: datetime | None = None
    gpt_duration_seconds: float | None = None
    gpt_ttft_seconds: float | None = None
    gpt_total_tokens: int = 0
    gpt_tokens_per_second: float | None = None
    gpt_success: bool = True
    gpt_error_type: str | None = None
    gpt_error_message: str | None = None
    gpt_response_text: str | None = None
