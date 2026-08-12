"""Run-time entities: Run, Task, Score.

Distinct from Phase 1's catalog types (which describe what gets measured).
Runtime types describe a single execution and its outputs.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RunTrigger = Literal["manual", "launchd-weekly", "launchd-monthly", "ci"]
ScoringMethod = Literal["deterministic"]  # v2 adds "judge_ensemble"
SampleAggregation = Literal["single", "majority_vote", "median", "mean"]

# Run id: YYYY-MM-DD-{trigger}, e.g. 2026-04-29-manual
_RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z][a-z-]*$")


class RunStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    trigger: RunTrigger
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.IN_PROGRESS
    estimated_cost_usd: float | None = None

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _RUN_ID_RE.match(v):
            raise ValueError(f"Run id must be YYYY-MM-DD-<trigger>; got {v!r}")
        return v


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    prompt_id: str
    model_id: str
    sample_n: int = Field(ge=0)
    status: TaskStatus
    text: str
    """Full response body (or empty string on failure)."""
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd_estimate: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    error: str | None = None
    """Set on TaskStatus.FAILED; None on SUCCESS."""

    def key(self) -> tuple[str, str, str, int]:
        """Stable identity for resume: same key → same task to dispatch."""
        return (self.run_id, self.prompt_id, self.model_id, self.sample_n)


class Score(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    prompt_id: str
    model_id: str
    subject_id: str
    metric: str
    """e.g. mention_presence, mention_presence_rate, ordinal_rank, share_of_voice."""
    value: bool | int | float | None
    scoring_method: ScoringMethod = "deterministic"
    sample_aggregation: SampleAggregation = "single"
