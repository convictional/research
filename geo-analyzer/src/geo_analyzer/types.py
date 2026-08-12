"""Pydantic models for the catalog and run-time entities.

These types are the contract between catalog YAML on disk and every other
component (loader, scoring, runner, reports). Strict by default — wrong YAML
should fail at load-time, not at run-time.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PromptTier = Literal["L1", "L2", "L3", "L4"]
ModelMode = Literal["grounded", "ungrounded"]


_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PROMPT_ID_RE = re.compile(r"^prompt\.[a-z0-9_-]+(\.[a-z0-9_-]+)*$")
_MODEL_ID_RE = re.compile(r"^[a-z0-9_-]+:[a-zA-Z0-9._-]+:(grounded|ungrounded)$")


class SubjectKind(str, Enum):
    BRAND = "brand"
    CATEGORY = "category"
    ANTI_BRAND = "anti_brand"


class Subject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    kind: SubjectKind
    aliases: list[str] = Field(min_length=1)
    definition: str = Field(min_length=1)
    competitors: list[str] = Field(default_factory=list)
    owned_domains: list[str] = Field(default_factory=list)
    legacy_of: str | None = None
    """For anti_brand subjects: the brand id this anti-brand is the legacy of.
    The conflation metric only fires when the named brand and this anti-brand
    co-occur. Setting this on a non-anti_brand subject is a catalog error."""

    @field_validator("id")
    @classmethod
    def _id_snake(cls, v: str) -> str:
        if not _SNAKE_RE.match(v):
            raise ValueError(f"Subject id must be snake_case: got {v!r}")
        return v


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tier: PromptTier
    text: str = Field(min_length=1)
    targets: list[str] = Field(min_length=1)
    version: int = Field(ge=1)
    authored_at: date

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _PROMPT_ID_RE.match(v):
            raise ValueError(f"Prompt id must match prompt.<tier>.<slug>...; got {v!r}")
        return v


class RetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(default=3, ge=1)
    backoff_base_s: float = Field(default=2.0, gt=0)


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concurrency: int = Field(ge=1)
    retry: RetryConfig = Field(default_factory=RetryConfig)


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n: int = Field(ge=1)
    temperature: float | None
    seed: int | None = None


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    provider: str
    model_name: str
    mode: ModelMode
    active: bool
    config: dict[str, Any] = Field(default_factory=dict)
    sampling: SamplingConfig

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _MODEL_ID_RE.match(v):
            raise ValueError(f"Model id must be 'provider:model_name:(grounded|ungrounded)'; got {v!r}")
        return v


class Catalog(BaseModel):
    """In-memory representation of the full catalog after loading + validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subjects: list[Subject]
    prompts: list[Prompt]
    providers: dict[str, ProviderConfig]
    models: list[ModelSpec]


GoalDirection = Literal["above", "below"]


class Goal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    subject: str
    metric: str
    tier: PromptTier
    target: float
    created_at: date
    target_date: date
    direction: GoalDirection = "above"
    notes: str | None = None
