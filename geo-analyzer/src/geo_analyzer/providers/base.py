"""Provider abstraction. Adapters wrap each LLM SDK and return a uniform shape.

Phase 2 adapters return text + token counts + cost + latency. Citation extraction
on the text body is Phase 3's job (see geo_analyzer.scoring.extract_citations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from geo_analyzer.types import ModelSpec


class ProviderError(RuntimeError):
    """Base error for provider problems (unknown name, auth, transient API errors).

    Adapters convert SDK exceptions into ProviderError so callers see one type.
    """


@dataclass(frozen=True)
class ProbeRequest:
    """One sample's worth of input. The adapter is N=1 — to get N samples,
    the caller invokes `provider.call()` N times."""

    model: ModelSpec
    prompt: str
    temperature_override: float | None = None
    """If not None, overrides model.sampling.temperature for this call only."""
    seed_override: int | None = None
    """If not None, overrides model.sampling.seed for this call only."""


@dataclass(frozen=True)
class ProviderResponse:
    """Uniform shape returned by every adapter."""

    text: str
    """Full response body. Empty string is allowed (e.g., refusal). Never None."""
    tokens_in: int
    tokens_out: int
    cost_usd_estimate: float
    """USD; computed by `geo_analyzer.providers.pricing.estimate_cost`."""
    latency_ms: int
    raw: dict[str, Any]
    """Provider-specific raw response (kept for forensics; not schema-checked)."""


@runtime_checkable
class Provider(Protocol):
    """Async provider adapter contract."""

    name: str  # lowercase, matches model.provider (e.g., "openai")

    async def call(self, request: ProbeRequest) -> ProviderResponse:
        """Execute one prompt against the wrapped SDK; return one sample."""
        ...
