"""Google adapter. Wraps google.genai.Client for both modes.

The new google-genai SDK (>=1.0) exposes async via `client.aio.models.generate_content`.
Generation parameters (temperature, tools) are passed via the `config` dict.
"""

from __future__ import annotations

import time
from typing import Any

from google import genai

from geo_analyzer.providers.base import (
    ProbeRequest,
    ProviderError,
    ProviderResponse,
)
from geo_analyzer.providers.pricing import estimate_cost


class GoogleProvider:
    name = "google"

    def __init__(self, api_key: str, *, client: genai.Client | Any | None = None) -> None:
        self._client = client if client is not None else genai.Client(api_key=api_key)

    async def call(self, request: ProbeRequest) -> ProviderResponse:
        m = request.model
        config: dict[str, Any] = {}
        temp = request.temperature_override if request.temperature_override is not None else m.sampling.temperature
        if temp is not None:
            config["temperature"] = temp
        if m.mode == "grounded":
            tools = m.config.get("tools")
            if tools:
                config["tools"] = tools

        t0 = time.monotonic()
        try:
            response = await self._client.aio.models.generate_content(  # type: ignore[union-attr]
                model=m.model_name,
                contents=request.prompt,
                config=config,  # type: ignore[arg-type]
            )
        except Exception as e:
            raise ProviderError(f"google call failed: {e}") from e
        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            text = response.text or ""
            tokens_in = int(response.usage_metadata.prompt_token_count)  # type: ignore[union-attr]
            tokens_out = int(response.usage_metadata.candidates_token_count)  # type: ignore[union-attr]
        except (AttributeError, TypeError) as e:
            raise ProviderError(f"google response malformed: {e}") from e

        raw_dump: dict[str, Any] = {}
        try:
            raw_dump = response.model_dump()
        except Exception:
            raw_dump = {}

        return ProviderResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd_estimate=estimate_cost(m.model_name, tokens_in=tokens_in, tokens_out=tokens_out),
            latency_ms=latency_ms,
            raw=raw_dump,
        )
