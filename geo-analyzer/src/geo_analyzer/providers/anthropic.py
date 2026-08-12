"""Anthropic adapter. Wraps anthropic.AsyncAnthropic for both modes.

The Messages API returns content as a list of typed blocks (text, tool_use,
tool_result, etc.). We concatenate every block where type == "text" to get the
visible response. Tool-use blocks are kept in the raw dump for debugging but
not surfaced as text.
"""

from __future__ import annotations

import time
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import Message

from geo_analyzer.providers.base import (
    ProbeRequest,
    ProviderError,
    ProviderResponse,
)
from geo_analyzer.providers.pricing import estimate_cost

_DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, *, client: AsyncAnthropic | Any | None = None) -> None:
        self._client = client if client is not None else AsyncAnthropic(api_key=api_key)

    async def call(self, request: ProbeRequest) -> ProviderResponse:
        m = request.model
        kwargs: dict[str, Any] = {
            "model": m.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": _DEFAULT_MAX_TOKENS,
        }
        temp = request.temperature_override if request.temperature_override is not None else m.sampling.temperature
        # Reasoning models (e.g., claude-opus-4-7) reject `temperature` entirely.
        # Catalog can set config.skip_temperature=true to suppress sending it.
        skip_temperature = bool(m.config.get("skip_temperature", False))
        if temp is not None and not skip_temperature:
            kwargs["temperature"] = temp
        # Anthropic does not support seed in current SDK; ignored even if set.
        if m.mode == "grounded":
            tools = m.config.get("tools")
            if tools:
                kwargs["tools"] = tools

        t0 = time.monotonic()
        try:
            msg = cast(Message, await self._client.messages.create(**kwargs))
        except Exception as e:
            raise ProviderError(f"anthropic call failed: {e}") from e
        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
            tokens_in = int(msg.usage.input_tokens)
            tokens_out = int(msg.usage.output_tokens)
        except (AttributeError, TypeError) as e:
            raise ProviderError(f"anthropic response malformed: {e}") from e

        raw_dump: dict[str, Any] = {}
        try:
            raw_dump = msg.model_dump()
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
