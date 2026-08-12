"""OpenAI adapter. Wraps openai.AsyncOpenAI for both ungrounded and grounded modes.

Ungrounded → `client.chat.completions.create(...)` (the classic chat endpoint).
Grounded   → `client.responses.create(...)` (the newer Responses API, which is
where built-in tools like `web_search` actually live; chat.completions only
accepts `function`/`custom` tools and rejects `web_search`).

Token-count field names differ between the two endpoints:
  Chat Completions:  usage.prompt_tokens  / usage.completion_tokens
  Responses API:     usage.input_tokens   / usage.output_tokens
"""

from __future__ import annotations

import time
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from geo_analyzer.providers.base import (
    ProbeRequest,
    ProviderError,
    ProviderResponse,
)
from geo_analyzer.providers.pricing import estimate_cost


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, *, client: Any | None = None) -> None:
        self._client: AsyncOpenAI = client if client is not None else AsyncOpenAI(api_key=api_key)

    async def call(self, request: ProbeRequest) -> ProviderResponse:
        m = request.model
        # Temperature: override > model.sampling.temperature > omit (SDK default)
        temp = request.temperature_override if request.temperature_override is not None else m.sampling.temperature
        # Seed: override > sampling.seed > omit
        seed = request.seed_override if request.seed_override is not None else m.sampling.seed

        if m.mode == "grounded":
            return await self._call_responses_api(m=m, prompt=request.prompt, temperature=temp)
        return await self._call_chat_completions(m=m, prompt=request.prompt, temperature=temp, seed=seed)

    async def _call_chat_completions(
        self,
        *,
        m: Any,
        prompt: str,
        temperature: float | None,
        seed: int | None,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": m.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed

        t0 = time.monotonic()
        try:
            completion = cast(ChatCompletion, await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            raise ProviderError(f"openai call failed: {e}") from e
        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            text: str = completion.choices[0].message.content or ""
            tokens_in: int = completion.usage.prompt_tokens  # type: ignore[union-attr]
            tokens_out: int = completion.usage.completion_tokens  # type: ignore[union-attr]
        except (AttributeError, IndexError, TypeError) as e:
            raise ProviderError(f"openai response malformed: {e}") from e

        return ProviderResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd_estimate=estimate_cost(m.model_name, tokens_in=tokens_in, tokens_out=tokens_out),
            latency_ms=latency_ms,
            raw=_safe_dump(completion),
        )

    async def _call_responses_api(
        self,
        *,
        m: Any,
        prompt: str,
        temperature: float | None,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": m.model_name,
            "input": prompt,
        }
        # Tools: pass verbatim from model.config (e.g., [{"type": "web_search"}]).
        tools = m.config.get("tools")
        if tools:
            kwargs["tools"] = tools
        # Temperature is optional on the Responses API and may not be supported for
        # every model (reasoning models often reject it). Only pass when explicitly set.
        if temperature is not None:
            kwargs["temperature"] = temperature

        t0 = time.monotonic()
        try:
            response = cast(Response, await self._client.responses.create(**kwargs))
        except Exception as e:
            raise ProviderError(f"openai call failed: {e}") from e
        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            text: str = response.output_text or ""
            tokens_in: int = response.usage.input_tokens  # type: ignore[union-attr]
            tokens_out: int = response.usage.output_tokens  # type: ignore[union-attr]
        except (AttributeError, TypeError) as e:
            raise ProviderError(f"openai response malformed: {e}") from e

        return ProviderResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd_estimate=estimate_cost(m.model_name, tokens_in=tokens_in, tokens_out=tokens_out),
            latency_ms=latency_ms,
            raw=_safe_dump(response),
        )


def _safe_dump(obj: Any) -> dict[str, Any]:
    """Best-effort raw capture for debugging; some test stubs lack model_dump."""
    try:
        return cast(dict[str, Any], obj.model_dump())
    except Exception:
        return {}
