from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from geo_analyzer.providers.base import ProbeRequest
from geo_analyzer.providers.openai import OpenAIProvider
from geo_analyzer.types import ModelSpec, SamplingConfig


def _ungrounded_model() -> ModelSpec:
    return ModelSpec(
        id="openai:gpt-5.1:ungrounded",
        provider="openai",
        model_name="gpt-5.1",
        mode="ungrounded",
        active=True,
        config={},
        sampling=SamplingConfig(n=1, temperature=0, seed=42),
    )


def _grounded_model() -> ModelSpec:
    return ModelSpec(
        id="openai:gpt-5.1:grounded",
        provider="openai",
        model_name="gpt-5.1",
        mode="grounded",
        active=True,
        config={"tools": [{"type": "web_search"}]},
        sampling=SamplingConfig(n=3, temperature=None, seed=42),
    )


def _stub_completion(text: str, prompt_tokens: int, completion_tokens: int) -> Any:
    """Build a duck-typed object mimicking openai.types.chat.ChatCompletion."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    completion.model_dump = MagicMock(return_value={"id": "stub-1", "choices": [{"message": {"content": text}}]})
    return completion


def _stub_response(text: str, input_tokens: int, output_tokens: int) -> Any:
    """Build a duck-typed object mimicking openai.types.responses.Response."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response = MagicMock()
    response.output_text = text
    response.usage = usage
    response.model_dump = MagicMock(return_value={"id": "stub-resp-1", "output_text": text})
    return response


class TestOpenAIProviderUngrounded:
    @pytest.mark.asyncio
    async def test_returns_text_and_tokens(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_stub_completion("Convictional helps leaders.", 100, 50)
        )
        provider = OpenAIProvider(api_key="sk-test", client=mock_client)
        req = ProbeRequest(model=_ungrounded_model(), prompt="What is org health?")
        resp = await provider.call(req)
        assert resp.text == "Convictional helps leaders."
        assert resp.tokens_in == 100
        assert resp.tokens_out == 50
        assert resp.cost_usd_estimate > 0
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_passes_temperature_zero_to_sdk(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_stub_completion("ok", 5, 5))
        provider = OpenAIProvider(api_key="sk-test", client=mock_client)
        await provider.call(ProbeRequest(model=_ungrounded_model(), prompt="hi"))
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-5.1"
        assert kwargs["temperature"] == 0
        assert kwargs["seed"] == 42
        assert "tools" not in kwargs  # ungrounded must not send tools

    @pytest.mark.asyncio
    async def test_temperature_override_takes_precedence(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_stub_completion("ok", 5, 5))
        provider = OpenAIProvider(api_key="sk-test", client=mock_client)
        await provider.call(
            ProbeRequest(
                model=_ungrounded_model(),
                prompt="hi",
                temperature_override=0.7,
                seed_override=1234,
            )
        )
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["seed"] == 1234


class TestOpenAIProviderGrounded:
    @pytest.mark.asyncio
    async def test_uses_responses_api_with_tools(self) -> None:
        # Grounded mode uses the Responses API (where built-in tools like
        # web_search live). Chat Completions rejects {"type": "web_search"}.
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=_stub_response("Per https://x.com, ...", 200, 80))
        provider = OpenAIProvider(api_key="sk-test", client=mock_client)
        resp = await provider.call(ProbeRequest(model=_grounded_model(), prompt="hi"))

        # Chat Completions must NOT have been touched.
        mock_client.chat.completions.create.assert_not_called()

        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["model"] == "gpt-5.1"
        assert kwargs["input"] == "hi"
        assert kwargs["tools"] == [{"type": "web_search"}]
        # Grounded model has sampling.temperature=None (provider default) —
        # the adapter omits the key entirely rather than passing None.
        assert "temperature" not in kwargs

        assert resp.text == "Per https://x.com, ..."
        assert resp.tokens_in == 200
        assert resp.tokens_out == 80
        assert resp.cost_usd_estimate > 0

    @pytest.mark.asyncio
    async def test_grounded_temperature_override_is_passed(self) -> None:
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=_stub_response("ok", 5, 5))
        provider = OpenAIProvider(api_key="sk-test", client=mock_client)
        await provider.call(ProbeRequest(model=_grounded_model(), prompt="hi", temperature_override=0.7))
        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7


class TestRegistry:
    @pytest.mark.asyncio
    async def test_get_provider_returns_openai_adapter(self) -> None:
        from geo_analyzer.providers import get_provider

        # Importing geo_analyzer.providers.openai (which Task 4 just made happen
        # via __init__.py) registers the adapter as a side effect.
        p = get_provider("openai", api_key="sk-test")
        assert p.name == "openai"
