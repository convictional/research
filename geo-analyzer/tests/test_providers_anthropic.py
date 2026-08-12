from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from geo_analyzer.providers.anthropic import AnthropicProvider
from geo_analyzer.providers.base import ProbeRequest
from geo_analyzer.types import ModelSpec, SamplingConfig


def _ungrounded_model() -> ModelSpec:
    return ModelSpec(
        id="anthropic:claude-opus-4-7:ungrounded",
        provider="anthropic",
        model_name="claude-opus-4-7",
        mode="ungrounded",
        active=True,
        config={},
        sampling=SamplingConfig(n=1, temperature=0),
    )


def _grounded_model() -> ModelSpec:
    return ModelSpec(
        id="anthropic:claude-opus-4-7:grounded",
        provider="anthropic",
        model_name="claude-opus-4-7",
        mode="grounded",
        active=True,
        config={"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
        sampling=SamplingConfig(n=3, temperature=None),
    )


def _stub_message(text: str, input_tokens: int, output_tokens: int) -> Any:
    block = MagicMock()
    block.type = "text"
    block.text = text
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    msg = MagicMock()
    msg.content = [block]
    msg.usage = usage
    msg.model_dump = MagicMock(return_value={"content": [{"type": "text", "text": text}]})
    return msg


class TestAnthropicProviderUngrounded:
    @pytest.mark.asyncio
    async def test_returns_text_and_tokens(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_stub_message("Convictional repositions.", 100, 50))
        provider = AnthropicProvider(api_key="sk-ant-test", client=mock_client)
        req = ProbeRequest(model=_ungrounded_model(), prompt="hi")
        resp = await provider.call(req)
        assert resp.text == "Convictional repositions."
        assert resp.tokens_in == 100
        assert resp.tokens_out == 50
        assert resp.cost_usd_estimate > 0

    @pytest.mark.asyncio
    async def test_passes_temperature_zero(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_stub_message("ok", 5, 5))
        provider = AnthropicProvider(api_key="x", client=mock_client)
        await provider.call(ProbeRequest(model=_ungrounded_model(), prompt="hi"))
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-opus-4-7"
        assert kwargs["temperature"] == 0
        assert "tools" not in kwargs


class TestAnthropicProviderGrounded:
    @pytest.mark.asyncio
    async def test_passes_tools(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_stub_message("ok", 5, 5))
        provider = AnthropicProvider(api_key="x", client=mock_client)
        await provider.call(ProbeRequest(model=_grounded_model(), prompt="hi"))
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["tools"] == [{"type": "web_search_20250305", "name": "web_search"}]


class TestAnthropicTextBlockExtraction:
    @pytest.mark.asyncio
    async def test_concatenates_multiple_text_blocks(self) -> None:
        # Anthropic responses can contain multiple blocks (e.g., tool_use + text).
        # We concatenate all blocks where block.type == "text".
        block1 = MagicMock()
        block1.type = "text"
        block1.text = "Hello "
        tool_block = MagicMock()
        tool_block.type = "tool_use"  # no .text
        block2 = MagicMock()
        block2.type = "text"
        block2.text = "world."
        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 5
        msg = MagicMock()
        msg.content = [block1, tool_block, block2]
        msg.usage = usage
        msg.model_dump = MagicMock(return_value={})

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=msg)
        provider = AnthropicProvider(api_key="x", client=mock_client)
        resp = await provider.call(ProbeRequest(model=_ungrounded_model(), prompt="hi"))
        assert resp.text == "Hello world."


class TestRegistry:
    @pytest.mark.asyncio
    async def test_get_provider_returns_anthropic_adapter(self) -> None:
        from geo_analyzer.providers import get_provider

        p = get_provider("anthropic", api_key="sk-ant-test")
        assert p.name == "anthropic"


class TestSkipTemperatureFlag:
    @pytest.mark.asyncio
    async def test_skip_temperature_omits_param(self) -> None:
        # Reasoning models (opus-4-7) reject `temperature`; catalog can opt out
        # via config.skip_temperature=true.
        model = ModelSpec(
            id="anthropic:claude-opus-4-7:ungrounded",
            provider="anthropic",
            model_name="claude-opus-4-7",
            mode="ungrounded",
            active=True,
            config={"skip_temperature": True},
            sampling=SamplingConfig(n=1, temperature=0),
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_stub_message("ok", 5, 5))
        provider = AnthropicProvider(api_key="x", client=mock_client)
        await provider.call(ProbeRequest(model=model, prompt="hi"))
        kwargs = mock_client.messages.create.call_args.kwargs
        assert "temperature" not in kwargs
