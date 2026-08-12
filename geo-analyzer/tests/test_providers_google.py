from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from geo_analyzer.providers.base import ProbeRequest
from geo_analyzer.providers.google import GoogleProvider
from geo_analyzer.types import ModelSpec, SamplingConfig


def _ungrounded_model() -> ModelSpec:
    return ModelSpec(
        id="google:gemini-2.5-pro:ungrounded",
        provider="google",
        model_name="gemini-2.5-pro",
        mode="ungrounded",
        active=True,
        config={},
        sampling=SamplingConfig(n=1, temperature=0),
    )


def _grounded_model() -> ModelSpec:
    return ModelSpec(
        id="google:gemini-2.5-pro:grounded",
        provider="google",
        model_name="gemini-2.5-pro",
        mode="grounded",
        active=True,
        config={"tools": [{"google_search": {}}]},
        sampling=SamplingConfig(n=3, temperature=None),
    )


def _stub_response(text: str, prompt_tokens: int, completion_tokens: int) -> Any:
    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = completion_tokens
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = usage
    resp.model_dump = MagicMock(return_value={"text": text})
    return resp


class TestGoogleProviderUngrounded:
    @pytest.mark.asyncio
    async def test_returns_text_and_tokens(self) -> None:
        mock_aio = MagicMock()
        mock_aio.models.generate_content = AsyncMock(return_value=_stub_response("Org health is a category.", 100, 50))
        mock_client = MagicMock()
        mock_client.aio = mock_aio

        provider = GoogleProvider(api_key="g-test", client=mock_client)
        req = ProbeRequest(model=_ungrounded_model(), prompt="hi")
        resp = await provider.call(req)
        assert resp.text == "Org health is a category."
        assert resp.tokens_in == 100
        assert resp.tokens_out == 50
        assert resp.cost_usd_estimate > 0

    @pytest.mark.asyncio
    async def test_passes_temperature_to_config(self) -> None:
        mock_aio = MagicMock()
        mock_aio.models.generate_content = AsyncMock(return_value=_stub_response("ok", 5, 5))
        mock_client = MagicMock()
        mock_client.aio = mock_aio

        provider = GoogleProvider(api_key="g-test", client=mock_client)
        await provider.call(ProbeRequest(model=_ungrounded_model(), prompt="hi"))
        kwargs = mock_aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-pro"
        # google-genai passes generation config and tools via kwargs (config dict
        # or GenerateContentConfig). We check the merged kwargs structure.
        config = cast(dict[str, Any], kwargs.get("config", {}))
        # Different versions of google-genai accept either a dict or a typed
        # config object. The adapter passes a dict for portability.
        assert isinstance(config, dict)
        assert config.get("temperature") == 0


class TestGoogleProviderGrounded:
    @pytest.mark.asyncio
    async def test_passes_tools(self) -> None:
        mock_aio = MagicMock()
        mock_aio.models.generate_content = AsyncMock(return_value=_stub_response("ok", 5, 5))
        mock_client = MagicMock()
        mock_client.aio = mock_aio

        provider = GoogleProvider(api_key="g-test", client=mock_client)
        await provider.call(ProbeRequest(model=_grounded_model(), prompt="hi"))
        kwargs = mock_aio.models.generate_content.call_args.kwargs
        config = cast(dict[str, Any], kwargs.get("config", {}))
        assert config.get("tools") == [{"google_search": {}}]


class TestRegistry:
    @pytest.mark.asyncio
    async def test_get_provider_returns_google_adapter(self) -> None:
        from geo_analyzer.providers import get_provider

        p = get_provider("google", api_key="g-test")
        assert p.name == "google"
