from __future__ import annotations

import pytest

from geo_analyzer.providers import ProviderError, get_provider
from geo_analyzer.providers.base import ProbeRequest, ProviderResponse
from geo_analyzer.types import ModelSpec, SamplingConfig


class TestProbeRequestProviderResponse:
    def test_probe_request_minimal(self) -> None:
        model = ModelSpec(
            id="openai:gpt-5.1:ungrounded",
            provider="openai",
            model_name="gpt-5.1",
            mode="ungrounded",
            active=True,
            config={},
            sampling=SamplingConfig(n=1, temperature=0, seed=42),
        )
        req = ProbeRequest(model=model, prompt="hi", temperature_override=None, seed_override=None)
        assert req.prompt == "hi"

    def test_provider_response_minimal(self) -> None:
        resp = ProviderResponse(
            text="hello",
            tokens_in=10,
            tokens_out=5,
            cost_usd_estimate=0.001,
            latency_ms=120,
            raw={"any": "shape"},
        )
        assert resp.text == "hello"
        assert resp.tokens_in == 10


class TestRegistry:
    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ProviderError, match="unknown provider"):
            get_provider("nope", api_key="x")

    def test_openai_registered_by_task_4(self) -> None:
        # Task 4 registered the OpenAI adapter as an import side effect.
        # Just confirm it resolves.
        p = get_provider("openai", api_key="sk-test")
        assert p.name == "openai"
