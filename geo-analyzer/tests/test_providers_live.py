"""Live integration tests — opt-in. Run with: pytest -m live

These hit real provider APIs and require API keys in the environment (or .env).
Skipped by default. Each test asserts the response has plausible shape (non-empty
text, non-zero token counts, non-negative cost) — not exact content.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from geo_analyzer.catalog import load_catalog
from geo_analyzer.providers import ProbeRequest, get_provider
from geo_analyzer.types import ModelSpec

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def catalog_models() -> dict[str, ModelSpec]:
    """Load the seed catalog once per module run."""
    load_dotenv()
    project_root = Path(__file__).resolve().parents[1]
    cat = load_catalog(project_root / "catalog")
    return {m.id: m for m in cat.models}


_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def _require_key(provider: str) -> str:
    env_var = _API_KEY_ENV[provider]
    key = os.environ.get(env_var)
    if not key:
        pytest.skip(f"{env_var} not set; skipping live test")
    return key


_LIVE_MODEL_IDS = [
    "openai:gpt-5.1:ungrounded",
    "openai:gpt-5.1:grounded",
    "anthropic:claude-sonnet-4-6:ungrounded",
    "anthropic:claude-sonnet-4-6:grounded",
    "google:gemini-2.5-flash:ungrounded",
    "google:gemini-2.5-flash:grounded",
]


@pytest.mark.parametrize("model_id", _LIVE_MODEL_IDS)
@pytest.mark.asyncio
async def test_live_probe_returns_plausible_response(catalog_models: dict[str, ModelSpec], model_id: str) -> None:
    model = catalog_models[model_id]
    api_key = _require_key(model.provider)
    provider = get_provider(model.provider, api_key=api_key)

    request = ProbeRequest(model=model, prompt="Say the word 'pong' and nothing else.")
    response = await provider.call(request)

    # Soft asserts — we don't pin exact text because the model is non-deterministic
    # in grounded mode and may add extra commentary.
    assert response.text, f"empty text from {model_id!r}"
    assert response.tokens_in > 0, f"tokens_in not reported by {model_id!r}"
    assert response.tokens_out > 0, f"tokens_out not reported by {model_id!r}"
    assert response.cost_usd_estimate >= 0
    assert response.latency_ms >= 0
