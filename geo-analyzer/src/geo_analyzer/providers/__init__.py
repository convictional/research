"""Provider adapter registry.

Each adapter module's import has the side effect of calling `_register`
to populate `_REGISTRY`. Importing `geo_analyzer.providers` brings every
registered adapter into the registry automatically.
"""

from __future__ import annotations

from collections.abc import Callable

from geo_analyzer.providers.base import (
    ProbeRequest,
    Provider,
    ProviderError,
    ProviderResponse,
)

_REGISTRY: dict[str, Callable[[str], Provider]] = {}


def _register(name: str, factory: Callable[[str], Provider]) -> None:
    _REGISTRY[name] = factory


def get_provider(name: str, *, api_key: str) -> Provider:
    factory = _REGISTRY.get(name)
    if factory is None:
        known = sorted(_REGISTRY) or ["<none registered>"]
        raise ProviderError(f"unknown provider {name!r}; known: {known}")
    return factory(api_key)


# --- adapter registration -------------------------------------------------
# Each provider module is imported here (with the registration side effect).
# Tasks 5 and 6 will append to this list.

from geo_analyzer.providers.anthropic import AnthropicProvider as _AnthropicProvider  # noqa: E402
from geo_analyzer.providers.google import GoogleProvider as _GoogleProvider  # noqa: E402
from geo_analyzer.providers.openai import OpenAIProvider as _OpenAIProvider  # noqa: E402

_register("anthropic", lambda key: _AnthropicProvider(api_key=key))
_register("google", lambda key: _GoogleProvider(api_key=key))
_register("openai", lambda key: _OpenAIProvider(api_key=key))


__all__ = [
    "ProbeRequest",
    "Provider",
    "ProviderError",
    "ProviderResponse",
    "get_provider",
]
