"""Per-provider concurrency caps. One asyncio.Semaphore per provider name."""

from __future__ import annotations

import asyncio


class ConcurrencyManager:
    def __init__(self, *, caps: dict[str, int]) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {name: asyncio.Semaphore(n) for name, n in caps.items()}

    def semaphore_for(self, provider: str) -> asyncio.Semaphore:
        try:
            return self._semaphores[provider]
        except KeyError as e:
            raise KeyError(f"no concurrency cap configured for provider {provider!r}") from e
