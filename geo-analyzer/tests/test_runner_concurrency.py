from __future__ import annotations

import asyncio

import pytest

from geo_analyzer.runner.concurrency import ConcurrencyManager


class TestConcurrencyManager:
    def test_get_returns_semaphore_for_provider(self) -> None:
        cm = ConcurrencyManager(caps={"openai": 8, "anthropic": 5})
        assert isinstance(cm.semaphore_for("openai"), asyncio.Semaphore)

    def test_unknown_provider_raises(self) -> None:
        cm = ConcurrencyManager(caps={"openai": 8})
        with pytest.raises(KeyError):
            cm.semaphore_for("nope")

    @pytest.mark.asyncio
    async def test_caps_concurrent_in_flight(self) -> None:
        cm = ConcurrencyManager(caps={"openai": 2})
        in_flight = 0
        max_in_flight = 0

        async def task() -> None:
            nonlocal in_flight, max_in_flight
            async with cm.semaphore_for("openai"):
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        await asyncio.gather(*(task() for _ in range(10)))
        assert max_in_flight <= 2
