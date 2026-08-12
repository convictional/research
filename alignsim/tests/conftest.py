"""Shared pytest fixtures for AlignSim tests."""

import random

import pytest

from alignsim.src.engine.game import GameEngine
from alignsim.src.models.scenario import CalibrationParams

from .factories import make_minimal_scenario


@pytest.fixture
def anyio_backend() -> str:
    """Pin anyio tests to asyncio only.

    pytest-anyio parametrizes @pytest.mark.anyio tests over asyncio AND trio by
    default. We only use asyncio (FastAPI, uvicorn, the orchestrator) and trio
    isn't a project dependency, so the trio variants always fail at import.
    """
    return "asyncio"


@pytest.fixture
def default_calibration() -> CalibrationParams:
    return CalibrationParams()


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def minimal_scenario():
    return make_minimal_scenario()


@pytest.fixture
def engine(minimal_scenario) -> GameEngine:
    return GameEngine(minimal_scenario)
