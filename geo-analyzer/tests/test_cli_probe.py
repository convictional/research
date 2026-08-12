from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from geo_analyzer.cli import app
from geo_analyzer.providers.base import ProviderResponse

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_dotenv() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Stop the CLI from loading the developer's real .env during tests.

    `probe` calls `load_dotenv()` which reads the actual filesystem .env into
    os.environ — that bleeds the developer's API keys into tests that try to
    assert "no key set" behavior. Patch it to a no-op for every test in this file.
    """
    with patch("geo_analyzer.cli.load_dotenv"):
        yield


def _fake_provider_response(text: str = "stub answer") -> ProviderResponse:
    return ProviderResponse(
        text=text,
        tokens_in=42,
        tokens_out=17,
        cost_usd_estimate=0.0001,
        latency_ms=120,
        raw={},
    )


def test_probe_unknown_model_fails(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = runner.invoke(
        app,
        [
            "probe",
            "what is convictional?",
            "--model",
            "nope:not-a-model:grounded",
            "--catalog-dir",
            str(project_root / "catalog"),
        ],
        env={"OPENAI_API_KEY": "x"},
    )
    assert result.exit_code != 0
    assert "model" in (result.stdout or "") or "model" in (result.stderr or "")


def test_probe_missing_api_key_fails(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = runner.invoke(
        app,
        [
            "probe",
            "what is convictional?",
            "--model",
            "openai:gpt-5.1:ungrounded",
            "--catalog-dir",
            str(project_root / "catalog"),
        ],
        env={},  # nothing set
    )
    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in (result.stdout or "") or "OPENAI_API_KEY" in (result.stderr or "")


def test_probe_calls_provider_and_prints_text() -> None:
    project_root = Path(__file__).resolve().parents[1]

    with patch("geo_analyzer.cli.get_provider") as mock_get:
        mock_provider: Any = type("P", (), {})()
        mock_provider.name = "openai"
        mock_provider.call = AsyncMock(return_value=_fake_provider_response("hello convictional"))
        mock_get.return_value = mock_provider

        result = runner.invoke(
            app,
            [
                "probe",
                "what is convictional?",
                "--model",
                "openai:gpt-5.1:ungrounded",
                "--catalog-dir",
                str(project_root / "catalog"),
            ],
            env={"OPENAI_API_KEY": "sk-test"},
        )

    assert result.exit_code == 0, result.stdout
    assert "hello convictional" in result.stdout
    # Cost and tokens should be visible somewhere in the output.
    assert "42" in result.stdout
    assert "17" in result.stdout


def test_probe_sensitivity_runs_multiple_samples() -> None:
    project_root = Path(__file__).resolve().parents[1]

    with patch("geo_analyzer.cli.get_provider") as mock_get:
        mock_provider: Any = type("P", (), {})()
        mock_provider.name = "openai"
        # Each call returns a different text so we can count invocations.
        responses = [
            _fake_provider_response("first"),
            _fake_provider_response("second"),
            _fake_provider_response("third"),
        ]
        mock_provider.call = AsyncMock(side_effect=responses)
        mock_get.return_value = mock_provider

        result = runner.invoke(
            app,
            [
                "probe",
                "hi",
                "--model",
                "openai:gpt-5.1:ungrounded",
                "--sensitivity-samples",
                "3",
                "--temperature",
                "0.7",
                "--catalog-dir",
                str(project_root / "catalog"),
            ],
            env={"OPENAI_API_KEY": "x"},
        )

    assert result.exit_code == 0, result.stdout
    assert mock_provider.call.await_count == 3
    # All three response texts should appear in the output.
    for label in ("first", "second", "third"):
        assert label in result.stdout


def test_probe_sensitivity_passes_temperature_override() -> None:
    project_root = Path(__file__).resolve().parents[1]

    captured: list[float | None] = []

    async def _capture_call(request: Any) -> ProviderResponse:
        captured.append(request.temperature_override)
        return _fake_provider_response("ok")

    with patch("geo_analyzer.cli.get_provider") as mock_get:
        mock_provider: Any = type("P", (), {})()
        mock_provider.name = "openai"
        mock_provider.call = _capture_call
        mock_get.return_value = mock_provider

        result = runner.invoke(
            app,
            [
                "probe",
                "hi",
                "--model",
                "openai:gpt-5.1:ungrounded",
                "--sensitivity-samples",
                "2",
                "--temperature",
                "0.5",
                "--catalog-dir",
                str(project_root / "catalog"),
            ],
            env={"OPENAI_API_KEY": "x"},
        )

    assert result.exit_code == 0
    assert captured == [0.5, 0.5]


def test_probe_sensitivity_requires_temperature() -> None:
    # If --sensitivity-samples > 1 but no --temperature is given, fail loudly:
    # the whole point of sensitivity mode is varying generation. Defaulting to
    # the model's sampling.temperature would silently make N samples identical
    # for ungrounded (temp=0).
    project_root = Path(__file__).resolve().parents[1]
    result = runner.invoke(
        app,
        [
            "probe",
            "hi",
            "--model",
            "openai:gpt-5.1:ungrounded",
            "--sensitivity-samples",
            "3",
            "--catalog-dir",
            str(project_root / "catalog"),
        ],
        env={"OPENAI_API_KEY": "x"},
    )
    assert result.exit_code != 0
    assert "temperature" in (result.stdout or "") or "temperature" in (result.stderr or "")
