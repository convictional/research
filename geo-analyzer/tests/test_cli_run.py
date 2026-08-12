from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from geo_analyzer.cli import app
from geo_analyzer.runner.orchestrator import RunSummary
from geo_analyzer.runtime import Run, RunStatus

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_dotenv() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    with patch("geo_analyzer.cli.load_dotenv"):
        yield


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stub_summary() -> RunSummary:
    run_obj = Run(
        id="2026-04-29-manual",
        trigger="manual",
        started_at=datetime(2026, 4, 29, tzinfo=UTC),
        finished_at=datetime(2026, 4, 29, 0, 1, tzinfo=UTC),
        status=RunStatus.COMPLETED,
    )
    return RunSummary(run=run_obj, n_success=12, n_failed=0)


def test_run_dry_run_does_not_call_orchestrator(project_root: Path, tmp_path: Path) -> None:
    with patch("geo_analyzer.cli.orchestrator_run") as mock_run:
        mock_run.side_effect = AssertionError("orchestrator should not be called for --dry-run")
        result = runner.invoke(
            app,
            [
                "run",
                "--dry-run",
                "--catalog-dir",
                str(project_root / "catalog"),
                "--data-dir",
                str(tmp_path),
                "--tier",
                "L1",
                "--model",
                "openai:gpt-5.1:ungrounded",
            ],
            env={"OPENAI_API_KEY": "sk-test"},
        )
    assert result.exit_code == 0, result.stdout
    assert "tasks" in result.stdout.lower() or "matrix" in result.stdout.lower()


def test_run_invokes_orchestrator(project_root: Path, tmp_path: Path) -> None:
    summary = _stub_summary()
    with patch("geo_analyzer.cli.orchestrator_run", new=AsyncMock(return_value=summary)) as mock_run:
        result = runner.invoke(
            app,
            [
                "run",
                "--catalog-dir",
                str(project_root / "catalog"),
                "--data-dir",
                str(tmp_path),
                "--tier",
                "L1",
                "--model",
                "openai:gpt-5.1:ungrounded",
                "--yes",
            ],
            env={"OPENAI_API_KEY": "sk-test"},
        )
    assert result.exit_code == 0, result.stdout
    mock_run.assert_awaited_once()


def test_run_missing_api_key_fails(project_root: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--catalog-dir",
            str(project_root / "catalog"),
            "--data-dir",
            str(tmp_path),
            "--tier",
            "L1",
            "--model",
            "openai:gpt-5.1:ungrounded",
            "--yes",
        ],
        env={},
    )
    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in (result.stdout or "")


def test_run_cost_gate_blocks_without_yes(project_root: Path, tmp_path: Path) -> None:
    with (
        patch("geo_analyzer.cli._estimate_run_cost", return_value=999.0),
        patch("geo_analyzer.cli.orchestrator_run") as mock_run,
    ):
        result = runner.invoke(
            app,
            [
                "run",
                "--catalog-dir",
                str(project_root / "catalog"),
                "--data-dir",
                str(tmp_path),
                "--tier",
                "L1",
                "--model",
                "openai:gpt-5.1:ungrounded",
            ],
            env={"OPENAI_API_KEY": "sk-test"},
            input="n\n",
        )
    assert result.exit_code != 0
    mock_run.assert_not_called()
