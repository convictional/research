from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from geo_analyzer.cli import app
from geo_analyzer.runtime import Run, RunStatus
from geo_analyzer.storage import (
    Manifest,
    append_jsonl,
    run_paths_for,
    write_manifest,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_dotenv() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    with patch("geo_analyzer.cli.load_dotenv"):
        yield


def _seed_run(tmp_path: Path, run_id: str = "2026-04-29-manual") -> Path:
    rp = run_paths_for(tmp_path, run_id)
    rp.ensure()
    started = datetime(2026, 4, 29, 9, tzinfo=UTC)
    finished = datetime(2026, 4, 29, 9, 5, tzinfo=UTC)
    run = Run(id=run_id, trigger="manual", started_at=started, finished_at=finished, status=RunStatus.COMPLETED)
    write_manifest(
        rp.manifest,
        Manifest(
            run=run,
            subject_ids=[],
            prompt_ids=[],
            model_ids=[],
            catalog_hash="x",
        ),
    )
    append_jsonl(
        rp.tasks_jsonl,
        {
            "run_id": run_id,
            "prompt_id": "prompt.broad.l1.companies-in-age-of-ai",
            "model_id": "openai:gpt-5.1:ungrounded",
            "sample_n": 0,
            "status": "success",
            "text": "hi",
            "tokens_in": 5,
            "tokens_out": 5,
            "cost_usd_estimate": 0.0001,
            "latency_ms": 100,
            "error": None,
        },
    )
    append_jsonl(
        rp.scores_jsonl,
        {
            "run_id": run_id,
            "prompt_id": "prompt.broad.l1.companies-in-age-of-ai",
            "model_id": "openai:gpt-5.1:ungrounded",
            "subject_id": "convictional_brand",
            "metric": "mention_presence",
            "value": True,
            "scoring_method": "deterministic",
            "sample_aggregation": "single",
        },
    )
    return rp.run_dir


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_report_writes_summary_for_run(project_root: Path, tmp_path: Path) -> None:
    run_dir = _seed_run(tmp_path)
    result = runner.invoke(
        app,
        [
            "report",
            "2026-04-29-manual",
            "--data-dir",
            str(tmp_path),
            "--catalog-dir",
            str(project_root / "catalog"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    summary = (run_dir / "summary.md").read_text()
    assert "Run 2026-04-29-manual" in summary
    assert "TL;DR" in summary

    # The HTML peer should also exist with the same content rendered.
    html = (run_dir / "summary.html").read_text()
    assert html.startswith("<!DOCTYPE html>")
    assert "Run 2026-04-29-manual" in html


def test_report_uses_latest_when_run_id_omitted(project_root: Path, tmp_path: Path) -> None:
    _seed_run(tmp_path, "2026-04-15-manual")
    _seed_run(tmp_path, "2026-04-29-manual")
    result = runner.invoke(
        app,
        [
            "report",
            "--data-dir",
            str(tmp_path),
            "--catalog-dir",
            str(project_root / "catalog"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    latest_summary = (tmp_path / "runs" / "2026-04-29-manual" / "summary.md").read_text()
    assert "Run 2026-04-29-manual" in latest_summary


def test_report_no_runs_fails(project_root: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "report",
            "--data-dir",
            str(tmp_path),
            "--catalog-dir",
            str(project_root / "catalog"),
        ],
    )
    assert result.exit_code != 0
    assert "no runs" in (result.stdout or "").lower() or "no runs" in (result.stderr or "").lower()


def test_report_unknown_run_id_fails(project_root: Path, tmp_path: Path) -> None:
    _seed_run(tmp_path)
    result = runner.invoke(
        app,
        [
            "report",
            "2026-99-99-not-real",
            "--data-dir",
            str(tmp_path),
            "--catalog-dir",
            str(project_root / "catalog"),
        ],
    )
    assert result.exit_code != 0
