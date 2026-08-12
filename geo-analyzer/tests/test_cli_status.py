from __future__ import annotations

import shutil
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


def _seed_run_with_score(tmp_path: Path, *, presence_value: bool) -> None:
    rp = run_paths_for(tmp_path, "2026-04-29-manual")
    rp.ensure()
    started = datetime(2026, 4, 29, 9, tzinfo=UTC)
    finished = datetime(2026, 4, 29, 9, 5, tzinfo=UTC)
    run = Run(
        id="2026-04-29-manual", trigger="manual", started_at=started, finished_at=finished, status=RunStatus.COMPLETED
    )
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
        rp.scores_jsonl,
        {
            "run_id": "2026-04-29-manual",
            "prompt_id": "prompt.broad.l1.companies-in-age-of-ai",
            "model_id": "openai:gpt-5.1:ungrounded",
            "subject_id": "convictional_brand",
            "metric": "mention_presence",
            "value": presence_value,
            "scoring_method": "deterministic",
            "sample_aggregation": "single",
        },
    )


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_catalog_with_goals(tmp_path: Path, project_root: Path) -> Path:
    """Copy the real catalog into tmp_path and write a test goals.yaml inside.

    Goals now live under catalog/, so tests need a writable catalog dir to
    inject test-specific goals without mutating the committed catalog.
    """
    cat_dir = tmp_path / "catalog"
    shutil.copytree(project_root / "catalog", cat_dir)
    (cat_dir / "goals.yaml").write_text(
        "- id: g1\n"
        "  subject: convictional_brand\n"
        "  metric: mention_presence\n"
        "  tier: L1\n"
        "  target: 0.5\n"
        "  direction: above\n"
        "  created_at: 2026-01-01\n"
        "  target_date: 2026-12-31\n"
    )
    return cat_dir


def _build_catalog_without_goals(tmp_path: Path, project_root: Path) -> Path:
    """Copy of the real catalog with goals.yaml stripped — used by the
    'no goals defined' test path."""
    cat_dir = tmp_path / "catalog_no_goals"
    shutil.copytree(project_root / "catalog", cat_dir)
    goals_path = cat_dir / "goals.yaml"
    goals_path.unlink(missing_ok=True)
    return cat_dir


def test_status_green_goal_exits_zero(project_root: Path, tmp_path: Path) -> None:
    _seed_run_with_score(tmp_path, presence_value=True)
    cat_dir = _build_catalog_with_goals(tmp_path, project_root)
    result = runner.invoke(
        app,
        [
            "status",
            "--data-dir",
            str(tmp_path),
            "--catalog-dir",
            str(cat_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "g1" in result.stdout
    assert "GREEN" in result.stdout


def test_status_red_goal_exits_three(project_root: Path, tmp_path: Path) -> None:
    _seed_run_with_score(tmp_path, presence_value=False)
    cat_dir = _build_catalog_with_goals(tmp_path, project_root)
    result = runner.invoke(
        app,
        [
            "status",
            "--data-dir",
            str(tmp_path),
            "--catalog-dir",
            str(cat_dir),
        ],
    )
    assert result.exit_code == 3
    assert "g1" in result.stdout
    assert "RED" in result.stdout


def test_status_no_goals_exits_zero(project_root: Path, tmp_path: Path) -> None:
    _seed_run_with_score(tmp_path, presence_value=True)
    cat_dir = _build_catalog_without_goals(tmp_path, project_root)
    result = runner.invoke(
        app,
        [
            "status",
            "--data-dir",
            str(tmp_path),
            "--catalog-dir",
            str(cat_dir),
        ],
    )
    assert result.exit_code == 0
    assert "no goals" in (result.stdout or "").lower() or result.stdout.strip() == ""


def test_status_no_runs_fails(project_root: Path, tmp_path: Path) -> None:
    cat_dir = _build_catalog_with_goals(tmp_path, project_root)
    result = runner.invoke(
        app,
        [
            "status",
            "--data-dir",
            str(tmp_path),
            "--catalog-dir",
            str(cat_dir),
        ],
    )
    assert result.exit_code != 0
    assert "no runs" in (result.stdout or "").lower() or "no runs" in (result.stderr or "").lower()
