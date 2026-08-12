from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest

from geo_analyzer.storage.paths import (
    build_run_id,
    run_paths_for,
)


class TestBuildRunId:
    def test_basic(self) -> None:
        assert build_run_id(date(2026, 4, 29), trigger="manual") == "2026-04-29-manual"

    def test_launchd_weekly(self) -> None:
        assert build_run_id(date(2026, 4, 29), trigger="launchd-weekly") == "2026-04-29-launchd-weekly"


class TestRunPaths:
    def test_paths_under_run_dir(self, tmp_path: Path) -> None:
        rp = run_paths_for(tmp_path, "2026-04-29-manual")
        assert rp.run_dir == tmp_path / "runs" / "2026-04-29-manual"
        assert rp.manifest == rp.run_dir / "manifest.json"
        assert rp.tasks_jsonl == rp.run_dir / "tasks.jsonl"
        assert rp.scores_jsonl == rp.run_dir / "scores.jsonl"
        assert rp.tasks_csv == rp.run_dir / "tasks.csv"
        assert rp.scores_csv == rp.run_dir / "scores.csv"

    def test_ensure_creates_dir(self, tmp_path: Path) -> None:
        rp = run_paths_for(tmp_path, "2026-04-29-manual")
        rp.ensure()
        assert rp.run_dir.is_dir()

    def test_ensure_is_idempotent(self, tmp_path: Path) -> None:
        rp = run_paths_for(tmp_path, "2026-04-29-manual")
        rp.ensure()
        rp.ensure()  # should not raise
        assert rp.run_dir.is_dir()

    def test_immutable_dataclass(self) -> None:
        rp = run_paths_for(Path("/tmp"), "x-y")
        with pytest.raises(FrozenInstanceError):
            rp.run_dir = Path("/somewhere-else")  # type: ignore[misc]
