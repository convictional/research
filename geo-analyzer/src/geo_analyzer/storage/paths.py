"""Run directory layout under data/runs/<run-id>/.

The conventions live here so the runner, CLI, and any future analysis script
agree on filenames.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


def build_run_id(d: date, *, trigger: str) -> str:
    """Format: YYYY-MM-DD-<trigger>."""
    return f"{d.isoformat()}-{trigger}"


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def tasks_jsonl(self) -> Path:
        return self.run_dir / "tasks.jsonl"

    @property
    def scores_jsonl(self) -> Path:
        return self.run_dir / "scores.jsonl"

    @property
    def tasks_csv(self) -> Path:
        return self.run_dir / "tasks.csv"

    @property
    def scores_csv(self) -> Path:
        return self.run_dir / "scores.csv"

    def ensure(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)


def run_paths_for(data_dir: Path, run_id: str) -> RunPaths:
    """`data_dir` is the geo-analyzer/data root (parent of runs/)."""
    return RunPaths(run_dir=data_dir / "runs" / run_id)
