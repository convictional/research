"""Read scores back from a run dir; discover and rank run ids."""

from __future__ import annotations

from pathlib import Path

from geo_analyzer.runtime import Score
from geo_analyzer.storage import read_jsonl_dicts


def read_scores_jsonl(path: Path) -> list[Score]:
    """Read scores.jsonl back into typed Score models. Missing file → []."""
    return [Score.model_validate(d) for d in read_jsonl_dicts(path)]


def list_run_ids(data_dir: Path) -> list[str]:
    """Return run ids under `data_dir/runs/`, sorted ascending (latest last).

    Sorts lexicographically — works because run ids start with ISO date.
    Non-directories under `runs/` are skipped.
    """
    runs_dir = data_dir / "runs"
    if not runs_dir.is_dir():
        return []
    return sorted(p.name for p in runs_dir.iterdir() if p.is_dir())


def latest_run_id(data_dir: Path) -> str | None:
    """Return the most recent run id (lexicographically last), or None."""
    ids = list_run_ids(data_dir)
    return ids[-1] if ids else None
