"""Multi-run trends: walk data/runs/ and emit one row per (run, subject, metric)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from geo_analyzer.reports.loader import list_run_ids, read_scores_jsonl
from geo_analyzer.reports.topline import compute_topline
from geo_analyzer.storage import run_paths_for


@dataclass(frozen=True)
class MultiRunRow:
    run_id: str
    run_date: date
    subject_id: str
    metric: str
    n: int
    prompt_level_rate: float | None
    interaction_level_rate: float | None
    mean_value: float | None


def _parse_run_date(run_id: str) -> date:
    """Run id is 'YYYY-MM-DD-<trigger>'; first 10 chars are the date."""
    return date.fromisoformat(run_id[:10])


def compute_multi_run_trends(
    data_dir: Path,
    *,
    since: date | None,
) -> list[MultiRunRow]:
    """Walk data_dir/runs/, compute per-run topline, return one row per
    (run, subject, metric). `since=None` means include all runs."""
    rows: list[MultiRunRow] = []
    for run_id in list_run_ids(data_dir):
        run_date = _parse_run_date(run_id)
        if since is not None and run_date < since:
            continue
        rp = run_paths_for(data_dir, run_id)
        scores = read_scores_jsonl(rp.scores_jsonl)
        for tl in compute_topline(scores):
            rows.append(
                MultiRunRow(
                    run_id=run_id,
                    run_date=run_date,
                    subject_id=tl.subject_id,
                    metric=tl.metric,
                    n=tl.n,
                    prompt_level_rate=tl.prompt_level_rate,
                    interaction_level_rate=tl.interaction_level_rate,
                    mean_value=tl.mean_value,
                )
            )
    return rows
