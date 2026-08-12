"""End-of-run CSV emit. tasks.csv excludes the response `text` column to keep
the file small (raw bodies stay in tasks.jsonl). scores.csv has every field.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from geo_analyzer.runtime import Score, Task

_TASK_COLUMNS = [
    "run_id",
    "prompt_id",
    "model_id",
    "sample_n",
    "status",
    "tokens_in",
    "tokens_out",
    "cost_usd_estimate",
    "latency_ms",
    "error",
]

_SCORE_COLUMNS = [
    "run_id",
    "prompt_id",
    "model_id",
    "subject_id",
    "metric",
    "value",
    "scoring_method",
    "sample_aggregation",
]


def _row_from(model: Any, columns: list[str]) -> dict[str, Any]:
    """Project a Pydantic model down to the given column list. None values
    serialize as '' (csv.DictWriter default — empty cell)."""
    raw = model.model_dump(mode="json")
    return {c: raw.get(c, "") if raw.get(c) is not None else "" for c in columns}


def write_tasks_csv(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TASK_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for t in tasks:
            w.writerow(_row_from(t, _TASK_COLUMNS))


def write_scores_csv(path: Path, scores: list[Score]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SCORE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for s in scores:
            w.writerow(_row_from(s, _SCORE_COLUMNS))
