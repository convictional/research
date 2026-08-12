"""JSONL helpers. One JSON object per line.

`append_jsonl` is the right writer for genuinely append-only logs (tasks.jsonl
during a run, where each task is recorded once as it completes). `write_jsonl`
is the right writer for derived data that's regenerated each run (scores.jsonl,
which is recomputed from tasks at end of run) — appending there causes
duplicate accumulation across re-runs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from geo_analyzer.runtime import Task


def _serialize(record: dict[str, Any]) -> str:
    return json.dumps(record, separators=(",", ":"), default=str)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON object as a line. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_serialize(record) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Truncate and rewrite. Use for derived data regenerated each run.

    Counterpart to `append_jsonl` — same on-wire format, but overwrites instead
    of appending so re-runs don't accumulate duplicates of derived records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(_serialize(record) + "\n")


def read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    """Read every line as a JSON dict. Missing file or empty file → []."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj: Any = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: invalid JSONL line: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line is not a JSON object: {stripped!r}")
            out.append(cast(dict[str, Any], obj))
    return out


def read_tasks_jsonl(path: Path) -> list[Task]:
    """Read tasks.jsonl and parse each line into a Task, deduped by Task.key().

    A given (run_id, prompt_id, model_id, sample_n) may appear multiple times
    if a task was retried across runs — typically a FAILED entry from run N
    followed by a SUCCESS entry from run N+1. We keep the LATEST entry per
    key (last write wins), matching the user's mental model: "what is the
    current state of each task?"
    """
    tasks_by_key: dict[tuple[str, str, str, int], Task] = {}
    for d in read_jsonl_dicts(path):
        t = Task.model_validate(d)
        tasks_by_key[t.key()] = t
    return list(tasks_by_key.values())
