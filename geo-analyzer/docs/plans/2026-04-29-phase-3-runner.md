# GEO Analyzer — Phase 3: Runner + Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `uv run geo-analyzer run` execute the full catalog × matrix (~30 prompts × 10 models × 1 or 3 samples ≈ 600 calls) concurrently, persist every task and score to JSONL/CSV under `data/runs/<run-id>/`, and resume cleanly after a crash — so a teammate can run the harness once a week and have raw data on disk for analysis.

**Architecture:** Two new packages — `geo_analyzer.storage` (run dir creation, manifest, append-only JSONL, end-of-run CSV emit) and `geo_analyzer.runner` (matrix expansion, exponential-backoff retry, per-provider asyncio semaphores, scoring pipeline that calls Phase 1 extractors + aggregations, and the orchestrator that ties it all together). One new top-level module `runtime.py` defines `Run`/`Task`/`Score` dataclasses kept distinct from Phase 1's catalog types. The CLI gets one new subcommand: `run`, with `--dry-run`, `--tier`, `--subject`, `--model`, `--resume`/`--no-resume`, `--yes` flags. No new third-party dependencies — uses stdlib `csv`/`json`/`asyncio` and the existing Pydantic/typer/rich.

**Tech Stack:** Python 3.13, asyncio, stdlib `csv`/`json`/`pathlib`, plus everything from Phase 1+2.

**Out of scope for Phase 3 (Phase 4):**

- `summary.md` generation (the human dashboard) — Phase 4
- Goal traffic-lights / `geo-analyzer status` — Phase 4
- Multi-run trends / `report --since` — Phase 4
- launchd plist + GitHub Actions CI workflow — Phase 5

---

## File Structure

Files created in this phase:

```
experiments/geo-analyzer/
├── src/geo_analyzer/
│   ├── runtime.py                          # Run, Task, Score, RunStatus types
│   ├── storage/
│   │   ├── __init__.py                     # public API
│   │   ├── paths.py                        # run id + run dir resolver
│   │   ├── manifest.py                     # read/write manifest.json
│   │   ├── jsonl.py                        # append-only JSONL helpers
│   │   └── csv_export.py                   # tasks.csv + scores.csv emit
│   ├── runner/
│   │   ├── __init__.py                     # public API
│   │   ├── matrix.py                       # catalog → list[Task]
│   │   ├── retry.py                        # exponential backoff helper
│   │   ├── concurrency.py                  # per-provider asyncio.Semaphore manager
│   │   ├── scoring.py                      # apply Phase 1 extractors + aggregations
│   │   └── orchestrator.py                 # the run loop
│   └── cli.py                              # add `run` subcommand
└── tests/
    ├── test_runtime.py                     # type validation
    ├── test_storage_paths.py
    ├── test_storage_jsonl.py
    ├── test_storage_manifest.py
    ├── test_storage_csv.py
    ├── test_runner_matrix.py
    ├── test_runner_retry.py
    ├── test_runner_concurrency.py
    ├── test_runner_scoring.py
    ├── test_runner_orchestrator.py
    └── test_cli_run.py
```

Files modified:
- `experiments/geo-analyzer/src/geo_analyzer/cli.py` (add `run` command)
- `experiments/geo-analyzer/README.md` (Phase 3 status section)

---

## Working Directory

`uv run` commands assume cwd is `/Users/billtarbell/Code/decide/experiments/geo-analyzer/`. Branch is `geo-analyzer-implementation-v1` — stacking on Phase 1+2.

---

### Task 1: Runtime types (Run, Task, Score)

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/runtime.py`
- Create: `experiments/geo-analyzer/tests/test_runtime.py`

These are run-time entities — distinct from Phase 1's catalog types. Keep them in a separate module so `types.py` stays catalog-focused.

- [ ] **Step 1.1: Write failing tests**

Create `tests/test_runtime.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from geo_analyzer.runtime import (
    Run,
    RunStatus,
    Score,
    Task,
    TaskStatus,
)


class TestRun:
    def test_minimal(self) -> None:
        r = Run(
            id="2026-04-29-manual",
            trigger="manual",
            started_at=datetime(2026, 4, 29, 9, 0, tzinfo=timezone.utc),
        )
        assert r.id == "2026-04-29-manual"
        assert r.trigger == "manual"
        assert r.status == RunStatus.IN_PROGRESS  # default

    def test_id_must_match_format(self) -> None:
        with pytest.raises(ValidationError):
            Run(
                id="not-a-date",
                trigger="manual",
                started_at=datetime.now(timezone.utc),
            )

    def test_unknown_trigger_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Run(
                id="2026-04-29-manual",
                trigger="bogus",  # type: ignore[arg-type]
                started_at=datetime.now(timezone.utc),
            )


class TestTask:
    def test_minimal(self) -> None:
        t = Task(
            run_id="2026-04-29-manual",
            prompt_id="prompt.broad.l1.companies-run-poorly",
            model_id="openai:gpt-5.1:ungrounded",
            sample_n=0,
            status=TaskStatus.SUCCESS,
            text="Convictional helps...",
            tokens_in=100,
            tokens_out=50,
            cost_usd_estimate=0.003,
            latency_ms=500,
        )
        assert t.sample_n == 0
        assert t.status == TaskStatus.SUCCESS

    def test_failure_task_can_have_error(self) -> None:
        t = Task(
            run_id="2026-04-29-manual",
            prompt_id="prompt.broad.l1.companies-run-poorly",
            model_id="openai:gpt-5.1:ungrounded",
            sample_n=0,
            status=TaskStatus.FAILED,
            error="rate limit",
            text="",
            tokens_in=0,
            tokens_out=0,
            cost_usd_estimate=0.0,
            latency_ms=0,
        )
        assert t.error == "rate limit"

    def test_negative_sample_n_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Task(
                run_id="2026-04-29-manual",
                prompt_id="x",
                model_id="x:x:grounded",
                sample_n=-1,
                status=TaskStatus.SUCCESS,
                text="",
                tokens_in=0,
                tokens_out=0,
                cost_usd_estimate=0.0,
                latency_ms=0,
            )

    def test_task_key_is_stable(self) -> None:
        t1 = Task(
            run_id="r", prompt_id="p", model_id="m:n:grounded", sample_n=1,
            status=TaskStatus.SUCCESS, text="", tokens_in=0, tokens_out=0,
            cost_usd_estimate=0.0, latency_ms=0,
        )
        t2 = Task(
            run_id="r", prompt_id="p", model_id="m:n:grounded", sample_n=1,
            status=TaskStatus.SUCCESS, text="different", tokens_in=1, tokens_out=1,
            cost_usd_estimate=0.0, latency_ms=0,
        )
        assert t1.key() == t2.key()
        assert t1.key() == ("r", "p", "m:n:grounded", 1)


class TestScore:
    def test_minimal(self) -> None:
        s = Score(
            run_id="2026-04-29-manual",
            prompt_id="p",
            model_id="m:n:grounded",
            subject_id="convictional_brand",
            metric="mention_presence",
            value=True,
            scoring_method="deterministic",
            sample_aggregation="majority_vote",
        )
        assert s.value is True
        assert s.scoring_method == "deterministic"

    def test_value_can_be_int_float_bool_or_none(self) -> None:
        # ordinal_rank can be int|None; SoV can be float|None; presence bool.
        for v in (None, 0, 1, 0.5, True, False):
            Score(
                run_id="r", prompt_id="p", model_id="m:n:grounded",
                subject_id="s", metric="x", value=v,
                scoring_method="deterministic", sample_aggregation="single",
            )
```

- [ ] **Step 1.2: Run, verify they fail**

```bash
uv run pytest tests/test_runtime.py -v
```

Expected: ImportError on `geo_analyzer.runtime`.

- [ ] **Step 1.3: Implement `runtime.py`**

```python
"""Run-time entities: Run, Task, Score.

Distinct from Phase 1's catalog types (which describe what gets measured).
Runtime types describe a single execution and its outputs.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RunTrigger = Literal["manual", "launchd-weekly", "launchd-monthly", "ci"]
ScoringMethod = Literal["deterministic"]  # v2 adds "judge_ensemble"
SampleAggregation = Literal["single", "majority_vote", "median", "mean"]

# Run id: YYYY-MM-DD-{trigger}, e.g. 2026-04-29-manual
_RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z][a-z-]*$")


class RunStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    trigger: RunTrigger
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.IN_PROGRESS
    estimated_cost_usd: float | None = None

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _RUN_ID_RE.match(v):
            raise ValueError(
                f"Run id must be YYYY-MM-DD-<trigger>; got {v!r}"
            )
        return v


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    prompt_id: str
    model_id: str
    sample_n: int = Field(ge=0)
    status: TaskStatus
    text: str
    """Full response body (or empty string on failure)."""
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd_estimate: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    error: str | None = None
    """Set on TaskStatus.FAILED; None on SUCCESS."""

    def key(self) -> tuple[str, str, str, int]:
        """Stable identity for resume: same key → same task to dispatch."""
        return (self.run_id, self.prompt_id, self.model_id, self.sample_n)


class Score(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    prompt_id: str
    model_id: str
    subject_id: str
    metric: str
    """e.g. mention_presence, mention_presence_rate, ordinal_rank, share_of_voice."""
    value: bool | int | float | None
    scoring_method: ScoringMethod = "deterministic"
    sample_aggregation: SampleAggregation = "single"
```

- [ ] **Step 1.4: Run tests, verify pass**

```bash
uv run pytest tests/test_runtime.py -v
uv run pyright
uv run ruff check src tests
```

Expected: 9 pass, pyright clean.

- [ ] **Step 1.5: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/runtime.py \
        experiments/geo-analyzer/tests/test_runtime.py
git commit -m "geo-analyzer: runtime types (Run, Task, Score)"
```

---

### Task 2: Storage paths and run directory creation

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/storage/__init__.py`
- Create: `experiments/geo-analyzer/src/geo_analyzer/storage/paths.py`
- Create: `experiments/geo-analyzer/tests/test_storage_paths.py`

Centralizes the `data/runs/<run-id>/` layout. One module, no logic beyond `pathlib`.

- [ ] **Step 2.1: Write failing tests**

```python
# tests/test_storage_paths.py
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from geo_analyzer.storage.paths import (
    RunPaths,
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
        with pytest.raises(Exception):
            rp.run_dir = Path("/somewhere-else")  # type: ignore[misc]
```

- [ ] **Step 2.2: Run, verify fail**

```bash
uv run pytest tests/test_storage_paths.py -v
```

- [ ] **Step 2.3: Implement `paths.py`**

```python
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
```

- [ ] **Step 2.4: Implement `storage/__init__.py`**

```python
"""Storage primitives: run dir layout, manifest, JSONL writers, CSV emit."""

from geo_analyzer.storage.paths import RunPaths, build_run_id, run_paths_for

__all__ = ["RunPaths", "build_run_id", "run_paths_for"]
```

- [ ] **Step 2.5: Run tests + lint**

```bash
uv run pytest tests/test_storage_paths.py -v
uv run pyright
uv run ruff check src tests
```

- [ ] **Step 2.6: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/storage/ \
        experiments/geo-analyzer/tests/test_storage_paths.py
git commit -m "geo-analyzer: storage paths and run directory layout"
```

---

### Task 3: Append-only JSONL reader/writer

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/storage/jsonl.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/storage/__init__.py` (re-exports)
- Create: `experiments/geo-analyzer/tests/test_storage_jsonl.py`

The runner appends one line per task and one per score. Resume reads existing lines to know what's done.

- [ ] **Step 3.1: Write failing tests**

```python
# tests/test_storage_jsonl.py
from __future__ import annotations

from pathlib import Path

import pytest

from geo_analyzer.runtime import Task, TaskStatus
from geo_analyzer.storage.jsonl import (
    append_jsonl,
    read_jsonl_dicts,
    read_tasks_jsonl,
)


def _task(prompt_id: str, sample_n: int = 0) -> Task:
    return Task(
        run_id="2026-04-29-manual",
        prompt_id=prompt_id,
        model_id="openai:gpt-5.1:ungrounded",
        sample_n=sample_n,
        status=TaskStatus.SUCCESS,
        text="hello",
        tokens_in=10,
        tokens_out=5,
        cost_usd_estimate=0.001,
        latency_ms=100,
    )


class TestAppendJsonl:
    def test_appends_a_line(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        append_jsonl(path, _task("p1").model_dump(mode="json"))
        text = path.read_text()
        assert text.count("\n") == 1
        assert "p1" in text

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "more" / "tasks.jsonl"
        append_jsonl(path, {"a": 1})
        assert path.exists()

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"b": 2})
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2


class TestReadJsonlDicts:
    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.touch()
        assert read_jsonl_dicts(path) == []

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_jsonl_dicts(tmp_path / "nope.jsonl") == []

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"b": 2})
        assert read_jsonl_dicts(path) == [{"a": 1}, {"b": 2}]


class TestReadTasksJsonl:
    def test_parses_back_to_task_models(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        append_jsonl(path, _task("p1").model_dump(mode="json"))
        append_jsonl(path, _task("p2", sample_n=1).model_dump(mode="json"))
        tasks = read_tasks_jsonl(path)
        assert len(tasks) == 2
        assert {t.prompt_id for t in tasks} == {"p1", "p2"}

    def test_invalid_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        path.write_text("{not json\n")
        with pytest.raises(ValueError):
            read_tasks_jsonl(path)
```

- [ ] **Step 3.2: Run, verify fail**

```bash
uv run pytest tests/test_storage_jsonl.py -v
```

- [ ] **Step 3.3: Implement `jsonl.py`**

```python
"""Append-only JSONL helpers. One JSON object per line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geo_analyzer.runtime import Task


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON object as a line. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), default=str)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: invalid JSONL line: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line is not a JSON object: {stripped!r}")
            out.append(obj)
    return out


def read_tasks_jsonl(path: Path) -> list[Task]:
    """Read tasks.jsonl and parse each line into a Task."""
    return [Task.model_validate(d) for d in read_jsonl_dicts(path)]
```

- [ ] **Step 3.4: Update `storage/__init__.py`**

```python
"""Storage primitives: run dir layout, manifest, JSONL writers, CSV emit."""

from geo_analyzer.storage.jsonl import append_jsonl, read_jsonl_dicts, read_tasks_jsonl
from geo_analyzer.storage.paths import RunPaths, build_run_id, run_paths_for

__all__ = [
    "RunPaths",
    "append_jsonl",
    "build_run_id",
    "read_jsonl_dicts",
    "read_tasks_jsonl",
    "run_paths_for",
]
```

- [ ] **Step 3.5: Run + lint**

```bash
uv run pytest tests/test_storage_jsonl.py -v
uv run pyright
uv run ruff check src tests
```

- [ ] **Step 3.6: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/storage/ \
        experiments/geo-analyzer/tests/test_storage_jsonl.py
git commit -m "geo-analyzer: append-only JSONL reader/writer"
```

---

### Task 4: manifest.json read/write

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/storage/manifest.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/storage/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_storage_manifest.py`

`manifest.json` captures run metadata + a snapshot of catalog ids/versions, so an old run remains interpretable even if the catalog is later edited.

- [ ] **Step 4.1: Write failing tests**

```python
# tests/test_storage_manifest.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from geo_analyzer.runtime import Run, RunStatus
from geo_analyzer.storage.manifest import (
    Manifest,
    read_manifest,
    write_manifest,
)


def _run() -> Run:
    return Run(
        id="2026-04-29-manual",
        trigger="manual",
        started_at=datetime(2026, 4, 29, 9, 0, tzinfo=timezone.utc),
    )


class TestManifest:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        m = Manifest(
            run=_run(),
            subject_ids=["convictional_brand", "convictional_legacy_dropship"],
            prompt_ids=["prompt.broad.l1.companies-run-poorly"],
            model_ids=["openai:gpt-5.1:ungrounded"],
            catalog_hash="abc123",
        )
        write_manifest(path, m)
        loaded = read_manifest(path)
        assert loaded.run.id == m.run.id
        assert loaded.subject_ids == m.subject_ids
        assert loaded.catalog_hash == "abc123"

    def test_status_round_trips(self, tmp_path: Path) -> None:
        m = Manifest(
            run=_run().model_copy(update={"status": RunStatus.COMPLETED}),
            subject_ids=[],
            prompt_ids=[],
            model_ids=[],
            catalog_hash="x",
        )
        path = tmp_path / "manifest.json"
        write_manifest(path, m)
        assert read_manifest(path).run.status == RunStatus.COMPLETED
```

- [ ] **Step 4.2: Run, verify fail**

```bash
uv run pytest tests/test_storage_manifest.py -v
```

- [ ] **Step 4.3: Implement `manifest.py`**

```python
"""manifest.json — run metadata + catalog snapshot, written at run start
and updated on completion. Captures enough context that a stale run is
still interpretable after the catalog changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from geo_analyzer.runtime import Run


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: Run
    subject_ids: list[str]
    prompt_ids: list[str]
    model_ids: list[str]
    catalog_hash: str
    """SHA-256 hex of the concatenated YAML files at load time. Used by
    cross-run analysis to detect catalog drift between runs."""


def write_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def read_manifest(path: Path) -> Manifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(raw)
```

- [ ] **Step 4.4: Update `storage/__init__.py`**

```python
"""Storage primitives: run dir layout, manifest, JSONL writers, CSV emit."""

from geo_analyzer.storage.jsonl import append_jsonl, read_jsonl_dicts, read_tasks_jsonl
from geo_analyzer.storage.manifest import Manifest, read_manifest, write_manifest
from geo_analyzer.storage.paths import RunPaths, build_run_id, run_paths_for

__all__ = [
    "Manifest",
    "RunPaths",
    "append_jsonl",
    "build_run_id",
    "read_jsonl_dicts",
    "read_manifest",
    "read_tasks_jsonl",
    "run_paths_for",
    "write_manifest",
]
```

- [ ] **Step 4.5: Run + lint**

```bash
uv run pytest tests/test_storage_manifest.py -v
uv run pyright
uv run ruff check src tests
```

- [ ] **Step 4.6: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/storage/ \
        experiments/geo-analyzer/tests/test_storage_manifest.py
git commit -m "geo-analyzer: manifest.json read/write with catalog snapshot"
```

---

### Task 5: CSV emitter (tasks.csv + scores.csv)

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/storage/csv_export.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/storage/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_storage_csv.py`

Per DESIGN §7.1: `tasks.csv` excludes `text` (response bodies are huge); full text stays in `tasks.jsonl`. `scores.csv` includes everything.

- [ ] **Step 5.1: Write failing tests**

```python
# tests/test_storage_csv.py
from __future__ import annotations

import csv
from pathlib import Path

from geo_analyzer.runtime import Score, Task, TaskStatus
from geo_analyzer.storage.csv_export import write_scores_csv, write_tasks_csv


def _task(prompt_id: str = "p1") -> Task:
    return Task(
        run_id="r1", prompt_id=prompt_id, model_id="m:n:grounded",
        sample_n=0, status=TaskStatus.SUCCESS, text="long response here",
        tokens_in=10, tokens_out=5, cost_usd_estimate=0.001, latency_ms=100,
    )


def _score(metric: str = "mention_presence") -> Score:
    return Score(
        run_id="r1", prompt_id="p1", model_id="m:n:grounded",
        subject_id="s", metric=metric, value=True,
        scoring_method="deterministic", sample_aggregation="single",
    )


class TestWriteTasksCsv:
    def test_excludes_text_column(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.csv"
        write_tasks_csv(path, [_task(), _task("p2")])
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2
        assert "text" not in rows[0]
        # But everything else should be there.
        for col in ("run_id", "prompt_id", "model_id", "sample_n", "status",
                    "tokens_in", "tokens_out", "cost_usd_estimate", "latency_ms"):
            assert col in rows[0]

    def test_empty_list_writes_header_only(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.csv"
        write_tasks_csv(path, [])
        text = path.read_text()
        assert text.count("\n") == 1  # just header


class TestWriteScoresCsv:
    def test_writes_all_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.csv"
        write_scores_csv(path, [_score(), _score("ordinal_rank")])
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2
        for col in ("run_id", "prompt_id", "model_id", "subject_id",
                    "metric", "value", "scoring_method", "sample_aggregation"):
            assert col in rows[0]

    def test_value_serialization_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.csv"
        s_int = _score("ordinal_rank").model_copy(update={"value": 3})
        s_float = _score("share_of_voice").model_copy(update={"value": 0.5})
        s_none = _score("share_of_voice").model_copy(update={"value": None})
        write_scores_csv(path, [s_int, s_float, s_none])
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["value"] == "3"
        assert rows[1]["value"] == "0.5"
        assert rows[2]["value"] == ""  # None → empty cell
```

- [ ] **Step 5.2: Run, verify fail**

```bash
uv run pytest tests/test_storage_csv.py -v
```

- [ ] **Step 5.3: Implement `csv_export.py`**

```python
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
```

- [ ] **Step 5.4: Update `storage/__init__.py`** to re-export `write_tasks_csv` and `write_scores_csv`

Replace contents:

```python
"""Storage primitives: run dir layout, manifest, JSONL writers, CSV emit."""

from geo_analyzer.storage.csv_export import write_scores_csv, write_tasks_csv
from geo_analyzer.storage.jsonl import append_jsonl, read_jsonl_dicts, read_tasks_jsonl
from geo_analyzer.storage.manifest import Manifest, read_manifest, write_manifest
from geo_analyzer.storage.paths import RunPaths, build_run_id, run_paths_for

__all__ = [
    "Manifest",
    "RunPaths",
    "append_jsonl",
    "build_run_id",
    "read_jsonl_dicts",
    "read_manifest",
    "read_tasks_jsonl",
    "run_paths_for",
    "write_manifest",
    "write_scores_csv",
    "write_tasks_csv",
]
```

- [ ] **Step 5.5: Run + lint**

```bash
uv run pytest tests/test_storage_csv.py -v
uv run pyright
uv run ruff check src tests
```

- [ ] **Step 5.6: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/storage/ \
        experiments/geo-analyzer/tests/test_storage_csv.py
git commit -m "geo-analyzer: end-of-run CSV emitter (tasks.csv, scores.csv)"
```

---

### Task 6: Matrix expansion (catalog → tasks)

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/runner/__init__.py`
- Create: `experiments/geo-analyzer/src/geo_analyzer/runner/matrix.py`
- Create: `experiments/geo-analyzer/tests/test_runner_matrix.py`

Given a catalog and a run_id, produce the list of tasks to dispatch. Honors `--tier`, `--subject`, `--model` filters. Each ungrounded model emits 1 task per matching prompt; each grounded model emits N=`sampling.n` tasks per prompt.

- [ ] **Step 6.1: Write failing tests**

```python
# tests/test_runner_matrix.py
from __future__ import annotations

from pathlib import Path

import pytest

from geo_analyzer.catalog import load_catalog
from geo_analyzer.runner.matrix import (
    PendingTask,
    expand_matrix,
    filter_catalog,
)


@pytest.fixture(scope="module")
def real_catalog() -> object:
    """Load the actual seed catalog so tests reflect real shape."""
    project_root = Path(__file__).resolve().parents[1]
    return load_catalog(project_root / "catalog")


class TestFilterCatalog:
    def test_no_filters_returns_full_catalog(self, real_catalog: object) -> None:
        result = filter_catalog(real_catalog, tiers=None, subjects=None, model_ids=None)  # type: ignore[arg-type]
        assert len(result.prompts) == 12
        assert len(result.models) == 10

    def test_tier_filter(self, real_catalog: object) -> None:
        result = filter_catalog(real_catalog, tiers=["L1"], subjects=None, model_ids=None)  # type: ignore[arg-type]
        assert len(result.prompts) == 3
        assert all(p.tier == "L1" for p in result.prompts)

    def test_subject_filter_keeps_prompts_targeting_subject(self, real_catalog: object) -> None:
        result = filter_catalog(
            real_catalog,  # type: ignore[arg-type]
            tiers=None,
            subjects=["convictional_legacy_dropship"],
            model_ids=None,
        )
        # Only L4 brand prompts target the legacy subject in the seed catalog.
        assert all("convictional_legacy_dropship" in p.targets for p in result.prompts)

    def test_model_filter_inactive_models_dropped(self, real_catalog: object) -> None:
        result = filter_catalog(
            real_catalog,  # type: ignore[arg-type]
            tiers=None,
            subjects=None,
            model_ids=["openai:gpt-5.1:grounded"],
        )
        assert len(result.models) == 1
        assert result.models[0].id == "openai:gpt-5.1:grounded"


class TestExpandMatrix:
    def test_ungrounded_emits_one_per_prompt(self, real_catalog: object) -> None:
        result = filter_catalog(
            real_catalog,  # type: ignore[arg-type]
            tiers=["L1"],
            subjects=None,
            model_ids=["openai:gpt-5.1:ungrounded"],
        )
        tasks = expand_matrix(result, run_id="2026-04-29-manual")
        # 3 L1 prompts × 1 model × 1 sample
        assert len(tasks) == 3
        assert all(t.sample_n == 0 for t in tasks)

    def test_grounded_emits_n_samples_per_prompt(self, real_catalog: object) -> None:
        result = filter_catalog(
            real_catalog,  # type: ignore[arg-type]
            tiers=["L1"],
            subjects=None,
            model_ids=["openai:gpt-5.1:grounded"],
        )
        tasks = expand_matrix(result, run_id="2026-04-29-manual")
        # 3 L1 prompts × 1 model × 3 samples
        assert len(tasks) == 9
        sample_ns = sorted({t.sample_n for t in tasks})
        assert sample_ns == [0, 1, 2]

    def test_pending_task_has_required_fields(self, real_catalog: object) -> None:
        result = filter_catalog(
            real_catalog,  # type: ignore[arg-type]
            tiers=["L1"],
            subjects=None,
            model_ids=["openai:gpt-5.1:ungrounded"],
        )
        tasks = expand_matrix(result, run_id="r1")
        t = tasks[0]
        assert isinstance(t, PendingTask)
        assert t.run_id == "r1"
        assert t.prompt_id.startswith("prompt.")
        assert t.model_id == "openai:gpt-5.1:ungrounded"
        assert isinstance(t.sample_n, int)

    def test_inactive_model_excluded(self, real_catalog: object) -> None:
        # Mutate a copy: build a Catalog where one model is inactive
        from geo_analyzer.types import Catalog
        cat: Catalog = real_catalog  # type: ignore[assignment]
        # Make a new Catalog with one model flipped to active=False
        models = [m if m.id != "openai:gpt-5.1:ungrounded" else m.model_copy(update={"active": False})
                  for m in cat.models]
        new_cat = Catalog(
            subjects=cat.subjects,
            prompts=cat.prompts,
            providers=cat.providers,
            models=models,
        )
        tasks = expand_matrix(new_cat, run_id="r1")
        assert all(t.model_id != "openai:gpt-5.1:ungrounded" for t in tasks)
```

- [ ] **Step 6.2: Run, verify fail**

```bash
uv run pytest tests/test_runner_matrix.py -v
```

- [ ] **Step 6.3: Implement `runner/matrix.py`**

```python
"""Catalog × matrix → list of PendingTask.

A PendingTask is what the orchestrator dispatches. After the call resolves
it gets converted into a runtime Task (with status, tokens, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from geo_analyzer.types import Catalog, ModelSpec, Prompt, PromptTier


@dataclass(frozen=True)
class PendingTask:
    run_id: str
    prompt_id: str
    model_id: str
    sample_n: int


def filter_catalog(
    catalog: Catalog,
    *,
    tiers: list[PromptTier] | None,
    subjects: list[str] | None,
    model_ids: list[str] | None,
) -> Catalog:
    """Return a new Catalog with prompts/models narrowed by the given filters.

    - tiers=None means all tiers; otherwise keep only prompts whose tier matches.
    - subjects=None means all subjects; otherwise keep only prompts whose
      `targets` contains any of the named subject ids.
    - model_ids=None means all models; otherwise keep only those ids.
    Inactive models are always excluded regardless of `model_ids`.
    """
    prompts: list[Prompt] = list(catalog.prompts)
    if tiers is not None:
        tier_set = set(tiers)
        prompts = [p for p in prompts if p.tier in tier_set]
    if subjects is not None:
        subj_set = set(subjects)
        prompts = [p for p in prompts if any(t in subj_set for t in p.targets)]

    models: list[ModelSpec] = [m for m in catalog.models if m.active]
    if model_ids is not None:
        id_set = set(model_ids)
        models = [m for m in models if m.id in id_set]

    return Catalog(
        subjects=catalog.subjects,
        prompts=prompts,
        providers=catalog.providers,
        models=models,
    )


def expand_matrix(catalog: Catalog, *, run_id: str) -> list[PendingTask]:
    """Cartesian-product prompts × models, with N samples per (prompt, model).

    N comes from model.sampling.n — typically 1 for ungrounded, 3 for grounded.
    """
    out: list[PendingTask] = []
    for prompt in catalog.prompts:
        for model in catalog.models:
            for n in range(model.sampling.n):
                out.append(
                    PendingTask(
                        run_id=run_id,
                        prompt_id=prompt.id,
                        model_id=model.id,
                        sample_n=n,
                    )
                )
    return out
```

- [ ] **Step 6.4: Implement `runner/__init__.py`**

```python
"""Runner: matrix expansion, retry, concurrency, scoring, orchestration."""

from geo_analyzer.runner.matrix import PendingTask, expand_matrix, filter_catalog

__all__ = ["PendingTask", "expand_matrix", "filter_catalog"]
```

- [ ] **Step 6.5: Run + lint**

```bash
uv run pytest tests/test_runner_matrix.py -v
uv run pyright
uv run ruff check src tests
```

- [ ] **Step 6.6: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/runner/ \
        experiments/geo-analyzer/tests/test_runner_matrix.py
git commit -m "geo-analyzer: matrix expansion (catalog × prompts × samples)"
```

---

### Task 7: Retry helper (exponential backoff)

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/runner/retry.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/runner/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_runner_retry.py`

Wraps an awaitable in retry logic. Per DESIGN §5.2: max_retries=3 default, exponential backoff with `backoff_base_s` from `RetryConfig`.

- [ ] **Step 7.1: Write failing tests**

```python
# tests/test_runner_retry.py
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from geo_analyzer.providers.base import ProviderError
from geo_analyzer.runner.retry import retry_with_backoff


class TestRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self) -> None:
        called = AsyncMock(return_value="ok")
        result = await retry_with_backoff(called, max_attempts=3, backoff_base_s=0.0)
        assert result == "ok"
        assert called.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_provider_error(self) -> None:
        attempts: list[int] = []

        async def flaky() -> str:
            attempts.append(len(attempts))
            if len(attempts) < 3:
                raise ProviderError("transient")
            return "ok"

        with patch("geo_analyzer.runner.retry.asyncio.sleep", new=AsyncMock()):
            result = await retry_with_backoff(flaky, max_attempts=3, backoff_base_s=1.0)
        assert result == "ok"
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self) -> None:
        async def always_fails() -> str:
            raise ProviderError("permanent")

        with patch("geo_analyzer.runner.retry.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ProviderError, match="permanent"):
                await retry_with_backoff(always_fails, max_attempts=3, backoff_base_s=0.0)

    @pytest.mark.asyncio
    async def test_does_not_retry_unrelated_exceptions(self) -> None:
        async def raises_value_error() -> str:
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            await retry_with_backoff(raises_value_error, max_attempts=3, backoff_base_s=0.0)

    @pytest.mark.asyncio
    async def test_backoff_grows_exponentially(self) -> None:
        sleep_calls: list[float] = []

        async def fail_twice_then_ok() -> str:
            if len(sleep_calls) < 2:
                raise ProviderError("transient")
            return "ok"

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch("geo_analyzer.runner.retry.asyncio.sleep", new=fake_sleep):
            await retry_with_backoff(fail_twice_then_ok, max_attempts=4, backoff_base_s=2.0)

        # Backoff: base * 2^(attempt-1) → 2.0, 4.0
        assert sleep_calls == [2.0, 4.0]
```

- [ ] **Step 7.2: Run, verify fail**

```bash
uv run pytest tests/test_runner_retry.py -v
```

- [ ] **Step 7.3: Implement `runner/retry.py`**

```python
"""Exponential-backoff retry for provider calls.

Only retries `ProviderError` (which adapters raise on 429/5xx/timeout/network).
Non-provider exceptions (programming errors) propagate immediately.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from geo_analyzer.providers.base import ProviderError

_T = TypeVar("_T")


async def retry_with_backoff(
    fn: Callable[[], Awaitable[_T]],
    *,
    max_attempts: int,
    backoff_base_s: float,
) -> _T:
    """Call `fn()` up to `max_attempts` times. Sleeps backoff_base_s * 2^(attempt-1)
    between failures. Re-raises the last ProviderError if all attempts fail.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last: ProviderError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except ProviderError as e:
            last = e
            if attempt == max_attempts:
                break
            delay = backoff_base_s * (2 ** (attempt - 1))
            await asyncio.sleep(delay)
    assert last is not None
    raise last
```

- [ ] **Step 7.4: Update `runner/__init__.py`**

```python
"""Runner: matrix expansion, retry, concurrency, scoring, orchestration."""

from geo_analyzer.runner.matrix import PendingTask, expand_matrix, filter_catalog
from geo_analyzer.runner.retry import retry_with_backoff

__all__ = ["PendingTask", "expand_matrix", "filter_catalog", "retry_with_backoff"]
```

- [ ] **Step 7.5: Run + lint**

```bash
uv run pytest tests/test_runner_retry.py -v
uv run pyright
uv run ruff check src tests
```

- [ ] **Step 7.6: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/runner/ \
        experiments/geo-analyzer/tests/test_runner_retry.py
git commit -m "geo-analyzer: exponential-backoff retry for provider calls"
```

---

### Task 8: Per-provider concurrency (asyncio.Semaphore manager)

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/runner/concurrency.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/runner/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_runner_concurrency.py`

DESIGN §5.2: per-provider semaphore caps in-flight requests (e.g., `openai=8, anthropic=5`). `ConcurrencyManager` is a tiny dict of `asyncio.Semaphore`.

- [ ] **Step 8.1: Write failing tests**

```python
# tests/test_runner_concurrency.py
from __future__ import annotations

import asyncio

import pytest

from geo_analyzer.runner.concurrency import ConcurrencyManager


class TestConcurrencyManager:
    def test_get_returns_semaphore_for_provider(self) -> None:
        cm = ConcurrencyManager(caps={"openai": 8, "anthropic": 5})
        assert isinstance(cm.semaphore_for("openai"), asyncio.Semaphore)

    def test_unknown_provider_raises(self) -> None:
        cm = ConcurrencyManager(caps={"openai": 8})
        with pytest.raises(KeyError):
            cm.semaphore_for("nope")

    @pytest.mark.asyncio
    async def test_caps_concurrent_in_flight(self) -> None:
        cm = ConcurrencyManager(caps={"openai": 2})
        in_flight = 0
        max_in_flight = 0

        async def task() -> None:
            nonlocal in_flight, max_in_flight
            async with cm.semaphore_for("openai"):
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        await asyncio.gather(*(task() for _ in range(10)))
        assert max_in_flight <= 2
```

- [ ] **Step 8.2: Run, verify fail**

```bash
uv run pytest tests/test_runner_concurrency.py -v
```

- [ ] **Step 8.3: Implement `runner/concurrency.py`**

```python
"""Per-provider concurrency caps. One asyncio.Semaphore per provider name."""

from __future__ import annotations

import asyncio


class ConcurrencyManager:
    def __init__(self, *, caps: dict[str, int]) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(n) for name, n in caps.items()
        }

    def semaphore_for(self, provider: str) -> asyncio.Semaphore:
        try:
            return self._semaphores[provider]
        except KeyError as e:
            raise KeyError(f"no concurrency cap configured for provider {provider!r}") from e
```

- [ ] **Step 8.4: Update `runner/__init__.py`**

```python
"""Runner: matrix expansion, retry, concurrency, scoring, orchestration."""

from geo_analyzer.runner.concurrency import ConcurrencyManager
from geo_analyzer.runner.matrix import PendingTask, expand_matrix, filter_catalog
from geo_analyzer.runner.retry import retry_with_backoff

__all__ = [
    "ConcurrencyManager",
    "PendingTask",
    "expand_matrix",
    "filter_catalog",
    "retry_with_backoff",
]
```

- [ ] **Step 8.5: Run + lint**

```bash
uv run pytest tests/test_runner_concurrency.py -v
uv run pyright
uv run ruff check src tests
```

- [ ] **Step 8.6: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/runner/ \
        experiments/geo-analyzer/tests/test_runner_concurrency.py
git commit -m "geo-analyzer: per-provider concurrency manager"
```

---

### Task 9: Scoring pipeline

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/runner/scoring.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/runner/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_runner_scoring.py`

Given a list of completed `Task`s, produce per-(prompt, model, subject, metric) `Score`s. Per-sample scoring uses Phase 1's extractors; aggregation across samples uses Phase 1's `aggregation.py`.

Scope (matches DESIGN §6.1, §5.3):
- Per task, per subject in `prompt.targets`: compute `mention_presence`, `ordinal_rank`, `share_of_voice`. For anti-brand subjects, also compute `brand_legacy_conflation` against the brand subject `convictional_brand` (only if both subjects exist in catalog).
- Aggregate per (prompt, model, subject):
  - mention_presence (bool, majority_vote) + mention_presence_rate (float, mean) for grounded N≥2
  - ordinal_rank (median) + share_of_voice (mean)
  - brand_legacy_conflation (majority_vote bool) + rate (mean) for grounded
  - Ungrounded (N=1) emits only the binary, not the rate.

- [ ] **Step 9.1: Write failing tests**

```python
# tests/test_runner_scoring.py
from __future__ import annotations

from geo_analyzer.runner.scoring import score_run
from geo_analyzer.runtime import Task, TaskStatus
from geo_analyzer.types import Subject, SubjectKind


_RUN_ID = "r1"


def _task(text: str, prompt_id: str = "p1", model_id: str = "openai:gpt-5.1:ungrounded",
          sample_n: int = 0) -> Task:
    return Task(
        run_id="r", prompt_id=prompt_id, model_id=model_id, sample_n=sample_n,
        status=TaskStatus.SUCCESS, text=text,
        tokens_in=10, tokens_out=10, cost_usd_estimate=0.0, latency_ms=0,
    )


def _brand() -> Subject:
    return Subject(
        id="convictional_brand", kind=SubjectKind.BRAND,
        aliases=["Convictional"], definition="x",
        competitors=["lattice"],
    )


def _lattice() -> Subject:
    return Subject(
        id="lattice", kind=SubjectKind.BRAND, aliases=["Lattice"], definition="x",
    )


def _legacy() -> Subject:
    return Subject(
        id="convictional_legacy_dropship", kind=SubjectKind.ANTI_BRAND,
        aliases=["dropship"], definition="x",
    )


class TestScoreRun:
    def test_mention_presence_emitted(self) -> None:
        tasks = [_task("Convictional is great.")]
        # prompt p1 targets convictional_brand only
        prompt_targets = {"p1": ["convictional_brand"]}
        subjects = {s.id: s for s in [_brand(), _lattice()]}
        scores = score_run(tasks, run_id=_RUN_ID, prompt_targets=prompt_targets, subjects=subjects)
        metrics = {(s.subject_id, s.metric, s.value) for s in scores}
        assert ("convictional_brand", "mention_presence", True) in metrics

    def test_ungrounded_does_not_emit_rate(self) -> None:
        tasks = [_task("Convictional helps.")]
        prompt_targets = {"p1": ["convictional_brand"]}
        subjects = {"convictional_brand": _brand(), "lattice": _lattice()}
        scores = score_run(tasks, run_id=_RUN_ID, prompt_targets=prompt_targets, subjects=subjects)
        metrics = {s.metric for s in scores}
        assert "mention_presence_rate" not in metrics

    def test_grounded_three_samples_emits_rate_and_majority(self) -> None:
        tasks = [
            _task("Convictional helps.", model_id="openai:gpt-5.1:grounded", sample_n=0),
            _task("Convictional helps.", model_id="openai:gpt-5.1:grounded", sample_n=1),
            _task("nothing relevant",    model_id="openai:gpt-5.1:grounded", sample_n=2),
        ]
        prompt_targets = {"p1": ["convictional_brand"]}
        subjects = {"convictional_brand": _brand(), "lattice": _lattice()}
        scores = score_run(tasks, run_id=_RUN_ID, prompt_targets=prompt_targets, subjects=subjects)
        by_metric = {s.metric: s for s in scores if s.subject_id == "convictional_brand"}
        # 2/3 samples have presence → majority True, rate = 2/3
        assert by_metric["mention_presence"].value is True
        assert abs((by_metric["mention_presence_rate"].value or 0.0) - (2 / 3)) < 1e-9

    def test_brand_legacy_conflation_emitted_only_when_anti_brand_present(self) -> None:
        tasks = [_task("Convictional was a dropship platform.")]
        # Even though prompt targets only the brand, conflation is keyed on the
        # presence of an anti_brand subject in the catalog, not in targets.
        prompt_targets = {"p1": ["convictional_brand"]}
        subjects = {
            "convictional_brand": _brand(),
            "lattice": _lattice(),
            "convictional_legacy_dropship": _legacy(),
        }
        scores = score_run(tasks, run_id=_RUN_ID, prompt_targets=prompt_targets, subjects=subjects)
        metrics = {s.metric for s in scores}
        assert "brand_legacy_conflation" in metrics

    def test_failed_tasks_skipped(self) -> None:
        from geo_analyzer.runtime import TaskStatus as TS
        tasks = [_task("ok").model_copy(update={"status": TS.FAILED})]
        prompt_targets = {"p1": ["convictional_brand"]}
        subjects = {"convictional_brand": _brand(), "lattice": _lattice()}
        scores = score_run(tasks, run_id=_RUN_ID, prompt_targets=prompt_targets, subjects=subjects)
        assert scores == []
```

- [ ] **Step 9.2: Run, verify fail**

```bash
uv run pytest tests/test_runner_scoring.py -v
```

- [ ] **Step 9.3: Implement `runner/scoring.py`**

```python
"""Scoring pipeline. Apply Phase 1 extractors per sample, aggregate per
(prompt, model, subject) per DESIGN §5.3.

Inputs:
  - tasks: every successful task for a run (failed tasks are skipped).
  - run_id: tagged onto every emitted Score.
  - prompt_targets: prompt_id → list of subject ids the prompt targets.
  - subjects: subject_id → Subject (full catalog map; needed for SoV
    competitor lookups and the anti_brand detection).

Output: list[Score].
"""

from __future__ import annotations

from collections import defaultdict
from typing import cast

from geo_analyzer.runtime import (
    SampleAggregation,
    Score,
    Task,
    TaskStatus,
)
from geo_analyzer.scoring import (
    brand_legacy_conflation,
    mention_presence,
    ordinal_rank,
    share_of_voice,
)
from geo_analyzer.scoring.aggregation import (
    majority_vote,
    mean_of_floats,
    mean_rate,
    median_or_none,
)
from geo_analyzer.types import Subject, SubjectKind

_BRAND_LIKE = {SubjectKind.BRAND, SubjectKind.CATEGORY}


def score_run(
    tasks: list[Task],
    *,
    run_id: str,
    prompt_targets: dict[str, list[str]],
    subjects: dict[str, Subject],
) -> list[Score]:
    """Compute aggregated scores from the run's task results."""
    successful = [t for t in tasks if t.status == TaskStatus.SUCCESS]
    if not successful:
        return []

    # Group tasks by (prompt_id, model_id) — these are the aggregation cohorts.
    cohorts: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for t in successful:
        cohorts[(t.prompt_id, t.model_id)].append(t)

    # Identify the brand and anti_brand subjects for conflation (if any exist).
    anti_brands = [s for s in subjects.values() if s.kind == SubjectKind.ANTI_BRAND]
    brands = [s for s in subjects.values() if s.kind == SubjectKind.BRAND]

    out: list[Score] = []

    for (prompt_id, model_id), cohort in cohorts.items():
        target_ids = prompt_targets.get(prompt_id, [])
        n = len(cohort)
        is_grounded_multi = n > 1

        # --- per-target subject metrics ---
        for sid in target_ids:
            subj = subjects.get(sid)
            if subj is None:
                continue
            if subj.kind not in _BRAND_LIKE:
                continue

            presence_samples = [mention_presence(t.text, subj).present for t in cohort]
            rank_samples = [ordinal_rank(t.text, subj).rank for t in cohort]
            sov_samples = [share_of_voice(t.text, subj, subjects).value for t in cohort]

            out.append(_score(
                run_id, prompt_id, model_id, sid,
                metric="mention_presence",
                value=majority_vote(presence_samples) if is_grounded_multi else presence_samples[0],
                aggregation="majority_vote" if is_grounded_multi else "single",
            ))
            if is_grounded_multi:
                out.append(_score(
                    run_id, prompt_id, model_id, sid,
                    metric="mention_presence_rate",
                    value=mean_rate(presence_samples),
                    aggregation="mean",
                ))

            out.append(_score(
                run_id, prompt_id, model_id, sid,
                metric="ordinal_rank",
                value=median_or_none(rank_samples) if is_grounded_multi else rank_samples[0],
                aggregation="median" if is_grounded_multi else "single",
            ))

            out.append(_score(
                run_id, prompt_id, model_id, sid,
                metric="share_of_voice",
                value=mean_of_floats(sov_samples) if is_grounded_multi else sov_samples[0],
                aggregation="mean" if is_grounded_multi else "single",
            ))

        # --- conflation: brand × anti_brand co-occurrence ---
        # Emit one conflation score per (brand, anti_brand) pair regardless of
        # what the prompt explicitly targets; the metric is about the response.
        for brand in brands:
            for anti in anti_brands:
                conflation_samples = [
                    brand_legacy_conflation(t.text, brand, anti).fired for t in cohort
                ]
                out.append(_score(
                    run_id, prompt_id, model_id, anti.id,
                    metric="brand_legacy_conflation",
                    value=majority_vote(conflation_samples) if is_grounded_multi else conflation_samples[0],
                    aggregation="majority_vote" if is_grounded_multi else "single",
                ))
                if is_grounded_multi:
                    out.append(_score(
                        run_id, prompt_id, model_id, anti.id,
                        metric="brand_legacy_conflation_rate",
                        value=mean_rate(conflation_samples),
                        aggregation="mean",
                    ))

    return out


def _score(
    run_id: str,
    prompt_id: str,
    model_id: str,
    subject_id: str,
    *,
    metric: str,
    value: bool | int | float | None,
    aggregation: str,
) -> Score:
    return Score(
        run_id=run_id,
        prompt_id=prompt_id,
        model_id=model_id,
        subject_id=subject_id,
        metric=metric,
        value=value,
        scoring_method="deterministic",
        sample_aggregation=cast(SampleAggregation, aggregation),
    )
```

- [ ] **Step 9.4: Update `runner/__init__.py`**

```python
"""Runner: matrix expansion, retry, concurrency, scoring, orchestration."""

from geo_analyzer.runner.concurrency import ConcurrencyManager
from geo_analyzer.runner.matrix import PendingTask, expand_matrix, filter_catalog
from geo_analyzer.runner.retry import retry_with_backoff
from geo_analyzer.runner.scoring import score_run

__all__ = [
    "ConcurrencyManager",
    "PendingTask",
    "expand_matrix",
    "filter_catalog",
    "retry_with_backoff",
    "score_run",
]
```

- [ ] **Step 9.5: Run + lint**

```bash
uv run pytest tests/test_runner_scoring.py -v
uv run pyright
uv run ruff check src tests
```

- [ ] **Step 9.6: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/runner/ \
        experiments/geo-analyzer/tests/test_runner_scoring.py
git commit -m "geo-analyzer: scoring pipeline (per-task extract + per-cohort aggregate)"
```

---

### Task 10: Orchestrator (the run loop)

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/runner/orchestrator.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/runner/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_runner_orchestrator.py`

The integration point. Steps:
1. Resolve run paths, ensure dir.
2. Read existing `tasks.jsonl` (resume).
3. Compute pending tasks = expanded - completed.
4. For each pending task: acquire provider semaphore, retry-call provider, write to `tasks.jsonl`.
5. After all done: read `tasks.jsonl` back as Tasks, call `score_run`, write `scores.jsonl`, write CSVs, update manifest with finished_at + status.
6. Return a summary tuple (run, n_success, n_failed).

- [ ] **Step 10.1: Write failing tests**

```python
# tests/test_runner_orchestrator.py
from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from geo_analyzer.catalog import load_catalog
from geo_analyzer.providers.base import ProbeRequest, ProviderError, ProviderResponse
from geo_analyzer.runner.matrix import filter_catalog
from geo_analyzer.runner.orchestrator import RunSummary, run as orchestrator_run
from geo_analyzer.runtime import RunStatus, TaskStatus
from geo_analyzer.storage import read_jsonl_dicts, read_manifest, run_paths_for


class _FakeProvider:
    """Records every call. Returns a deterministic response per (prompt, model, sample)."""

    def __init__(self, name: str = "openai") -> None:
        self.name = name
        self.calls: list[ProbeRequest] = []

    async def call(self, request: ProbeRequest) -> ProviderResponse:
        self.calls.append(request)
        # Embed identifying info in the text so tests can assert per-task content.
        text = (
            f"Convictional response for {request.model.id} prompt={request.prompt!r}"
        )
        return ProviderResponse(
            text=text, tokens_in=10, tokens_out=20,
            cost_usd_estimate=0.001, latency_ms=50, raw={},
        )


@pytest.fixture(scope="module")
def real_catalog() -> object:
    project_root = Path(__file__).resolve().parents[1]
    return load_catalog(project_root / "catalog")


@pytest.mark.asyncio
async def test_orchestrator_writes_tasks_and_scores(real_catalog: object, tmp_path: Path) -> None:
    cat = filter_catalog(
        real_catalog,  # type: ignore[arg-type]
        tiers=["L1"],
        subjects=None,
        model_ids=["openai:gpt-5.1:ungrounded"],
    )
    fake = _FakeProvider()

    with patch("geo_analyzer.runner.orchestrator.get_provider", return_value=fake):
        summary: RunSummary = await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
        )

    assert summary.run.status == RunStatus.COMPLETED
    assert summary.n_success == 3  # 3 L1 prompts × 1 model × 1 sample
    assert summary.n_failed == 0

    rp = run_paths_for(tmp_path, "2026-04-29-manual")
    tasks_lines = read_jsonl_dicts(rp.tasks_jsonl)
    scores_lines = read_jsonl_dicts(rp.scores_jsonl)
    assert len(tasks_lines) == 3
    assert len(scores_lines) > 0
    assert rp.tasks_csv.exists()
    assert rp.scores_csv.exists()

    manifest = read_manifest(rp.manifest)
    assert manifest.run.status == RunStatus.COMPLETED
    assert len(manifest.prompt_ids) == 3


@pytest.mark.asyncio
async def test_orchestrator_resumes_partial_run(real_catalog: object, tmp_path: Path) -> None:
    cat = filter_catalog(
        real_catalog,  # type: ignore[arg-type]
        tiers=["L1"],
        subjects=None,
        model_ids=["openai:gpt-5.1:ungrounded"],
    )
    fake = _FakeProvider()

    with patch("geo_analyzer.runner.orchestrator.get_provider", return_value=fake):
        # First run: complete normally.
        await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
        )
        first_call_count = len(fake.calls)
        assert first_call_count == 3

        # Second run with resume=True: tasks already complete, no new calls.
        await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
        )
        assert len(fake.calls) == first_call_count  # no additional calls


@pytest.mark.asyncio
async def test_orchestrator_records_failed_tasks(real_catalog: object, tmp_path: Path) -> None:
    cat = filter_catalog(
        real_catalog,  # type: ignore[arg-type]
        tiers=["L1"],
        subjects=None,
        model_ids=["openai:gpt-5.1:ungrounded"],
    )

    class _AlwaysFails:
        name = "openai"
        async def call(self, request: ProbeRequest) -> ProviderResponse:
            raise ProviderError("rate limit")

    with patch("geo_analyzer.runner.orchestrator.get_provider", return_value=_AlwaysFails()):
        summary = await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
        )

    assert summary.n_failed == 3
    assert summary.n_success == 0
    rp = run_paths_for(tmp_path, "2026-04-29-manual")
    tasks_lines = read_jsonl_dicts(rp.tasks_jsonl)
    assert all(t["status"] == TaskStatus.FAILED.value for t in tasks_lines)
```

- [ ] **Step 10.2: Run, verify fail**

```bash
uv run pytest tests/test_runner_orchestrator.py -v
```

- [ ] **Step 10.3: Implement `runner/orchestrator.py`**

```python
"""The run loop. Composes matrix + retry + concurrency + scoring + storage."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from geo_analyzer.providers import ProbeRequest, ProviderError, get_provider
from geo_analyzer.runner.concurrency import ConcurrencyManager
from geo_analyzer.runner.matrix import PendingTask, expand_matrix
from geo_analyzer.runner.retry import retry_with_backoff
from geo_analyzer.runner.scoring import score_run
from geo_analyzer.runtime import (
    Run,
    RunStatus,
    RunTrigger,
    Task,
    TaskStatus,
)
from geo_analyzer.storage import (
    Manifest,
    append_jsonl,
    build_run_id,
    read_tasks_jsonl,
    run_paths_for,
    write_manifest,
    write_scores_csv,
    write_tasks_csv,
)
from geo_analyzer.types import Catalog, ModelSpec, Prompt


@dataclass(frozen=True)
class RunSummary:
    run: Run
    n_success: int
    n_failed: int


async def run(
    *,
    catalog: Catalog,
    data_dir: Path,
    run_date: date,
    trigger: RunTrigger,
    api_keys: dict[str, str],
    resume: bool,
) -> RunSummary:
    """Execute a run end-to-end.

    Side effects: creates `data_dir/runs/<run-id>/` and writes manifest.json,
    tasks.jsonl, scores.jsonl, tasks.csv, scores.csv inside it.
    """
    run_id = build_run_id(run_date, trigger=trigger)
    rp = run_paths_for(data_dir, run_id)
    rp.ensure()

    started = datetime.now(timezone.utc)
    run_obj = Run(id=run_id, trigger=trigger, started_at=started, status=RunStatus.IN_PROGRESS)

    # Persist initial manifest so partial runs are interpretable.
    manifest = Manifest(
        run=run_obj,
        subject_ids=[s.id for s in catalog.subjects],
        prompt_ids=[p.id for p in catalog.prompts],
        model_ids=[m.id for m in catalog.models],
        catalog_hash=_hash_catalog(catalog),
    )
    write_manifest(rp.manifest, manifest)

    # Resume: load existing completed task keys.
    completed_keys: set[tuple[str, str, str, int]] = set()
    if resume and rp.tasks_jsonl.exists():
        for t in read_tasks_jsonl(rp.tasks_jsonl):
            if t.status == TaskStatus.SUCCESS:
                completed_keys.add(t.key())

    pending_all = expand_matrix(catalog, run_id=run_id)
    pending = [pt for pt in pending_all if (run_id, pt.prompt_id, pt.model_id, pt.sample_n) not in completed_keys]

    cm = ConcurrencyManager(caps={name: pc.concurrency for name, pc in catalog.providers.items()})
    providers_by_name = {name: get_provider(name, api_key=api_keys[name]) for name in catalog.providers}

    prompts_by_id: dict[str, Prompt] = {p.id: p for p in catalog.prompts}
    models_by_id: dict[str, ModelSpec] = {m.id: m for m in catalog.models}

    async def _execute(pt: PendingTask) -> None:
        prompt = prompts_by_id[pt.prompt_id]
        model = models_by_id[pt.model_id]
        provider = providers_by_name[model.provider]
        retry_cfg = catalog.providers[model.provider].retry

        async def _do_call() -> None:
            async with cm.semaphore_for(model.provider):
                req = ProbeRequest(model=model, prompt=prompt.text)
                resp = await provider.call(req)
                _append_task(rp.tasks_jsonl, pt, status=TaskStatus.SUCCESS,
                             text=resp.text, tokens_in=resp.tokens_in,
                             tokens_out=resp.tokens_out,
                             cost_usd_estimate=resp.cost_usd_estimate,
                             latency_ms=resp.latency_ms, error=None)

        try:
            await retry_with_backoff(
                _do_call,
                max_attempts=retry_cfg.max_attempts,
                backoff_base_s=retry_cfg.backoff_base_s,
            )
        except ProviderError as e:
            _append_task(rp.tasks_jsonl, pt, status=TaskStatus.FAILED,
                         text="", tokens_in=0, tokens_out=0,
                         cost_usd_estimate=0.0, latency_ms=0, error=str(e))

    await asyncio.gather(*(_execute(pt) for pt in pending))

    # Re-read all tasks (including ones from prior partial runs) for scoring.
    all_tasks = read_tasks_jsonl(rp.tasks_jsonl)

    subjects = {s.id: s for s in catalog.subjects}
    prompt_targets = {p.id: list(p.targets) for p in catalog.prompts}
    scores = score_run(all_tasks, run_id=run_id, prompt_targets=prompt_targets, subjects=subjects)
    for s in scores:
        append_jsonl(rp.scores_jsonl, s.model_dump(mode="json"))

    write_tasks_csv(rp.tasks_csv, all_tasks)
    write_scores_csv(rp.scores_csv, scores)

    n_success = sum(1 for t in all_tasks if t.status == TaskStatus.SUCCESS)
    n_failed = sum(1 for t in all_tasks if t.status == TaskStatus.FAILED)

    finished_run = run_obj.model_copy(update={
        "status": RunStatus.COMPLETED if n_failed == 0 else RunStatus.COMPLETED,
        "finished_at": datetime.now(timezone.utc),
    })
    final_manifest = manifest.model_copy(update={"run": finished_run})
    write_manifest(rp.manifest, final_manifest)

    return RunSummary(run=finished_run, n_success=n_success, n_failed=n_failed)


def _append_task(
    path: Path,
    pt: PendingTask,
    *,
    status: TaskStatus,
    text: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd_estimate: float,
    latency_ms: int,
    error: str | None,
) -> None:
    t = Task(
        run_id=pt.run_id,
        prompt_id=pt.prompt_id,
        model_id=pt.model_id,
        sample_n=pt.sample_n,
        status=status,
        text=text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd_estimate=cost_usd_estimate,
        latency_ms=latency_ms,
        error=error,
    )
    append_jsonl(path, t.model_dump(mode="json"))


def _hash_catalog(catalog: Catalog) -> str:
    """SHA-256 of the catalog's deterministic JSON representation."""
    payload = catalog.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 10.4: Update `runner/__init__.py`**

```python
"""Runner: matrix expansion, retry, concurrency, scoring, orchestration."""

from geo_analyzer.runner.concurrency import ConcurrencyManager
from geo_analyzer.runner.matrix import PendingTask, expand_matrix, filter_catalog
from geo_analyzer.runner.orchestrator import RunSummary, run
from geo_analyzer.runner.retry import retry_with_backoff
from geo_analyzer.runner.scoring import score_run

__all__ = [
    "ConcurrencyManager",
    "PendingTask",
    "RunSummary",
    "expand_matrix",
    "filter_catalog",
    "retry_with_backoff",
    "run",
    "score_run",
]
```

- [ ] **Step 10.5: Run + lint**

```bash
uv run pytest tests/test_runner_orchestrator.py -v
uv run pytest -q
uv run pyright
uv run ruff check src tests
```

- [ ] **Step 10.6: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/runner/ \
        experiments/geo-analyzer/tests/test_runner_orchestrator.py
git commit -m "geo-analyzer: orchestrator (matrix → providers → tasks.jsonl → scores)"
```

---

### Task 11: `geo-analyzer run` CLI command (with cost gate)

**Files:**
- Modify: `experiments/geo-analyzer/src/geo_analyzer/cli.py`
- Create: `experiments/geo-analyzer/tests/test_cli_run.py`

CLI surface from DESIGN §8 + §5.6:
```
geo-analyzer run [--tier L1,L2,L3,L4] [--subject S,S] [--model M,M]
                 [--dry-run] [--resume / --no-resume] [--yes]
                 [--trigger manual | launchd-weekly | launchd-monthly]
```
Cost gate: if estimated cost > `$5`, require interactive confirmation (or `--yes`).

`--dry-run` prints the matrix expansion + estimated cost, exits 0 without calling providers.

- [ ] **Step 11.1: Write failing tests**

```python
# tests/test_cli_run.py
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
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
    from datetime import datetime, timezone
    run_obj = Run(
        id="2026-04-29-manual",
        trigger="manual",
        started_at=datetime(2026, 4, 29, tzinfo=timezone.utc),
        finished_at=datetime(2026, 4, 29, 0, 1, tzinfo=timezone.utc),
        status=RunStatus.COMPLETED,
    )
    return RunSummary(run=run_obj, n_success=12, n_failed=0)


def test_run_dry_run_does_not_call_orchestrator(project_root: Path, tmp_path: Path) -> None:
    with patch("geo_analyzer.cli.orchestrator_run") as mock_run:
        mock_run.side_effect = AssertionError("orchestrator should not be called for --dry-run")
        result = runner.invoke(
            app,
            [
                "run", "--dry-run",
                "--catalog-dir", str(project_root / "catalog"),
                "--data-dir", str(tmp_path),
                "--tier", "L1",
                "--model", "openai:gpt-5.1:ungrounded",
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
                "--catalog-dir", str(project_root / "catalog"),
                "--data-dir", str(tmp_path),
                "--tier", "L1",
                "--model", "openai:gpt-5.1:ungrounded",
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
            "--catalog-dir", str(project_root / "catalog"),
            "--data-dir", str(tmp_path),
            "--tier", "L1",
            "--model", "openai:gpt-5.1:ungrounded",
            "--yes",
        ],
        env={},
    )
    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in (result.stdout or "")


def test_run_cost_gate_blocks_without_yes(project_root: Path, tmp_path: Path) -> None:
    # Pretend the estimated cost is huge — patch the estimator to return $999.
    with patch("geo_analyzer.cli._estimate_run_cost", return_value=999.0), \
         patch("geo_analyzer.cli.orchestrator_run") as mock_run:
        result = runner.invoke(
            app,
            [
                "run",
                "--catalog-dir", str(project_root / "catalog"),
                "--data-dir", str(tmp_path),
                "--tier", "L1",
                "--model", "openai:gpt-5.1:ungrounded",
                # No --yes: should require confirmation, which CliRunner can't supply
            ],
            env={"OPENAI_API_KEY": "sk-test"},
            input="n\n",  # decline confirmation
        )
    assert result.exit_code != 0
    mock_run.assert_not_called()
```

- [ ] **Step 11.2: Run, verify fail**

```bash
uv run pytest tests/test_cli_run.py -v
```

- [ ] **Step 11.3: Extend `cli.py`**

Add the following imports near the existing imports (don't remove existing):

```python
from datetime import date

from geo_analyzer.providers.pricing import estimate_cost
from geo_analyzer.runner import RunSummary, expand_matrix, filter_catalog
from geo_analyzer.runner.orchestrator import run as orchestrator_run
from geo_analyzer.runtime import RunTrigger
```

Add a constant near `_API_KEY_ENV`:

```python
_DEFAULT_COST_WARNING_USD = 5.0
_DRY_RUN_AVG_TOKENS_IN = 100
_DRY_RUN_AVG_TOKENS_OUT = 200
```

Add the `run` command after the existing `probe` command:

```python
@app.command("run")
def run_command(
    catalog_dir: Annotated[Path, typer.Option("--catalog-dir")] = Path("catalog"),
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    trigger: Annotated[
        str,
        typer.Option("--trigger", help="manual | launchd-weekly | launchd-monthly | ci"),
    ] = "manual",
    tiers: Annotated[
        list[str] | None,
        typer.Option("--tier", help="Filter to specific tiers (repeatable). Default: all."),
    ] = None,
    subjects: Annotated[
        list[str] | None,
        typer.Option("--subject", help="Filter prompts to those targeting these subject ids (repeatable)."),
    ] = None,
    model_ids: Annotated[
        list[str] | None,
        typer.Option("--model", help="Filter to specific model ids (repeatable). Default: all active."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print matrix size + estimated cost; do not call providers."),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume any partial run on the same date."),
    ] = True,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip cost confirmation prompt."),
    ] = False,
) -> None:
    """Run the full catalog × matrix and persist artifacts under data/runs/<run-id>/."""
    load_dotenv()

    try:
        full_catalog = load_catalog(catalog_dir)
    except (CatalogError, ValidationError) as e:
        _err_console.print(f"[red]catalog error:[/red] {e}")
        raise typer.Exit(code=1) from e

    cat = filter_catalog(
        full_catalog,
        tiers=tiers,  # type: ignore[arg-type]
        subjects=subjects,
        model_ids=model_ids,
    )
    if not cat.prompts or not cat.models:
        _err_console.print("[red]error:[/red] filter produced empty matrix (no prompts or no models).")
        raise typer.Exit(code=1)

    pending = expand_matrix(cat, run_id="dry-run")  # run_id only used in PendingTask
    estimated_cost = _estimate_run_cost(cat, n_tasks=len(pending))

    _console.print(
        f"[bold]matrix:[/bold] prompts={len(cat.prompts)} models={len(cat.models)} "
        f"tasks={len(pending)} est_cost=${estimated_cost:.2f}"
    )

    if dry_run:
        return

    if estimated_cost > _DEFAULT_COST_WARNING_USD and not yes:
        confirm = typer.confirm(
            f"Estimated cost ${estimated_cost:.2f} exceeds threshold "
            f"${_DEFAULT_COST_WARNING_USD:.2f}. Continue?",
            default=False,
        )
        if not confirm:
            _err_console.print("[yellow]aborted.[/yellow]")
            raise typer.Exit(code=1)

    api_keys: dict[str, str] = {}
    for provider_name in cat.providers:
        env_var = _API_KEY_ENV.get(provider_name)
        if env_var is None:
            _err_console.print(f"[red]error:[/red] no API key env var configured for {provider_name!r}.")
            raise typer.Exit(code=1)
        key = os.environ.get(env_var)
        if not key:
            _err_console.print(f"[red]error:[/red] {env_var} is not set.")
            raise typer.Exit(code=1)
        api_keys[provider_name] = key

    summary: RunSummary = asyncio.run(
        orchestrator_run(
            catalog=cat,
            data_dir=data_dir,
            run_date=date.today(),
            trigger=cast(RunTrigger, trigger),
            api_keys=api_keys,
            resume=resume,
        )
    )

    _console.print(
        f"[green]done[/green] run_id={summary.run.id} "
        f"success={summary.n_success} failed={summary.n_failed}"
    )
    if summary.n_failed > 0:
        raise typer.Exit(code=2)


def _estimate_run_cost(catalog: Any, *, n_tasks: int) -> float:
    """Crude estimate using DEFAULT_AVG_TOKENS_{IN,OUT} and the pricing table."""
    total = 0.0
    # Average across all active models — every task picks one of them; this is
    # a back-of-the-envelope estimate, not billing.
    if not catalog.models:
        return 0.0
    per_task = sum(
        estimate_cost(m.model_name, tokens_in=_DRY_RUN_AVG_TOKENS_IN, tokens_out=_DRY_RUN_AVG_TOKENS_OUT)
        for m in catalog.models
    ) / len(catalog.models)
    total = per_task * n_tasks
    return total
```

You also need to add `cast` to the existing imports if not already there:

```python
from typing import Annotated, cast
```

- [ ] **Step 11.4: Run + lint + manual smoke**

```bash
uv run pytest tests/test_cli_run.py -v
uv run pytest -q
uv run pyright
uv run ruff check src tests
```

Manual smoke (no API calls):

```bash
uv run geo-analyzer run --dry-run --tier L1 --model openai:gpt-5.1:ungrounded
```

Expected: prints matrix size + estimated cost, exits 0.

- [ ] **Step 11.5: Commit**

```bash
git add experiments/geo-analyzer/src/geo_analyzer/cli.py \
        experiments/geo-analyzer/tests/test_cli_run.py
git commit -m "geo-analyzer: 'run' CLI command with dry-run, filters, cost gate"
```

---

### Task 12: README + final pass

**Files:**
- Modify: `experiments/geo-analyzer/README.md`

- [ ] **Step 12.1: Run full suite + lint + typecheck**

```bash
uv run pytest -q
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests
```

If `format --check` reports diffs, run `uv run ruff format src tests`.

- [ ] **Step 12.2: Verify CLI surface**

```bash
uv run geo-analyzer --help
uv run geo-analyzer run --help
uv run geo-analyzer run --dry-run --tier L1 --model openai:gpt-5.1:ungrounded
```

- [ ] **Step 12.3: Optional real-API smoke**

If keys are set:

```bash
uv run geo-analyzer run --tier L4 --model openai:gpt-5.1:ungrounded --yes
ls data/runs/$(date +%Y-%m-%d)-manual/
```

Expected: directory contains `manifest.json`, `tasks.jsonl`, `tasks.csv`, `scores.jsonl`, `scores.csv`.

- [ ] **Step 12.4: Update README**

Replace the "## Status (Phase 2 complete)" section heading with "## Status (Phase 3 complete)" and replace its body with:

```markdown
## Status (Phase 3 complete)

Working today:

- `uv run geo-analyzer catalog validate` — loads + cross-checks the seed catalog.
- `uv run geo-analyzer probe "<prompt>" --model <id>` — single prompt against any provider.
- `uv run geo-analyzer probe ... --sensitivity-samples N --temperature T` — N samples
  to inspect generation variance (per DESIGN §5.7).
- `uv run geo-analyzer run [filters] [--dry-run] [--yes]` — execute the full
  catalog × matrix concurrently and persist artifacts under `data/runs/<run-id>/`:
  - `manifest.json` (run metadata + catalog snapshot)
  - `tasks.jsonl` / `tasks.csv` (one row per task; full response in JSONL only)
  - `scores.jsonl` / `scores.csv` (per (prompt, model, subject, metric) — both
    binary and rate variants for grounded N=3 cohorts)
  - `--dry-run` prints the matrix size and estimated cost without calling providers.
  - Cost gate at $5 (override with `--yes`).
  - Resume on partial runs by default (`--no-resume` to start clean).
- Three provider adapters with grounded + ungrounded modes; OpenAI grounded uses
  the Responses API.
- Deterministic scoring (mention presence, ordinal rank, share of voice,
  brand-legacy conflation, citations) and N=3 sample aggregation.

Not yet (Phase 4):

- `geo-analyzer report` / `summary.md` — human dashboard
- `geo-analyzer status` — goal traffic-lights
- Multi-run trend reports
- launchd plist / GitHub Actions CI workflow (Phase 5)
```

- [ ] **Step 12.5: Commit**

```bash
git add experiments/geo-analyzer/README.md
git commit -m "geo-analyzer: document Phase 3 surface area in README"
```

---

## Done state for Phase 3

When all tasks are committed and green:

1. `uv run geo-analyzer run --dry-run --tier L1` prints the matrix size and estimated cost and exits 0.
2. `uv run geo-analyzer run --tier L4 --model openai:gpt-5.1:ungrounded --yes` (with API key set) produces a populated `data/runs/<today>-manual/` directory with manifest, tasks.jsonl, scores.jsonl, tasks.csv, scores.csv.
3. Re-running the same command resumes — no provider calls are made for tasks already complete.
4. `uv run pytest` is green with ~150 tests.
5. `uv run pyright` clean.
6. `uv run ruff check` and `ruff format --check` clean.

Phase 4 picks up here: `summary.md` generator, goal traffic-light status, multi-run `report --since` aggregation.
