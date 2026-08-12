# GEO Analyzer — Phase 4: Reports + Goals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `uv run geo-analyzer report` produce a one-page `summary.md` for any run (top-line scoreboard, worst prompts, funnel, grounded-vs-ungrounded gap, goal traffic-lights, cost & runtime), and make `uv run geo-analyzer status` give a goal scorecard against the latest run — so a Convictional teammate can open the markdown after each weekly run and see the headline numbers without ever looking at JSONL.

**Architecture:** New package `geo_analyzer.reports` containing pure functions (no I/O beyond reading the local run dir): `loader.py` (read scores/tasks back into typed models, list/resolve runs), `topline.py`, `worst_prompts.py`, `funnel.py`, `grounded_gap.py`, `goals.py`, `multi_run.py`, plus `summary.py` (assembles markdown sections into the final document). The CLI gets two new subcommands (`report`, `status`). All section computers take `(scores, catalog)` or similar focused inputs — they don't know about disks. The `summary.py` renderer assembles them into the final markdown. Tests use `tmp_path` plus the seed catalog plus hand-built `Score` lists; no real run directories are required.

**Tech Stack:** Python 3.13, plus everything from Phases 1-3.

**Out of scope for Phase 4 (Phase 5):**

- launchd plist examples
- GitHub Actions CI workflow (`.github/workflows/geo-analyzer-pr.yml`)
- Any change to `geo-analyzer run`'s execution path

---

## File Structure

Files created or modified:

```
experiments/geo-analyzer/
├── src/geo_analyzer/
│   ├── reports/
│   │   ├── __init__.py                     # public API
│   │   ├── loader.py                       # read scores.jsonl back; list runs; latest run
│   │   ├── topline.py                      # per-subject scoreboard rows
│   │   ├── worst_prompts.py                # per-metric prompt ranking
│   │   ├── funnel.py                       # per-subject tier progression + sparkline
│   │   ├── grounded_gap.py                 # grounded vs ungrounded gap
│   │   ├── goals.py                        # traffic-light evaluator (linear interpolation)
│   │   ├── multi_run.py                    # --since aggregation across runs
│   │   └── summary.py                      # assemble summary.md from sections
│   └── cli.py                              # add `report` and `status` subcommands
└── tests/
    ├── test_reports_loader.py
    ├── test_reports_topline.py
    ├── test_reports_worst_prompts.py
    ├── test_reports_funnel.py
    ├── test_reports_grounded_gap.py
    ├── test_reports_goals.py
    ├── test_reports_multi_run.py
    ├── test_reports_summary.py
    ├── test_cli_report.py
    └── test_cli_status.py
```

The `reports/` package mirrors `runner/`'s organization: each section is its own module so the controller and analyst can both reason about pieces independently.

---

## Working Directory

`uv run` from `/Users/billtarbell/Code/decide/experiments/geo-analyzer/`. Branch `geo-analyzer-implementation-v1` — stack on Phases 1-3.

---

### Task 1: Score loader + run discovery

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/reports/__init__.py`
- Create: `experiments/geo-analyzer/src/geo_analyzer/reports/loader.py`
- Create: `experiments/geo-analyzer/tests/test_reports_loader.py`

Reads `scores.jsonl` back into `list[Score]`, lists run ids in chronological order, resolves "latest". Phase 3 already has `read_jsonl_dicts` and `read_tasks_jsonl` — this task adds a `read_scores_jsonl` peer plus run discovery helpers.

- [ ] **Step 1.1: Write failing tests**

Create `tests/test_reports_loader.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from geo_analyzer.reports.loader import (
    latest_run_id,
    list_run_ids,
    read_scores_jsonl,
)
from geo_analyzer.runtime import Score
from geo_analyzer.storage import append_jsonl, run_paths_for


def _score(prompt_id: str = "p", metric: str = "mention_presence") -> Score:
    return Score(
        run_id="r",
        prompt_id=prompt_id,
        model_id="openai:gpt-5.1:ungrounded",
        subject_id="convictional_brand",
        metric=metric,
        value=True,
        scoring_method="deterministic",
        sample_aggregation="single",
    )


class TestReadScoresJsonl:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.jsonl"
        append_jsonl(path, _score("p1").model_dump(mode="json"))
        append_jsonl(path, _score("p2", "share_of_voice").model_dump(mode="json"))
        loaded = read_scores_jsonl(path)
        assert len(loaded) == 2
        assert {s.prompt_id for s in loaded} == {"p1", "p2"}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_scores_jsonl(tmp_path / "nope.jsonl") == []


class TestListRunIds:
    def test_returns_sorted_ids(self, tmp_path: Path) -> None:
        for run_id in ("2026-04-29-manual", "2026-04-22-launchd-weekly", "2026-04-15-manual"):
            run_paths_for(tmp_path, run_id).ensure()
        ids = list_run_ids(tmp_path)
        # Sorted ascending so "latest" is last.
        assert ids == ["2026-04-15-manual", "2026-04-22-launchd-weekly", "2026-04-29-manual"]

    def test_empty_data_dir_returns_empty(self, tmp_path: Path) -> None:
        assert list_run_ids(tmp_path) == []

    def test_skips_non_directories(self, tmp_path: Path) -> None:
        run_paths_for(tmp_path, "2026-04-29-manual").ensure()
        # A stray file under runs/ should be ignored.
        (tmp_path / "runs" / "stray.txt").write_text("noise")
        ids = list_run_ids(tmp_path)
        assert ids == ["2026-04-29-manual"]


class TestLatestRunId:
    def test_returns_most_recent(self, tmp_path: Path) -> None:
        for run_id in ("2026-04-15-manual", "2026-04-29-manual"):
            run_paths_for(tmp_path, run_id).ensure()
        assert latest_run_id(tmp_path) == "2026-04-29-manual"

    def test_no_runs_returns_none(self, tmp_path: Path) -> None:
        assert latest_run_id(tmp_path) is None
```

- [ ] **Step 1.2: Run, verify they fail**

```bash
uv run pytest tests/test_reports_loader.py -v
```

- [ ] **Step 1.3: Implement `loader.py`**

```python
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
```

- [ ] **Step 1.4: Implement `reports/__init__.py`**

```python
"""Reports: load run artifacts, compute sections, render summary.md."""

from geo_analyzer.reports.loader import (
    latest_run_id,
    list_run_ids,
    read_scores_jsonl,
)

__all__ = ["latest_run_id", "list_run_ids", "read_scores_jsonl"]
```

- [ ] **Step 1.5: Run + lint + commit**

```bash
uv run pytest tests/test_reports_loader.py -v
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests

git add experiments/geo-analyzer/src/geo_analyzer/reports/ \
        experiments/geo-analyzer/tests/test_reports_loader.py
git commit -m "geo-analyzer: scores.jsonl loader + run discovery helpers"
```

If denied: report diff.

---

### Task 2: Top-line metrics rollup

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/reports/topline.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/reports/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_reports_topline.py`

DESIGN §9.1 section 1 — per-subject scoreboard. For each `(subject_id, metric)` pair, emit:
- **prompt-level rate** — fraction of (prompt × model) cohorts where the binary metric is True (computed from `_presence` / `_conflation` style boolean scores)
- **interaction-level rate** — mean of `_rate` scores (which already encode "fraction of samples where it fired")
- **mean** — for non-binary metrics like `share_of_voice` or `ordinal_rank` (median of ints, mean of floats)
- **n** — number of cohorts contributing

The renderer (Task 8) decides which fields to surface for which metric.

- [ ] **Step 2.1: Write failing tests**

```python
# tests/test_reports_topline.py
from __future__ import annotations

from geo_analyzer.reports.topline import TopLineRow, compute_topline
from geo_analyzer.runtime import Score


def _score(metric: str, value: bool | float | int | None,
           subject: str = "convictional_brand", prompt: str = "p1",
           model: str = "openai:gpt-5.1:ungrounded",
           agg: str = "single") -> Score:
    return Score(
        run_id="r", prompt_id=prompt, model_id=model, subject_id=subject,
        metric=metric, value=value, scoring_method="deterministic",
        sample_aggregation=agg,  # type: ignore[arg-type]
    )


class TestComputeTopline:
    def test_binary_metric_prompt_level_rate(self) -> None:
        # 3 cohorts: 2 True, 1 False → prompt-level rate = 2/3.
        scores = [
            _score("mention_presence", True, prompt="p1"),
            _score("mention_presence", True, prompt="p2"),
            _score("mention_presence", False, prompt="p3"),
        ]
        rows = compute_topline(scores)
        row = next(r for r in rows if r.metric == "mention_presence")
        assert row.prompt_level_rate == pytest_approx(2 / 3)
        assert row.n == 3

    def test_rate_metric_interaction_level(self) -> None:
        # _rate metrics: mean across cohorts.
        scores = [
            _score("mention_presence_rate", 1.0, prompt="p1", agg="mean"),
            _score("mention_presence_rate", 0.5, prompt="p2", agg="mean"),
        ]
        rows = compute_topline(scores)
        row = next(r for r in rows if r.metric == "mention_presence_rate")
        assert row.interaction_level_rate == pytest_approx(0.75)

    def test_share_of_voice_mean(self) -> None:
        scores = [
            _score("share_of_voice", 0.3),
            _score("share_of_voice", 0.5),
            _score("share_of_voice", None),  # None values skipped from the mean
        ]
        rows = compute_topline(scores)
        row = next(r for r in rows if r.metric == "share_of_voice")
        assert row.mean_value == pytest_approx(0.4)

    def test_groups_by_subject_and_metric(self) -> None:
        scores = [
            _score("mention_presence", True, subject="convictional_brand"),
            _score("mention_presence", False, subject="lattice"),
        ]
        rows = compute_topline(scores)
        keys = {(r.subject_id, r.metric) for r in rows}
        assert keys == {("convictional_brand", "mention_presence"),
                        ("lattice", "mention_presence")}

    def test_empty_scores_returns_empty(self) -> None:
        assert compute_topline([]) == []


# `pytest_approx` is just pytest.approx aliased to keep tests terse.
import pytest
pytest_approx = pytest.approx
```

- [ ] **Step 2.2: Run, fail**

```bash
uv run pytest tests/test_reports_topline.py -v
```

- [ ] **Step 2.3: Implement `topline.py`**

```python
"""Per-subject top-line scoreboard.

For each (subject_id, metric) pair across the run, compute:
  - prompt_level_rate: for binary metrics (mention_presence, brand_legacy_conflation),
    fraction of (prompt × model) cohorts that fired True.
  - interaction_level_rate: for the matching `_rate` metrics, mean across cohorts.
  - mean_value: for share_of_voice and other float-valued metrics, mean across
    non-None cohort values; for ordinal_rank (int|None), mean of non-None values.
  - n: cohort count.
The renderer chooses which to surface for which metric (see DESIGN §9.1).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from geo_analyzer.runtime import Score

_BINARY_METRICS = {"mention_presence", "brand_legacy_conflation"}
_RATE_METRICS = {"mention_presence_rate", "brand_legacy_conflation_rate"}


@dataclass(frozen=True)
class TopLineRow:
    subject_id: str
    metric: str
    n: int
    prompt_level_rate: float | None = None
    """Fraction of cohorts where binary metric == True (None for non-binary)."""
    interaction_level_rate: float | None = None
    """Mean of _rate metric values across cohorts (None for non-rate metrics)."""
    mean_value: float | None = None
    """Mean of non-None float values; mean of non-None int values for ranks."""


def compute_topline(scores: list[Score]) -> list[TopLineRow]:
    """Group scores by (subject_id, metric) and compute aggregates."""
    grouped: dict[tuple[str, str], list[Score]] = defaultdict(list)
    for s in scores:
        grouped[(s.subject_id, s.metric)].append(s)

    rows: list[TopLineRow] = []
    for (subject_id, metric), bucket in grouped.items():
        n = len(bucket)
        prompt_rate: float | None = None
        interaction_rate: float | None = None
        mean: float | None = None

        if metric in _BINARY_METRICS:
            trues = sum(1 for s in bucket if s.value is True)
            prompt_rate = trues / n if n > 0 else None
        elif metric in _RATE_METRICS:
            float_values = [float(s.value) for s in bucket if isinstance(s.value, (int, float)) and not isinstance(s.value, bool)]
            interaction_rate = sum(float_values) / len(float_values) if float_values else None
        else:
            # share_of_voice, ordinal_rank, etc. — mean of non-None numeric values.
            numeric = [
                float(s.value)
                for s in bucket
                if isinstance(s.value, (int, float)) and not isinstance(s.value, bool)
            ]
            mean = sum(numeric) / len(numeric) if numeric else None

        rows.append(TopLineRow(
            subject_id=subject_id,
            metric=metric,
            n=n,
            prompt_level_rate=prompt_rate,
            interaction_level_rate=interaction_rate,
            mean_value=mean,
        ))
    return rows
```

- [ ] **Step 2.4: Update `reports/__init__.py`**

Add `TopLineRow` and `compute_topline` to the imports + `__all__`.

- [ ] **Step 2.5: Run + lint + commit**

```bash
uv run pytest tests/test_reports_topline.py -v
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests

git add experiments/geo-analyzer/src/geo_analyzer/reports/ \
        experiments/geo-analyzer/tests/test_reports_topline.py
git commit -m "geo-analyzer: top-line per-subject metrics rollup"
```

---

### Task 3: Worst prompts ranker

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/reports/worst_prompts.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/reports/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_reports_worst_prompts.py`

DESIGN §9.1 section 2 — for each binary metric (mention_presence, brand_legacy_conflation), rank prompts by their interaction-level rate descending. Surfaces specific topics where the model misframes most often.

- [ ] **Step 3.1: Write failing tests**

```python
# tests/test_reports_worst_prompts.py
from __future__ import annotations

import pytest

from geo_analyzer.reports.worst_prompts import (
    WorstPromptRow,
    compute_worst_prompts,
)
from geo_analyzer.runtime import Score


def _rate_score(prompt: str, value: float, metric: str = "mention_presence_rate",
                subject: str = "convictional_brand",
                model: str = "openai:gpt-5.1:grounded") -> Score:
    return Score(
        run_id="r", prompt_id=prompt, model_id=model, subject_id=subject,
        metric=metric, value=value, scoring_method="deterministic",
        sample_aggregation="mean",
    )


class TestComputeWorstPrompts:
    def test_orders_by_rate_descending(self) -> None:
        # Two prompts: p1 has higher rate. Expect p1 first.
        scores = [
            _rate_score("p1", 0.9),
            _rate_score("p2", 0.3),
        ]
        worst = compute_worst_prompts(
            scores,
            metric="mention_presence_rate",
            subject_id="convictional_brand",
            top_n=10,
        )
        assert [w.prompt_id for w in worst] == ["p1", "p2"]

    def test_aggregates_across_models_per_prompt(self) -> None:
        # Two models, same prompt — mean across them.
        scores = [
            _rate_score("p1", 0.4, model="openai:gpt-5.1:grounded"),
            _rate_score("p1", 0.6, model="anthropic:claude-opus-4-7:grounded"),
        ]
        worst = compute_worst_prompts(
            scores,
            metric="mention_presence_rate",
            subject_id="convictional_brand",
            top_n=10,
        )
        assert len(worst) == 1
        assert worst[0].prompt_id == "p1"
        assert worst[0].rate == pytest.approx(0.5)
        assert worst[0].n_models == 2

    def test_top_n_truncates(self) -> None:
        scores = [_rate_score(f"p{i}", i * 0.1) for i in range(1, 6)]
        worst = compute_worst_prompts(
            scores,
            metric="mention_presence_rate",
            subject_id="convictional_brand",
            top_n=3,
        )
        assert len(worst) == 3
        # Highest 3 rates: 0.5, 0.4, 0.3
        assert [w.prompt_id for w in worst] == ["p5", "p4", "p3"]

    def test_filters_by_subject(self) -> None:
        scores = [
            _rate_score("p1", 0.9, subject="convictional_brand"),
            _rate_score("p1", 0.3, subject="lattice"),
        ]
        worst = compute_worst_prompts(
            scores,
            metric="mention_presence_rate",
            subject_id="convictional_brand",
            top_n=10,
        )
        assert len(worst) == 1
        assert worst[0].rate == pytest.approx(0.9)

    def test_no_matching_scores_returns_empty(self) -> None:
        assert compute_worst_prompts(
            [], metric="mention_presence_rate",
            subject_id="convictional_brand", top_n=5,
        ) == []
```

- [ ] **Step 3.2: Run, fail**

```bash
uv run pytest tests/test_reports_worst_prompts.py -v
```

- [ ] **Step 3.3: Implement `worst_prompts.py`**

```python
"""Per-metric prompt ranking. Highest interaction-level rate first."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from geo_analyzer.runtime import Score


@dataclass(frozen=True)
class WorstPromptRow:
    prompt_id: str
    subject_id: str
    metric: str
    rate: float
    """Mean of the _rate metric across all models that scored this prompt."""
    n_models: int


def compute_worst_prompts(
    scores: list[Score],
    *,
    metric: str,
    subject_id: str,
    top_n: int,
) -> list[WorstPromptRow]:
    """Return up to top_n prompts ordered by interaction-level rate desc.

    Aggregates across models for each prompt by taking the mean of the rate values.
    """
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        if s.metric != metric or s.subject_id != subject_id:
            continue
        if isinstance(s.value, bool) or not isinstance(s.value, (int, float)):
            continue
        by_prompt[s.prompt_id].append(float(s.value))

    rows = [
        WorstPromptRow(
            prompt_id=prompt_id,
            subject_id=subject_id,
            metric=metric,
            rate=sum(values) / len(values),
            n_models=len(values),
        )
        for prompt_id, values in by_prompt.items()
    ]
    rows.sort(key=lambda r: r.rate, reverse=True)
    return rows[:top_n]
```

- [ ] **Step 3.4: Update `__init__.py`**

Add `WorstPromptRow` and `compute_worst_prompts` to imports + `__all__`.

- [ ] **Step 3.5: Run + lint + commit**

```bash
uv run pytest tests/test_reports_worst_prompts.py -v
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests

git add experiments/geo-analyzer/src/geo_analyzer/reports/ \
        experiments/geo-analyzer/tests/test_reports_worst_prompts.py
git commit -m "geo-analyzer: worst-prompts ranking by interaction-level rate"
```

---

### Task 4: Funnel sparkline (per-subject tier progression)

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/reports/funnel.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/reports/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_reports_funnel.py`

DESIGN §9.1 section 3 — for each subject, mean mention_presence rate at each tier (L1, L2, L3, L4). Renders as an ASCII sparkline. The funnel shape tells you where in the prospect's funnel the model surfaces you.

- [ ] **Step 4.1: Write failing tests**

```python
# tests/test_reports_funnel.py
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from geo_analyzer.catalog import load_catalog
from geo_analyzer.reports.funnel import (
    FunnelTier,
    compute_funnel,
    render_sparkline,
)
from geo_analyzer.runtime import Score
from geo_analyzer.types import Catalog


@pytest.fixture(scope="module")
def real_catalog() -> Catalog:
    project_root = Path(__file__).resolve().parents[1]
    return load_catalog(project_root / "catalog")


def _binary_score(prompt_id: str, value: bool,
                  subject: str = "convictional_brand") -> Score:
    return Score(
        run_id="r", prompt_id=prompt_id,
        model_id="openai:gpt-5.1:ungrounded",
        subject_id=subject, metric="mention_presence", value=value,
        scoring_method="deterministic", sample_aggregation="single",
    )


class TestComputeFunnel:
    def test_tiers_use_catalog_lookup(self, real_catalog: Catalog) -> None:
        # Use real prompt ids from the seed catalog so tier lookup works.
        # L1: prompt.broad.l1.companies-run-poorly
        # L4: prompt.brand.l4.what-is-convictional
        scores = [
            _binary_score("prompt.broad.l1.companies-run-poorly", False),
            _binary_score("prompt.brand.l4.what-is-convictional", True),
        ]
        funnel = compute_funnel(scores, real_catalog, subject_id="convictional_brand")
        # FunnelTier values should reflect mean presence per tier
        l1 = next(t for t in funnel if t.tier == "L1")
        l4 = next(t for t in funnel if t.tier == "L4")
        assert l1.rate == 0.0
        assert l4.rate == 1.0

    def test_skips_subjects_with_no_scores(self, real_catalog: Catalog) -> None:
        funnel = compute_funnel([], real_catalog, subject_id="convictional_brand")
        # All tiers present but rate=None for empty cohorts.
        tiers = {t.tier: t.rate for t in funnel}
        assert tiers == {"L1": None, "L2": None, "L3": None, "L4": None}

    def test_unknown_prompt_id_skipped(self, real_catalog: Catalog) -> None:
        scores = [_binary_score("prompt.does.not.exist", True)]
        funnel = compute_funnel(scores, real_catalog, subject_id="convictional_brand")
        # Unknown prompt → no tier, no contribution. All rates None.
        assert all(t.rate is None for t in funnel)


class TestRenderSparkline:
    def test_basic_progression(self) -> None:
        # Bars proportional to value 0..1. Same-length input/output.
        bars = render_sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
        assert len(bars) == 5

    def test_none_renders_as_blank(self) -> None:
        bars = render_sparkline([None, 0.5, None])
        # Blank for None — exact char depends on implementation but must be
        # distinguishable from filled bars.
        assert len(bars) == 3
```

- [ ] **Step 4.2: Run, fail**

```bash
uv run pytest tests/test_reports_funnel.py -v
```

- [ ] **Step 4.3: Implement `funnel.py`**

```python
"""Per-subject funnel: mean mention_presence rate across L1->L4 tiers.

A healthy `convictional_brand` curve climbs from L3 to L4; a healthy
`organizational_health_category` curve rises at L2/L3 over time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import get_args

from geo_analyzer.runtime import Score
from geo_analyzer.types import Catalog, PromptTier

# Block-character bars for sparkline; index 0 = lowest, index 7 = highest.
_BARS = "▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class FunnelTier:
    tier: PromptTier
    rate: float | None
    n: int
    """Number of (prompt × model) cohorts contributing."""


def compute_funnel(
    scores: list[Score],
    catalog: Catalog,
    *,
    subject_id: str,
) -> list[FunnelTier]:
    """Return one FunnelTier per L1..L4 (always all four, even when empty)."""
    prompt_tier: dict[str, PromptTier] = {p.id: p.tier for p in catalog.prompts}

    by_tier: dict[PromptTier, list[bool]] = defaultdict(list)
    for s in scores:
        if s.subject_id != subject_id:
            continue
        if s.metric != "mention_presence":
            continue
        if not isinstance(s.value, bool):
            continue
        tier = prompt_tier.get(s.prompt_id)
        if tier is None:
            continue
        by_tier[tier].append(s.value)

    out: list[FunnelTier] = []
    for tier in get_args(PromptTier):  # ("L1", "L2", "L3", "L4") in order
        bucket = by_tier.get(tier, [])
        rate = sum(1 for v in bucket if v) / len(bucket) if bucket else None
        out.append(FunnelTier(tier=tier, rate=rate, n=len(bucket)))
    return out


def render_sparkline(values: list[float | None]) -> str:
    """Render a list of [0, 1] values as a unicode block sparkline. None → space."""
    chars: list[str] = []
    for v in values:
        if v is None:
            chars.append(" ")
            continue
        clamped = max(0.0, min(1.0, v))
        idx = int(round(clamped * (len(_BARS) - 1)))
        chars.append(_BARS[idx])
    return "".join(chars)
```

- [ ] **Step 4.4: Update `__init__.py`**

Add `FunnelTier`, `compute_funnel`, `render_sparkline` to imports + `__all__`.

- [ ] **Step 4.5: Run + lint + commit**

```bash
uv run pytest tests/test_reports_funnel.py -v
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests

git add experiments/geo-analyzer/src/geo_analyzer/reports/ \
        experiments/geo-analyzer/tests/test_reports_funnel.py
git commit -m "geo-analyzer: per-subject funnel sparkline (L1-L4)"
```

---

### Task 5: Grounded vs ungrounded gap

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/reports/grounded_gap.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/reports/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_reports_grounded_gap.py`

DESIGN §9.1 section 5. For each (prompt, model_name, subject, metric) tuple, compare grounded vs ungrounded values. The model id encodes the mode in its third segment — `openai:gpt-5.1:grounded` vs `:ungrounded`.

A model_name's "stem" pairs grounded ↔ ungrounded. Compute `gap = grounded_value - ungrounded_value`. Return top-N by absolute gap.

- [ ] **Step 5.1: Write failing tests**

```python
# tests/test_reports_grounded_gap.py
from __future__ import annotations

from geo_analyzer.reports.grounded_gap import (
    GroundedGapRow,
    compute_grounded_gaps,
)
from geo_analyzer.runtime import Score


def _score(prompt: str, model: str, value: float | bool,
           metric: str = "mention_presence_rate",
           subject: str = "convictional_brand") -> Score:
    return Score(
        run_id="r", prompt_id=prompt, model_id=model,
        subject_id=subject, metric=metric, value=value,
        scoring_method="deterministic", sample_aggregation="mean",
    )


class TestComputeGroundedGaps:
    def test_single_pair(self) -> None:
        scores = [
            _score("p1", "openai:gpt-5.1:grounded", 0.8),
            _score("p1", "openai:gpt-5.1:ungrounded", 0.3),
        ]
        gaps = compute_grounded_gaps(scores, top_n=10)
        assert len(gaps) == 1
        g = gaps[0]
        assert g.prompt_id == "p1"
        assert g.subject_id == "convictional_brand"
        assert g.grounded_value == 0.8
        assert g.ungrounded_value == 0.3
        assert g.gap == pytest_approx(0.5)

    def test_orders_by_absolute_gap(self) -> None:
        scores = [
            _score("p_small", "openai:gpt-5.1:grounded", 0.5),
            _score("p_small", "openai:gpt-5.1:ungrounded", 0.4),  # gap = 0.1
            _score("p_big",   "openai:gpt-5.1:grounded", 0.9),
            _score("p_big",   "openai:gpt-5.1:ungrounded", 0.1),  # gap = 0.8
            _score("p_neg",   "openai:gpt-5.1:grounded", 0.1),
            _score("p_neg",   "openai:gpt-5.1:ungrounded", 0.6),  # gap = -0.5
        ]
        gaps = compute_grounded_gaps(scores, top_n=10)
        # Ordered by abs(gap) desc: 0.8, 0.5, 0.1
        assert [g.prompt_id for g in gaps] == ["p_big", "p_neg", "p_small"]

    def test_only_pairs_when_both_modes_present(self) -> None:
        scores = [
            _score("p1", "openai:gpt-5.1:grounded", 0.8),
            # No ungrounded pair → skip.
        ]
        assert compute_grounded_gaps(scores, top_n=10) == []

    def test_top_n_limits(self) -> None:
        scores: list[Score] = []
        for i in range(5):
            scores.append(_score(f"p{i}", "openai:gpt-5.1:grounded", float(i) / 5))
            scores.append(_score(f"p{i}", "openai:gpt-5.1:ungrounded", 0.0))
        gaps = compute_grounded_gaps(scores, top_n=3)
        assert len(gaps) == 3


import pytest
pytest_approx = pytest.approx
```

- [ ] **Step 5.2: Run, fail**

```bash
uv run pytest tests/test_reports_grounded_gap.py -v
```

- [ ] **Step 5.3: Implement `grounded_gap.py`**

```python
"""Grounded vs ungrounded gap per (prompt, model_name, subject, metric).

Pairs scores by stripping the `:grounded`/`:ungrounded` suffix from model_id.
Returns top-N by |gap|; positive gap = grounded > ungrounded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from geo_analyzer.runtime import Score


@dataclass(frozen=True)
class GroundedGapRow:
    prompt_id: str
    model_stem: str
    """Model id with the trailing ':grounded'/':ungrounded' stripped."""
    subject_id: str
    metric: str
    grounded_value: float
    ungrounded_value: float
    gap: float
    """grounded_value - ungrounded_value. Positive means grounded scored higher."""


def _stem_and_mode(model_id: str) -> tuple[str, str] | None:
    parts = model_id.rsplit(":", 1)
    if len(parts) != 2 or parts[1] not in ("grounded", "ungrounded"):
        return None
    return parts[0], parts[1]


def compute_grounded_gaps(scores: list[Score], *, top_n: int) -> list[GroundedGapRow]:
    """Return up to top_n (prompt × model_stem × subject × metric) rows where
    both grounded and ungrounded values exist, ordered by abs(gap) descending."""
    # Group: (prompt_id, model_stem, subject_id, metric) -> {"grounded": v, "ungrounded": v}
    pairs: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for s in scores:
        sm = _stem_and_mode(s.model_id)
        if sm is None:
            continue
        if isinstance(s.value, bool) or not isinstance(s.value, (int, float)):
            continue
        stem, mode = sm
        pairs[(s.prompt_id, stem, s.subject_id, s.metric)][mode] = float(s.value)

    rows: list[GroundedGapRow] = []
    for (prompt_id, stem, subject_id, metric), modes in pairs.items():
        if "grounded" not in modes or "ungrounded" not in modes:
            continue
        g = modes["grounded"]
        u = modes["ungrounded"]
        rows.append(GroundedGapRow(
            prompt_id=prompt_id,
            model_stem=stem,
            subject_id=subject_id,
            metric=metric,
            grounded_value=g,
            ungrounded_value=u,
            gap=g - u,
        ))
    rows.sort(key=lambda r: abs(r.gap), reverse=True)
    return rows[:top_n]
```

- [ ] **Step 5.4: Update `__init__.py`**

Add `GroundedGapRow` and `compute_grounded_gaps` to imports + `__all__`.

- [ ] **Step 5.5: Run + lint + commit**

```bash
uv run pytest tests/test_reports_grounded_gap.py -v
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests

git add experiments/geo-analyzer/src/geo_analyzer/reports/ \
        experiments/geo-analyzer/tests/test_reports_grounded_gap.py
git commit -m "geo-analyzer: grounded vs ungrounded gap analysis"
```

---

### Task 6: Goal traffic-light evaluator

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/reports/goals.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/reports/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_reports_goals.py`

DESIGN §10. Linear interpolation: at fraction `f` of elapsed time between `created_at` and `target_date`, you should be at `f` of the way from baseline to target. v1 simplification: baseline = 0 for `direction=above`, baseline = 1.0 for `direction=below` (note this in code; v2 stores actual baseline).

Status:
- `pending` — `today < goal.created_at`, OR no scores match the goal
- `green` — actual is at or beyond `expected` in the goal direction
- `yellow` — actual has moved from baseline but not enough to be on track
- `red` — actual hasn't moved past baseline (or moved backwards)

- [ ] **Step 6.1: Write failing tests**

```python
# tests/test_reports_goals.py
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from geo_analyzer.catalog import load_catalog
from geo_analyzer.reports.goals import (
    GoalEvaluation,
    GoalStatus,
    evaluate_goal,
)
from geo_analyzer.runtime import Score
from geo_analyzer.types import Catalog, Goal


@pytest.fixture(scope="module")
def real_catalog() -> Catalog:
    project_root = Path(__file__).resolve().parents[1]
    return load_catalog(project_root / "catalog")


def _grow_goal() -> Goal:
    return Goal(
        id="g1",
        subject="convictional_brand",
        metric="mention_presence",
        tier="L1",
        target=0.5,
        direction="above",
        created_at=date(2026, 1, 1),
        target_date=date(2026, 12, 31),
    )


def _shrink_goal() -> Goal:
    return Goal(
        id="g2",
        subject="convictional_legacy_dropship",
        metric="mention_presence",
        tier="L4",
        target=0.05,
        direction="below",
        created_at=date(2026, 1, 1),
        target_date=date(2026, 12, 31),
    )


def _binary(prompt: str, value: bool,
            subject: str = "convictional_brand",
            metric: str = "mention_presence") -> Score:
    return Score(
        run_id="r", prompt_id=prompt,
        model_id="openai:gpt-5.1:ungrounded",
        subject_id=subject, metric=metric, value=value,
        scoring_method="deterministic", sample_aggregation="single",
    )


class TestEvaluateGoal:
    def test_pending_when_no_matching_scores(self, real_catalog: Catalog) -> None:
        result = evaluate_goal(
            _grow_goal(),
            scores=[],
            catalog=real_catalog,
            today=date(2026, 6, 1),
        )
        assert result.status == GoalStatus.PENDING

    def test_pending_when_today_before_created_at(self, real_catalog: Catalog) -> None:
        scores = [_binary("prompt.broad.l1.companies-run-poorly", True)]
        result = evaluate_goal(
            _grow_goal(),
            scores=scores,
            catalog=real_catalog,
            today=date(2025, 12, 1),  # before created_at=2026-01-01
        )
        assert result.status == GoalStatus.PENDING

    def test_grow_goal_green_when_actual_above_expected(self, real_catalog: Catalog) -> None:
        # All 3 L1 prompts present → mean = 1.0; target = 0.5 → already past it.
        scores = [
            _binary("prompt.broad.l1.companies-run-poorly", True),
            _binary("prompt.broad.l1.improve-leadership", True),
            _binary("prompt.broad.l1.signs-unhealthy-org", True),
        ]
        result = evaluate_goal(
            _grow_goal(),
            scores=scores,
            catalog=real_catalog,
            today=date(2026, 6, 1),  # ~5 months in
        )
        assert result.status == GoalStatus.GREEN
        assert result.actual == pytest.approx(1.0)

    def test_grow_goal_red_when_actual_at_baseline(self, real_catalog: Catalog) -> None:
        # All False → actual = 0.0 = baseline. Should be red after 5 months.
        scores = [
            _binary("prompt.broad.l1.companies-run-poorly", False),
            _binary("prompt.broad.l1.improve-leadership", False),
            _binary("prompt.broad.l1.signs-unhealthy-org", False),
        ]
        result = evaluate_goal(
            _grow_goal(),
            scores=scores,
            catalog=real_catalog,
            today=date(2026, 6, 1),
        )
        assert result.status == GoalStatus.RED

    def test_shrink_goal_green_when_actual_low(self, real_catalog: Catalog) -> None:
        # No L4 prompts conflate → actual = 0.0; target = 0.05 → already past.
        scores = [
            _binary("prompt.brand.l4.what-is-convictional", False,
                    subject="convictional_legacy_dropship"),
            _binary("prompt.brand.l4.convictional-product", False,
                    subject="convictional_legacy_dropship"),
            _binary("prompt.brand.l4.convictional-pricing", False,
                    subject="convictional_legacy_dropship"),
        ]
        result = evaluate_goal(
            _shrink_goal(),
            scores=scores,
            catalog=real_catalog,
            today=date(2026, 6, 1),
        )
        assert result.status == GoalStatus.GREEN
```

- [ ] **Step 6.2: Run, fail**

```bash
uv run pytest tests/test_reports_goals.py -v
```

- [ ] **Step 6.3: Implement `goals.py`**

```python
"""Goal traffic-light evaluator. Linear interpolation per DESIGN §10.

v1 simplification: baseline is fixed at 0 (grow) or 1.0 (shrink). v2 should
store the actual value at goal creation time so the interpolation anchors on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from geo_analyzer.runtime import Score
from geo_analyzer.types import Catalog, Goal


class GoalStatus(str, Enum):
    PENDING = "pending"
    """Today is before goal.created_at, or no scores match the goal subject/metric/tier."""
    GREEN = "green"
    """Actual is at or beyond expected (in the goal direction)."""
    YELLOW = "yellow"
    """Actual has moved from baseline but not enough to be on track."""
    RED = "red"
    """Actual hasn't moved past baseline (or moved backwards)."""


@dataclass(frozen=True)
class GoalEvaluation:
    goal: Goal
    status: GoalStatus
    actual: float | None
    """Mean of matching score values (None when pending or no data)."""
    expected: float | None
    """Linear-interpolated expected value at `today`, or None when pending."""


def evaluate_goal(
    goal: Goal,
    *,
    scores: list[Score],
    catalog: Catalog,
    today: date,
) -> GoalEvaluation:
    if today < goal.created_at:
        return GoalEvaluation(goal=goal, status=GoalStatus.PENDING, actual=None, expected=None)

    prompt_tier = {p.id: p.tier for p in catalog.prompts}
    matching = [
        s for s in scores
        if s.subject_id == goal.subject
        and s.metric == goal.metric
        and prompt_tier.get(s.prompt_id) == goal.tier
    ]
    numeric_values: list[float] = []
    for s in matching:
        if isinstance(s.value, bool):
            numeric_values.append(1.0 if s.value else 0.0)
        elif isinstance(s.value, (int, float)):
            numeric_values.append(float(s.value))
    if not numeric_values:
        return GoalEvaluation(goal=goal, status=GoalStatus.PENDING, actual=None, expected=None)

    actual = sum(numeric_values) / len(numeric_values)

    total_days = (goal.target_date - goal.created_at).days
    elapsed_days = (today - goal.created_at).days
    fraction = min(1.0, max(0.0, elapsed_days / total_days)) if total_days > 0 else 1.0

    if goal.direction == "above":
        baseline = 0.0
        expected = baseline + (goal.target - baseline) * fraction
        if actual >= expected:
            status = GoalStatus.GREEN
        elif actual > baseline:
            status = GoalStatus.YELLOW
        else:
            status = GoalStatus.RED
    else:  # below — shrink
        baseline = 1.0
        expected = baseline + (goal.target - baseline) * fraction
        if actual <= expected:
            status = GoalStatus.GREEN
        elif actual < baseline:
            status = GoalStatus.YELLOW
        else:
            status = GoalStatus.RED

    return GoalEvaluation(goal=goal, status=status, actual=actual, expected=expected)
```

- [ ] **Step 6.4: Update `__init__.py`**

Add `GoalEvaluation`, `GoalStatus`, `evaluate_goal` to imports + `__all__`.

- [ ] **Step 6.5: Run + lint + commit**

```bash
uv run pytest tests/test_reports_goals.py -v
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests

git add experiments/geo-analyzer/src/geo_analyzer/reports/ \
        experiments/geo-analyzer/tests/test_reports_goals.py
git commit -m "geo-analyzer: goal traffic-light evaluator (linear interpolation)"
```

---

### Task 7: summary.md renderer

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/reports/summary.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/reports/__init__.py`
- Create: `experiments/geo-analyzer/tests/test_reports_summary.py`

Pulls together topline + worst prompts + funnel + grounded gap + goals + cost & runtime into a single markdown document. The renderer is a deterministic function; the CLI (Task 8) reads run artifacts off disk and feeds them in.

- [ ] **Step 7.1: Write failing tests**

```python
# tests/test_reports_summary.py
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from geo_analyzer.catalog import load_catalog
from geo_analyzer.reports.summary import SummaryInputs, render_summary
from geo_analyzer.runtime import Run, RunStatus, Score, Task, TaskStatus
from geo_analyzer.types import Catalog, Goal


@pytest.fixture(scope="module")
def real_catalog() -> Catalog:
    project_root = Path(__file__).resolve().parents[1]
    return load_catalog(project_root / "catalog")


def _run() -> Run:
    return Run(
        id="2026-04-29-manual",
        trigger="manual",
        started_at=datetime(2026, 4, 29, 9, 0, tzinfo=UTC),
        finished_at=datetime(2026, 4, 29, 9, 5, tzinfo=UTC),
        status=RunStatus.COMPLETED,
    )


def _task(prompt_id: str = "p1") -> Task:
    return Task(
        run_id="r", prompt_id=prompt_id,
        model_id="openai:gpt-5.1:ungrounded",
        sample_n=0, status=TaskStatus.SUCCESS,
        text="hi", tokens_in=10, tokens_out=20,
        cost_usd_estimate=0.001, latency_ms=200,
    )


def _score(metric: str = "mention_presence", value: bool | float = True,
           subject: str = "convictional_brand",
           prompt: str = "prompt.broad.l1.companies-run-poorly") -> Score:
    return Score(
        run_id="r", prompt_id=prompt,
        model_id="openai:gpt-5.1:ungrounded",
        subject_id=subject, metric=metric, value=value,
        scoring_method="deterministic", sample_aggregation="single",
    )


def _goal() -> Goal:
    return Goal(
        id="g1", subject="convictional_brand", metric="mention_presence",
        tier="L1", target=0.5, direction="above",
        created_at=date(2026, 1, 1), target_date=date(2026, 12, 31),
    )


class TestRenderSummary:
    def test_includes_run_id_and_top_line_section(self, real_catalog: Catalog) -> None:
        inputs = SummaryInputs(
            run=_run(),
            tasks=[_task()],
            scores=[_score()],
            catalog=real_catalog,
            goals=[],
            today=date(2026, 4, 29),
        )
        md = render_summary(inputs)
        assert "2026-04-29-manual" in md
        assert "Top line" in md
        assert "convictional_brand" in md

    def test_includes_goal_section_when_goals_present(self, real_catalog: Catalog) -> None:
        inputs = SummaryInputs(
            run=_run(),
            tasks=[_task()],
            scores=[_score()],
            catalog=real_catalog,
            goals=[_goal()],
            today=date(2026, 4, 29),
        )
        md = render_summary(inputs)
        assert "Goal progress" in md
        assert _goal().id in md  # "g1"

    def test_includes_cost_and_runtime(self, real_catalog: Catalog) -> None:
        inputs = SummaryInputs(
            run=_run(),
            tasks=[_task("p1"), _task("p2"), _task("p3")],
            scores=[],
            catalog=real_catalog,
            goals=[],
            today=date(2026, 4, 29),
        )
        md = render_summary(inputs)
        assert "Cost" in md or "cost" in md
        assert "tasks" in md

    def test_handles_empty_scores(self, real_catalog: Catalog) -> None:
        inputs = SummaryInputs(
            run=_run(),
            tasks=[],
            scores=[],
            catalog=real_catalog,
            goals=[],
            today=date(2026, 4, 29),
        )
        md = render_summary(inputs)
        assert "2026-04-29-manual" in md
```

- [ ] **Step 7.2: Run, fail**

```bash
uv run pytest tests/test_reports_summary.py -v
```

- [ ] **Step 7.3: Implement `summary.py`**

```python
"""Assemble summary.md from analytical sections.

Pure function over typed inputs (Run + tasks + scores + catalog + goals + today).
The CLI (Task 8) reads disks and calls this; the renderer never touches I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from geo_analyzer.reports.funnel import compute_funnel, render_sparkline
from geo_analyzer.reports.goals import GoalStatus, evaluate_goal
from geo_analyzer.reports.grounded_gap import compute_grounded_gaps
from geo_analyzer.reports.topline import compute_topline
from geo_analyzer.reports.worst_prompts import compute_worst_prompts
from geo_analyzer.runtime import Run, Score, Task, TaskStatus
from geo_analyzer.types import Catalog, Goal


@dataclass(frozen=True)
class SummaryInputs:
    run: Run
    tasks: list[Task]
    scores: list[Score]
    catalog: Catalog
    goals: list[Goal]
    today: date


def render_summary(inputs: SummaryInputs) -> str:
    sections: list[str] = []
    sections.append(_header(inputs.run))
    sections.append(_top_line(inputs))
    sections.append(_funnel_section(inputs))
    sections.append(_worst_prompts_section(inputs))
    sections.append(_grounded_gap_section(inputs))
    sections.append(_goals_section(inputs))
    sections.append(_cost_section(inputs))
    return "\n\n".join(s for s in sections if s).rstrip() + "\n"


def _header(run: Run) -> str:
    elapsed = ""
    if run.finished_at and run.started_at:
        delta = (run.finished_at - run.started_at).total_seconds()
        elapsed = f" • {delta:.0f}s"
    return f"# Run {run.id} ({run.status.value}{elapsed})"


def _top_line(inputs: SummaryInputs) -> str:
    rows = compute_topline(inputs.scores)
    if not rows:
        return "## Top line\n\n_No scores._"
    lines = ["## Top line", "", "| subject | metric | n | prompt-rate | interaction-rate | mean |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x.subject_id, x.metric)):
        lines.append(
            f"| {r.subject_id} | {r.metric} | {r.n} | "
            f"{_fmt_rate(r.prompt_level_rate)} | {_fmt_rate(r.interaction_level_rate)} | "
            f"{_fmt_value(r.mean_value)} |"
        )
    return "\n".join(lines)


def _funnel_section(inputs: SummaryInputs) -> str:
    subject_ids = sorted({s.id for s in inputs.catalog.subjects})
    if not subject_ids:
        return ""
    lines = ["## Funnel (mention_presence by tier)", ""]
    for sid in subject_ids:
        funnel = compute_funnel(inputs.scores, inputs.catalog, subject_id=sid)
        spark = render_sparkline([t.rate for t in funnel])
        tier_strs = "  ".join(
            f"{t.tier}={_fmt_rate(t.rate)}({t.n})" for t in funnel
        )
        lines.append(f"- **{sid}**: {spark}  {tier_strs}")
    return "\n".join(lines)


def _worst_prompts_section(inputs: SummaryInputs) -> str:
    """Top prompts by interaction-level rate for binary metrics."""
    sections: list[str] = ["## Worst prompts (interaction-level rate)"]
    any_rendered = False
    for metric in ("mention_presence_rate", "brand_legacy_conflation_rate"):
        for subject in sorted({s.subject_id for s in inputs.scores if s.metric == metric}):
            top = compute_worst_prompts(
                inputs.scores, metric=metric, subject_id=subject, top_n=5
            )
            if not top:
                continue
            any_rendered = True
            sections.append(f"\n**{subject} / {metric}**")
            for w in top:
                sections.append(f"- `{w.prompt_id}` — {_fmt_rate(w.rate)} (n_models={w.n_models})")
    if not any_rendered:
        return ""
    return "\n".join(sections)


def _grounded_gap_section(inputs: SummaryInputs) -> str:
    gaps = compute_grounded_gaps(inputs.scores, top_n=5)
    if not gaps:
        return ""
    lines = ["## Grounded vs ungrounded gap (top 5 by |gap|)", ""]
    for g in gaps:
        lines.append(
            f"- `{g.prompt_id}` × `{g.model_stem}` / {g.subject_id} / {g.metric}: "
            f"grounded={_fmt_rate(g.grounded_value)} ungrounded={_fmt_rate(g.ungrounded_value)} "
            f"gap={g.gap:+.3f}"
        )
    return "\n".join(lines)


def _goals_section(inputs: SummaryInputs) -> str:
    if not inputs.goals:
        return ""
    lines = ["## Goal progress", ""]
    for goal in inputs.goals:
        ev = evaluate_goal(goal, scores=inputs.scores, catalog=inputs.catalog, today=inputs.today)
        light = _LIGHTS[ev.status]
        actual = _fmt_value(ev.actual)
        expected = _fmt_value(ev.expected)
        lines.append(
            f"- {light} **{goal.id}** ({goal.subject}/{goal.metric}/{goal.tier}, "
            f"target={goal.target} by {goal.target_date}): "
            f"actual={actual} expected={expected}"
        )
    return "\n".join(lines)


def _cost_section(inputs: SummaryInputs) -> str:
    n_total = len(inputs.tasks)
    n_success = sum(1 for t in inputs.tasks if t.status == TaskStatus.SUCCESS)
    n_failed = n_total - n_success
    total_cost = sum(t.cost_usd_estimate for t in inputs.tasks)
    total_in = sum(t.tokens_in for t in inputs.tasks)
    total_out = sum(t.tokens_out for t in inputs.tasks)
    elapsed = ""
    if inputs.run.finished_at and inputs.run.started_at:
        seconds = (inputs.run.finished_at - inputs.run.started_at).total_seconds()
        elapsed = f"; wall_time={seconds:.0f}s"
    return (
        "## Cost & runtime\n\n"
        f"- tasks: total={n_total} success={n_success} failed={n_failed}\n"
        f"- tokens: in={total_in} out={total_out}\n"
        f"- estimated cost: ${total_cost:.4f}{elapsed}"
    )


_LIGHTS = {
    GoalStatus.GREEN: "[GREEN]",
    GoalStatus.YELLOW: "[YELLOW]",
    GoalStatus.RED: "[RED]",
    GoalStatus.PENDING: "[PENDING]",
}


def _fmt_rate(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def _fmt_value(v: float | int | None) -> str:
    if v is None:
        return "—"
    if isinstance(v, int):
        return str(v)
    return f"{v:.3f}"
```

(Note: status indicators are plain `[GREEN]`/`[YELLOW]`/`[RED]`/`[PENDING]` text — easy to grep, terminal-safe. If you want emoji indicators later, swap the `_LIGHTS` dict.)

- [ ] **Step 7.4: Update `__init__.py`**

Add `SummaryInputs` and `render_summary` to imports + `__all__`.

- [ ] **Step 7.5: Run + lint + commit**

```bash
uv run pytest tests/test_reports_summary.py -v
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests

git add experiments/geo-analyzer/src/geo_analyzer/reports/ \
        experiments/geo-analyzer/tests/test_reports_summary.py
git commit -m "geo-analyzer: summary.md renderer (top line, funnel, worst prompts, gap, goals, cost)"
```

---

### Task 8: `geo-analyzer report` CLI command

**Files:**
- Modify: `experiments/geo-analyzer/src/geo_analyzer/cli.py`
- Create: `experiments/geo-analyzer/tests/test_cli_report.py`

CLI surface:

```
geo-analyzer report [RUN_ID] [--data-dir PATH] [--catalog-dir PATH]
                    [--open-latest]
```

- No `RUN_ID` and no `--open-latest` → use latest run.
- `--open-latest` → open the resulting `summary.md` in a browser via `webbrowser.open`.
- Writes the rendered markdown to `data/runs/<run-id>/summary.md`.

- [ ] **Step 8.1: Write failing tests**

```python
# tests/test_cli_report.py
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
    write_manifest(rp.manifest, Manifest(
        run=run, subject_ids=[], prompt_ids=[], model_ids=[], catalog_hash="x",
    ))
    # Minimal task + score so the renderer has something to chew.
    append_jsonl(rp.tasks_jsonl, {
        "run_id": run_id, "prompt_id": "prompt.broad.l1.companies-run-poorly",
        "model_id": "openai:gpt-5.1:ungrounded", "sample_n": 0,
        "status": "success", "text": "hi", "tokens_in": 5, "tokens_out": 5,
        "cost_usd_estimate": 0.0001, "latency_ms": 100, "error": None,
    })
    append_jsonl(rp.scores_jsonl, {
        "run_id": run_id, "prompt_id": "prompt.broad.l1.companies-run-poorly",
        "model_id": "openai:gpt-5.1:ungrounded",
        "subject_id": "convictional_brand", "metric": "mention_presence",
        "value": True, "scoring_method": "deterministic", "sample_aggregation": "single",
    })
    return rp.run_dir


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_report_writes_summary_for_run(project_root: Path, tmp_path: Path) -> None:
    run_dir = _seed_run(tmp_path)
    result = runner.invoke(
        app,
        [
            "report", "2026-04-29-manual",
            "--data-dir", str(tmp_path),
            "--catalog-dir", str(project_root / "catalog"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    summary = (run_dir / "summary.md").read_text()
    assert "Run 2026-04-29-manual" in summary
    assert "Top line" in summary


def test_report_uses_latest_when_run_id_omitted(project_root: Path, tmp_path: Path) -> None:
    _seed_run(tmp_path, "2026-04-15-manual")
    _seed_run(tmp_path, "2026-04-29-manual")
    result = runner.invoke(
        app,
        [
            "report",
            "--data-dir", str(tmp_path),
            "--catalog-dir", str(project_root / "catalog"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    # Latest is 2026-04-29-manual; only its summary.md should have been written.
    latest_summary = (tmp_path / "runs" / "2026-04-29-manual" / "summary.md").read_text()
    assert "Run 2026-04-29-manual" in latest_summary


def test_report_no_runs_fails(project_root: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "report",
            "--data-dir", str(tmp_path),
            "--catalog-dir", str(project_root / "catalog"),
        ],
    )
    assert result.exit_code != 0
    assert "no runs" in (result.stdout or "").lower() or "no runs" in (result.stderr or "").lower()


def test_report_unknown_run_id_fails(project_root: Path, tmp_path: Path) -> None:
    _seed_run(tmp_path)
    result = runner.invoke(
        app,
        [
            "report", "2026-99-99-not-real",
            "--data-dir", str(tmp_path),
            "--catalog-dir", str(project_root / "catalog"),
        ],
    )
    assert result.exit_code != 0
```

- [ ] **Step 8.2: Run, fail**

```bash
uv run pytest tests/test_cli_report.py -v
```

- [ ] **Step 8.3: Extend `cli.py`**

Add to imports (don't remove existing):

```python
import webbrowser
from datetime import date as date_today

from geo_analyzer.goals import load_goals
from geo_analyzer.reports import (
    SummaryInputs,
    latest_run_id,
    list_run_ids,
    read_scores_jsonl,
    render_summary,
)
from geo_analyzer.storage import read_manifest, read_tasks_jsonl
```

Add the `report` command after the existing `run` command:

```python
@app.command("report")
def report_command(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Run id (e.g. 2026-04-29-manual). Defaults to latest run."),
    ] = None,
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    catalog_dir: Annotated[Path, typer.Option("--catalog-dir")] = Path("catalog"),
    open_latest: Annotated[
        bool,
        typer.Option("--open-latest", help="Open the rendered summary.md in a browser."),
    ] = False,
) -> None:
    """Render summary.md for a run (default: latest)."""
    try:
        cat = load_catalog(catalog_dir)
    except (CatalogError, ValidationError) as e:
        _err_console.print(f"[red]catalog error:[/red] {e}")
        raise typer.Exit(code=1) from e

    if run_id is None:
        resolved = latest_run_id(data_dir)
        if resolved is None:
            _err_console.print("[red]error:[/red] no runs found under " f"{data_dir / 'runs'}.")
            raise typer.Exit(code=1)
        run_id = resolved

    if run_id not in list_run_ids(data_dir):
        _err_console.print(f"[red]error:[/red] run {run_id!r} not found under {data_dir / 'runs'}.")
        raise typer.Exit(code=1)

    rp = run_paths_for(data_dir, run_id)
    manifest = read_manifest(rp.manifest)
    tasks = read_tasks_jsonl(rp.tasks_jsonl)
    scores = read_scores_jsonl(rp.scores_jsonl)
    goals = load_goals(data_dir / "goals.yaml")

    summary_md = render_summary(SummaryInputs(
        run=manifest.run,
        tasks=tasks,
        scores=scores,
        catalog=cat,
        goals=goals,
        today=date_today.today(),
    ))
    rp.run_dir.joinpath("summary.md").write_text(summary_md, encoding="utf-8")
    _console.print(f"[green]wrote[/green] {rp.run_dir / 'summary.md'}")

    if open_latest:
        webbrowser.open(f"file://{rp.run_dir / 'summary.md'}")
```

You also need to add `run_paths_for` to the existing `geo_analyzer.storage` import line if not already imported.

- [ ] **Step 8.4: Run + lint + manual smoke + commit**

```bash
uv run pytest tests/test_cli_report.py -v
uv run pytest -q
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests
```

Manual smoke (against the run created in Phase 3 testing):

```bash
uv run geo-analyzer report
cat data/runs/$(ls data/runs | tail -1)/summary.md | head -40
```

```bash
git add experiments/geo-analyzer/src/geo_analyzer/cli.py \
        experiments/geo-analyzer/tests/test_cli_report.py
git commit -m "geo-analyzer: 'report' CLI command for summary.md generation"
```

---

### Task 9: `geo-analyzer status` CLI command

**Files:**
- Modify: `experiments/geo-analyzer/src/geo_analyzer/cli.py`
- Create: `experiments/geo-analyzer/tests/test_cli_status.py`

CLI surface:

```
geo-analyzer status [--data-dir PATH] [--catalog-dir PATH]
```

Reads `data/goals.yaml`, evaluates each goal against the latest run's scores, prints one line per goal with its traffic-light color. Exits 3 if any goal is RED (per DESIGN §8.1: "exit 3 = at least one goal is red").

- [ ] **Step 9.1: Write failing tests**

```python
# tests/test_cli_status.py
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


def _seed_run_with_score(tmp_path: Path, *, presence_value: bool) -> None:
    rp = run_paths_for(tmp_path, "2026-04-29-manual")
    rp.ensure()
    started = datetime(2026, 4, 29, 9, tzinfo=UTC)
    finished = datetime(2026, 4, 29, 9, 5, tzinfo=UTC)
    run = Run(id="2026-04-29-manual", trigger="manual",
              started_at=started, finished_at=finished, status=RunStatus.COMPLETED)
    write_manifest(rp.manifest, Manifest(
        run=run, subject_ids=[], prompt_ids=[], model_ids=[], catalog_hash="x",
    ))
    append_jsonl(rp.scores_jsonl, {
        "run_id": "2026-04-29-manual",
        "prompt_id": "prompt.broad.l1.companies-run-poorly",
        "model_id": "openai:gpt-5.1:ungrounded",
        "subject_id": "convictional_brand", "metric": "mention_presence",
        "value": presence_value, "scoring_method": "deterministic",
        "sample_aggregation": "single",
    })


def _seed_goals_yaml(tmp_path: Path) -> Path:
    """Goals targeting convictional_brand mention_presence at L1 with target=0.5.

    With one True L1 score → actual=1.0 ≥ target=0.5 → green.
    With one False L1 score → actual=0.0 ≤ baseline=0.0 → red.
    """
    path = tmp_path / "goals.yaml"
    path.write_text(
        "- id: g1\n"
        "  subject: convictional_brand\n"
        "  metric: mention_presence\n"
        "  tier: L1\n"
        "  target: 0.5\n"
        "  direction: above\n"
        "  created_at: 2026-01-01\n"
        "  target_date: 2026-12-31\n"
    )
    return path


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_status_green_goal_exits_zero(project_root: Path, tmp_path: Path) -> None:
    _seed_run_with_score(tmp_path, presence_value=True)
    _seed_goals_yaml(tmp_path)
    result = runner.invoke(
        app,
        [
            "status",
            "--data-dir", str(tmp_path),
            "--catalog-dir", str(project_root / "catalog"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "g1" in result.stdout
    assert "GREEN" in result.stdout


def test_status_red_goal_exits_three(project_root: Path, tmp_path: Path) -> None:
    _seed_run_with_score(tmp_path, presence_value=False)
    _seed_goals_yaml(tmp_path)
    result = runner.invoke(
        app,
        [
            "status",
            "--data-dir", str(tmp_path),
            "--catalog-dir", str(project_root / "catalog"),
        ],
    )
    assert result.exit_code == 3
    assert "g1" in result.stdout
    assert "RED" in result.stdout


def test_status_no_goals_exits_zero(project_root: Path, tmp_path: Path) -> None:
    _seed_run_with_score(tmp_path, presence_value=True)
    # No goals.yaml file written.
    result = runner.invoke(
        app,
        [
            "status",
            "--data-dir", str(tmp_path),
            "--catalog-dir", str(project_root / "catalog"),
        ],
    )
    assert result.exit_code == 0
    assert "no goals" in (result.stdout or "").lower() or result.stdout.strip() == ""


def test_status_no_runs_fails(project_root: Path, tmp_path: Path) -> None:
    _seed_goals_yaml(tmp_path)
    result = runner.invoke(
        app,
        [
            "status",
            "--data-dir", str(tmp_path),
            "--catalog-dir", str(project_root / "catalog"),
        ],
    )
    assert result.exit_code != 0
    assert "no runs" in (result.stdout or "").lower() or "no runs" in (result.stderr or "").lower()
```

- [ ] **Step 9.2: Run, fail**

```bash
uv run pytest tests/test_cli_status.py -v
```

- [ ] **Step 9.3: Add `status` to `cli.py`**

Add the `status` command after the `report` command:

```python
@app.command("status")
def status_command(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    catalog_dir: Annotated[Path, typer.Option("--catalog-dir")] = Path("catalog"),
) -> None:
    """Print goal traffic-lights against the latest run.

    Exit codes:
      0 — all goals green/yellow/pending
      1 — operational error (no runs, catalog invalid)
      3 — at least one goal is RED
    """
    from geo_analyzer.reports import GoalStatus, evaluate_goal
    from datetime import date as _date

    try:
        cat = load_catalog(catalog_dir)
    except (CatalogError, ValidationError) as e:
        _err_console.print(f"[red]catalog error:[/red] {e}")
        raise typer.Exit(code=1) from e

    goals = load_goals(data_dir / "goals.yaml")
    if not goals:
        _console.print("no goals defined.")
        return

    run_id = latest_run_id(data_dir)
    if run_id is None:
        _err_console.print(f"[red]error:[/red] no runs found under {data_dir / 'runs'}.")
        raise typer.Exit(code=1)

    rp = run_paths_for(data_dir, run_id)
    scores = read_scores_jsonl(rp.scores_jsonl)
    today = _date.today()

    any_red = False
    _LIGHT_NAMES = {
        GoalStatus.GREEN: "[GREEN]",
        GoalStatus.YELLOW: "[YELLOW]",
        GoalStatus.RED: "[RED]",
        GoalStatus.PENDING: "[PENDING]",
    }
    for goal in goals:
        ev = evaluate_goal(goal, scores=scores, catalog=cat, today=today)
        light = _LIGHT_NAMES[ev.status]
        actual = "—" if ev.actual is None else f"{ev.actual:.3f}"
        expected = "—" if ev.expected is None else f"{ev.expected:.3f}"
        _console.print(
            f"{light}  {goal.id}  ({goal.subject}/{goal.metric}/{goal.tier})  "
            f"actual={actual} expected={expected} target={goal.target}"
        )
        if ev.status == GoalStatus.RED:
            any_red = True

    if any_red:
        raise typer.Exit(code=3)
```

- [ ] **Step 9.4: Run + lint + manual smoke + commit**

```bash
uv run pytest tests/test_cli_status.py -v
uv run pytest -q
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests
```

Manual smoke:

```bash
uv run geo-analyzer status
echo "exit=$?"
```

```bash
git add experiments/geo-analyzer/src/geo_analyzer/cli.py \
        experiments/geo-analyzer/tests/test_cli_status.py
git commit -m "geo-analyzer: 'status' CLI command (goal traffic-lights, exit 3 on red)"
```

---

### Task 10: Multi-run `--since` aggregation

**Files:**
- Create: `experiments/geo-analyzer/src/geo_analyzer/reports/multi_run.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/reports/__init__.py`
- Modify: `experiments/geo-analyzer/src/geo_analyzer/cli.py` (extend `report` with `--since`)
- Create: `experiments/geo-analyzer/tests/test_reports_multi_run.py`

DESIGN §9.2: `geo-analyzer report --since YYYY-MM-DD` walks all runs in the local directory and emits a trend report. v1 scope: per-(subject, metric) one row per run, ordered by date.

- [ ] **Step 10.1: Write failing tests**

```python
# tests/test_reports_multi_run.py
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from geo_analyzer.reports.multi_run import (
    MultiRunRow,
    compute_multi_run_trends,
)
from geo_analyzer.runtime import Run, RunStatus
from geo_analyzer.storage import (
    Manifest,
    append_jsonl,
    run_paths_for,
    write_manifest,
)


def _seed(tmp_path: Path, run_id: str, date_str: str, presence_value: bool) -> None:
    rp = run_paths_for(tmp_path, run_id)
    rp.ensure()
    started = datetime.fromisoformat(date_str + "T09:00:00+00:00")
    finished = started.replace(minute=5)
    run = Run(id=run_id, trigger="manual",
              started_at=started, finished_at=finished, status=RunStatus.COMPLETED)
    write_manifest(rp.manifest, Manifest(
        run=run, subject_ids=[], prompt_ids=[], model_ids=[], catalog_hash="x",
    ))
    append_jsonl(rp.scores_jsonl, {
        "run_id": run_id,
        "prompt_id": "prompt.broad.l1.companies-run-poorly",
        "model_id": "openai:gpt-5.1:ungrounded",
        "subject_id": "convictional_brand", "metric": "mention_presence",
        "value": presence_value, "scoring_method": "deterministic",
        "sample_aggregation": "single",
    })


class TestComputeMultiRunTrends:
    def test_returns_one_row_per_run(self, tmp_path: Path) -> None:
        _seed(tmp_path, "2026-04-15-manual", "2026-04-15", presence_value=False)
        _seed(tmp_path, "2026-04-22-manual", "2026-04-22", presence_value=True)
        _seed(tmp_path, "2026-04-29-manual", "2026-04-29", presence_value=True)
        rows = compute_multi_run_trends(tmp_path, since=None)
        # Three runs × one (subject, metric) = three rows
        assert len(rows) == 3
        assert [r.run_id for r in rows] == [
            "2026-04-15-manual", "2026-04-22-manual", "2026-04-29-manual",
        ]

    def test_since_filters_older(self, tmp_path: Path) -> None:
        from datetime import date
        _seed(tmp_path, "2026-04-15-manual", "2026-04-15", presence_value=False)
        _seed(tmp_path, "2026-04-29-manual", "2026-04-29", presence_value=True)
        rows = compute_multi_run_trends(tmp_path, since=date(2026, 4, 20))
        assert [r.run_id for r in rows] == ["2026-04-29-manual"]

    def test_no_runs_returns_empty(self, tmp_path: Path) -> None:
        assert compute_multi_run_trends(tmp_path, since=None) == []

    def test_row_carries_topline_fields(self, tmp_path: Path) -> None:
        _seed(tmp_path, "2026-04-29-manual", "2026-04-29", presence_value=True)
        rows = compute_multi_run_trends(tmp_path, since=None)
        row = rows[0]
        assert row.subject_id == "convictional_brand"
        assert row.metric == "mention_presence"
        assert row.prompt_level_rate == 1.0  # one True score → rate=1.0
```

- [ ] **Step 10.2: Run, fail**

```bash
uv run pytest tests/test_reports_multi_run.py -v
```

- [ ] **Step 10.3: Implement `multi_run.py`**

```python
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
    data_dir: Path, *, since: date | None,
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
            rows.append(MultiRunRow(
                run_id=run_id,
                run_date=run_date,
                subject_id=tl.subject_id,
                metric=tl.metric,
                n=tl.n,
                prompt_level_rate=tl.prompt_level_rate,
                interaction_level_rate=tl.interaction_level_rate,
                mean_value=tl.mean_value,
            ))
    return rows
```

- [ ] **Step 10.4: Update `reports/__init__.py`**

Add `MultiRunRow` and `compute_multi_run_trends` to imports + `__all__`.

- [ ] **Step 10.5: Extend `cli.py` `report` command with `--since`**

Add a `--since` option to the `report` command. The function signature gets one new parameter:

```python
since: Annotated[
    str | None,
    typer.Option("--since", help="Render a trend report across all runs since this YYYY-MM-DD."),
] = None,
```

At the top of the function body, branch on `since`:

```python
    if since is not None:
        from datetime import date as _date
        try:
            since_date = _date.fromisoformat(since)
        except ValueError as e:
            _err_console.print(f"[red]error:[/red] --since must be YYYY-MM-DD; got {since!r}")
            raise typer.Exit(code=1) from e
        from geo_analyzer.reports import compute_multi_run_trends
        rows = compute_multi_run_trends(data_dir, since=since_date)
        if not rows:
            _err_console.print(f"[yellow]no runs found since {since_date}.[/yellow]")
            return
        # Print trend table to stdout (no markdown file written for trend mode).
        _console.print(f"[bold]trend since {since_date}[/bold] — {len(rows)} rows")
        _console.print("run_id              subject              metric              n  prompt-rate  interaction-rate  mean")
        for r in rows:
            _console.print(
                f"{r.run_id:<20} {r.subject_id:<20} {r.metric:<20} {r.n:>2}  "
                f"{_fmt_or_dash(r.prompt_level_rate):>11}  "
                f"{_fmt_or_dash(r.interaction_level_rate):>16}  "
                f"{_fmt_or_dash(r.mean_value):>5}"
            )
        return
```

Add the helper next to the existing `_print_probe_result`:

```python
def _fmt_or_dash(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"
```

- [ ] **Step 10.6: Run + lint + commit**

```bash
uv run pytest tests/test_reports_multi_run.py -v
uv run pytest -q
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests

git add experiments/geo-analyzer/src/geo_analyzer/reports/ \
        experiments/geo-analyzer/src/geo_analyzer/cli.py \
        experiments/geo-analyzer/tests/test_reports_multi_run.py
git commit -m "geo-analyzer: 'report --since' multi-run trend aggregation"
```

---

### Task 11: README + final pass

**Files:**
- Modify: `experiments/geo-analyzer/README.md`

- [ ] **Step 11.1: Run full suite + lint + typecheck**

```bash
uv run pytest -q
uv run pyright
uv run ruff check src tests
uv run ruff format --check src tests
```

- [ ] **Step 11.2: Verify CLI surface end-to-end**

```bash
uv run geo-analyzer --help
uv run geo-analyzer report --help
uv run geo-analyzer status --help
uv run geo-analyzer report
uv run geo-analyzer status; echo "exit=$?"
```

If a previous run exists under `data/runs/`, paste a few lines of `data/runs/<latest>/summary.md` in the report.

- [ ] **Step 11.3: Update README**

Replace the existing "## Status (Phase 3 complete)" section's content with:

```markdown
## Status (Phase 4 complete)

Working today:

- `uv run geo-analyzer catalog validate` — loads + cross-checks the seed catalog.
- `uv run geo-analyzer probe "<prompt>" --model <id>` — single prompt against any provider.
- `uv run geo-analyzer probe ... --sensitivity-samples N --temperature T` — N samples
  to inspect generation variance (per DESIGN §5.7).
- `uv run geo-analyzer run [filters] [--dry-run] [--yes]` — execute the full
  catalog × matrix concurrently and persist artifacts under `data/runs/<run-id>/`.
- `uv run geo-analyzer report [RUN_ID] [--open-latest] [--since YYYY-MM-DD]` —
  render `summary.md` for a run (default: latest), or print a multi-run trend table.
- `uv run geo-analyzer status` — goal traffic-lights against the latest run
  (exit 3 if any goal is red).
- Three provider adapters with grounded + ungrounded modes.
- Deterministic scoring (mention presence, ordinal rank, share of voice,
  brand-legacy conflation, citations) and N=3 sample aggregation.
- Per-model token pricing and cost estimator.

Not yet (Phase 5):

- `launchd` plist example for scheduled local runs
- GitHub Actions CI workflow (lint/typecheck/unit tests on geo-analyzer/** changes)
```

Update the Quick start block to include the new commands:

```markdown
## Quick start

\`\`\`
cd experiments/geo-analyzer
uv sync
uv run geo-analyzer catalog validate
uv run pytest                    # ~210 tests, no API calls
uv run pytest -m live            # opt-in real API tests; needs keys
uv run geo-analyzer probe "What is Convictional?" --model openai:gpt-5.1:grounded
uv run geo-analyzer run --dry-run --tier L1 --model openai:gpt-5.1:ungrounded
uv run geo-analyzer run --tier L4 --model openai:gpt-5.1:ungrounded --yes
uv run geo-analyzer report                    # writes summary.md for latest run
uv run geo-analyzer status                    # goal traffic-lights
\`\`\`
```

(Replace the `\`\`\`` literals with actual triple-backticks when writing the file.)

- [ ] **Step 11.4: Commit**

```bash
git add experiments/geo-analyzer/README.md
git commit -m "geo-analyzer: document Phase 4 surface area in README"
```

---

## Done state for Phase 4

When all tasks are committed and green:

1. `uv run geo-analyzer report` writes a `summary.md` to the latest run dir with all six DESIGN §9.1 sections.
2. `uv run geo-analyzer report --since 2026-01-01` prints a multi-run trend table.
3. `uv run geo-analyzer status` prints one line per goal with traffic-light color; exits 3 if any goal is red.
4. `uv run pytest` is green with ~210 tests.
5. `uv run pyright` clean.
6. `uv run ruff check` and `ruff format --check` clean.

Phase 5 picks up here: launchd plist + GitHub Actions CI workflow (operational polish only — no new user-facing features).
