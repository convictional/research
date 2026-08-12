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
        run_id="r",
        prompt_id=prompt_id,
        model_id="openai:gpt-5.1:ungrounded",
        sample_n=0,
        status=TaskStatus.SUCCESS,
        text="hi",
        tokens_in=10,
        tokens_out=20,
        cost_usd_estimate=0.001,
        latency_ms=200,
    )


def _score(
    metric: str = "mention_presence",
    value: bool | float = True,
    subject: str = "convictional_brand",
    prompt: str = "prompt.broad.l1.companies-in-age-of-ai",
) -> Score:
    return Score(
        run_id="r",
        prompt_id=prompt,
        model_id="openai:gpt-5.1:ungrounded",
        subject_id=subject,
        metric=metric,
        value=value,
        scoring_method="deterministic",
        sample_aggregation="single",
    )


def _goal() -> Goal:
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


class TestRenderSummary:
    def test_includes_run_id_tldr_and_subject_section(self, real_catalog: Catalog) -> None:
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
        assert "TL;DR" in md
        # Each catalog subject gets its own H2 section.
        assert "## convictional_brand" in md

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

    def test_uses_plain_text_status_indicators(self, real_catalog: Catalog) -> None:
        # No emoji — plain [GREEN]/[RED]/[YELLOW]/[PENDING] labels.
        inputs = SummaryInputs(
            run=_run(),
            tasks=[_task()],
            scores=[_score()],
            catalog=real_catalog,
            goals=[_goal()],
            today=date(2026, 4, 29),
        )
        md = render_summary(inputs)
        # One of these must appear in the goals section.
        assert any(label in md for label in ("[GREEN]", "[RED]", "[YELLOW]", "[PENDING]"))
