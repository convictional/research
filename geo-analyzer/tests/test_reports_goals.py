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


def _binary(prompt: str, value: bool, subject: str = "convictional_brand", metric: str = "mention_presence") -> Score:
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
        scores = [_binary("prompt.broad.l1.companies-in-age-of-ai", True)]
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
            _binary("prompt.broad.l1.companies-in-age-of-ai", True),
            _binary("prompt.broad.l1.leader-role-in-age-of-ai", True),
            _binary("prompt.broad.l1.norms-for-ai-era-orgs", True),
        ]
        result = evaluate_goal(
            _grow_goal(),
            scores=scores,
            catalog=real_catalog,
            today=date(2026, 6, 1),  # ~5 months in
        )
        assert result.status == GoalStatus.GREEN
        assert result.actual == pytest.approx(1.0)  # type: ignore[misc]

    def test_grow_goal_red_when_actual_at_baseline(self, real_catalog: Catalog) -> None:
        # All False → actual = 0.0 = baseline. Should be red after 5 months.
        scores = [
            _binary("prompt.broad.l1.companies-in-age-of-ai", False),
            _binary("prompt.broad.l1.leader-role-in-age-of-ai", False),
            _binary("prompt.broad.l1.norms-for-ai-era-orgs", False),
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
            _binary("prompt.brand.l4.what-is-convictional", False, subject="convictional_legacy_dropship"),
            _binary("prompt.brand.l4.convictional-product", False, subject="convictional_legacy_dropship"),
            _binary("prompt.brand.l4.convictional-pricing", False, subject="convictional_legacy_dropship"),
        ]
        result = evaluate_goal(
            _shrink_goal(),
            scores=scores,
            catalog=real_catalog,
            today=date(2026, 6, 1),
        )
        assert result.status == GoalStatus.GREEN

    def test_evaluation_is_dataclass(self, real_catalog: Catalog) -> None:
        result = evaluate_goal(
            _grow_goal(),
            scores=[],
            catalog=real_catalog,
            today=date(2026, 6, 1),
        )
        assert isinstance(result, GoalEvaluation)
