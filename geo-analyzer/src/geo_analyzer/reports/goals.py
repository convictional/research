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
        s
        for s in scores
        if s.subject_id == goal.subject and s.metric == goal.metric and prompt_tier.get(s.prompt_id) == goal.tier
    ]
    numeric_values: list[float] = []
    for s in matching:
        if isinstance(s.value, bool):
            numeric_values.append(1.0 if s.value else 0.0)
        elif isinstance(s.value, int | float):
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
