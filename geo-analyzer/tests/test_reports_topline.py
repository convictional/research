from __future__ import annotations

import pytest

from geo_analyzer.reports.topline import compute_topline
from geo_analyzer.runtime import Score


def _score(
    metric: str,
    value: bool | float | int | None,
    subject: str = "convictional_brand",
    prompt: str = "p1",
    model: str = "openai:gpt-5.1:ungrounded",
    agg: str = "single",
) -> Score:
    return Score(
        run_id="r",
        prompt_id=prompt,
        model_id=model,
        subject_id=subject,
        metric=metric,
        value=value,
        scoring_method="deterministic",
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
        assert row.prompt_level_rate == pytest.approx(2 / 3)  # type: ignore[misc]
        assert row.n == 3

    def test_rate_metric_interaction_level(self) -> None:
        scores = [
            _score("mention_presence_rate", 1.0, prompt="p1", agg="mean"),
            _score("mention_presence_rate", 0.5, prompt="p2", agg="mean"),
        ]
        rows = compute_topline(scores)
        row = next(r for r in rows if r.metric == "mention_presence_rate")
        assert row.interaction_level_rate == pytest.approx(0.75)  # type: ignore[misc]

    def test_share_of_voice_mean(self) -> None:
        scores = [
            _score("share_of_voice", 0.3),
            _score("share_of_voice", 0.5),
            _score("share_of_voice", None),  # None values skipped from the mean
        ]
        rows = compute_topline(scores)
        row = next(r for r in rows if r.metric == "share_of_voice")
        assert row.mean_value == pytest.approx(0.4)  # type: ignore[misc]

    def test_groups_by_subject_and_metric(self) -> None:
        scores = [
            _score("mention_presence", True, subject="convictional_brand"),
            _score("mention_presence", False, subject="lattice"),
        ]
        rows = compute_topline(scores)
        keys = {(r.subject_id, r.metric) for r in rows}
        assert keys == {("convictional_brand", "mention_presence"), ("lattice", "mention_presence")}

    def test_empty_scores_returns_empty(self) -> None:
        assert compute_topline([]) == []
