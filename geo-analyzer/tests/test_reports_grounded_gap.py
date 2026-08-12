from __future__ import annotations

import pytest

from geo_analyzer.reports.grounded_gap import (
    GroundedGapRow,
    compute_grounded_gaps,
)
from geo_analyzer.runtime import Score


def _score(
    prompt: str,
    model: str,
    value: float | bool,
    metric: str = "mention_presence_rate",
    subject: str = "convictional_brand",
) -> Score:
    return Score(
        run_id="r",
        prompt_id=prompt,
        model_id=model,
        subject_id=subject,
        metric=metric,
        value=value,
        scoring_method="deterministic",
        sample_aggregation="mean",
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
        assert g.gap == pytest.approx(0.5)  # type: ignore[misc]

    def test_orders_by_absolute_gap(self) -> None:
        scores = [
            _score("p_small", "openai:gpt-5.1:grounded", 0.5),
            _score("p_small", "openai:gpt-5.1:ungrounded", 0.4),  # gap = 0.1
            _score("p_big", "openai:gpt-5.1:grounded", 0.9),
            _score("p_big", "openai:gpt-5.1:ungrounded", 0.1),  # gap = 0.8
            _score("p_neg", "openai:gpt-5.1:grounded", 0.1),
            _score("p_neg", "openai:gpt-5.1:ungrounded", 0.6),  # gap = -0.5
        ]
        gaps = compute_grounded_gaps(scores, top_n=10)
        assert [g.prompt_id for g in gaps] == ["p_big", "p_neg", "p_small"]

    def test_only_pairs_when_both_modes_present(self) -> None:
        scores = [
            _score("p1", "openai:gpt-5.1:grounded", 0.8),
        ]
        assert compute_grounded_gaps(scores, top_n=10) == []

    def test_top_n_limits(self) -> None:
        scores: list[Score] = []
        for i in range(5):
            scores.append(_score(f"p{i}", "openai:gpt-5.1:grounded", float(i) / 5))
            scores.append(_score(f"p{i}", "openai:gpt-5.1:ungrounded", 0.0))
        gaps = compute_grounded_gaps(scores, top_n=3)
        assert len(gaps) == 3

    def test_result_dataclass(self) -> None:
        scores = [
            _score("p1", "openai:gpt-5.1:grounded", 0.8),
            _score("p1", "openai:gpt-5.1:ungrounded", 0.3),
        ]
        gaps = compute_grounded_gaps(scores, top_n=10)
        assert isinstance(gaps[0], GroundedGapRow)
