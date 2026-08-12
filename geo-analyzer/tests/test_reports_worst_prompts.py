from __future__ import annotations

import pytest

from geo_analyzer.reports.worst_prompts import (
    WorstPromptRow,
    compute_worst_prompts,
)
from geo_analyzer.runtime import Score


def _rate_score(
    prompt: str,
    value: float,
    metric: str = "mention_presence_rate",
    subject: str = "convictional_brand",
    model: str = "openai:gpt-5.1:grounded",
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


class TestComputeWorstPrompts:
    def test_orders_by_rate_descending(self) -> None:
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
        assert worst[0].rate == pytest.approx(0.5)  # type: ignore[misc]
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
        assert worst[0].rate == pytest.approx(0.9)  # type: ignore[misc]

    def test_no_matching_scores_returns_empty(self) -> None:
        assert (
            compute_worst_prompts(
                [],
                metric="mention_presence_rate",
                subject_id="convictional_brand",
                top_n=5,
            )
            == []
        )

    def test_result_is_dataclass(self) -> None:
        scores = [_rate_score("p1", 0.5)]
        worst = compute_worst_prompts(
            scores,
            metric="mention_presence_rate",
            subject_id="convictional_brand",
            top_n=10,
        )
        assert isinstance(worst[0], WorstPromptRow)
