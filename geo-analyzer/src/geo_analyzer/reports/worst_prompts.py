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
        if isinstance(s.value, bool) or not isinstance(s.value, int | float):
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
