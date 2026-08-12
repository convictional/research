"""Per-subject top-line scoreboard.

For each (subject_id, metric) pair across the run, compute:
  - prompt_level_rate: for binary metrics (mention_presence, brand_legacy_conflation),
    fraction of (prompt x model) cohorts that fired True.
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
            float_values = [
                float(s.value) for s in bucket if isinstance(s.value, int | float) and not isinstance(s.value, bool)
            ]
            interaction_rate = sum(float_values) / len(float_values) if float_values else None
        else:
            # share_of_voice, ordinal_rank, etc. — mean of non-None numeric values.
            numeric = [
                float(s.value) for s in bucket if isinstance(s.value, int | float) and not isinstance(s.value, bool)
            ]
            mean = sum(numeric) / len(numeric) if numeric else None

        rows.append(
            TopLineRow(
                subject_id=subject_id,
                metric=metric,
                n=n,
                prompt_level_rate=prompt_rate,
                interaction_level_rate=interaction_rate,
                mean_value=mean,
            )
        )
    return rows
