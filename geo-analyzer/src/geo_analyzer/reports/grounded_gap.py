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
    """Return up to top_n (prompt x model_stem x subject x metric) rows where
    both grounded and ungrounded values exist, ordered by abs(gap) descending."""
    pairs: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for s in scores:
        sm = _stem_and_mode(s.model_id)
        if sm is None:
            continue
        if isinstance(s.value, bool) or not isinstance(s.value, int | float):
            continue
        stem, mode = sm
        pairs[(s.prompt_id, stem, s.subject_id, s.metric)][mode] = float(s.value)

    rows: list[GroundedGapRow] = []
    for (prompt_id, stem, subject_id, metric), modes in pairs.items():
        if "grounded" not in modes or "ungrounded" not in modes:
            continue
        g = modes["grounded"]
        u = modes["ungrounded"]
        rows.append(
            GroundedGapRow(
                prompt_id=prompt_id,
                model_stem=stem,
                subject_id=subject_id,
                metric=metric,
                grounded_value=g,
                ungrounded_value=u,
                gap=g - u,
            )
        )
    rows.sort(key=lambda r: abs(r.gap), reverse=True)
    return rows[:top_n]
