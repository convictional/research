"""Load goals.yaml. Goals are optional — missing file means zero goals."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, cast

import yaml

from geo_analyzer.types import Goal


class GoalsError(ValueError):
    """Raised when goals.yaml parses cleanly but fails cross-checks."""


def load_goals(path: Path) -> list[Goal]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f) or []
    if not isinstance(raw, list):
        raise GoalsError("goals.yaml must be a top-level list")
    items = cast(list[Any], raw)
    goals = [Goal.model_validate(item) for item in items]
    dupes = [g for g, c in Counter(x.id for x in goals).items() if c > 1]
    if dupes:
        raise GoalsError(f"duplicate goal id: {sorted(dupes)!r}")
    for g in goals:
        if g.target_date <= g.created_at:
            raise GoalsError(
                f"goal {g.id!r}: target_date ({g.target_date}) must be after " f"created_at ({g.created_at})"
            )
    return goals
