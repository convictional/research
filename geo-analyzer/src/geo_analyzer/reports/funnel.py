"""Per-subject funnel: mean mention_presence rate across L1->L4 tiers.

A healthy `convictional_brand` curve climbs from L3 to L4; a healthy
`organizational_health_category` curve rises at L2/L3 over time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import get_args

from geo_analyzer.runtime import Score
from geo_analyzer.types import Catalog, PromptTier


@dataclass(frozen=True)
class FunnelTier:
    tier: PromptTier
    rate: float | None
    n: int
    """Number of (prompt x model) cohorts contributing."""


def compute_funnel(
    scores: list[Score],
    catalog: Catalog,
    *,
    subject_id: str,
) -> list[FunnelTier]:
    """Return one FunnelTier per L1..L4 (always all four, even when empty)."""
    prompt_tier: dict[str, PromptTier] = {p.id: p.tier for p in catalog.prompts}

    by_tier: dict[PromptTier, list[bool]] = defaultdict(list)
    for s in scores:
        if s.subject_id != subject_id:
            continue
        if s.metric != "mention_presence":
            continue
        if not isinstance(s.value, bool):
            continue
        tier = prompt_tier.get(s.prompt_id)
        if tier is None:
            continue
        by_tier[tier].append(s.value)

    out: list[FunnelTier] = []
    for tier in get_args(PromptTier):  # ("L1", "L2", "L3", "L4") in order
        bucket = by_tier.get(tier, [])
        rate = sum(1 for v in bucket if v) / len(bucket) if bucket else None
        out.append(FunnelTier(tier=tier, rate=rate, n=len(bucket)))
    return out


# Unicode block characters for sparkline rendering. Index 0 is lowest height.
_SPARK_BARS = "▁▂▃▄▅▆▇█"


def render_sparkline(values: list[float | None]) -> str:
    """Render a list of [0, 1] values as a unicode block sparkline. None → space."""
    chars: list[str] = []
    for v in values:
        if v is None:
            chars.append(" ")
            continue
        clamped = max(0.0, min(1.0, v))
        idx = int(round(clamped * (len(_SPARK_BARS) - 1)))
        chars.append(_SPARK_BARS[idx])
    return "".join(chars)
