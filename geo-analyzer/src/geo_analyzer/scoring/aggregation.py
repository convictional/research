"""Per-sample → per-(prompt, model) aggregations for grounded N=3 rollups.

DESIGN §5.3:
  - mention_presence (bool): majority vote
  - mention_presence_rate (float): mean rate of True across samples
  - ordinal_rank (int|None): median, dropping Nones first
  - share_of_voice (float|None): mean across non-None samples
  - brand_legacy_conflation (bool): majority vote
  - brand_legacy_conflation_rate (float): mean rate of True
"""

from __future__ import annotations

from statistics import median


def majority_vote(samples: list[bool]) -> bool:
    """True iff strictly more than half of samples are True.

    For even-length inputs, ties resolve to True (favor signal presence).
    Note: normal grounded N=3 runs cannot produce ties by construction —
    the tie-break rule only applies if a caller passes N=2 deliberately.
    """
    if not samples:
        raise ValueError("majority_vote: empty samples")
    trues = sum(1 for s in samples if s)
    return trues * 2 >= len(samples)


def median_or_none(samples: list[int | None]) -> int | None:
    """Median of non-None samples, rounded to int (ranks are integral). None if no samples."""
    cleaned = [s for s in samples if s is not None]
    if not cleaned:
        return None
    return int(round(median(cleaned)))


def mean_of_floats(samples: list[float | None]) -> float | None:
    """Mean of non-None samples; None if all are None or list is empty."""
    cleaned = [s for s in samples if s is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def mean_rate(samples: list[bool]) -> float:
    """Mean rate of True. Raises ValueError on empty input — caller must guarantee N>=1."""
    if not samples:
        raise ValueError("mean_rate: empty samples")
    return sum(1 for s in samples if s) / len(samples)
