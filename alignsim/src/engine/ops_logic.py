"""Pure functions for operations (Ops) game mechanics.

All functions compute and return values — no state mutation.
"""

import math
import random

from alignsim.src.models.entities import ProcessProject, ProcessProjectStatus
from alignsim.src.models.game_state import ActiveProcessBonus


def compute_project_progress(project: ProcessProject, ops_capacity: int) -> bool:
    """Check if a project advances this turn and return whether it completes.

    A project advances if ops_capacity >= ops_capacity_cost. It completes when
    progress_turns (after increment) reaches duration_turns.

    Note: this function does NOT mutate the project. The caller should update
    project.progress_turns and project.status based on the return value.
    """
    if project.status == ProcessProjectStatus.completed:
        return False
    if ops_capacity < project.ops_capacity_cost:
        return False
    # Would advance: check if this turn completes it
    return (project.progress_turns + 1) >= project.duration_turns


def compute_process_bonus(project: ProcessProject, rng: random.Random) -> float:
    """Compute the bonus value from a completed project.

    Base scales logarithmically with target team investment. Variance decreases
    with investment — heavy investment is a risk-reduction strategy.
    """
    base = project.bonus_base + project.bonus_scale_factor * math.log1p(
        project.target_team_capacity_invested
    )
    variance = (project.bonus_max - base) * 0.4 * math.exp(
        -0.1 * project.target_team_capacity_invested
    )
    result = rng.gauss(base, max(variance, 0))
    return min(max(result, 0), project.bonus_max)


def compute_degradation_pct(bonus: ActiveProcessBonus) -> float:
    """How degraded the bonus is: 0.0 = fresh (full), 1.0 = fully expired."""
    if bonus.bonus_duration_turns <= 0:
        return 1.0
    return 1.0 - (bonus.turns_remaining / bonus.bonus_duration_turns)


def compute_effective_bonus(bonus: ActiveProcessBonus) -> float:
    """Current effective bonus: a permanent floor plus a decaying spike on top.

    effective = floor + (peak - floor) * (turns_remaining / bonus_duration_turns)
    where floor = bonus_value * permanent_floor_fraction. With permanent_floor_fraction == 0
    this is the original linear-to-zero decay. turns_remaining is floored at 0 because a
    floored bonus sits pinned at 0 (spike fully decayed) while the floor persists.
    """
    floor = bonus.bonus_value * bonus.permanent_floor_fraction
    if bonus.bonus_duration_turns <= 0:
        return floor
    fraction = max(bonus.turns_remaining, 0) / bonus.bonus_duration_turns
    return floor + (bonus.bonus_value - floor) * fraction


def compute_maintenance_cost(bonus: ActiveProcessBonus) -> int:
    """Ops capacity needed in a single action to fully refresh a degraded-but-active bonus."""
    return max(1, round(compute_degradation_pct(bonus) * bonus.original_ops_capacity_cost))


def get_active_bonus(
    bonuses: list[ActiveProcessBonus],
    function: str,
    bonus_type: str,
) -> float:
    """Strongest active (degraded) bonus of a given type for a function.

    Same-type process improvements REPLACE rather than stack: a higher-tier process
    supersedes a lower-tier one (process improvement is incremental, not additive), so the
    strongest currently-effective bonus wins. E.g. a tier-0 conversion bonus at 0.20 plus a
    tier-1 at 0.30 yields 0.30 (a net +0.10 win), not 0.50. Returns 0.0 if none are active.
    """
    return max(
        (
            compute_effective_bonus(b)
            for b in bonuses
            if b.target_function == function and b.bonus_type == bonus_type
        ),
        default=0.0,
    )


def tick_bonus_durations(
    bonuses: list[ActiveProcessBonus],
) -> list[ActiveProcessBonus]:
    """Decrement turns_remaining on all active bonuses.

    A non-floored bonus is removed once its spike fully decays (new_turns <= 0). A floored
    bonus (permanent_floor_fraction > 0) is instead pinned at turns_remaining == 0 and kept
    indefinitely — its spike is spent but its permanent floor persists (and remains
    maintenance-refreshable).
    """
    remaining = []
    for b in bonuses:
        new_turns = b.turns_remaining - 1
        if new_turns > 0:
            remaining.append(b.model_copy(update={"turns_remaining": new_turns}))
        elif b.permanent_floor_fraction > 0:
            remaining.append(b.model_copy(update={"turns_remaining": 0}))
    return remaining
