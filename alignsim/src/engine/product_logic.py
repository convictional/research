"""Pure functions for product/engineering game mechanics.

All functions compute and return values — no state mutation.
"""

import math
import random

from alignsim.src.engine.customer_generator import get_adjacent_features
from alignsim.src.models.entities import (
    Bug,
    BugSeverity,
    CustomerStage,
    EmergentNeed,
    Feature,
    FeatureStatus,
    QualityLevel,
)
from alignsim.src.models.scenario import CalibrationParams


def apply_build_progress(
    feature: Feature,
    capacity: int,
    target_quality: QualityLevel,
    calibration: "CalibrationParams | None" = None,
) -> tuple[float, FeatureStatus]:
    """Compute new progress and status after allocating build capacity.

    Applies diminishing returns when capacity exceeds optimal crew size.
    Enforces a max progress-per-turn cap and minimum turns requirement.

    Returns (new_progress, new_status).
    """
    total_cost = feature.cost.get(target_quality.value, 0)
    if total_cost == 0:
        return feature.progress, feature.status

    # If upgrading from a lower shipped quality, only pay the delta
    current_cost = _shipped_cost(feature)
    remaining_cost = max(total_cost - current_cost, 0)
    if remaining_cost == 0:
        return 100.0, _status_for_quality(target_quality)

    # Apply diminishing returns if calibration provided
    effective_capacity = float(capacity)
    if calibration is not None:
        optimal = calibration.build_optimal_capacity
        if capacity > optimal:
            excess = capacity - optimal
            efficiency = (optimal / capacity) ** calibration.build_overallocation_alpha
            effective_capacity = optimal + excess * efficiency
        # Cap: no more than build_max_progress_pct of remaining work in one turn
        max_capacity_this_turn = remaining_cost * (calibration.build_max_progress_pct / 100.0)
        effective_capacity = min(effective_capacity, max_capacity_this_turn)

    # Progress is percentage of remaining cost completed
    progress_increment = (effective_capacity / remaining_cost) * 100.0
    new_progress = min(feature.progress + progress_increment, 100.0)

    # Enforce minimum turns based on remaining work (delta cost), not total cost.
    # Upgrading MVP→solid should scale with the upgrade delta, not the full solid cost.
    if calibration is not None and new_progress >= 100.0:
        min_turns = max(2, math.ceil(remaining_cost * calibration.build_min_turns_factor))
        turns_worked_after = feature.turns_worked + 1  # this turn counts
        if turns_worked_after < min_turns:
            new_progress = 99.9
            return new_progress, FeatureStatus.in_progress

    if new_progress >= 100.0:
        return 100.0, _status_for_quality(target_quality)
    else:
        return new_progress, FeatureStatus.in_progress


def compute_tech_debt_delta(
    build_capacity_by_quality: dict[str, int],
    infra_capacity: int,
    calibration: CalibrationParams,
) -> float:
    """Compute the net change in tech debt for this turn.

    Args:
        build_capacity_by_quality: total capacity spent on building at each quality level
        infra_capacity: total capacity spent on infrastructure
        calibration: game calibration parameters
    """
    debt_increase = 0.0
    for quality, capacity in build_capacity_by_quality.items():
        if quality == QualityLevel.mvp.value:
            debt_increase += (capacity / 10.0) * calibration.debt_per_10_mvp_units
        elif quality == QualityLevel.solid.value:
            debt_increase += (capacity / 10.0) * calibration.debt_per_10_solid_units
        elif quality == QualityLevel.polished.value:
            debt_increase += (capacity / 10.0) * calibration.debt_per_10_polished_units

    debt_decrease = (infra_capacity / 5.0) * calibration.debt_reduction_per_5_infra

    return debt_increase - debt_decrease


def inject_bugs(
    debt_level: float,
    shipped_features: list[Feature],
    calibration: CalibrationParams,
    customers: dict,
    next_bug_id: int,
    turn: int,
    rng: random.Random,
    bug_rate_reduction: float = 0.0,
) -> list[Bug]:
    """Generate new bugs based on tech debt level and shipped feature complexity.

    Uses Poisson sampling with lambda = max(0.5, debt_level * multiplier * (1 - reduction)).
    The 0.5 floor is intentional: shipped software always has a baseline bug rate regardless
    of tech debt or process investment — zero-bug software doesn't exist.
    bug_rate_reduction (from Ops process bonus) reduces the lambda but cannot eliminate it.
    """
    if not shipped_features:
        return []

    lam = max(0.5, debt_level * calibration.bug_injection_multiplier * (1.0 - bug_rate_reduction))
    num_bugs = poisson_sample(lam, rng)

    bugs = []
    for i in range(num_bugs):
        # Weight feature selection by bug_rate_modifier
        weights = [f.bug_rate_modifier for f in shipped_features]
        total_weight = sum(weights)
        if total_weight == 0:
            continue
        feature = _weighted_choice(shipped_features, weights, rng)

        # Determine severity
        severity = _sample_severity(calibration, rng)

        # Determine affected customers (those who depend on this feature)
        affected = [
            cid for cid, customer in customers.items()
            if feature.id in customer.feature_needs and customer.stage == CustomerStage.customer
        ]

        bug = Bug(
            id=f"BUG_{next_bug_id + i:03d}",
            severity=severity,
            feature_id=feature.id,
            turn_injected=turn,
            affected_customers=affected,
        )
        bugs.append(bug)

    return bugs


def inject_emergent_needs(
    calibration: CalibrationParams,
    customers: dict,
    features: dict,
    existing_needs: list[EmergentNeed],
    next_id: int,
    turn: int,
    rng: random.Random,
) -> list[EmergentNeed]:
    """Generate new emergent feature needs on active customers.

    Mirrors inject_bugs: a Poisson-sampled count (deterministic per seed) with a
    floor so needs still appear with a small customer base.

        lam = max(emergent_need_injection_floor,
                  emergent_need_injection_rate * num_active_customers)

    Each draw seeds a need on a randomly chosen active customer for a feature they
    do NOT already know they need and do not already have an open need for. The
    candidate pool is constrained to DAG-adjacent features (plausible asks),
    falling back to any non-known feature. A draw is skipped if nothing is eligible.

    NOTE: this is hidden ground truth. The returned needs are revealed to CS only
    via a health_check action — callers must never surface an injection to an agent.
    """
    active_customers = [c for c in customers.values() if c.stage == CustomerStage.customer]
    if not active_customers:
        return []

    lam = max(
        calibration.emergent_need_injection_floor,
        calibration.emergent_need_injection_rate * len(active_customers),
    )
    num_needs = poisson_sample(lam, rng)
    if num_needs == 0:
        return []

    # Feature IDs each customer already has an open (unmet, unexpired) need for —
    # avoid stacking duplicates on the same customer/feature.
    open_by_customer: dict[str, set[str]] = {}
    for need in existing_needs:
        if not need.is_met and not need.is_expired:
            open_by_customer.setdefault(need.customer_id, set()).add(need.feature_id)

    shipped_statuses = {
        FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished,
    }

    def _eligible(fid: str, excluded: set[str]) -> bool:
        # Exclude known needs, already-open needs, and already-shipped features (a need for
        # something the customer already has would be degenerate / instantly met).
        feature = features.get(fid)
        return (
            feature is not None
            and fid not in excluded
            and feature.status not in shipped_statuses
        )

    new_needs: list[EmergentNeed] = []
    for _ in range(num_needs):
        customer = rng.choice(active_customers)
        excluded = set(customer.known_needs) | open_by_customer.get(customer.id, set())

        # Prefer features one DAG hop from the customer's existing known needs.
        adjacent: set[str] = set()
        for known in customer.known_needs:
            adjacent.update(get_adjacent_features(known, features))
        candidate_pool = [fid for fid in sorted(adjacent) if _eligible(fid, excluded)]

        # Fall back to any non-known, non-open, non-shipped feature.
        if not candidate_pool:
            candidate_pool = [fid for fid in sorted(features.keys()) if _eligible(fid, excluded)]

        if not candidate_pool:
            continue  # nothing eligible for this customer — skip the draw

        feature_id = rng.choice(candidate_pool)

        need = EmergentNeed(
            id=f"EN_{next_id + len(new_needs):03d}",
            customer_id=customer.id,
            feature_id=feature_id,
            turn_injected=turn,
        )
        new_needs.append(need)
        open_by_customer.setdefault(customer.id, set()).add(feature_id)

    return new_needs


def compute_bug_fix_progress(bug: Bug, capacity: int) -> bool:
    """Determine if a bug is fixed with the given capacity.

    Returns True if fixed.
    """
    required = _bug_fix_cost(bug.severity)
    return capacity >= required


def poisson_sample(lam: float, rng: random.Random) -> int:
    """Sample from a Poisson distribution using inverse transform method.

    Exact for lambda < 20 (well within our range since debt_level * 0.3 stays below ~6).
    """
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p < L:
            return k - 1


# --- Private helpers ---


def _shipped_cost(feature: Feature) -> int:
    """Get the cost already invested based on current shipped status."""
    if feature.status == FeatureStatus.shipped_polished:
        return feature.cost.get(QualityLevel.polished.value, 0)
    elif feature.status == FeatureStatus.shipped_solid:
        return feature.cost.get(QualityLevel.solid.value, 0)
    elif feature.status == FeatureStatus.shipped_mvp:
        return feature.cost.get(QualityLevel.mvp.value, 0)
    return 0


def _status_for_quality(quality: QualityLevel) -> FeatureStatus:
    return {
        QualityLevel.mvp: FeatureStatus.shipped_mvp,
        QualityLevel.solid: FeatureStatus.shipped_solid,
        QualityLevel.polished: FeatureStatus.shipped_polished,
    }[quality]


def _sample_severity(calibration: CalibrationParams, rng: random.Random) -> BugSeverity:
    roll = rng.random()
    if roll < calibration.bug_critical_pct:
        return BugSeverity.critical
    elif roll < calibration.bug_critical_pct + calibration.bug_major_pct:
        return BugSeverity.major
    return BugSeverity.minor


def _bug_fix_cost(severity: BugSeverity) -> int:
    return {
        BugSeverity.critical: 4,
        BugSeverity.major: 2,
        BugSeverity.minor: 1,
    }[severity]


def _weighted_choice[T](items: list[T], weights: list[float], rng: random.Random) -> T:
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if r <= cumulative:
            return item
    return items[-1]
