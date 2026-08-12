"""Pure functions for customer-related game mechanics.

All functions compute and return values — no state mutation.
"""

import math
import random

from alignsim.src.models.entities import (
    Bug,
    BugSeverity,
    Customer,
    CustomerStage,
    EmergentNeed,
    Engagement,
    Feature,
    FeatureStatus,
    QualityLevel,
)
from alignsim.src.models.scenario import CalibrationParams


def compute_rubric_satisfaction(customer: Customer, features: dict[str, Feature]) -> float:
    """Compute how well the current product satisfies this customer's rubric.

    Returns a score between 0.0 and 1.0.
    """
    # Feature coverage: weighted combination of breadth (how many needs met) and depth (quality)
    # Breadth: what fraction of the customer's needs have ANY shipped feature
    # Depth: average satisfaction score across the shipped features
    # This means unshipped features don't drag the score down — they just don't contribute.
    feature_coverage_score = 0.0
    if customer.feature_needs:
        total_needs = len(customer.feature_needs)
        needs_met = 0
        total_satisfaction = 0.0
        for feature_id, quality_scores in customer.feature_needs.items():
            feature = features.get(feature_id)
            if feature is None:
                continue
            shipped_quality = _get_shipped_quality(feature)
            if shipped_quality is not None and shipped_quality in quality_scores:
                needs_met += 1
                total_satisfaction += quality_scores[shipped_quality]
        breadth = needs_met / total_needs
        depth = total_satisfaction / needs_met if needs_met > 0 else 0.0
        feature_coverage_score = breadth * 0.4 + depth * 0.6  # depth matters more than breadth

    # Price score: inversely correlated with customer size (larger customers less price-sensitive)
    price_score = 0.5 + (customer.size * 0.1)  # 0.6 to 1.0

    # Maturity score: based on ratio of polished features to total shipped
    shipped_features = [f for f in features.values() if f.status in _SHIPPED_STATUSES]
    if shipped_features:
        polished_count = sum(1 for f in shipped_features if f.status == FeatureStatus.shipped_polished)
        solid_count = sum(1 for f in shipped_features if f.status == FeatureStatus.shipped_solid)
        maturity_score = (polished_count * 1.0 + solid_count * 0.6) / len(shipped_features)
    else:
        maturity_score = 0.0

    # Support score: based on customer health (proxy for support quality)
    support_score = customer.health / 10.0

    # Weighted composite using customer's rubric weights
    score = (
        customer.rubric.feature_coverage * feature_coverage_score
        + customer.rubric.price * price_score
        + customer.rubric.maturity * maturity_score
        + customer.rubric.support * support_score
    )

    # 30% satisfaction penalty per timeline reset, floored at 30% of base
    if customer.timeline_resets > 0:
        penalty_multiplier = max(0.3, 1.0 - 0.30 * customer.timeline_resets)
        score *= penalty_multiplier

    return min(max(score, 0.0), 1.0)


def has_dealbreakers_met(customer: Customer, features: dict[str, Feature]) -> bool:
    """Check if all of a customer's dealbreaker features are shipped."""
    for feature_id in customer.dealbreakers:
        feature = features.get(feature_id)
        if feature is None or feature.status in (FeatureStatus.not_started, FeatureStatus.in_progress):
            return False
    return True


def compute_pricing_modifier(proposed: int, desired: int, calibration: CalibrationParams) -> float:
    """Compute how proposed price affects conversion probability.

    Returns a multiplier: 1.0 = no effect, <1.0 = penalty, >1.0 = bonus.
    When desired <= 0 (no pricing), returns 1.0.
    """
    if desired <= 0:
        return 1.0

    delta = (proposed - desired) / desired

    if abs(delta) < calibration.pricing_dead_zone:
        return 1.0

    if delta > 0:
        effective_delta = delta - calibration.pricing_dead_zone
        modifier = (
            calibration.pricing_penalty_floor
            + (1.0 - calibration.pricing_penalty_floor) * math.exp(-calibration.pricing_penalty_steepness * effective_delta)
        )
        return modifier

    effective_delta = abs(delta) - calibration.pricing_dead_zone
    modifier = 1.0 + (calibration.pricing_bonus_cap - 1.0) * (
        1.0 - math.exp(-calibration.pricing_bonus_steepness * effective_delta)
    )
    return modifier


def compute_sandbagged_price(desired: int, calibration: CalibrationParams, rng: random.Random) -> int:
    """Compute a sandbagged price hint that's always at or below the proposed price."""
    sandbag = max(0.0, calibration.pricing_sandbag_factor + rng.uniform(
        -calibration.pricing_sandbag_jitter, calibration.pricing_sandbag_jitter,
    ))
    return int(desired * (1.0 - sandbag))


def compute_conversion_probability(
    customer: Customer,
    sell_action_type: str,
    satisfaction: float,
    calibration: CalibrationParams,
    capacity_allocated: int = 0,
    sales_momentum: float = 0.0,
    process_bonus: float = 0.0,
    pricing_modifier: float = 1.0,
) -> float:
    """Compute the probability of a customer advancing to the next pipeline stage."""
    base_rate = _get_base_rate_for_stage(customer.stage, sell_action_type, calibration)
    if base_rate == 0.0:
        return 0.0

    # Apply rubric satisfaction modifier
    probability = base_rate * satisfaction

    # Engagement modifier
    if customer.engagement == Engagement.hot:
        probability *= 1.3
    elif customer.engagement == Engagement.cold:
        probability *= 0.4

    # Competitive pressure modifier
    probability *= max(0.3, 1.0 - customer.competitive_pressure * 0.3)

    # Demo capacity bonus: extra capacity above minimum provides diminishing returns boost
    if sell_action_type == "demo" and capacity_allocated > 0:
        min_cap = compute_sell_minimum_capacity(customer, "demo", calibration)
        extra = max(0, capacity_allocated - min_cap)
        if extra > 0:
            probability += calibration.demo_extra_capacity_bonus * math.log1p(extra)

    # Sales momentum: accumulated social proof and market presence
    probability *= (1.0 + sales_momentum)

    # Process bonus from ops sales projects
    probability *= (1.0 + process_bonus)

    # Pricing modifier
    probability *= pricing_modifier

    return min(max(probability, 0.0), calibration.max_close_probability)


def advance_pipeline_stage(
    customer: Customer,
    sell_action_type: str,
    probability: float,
    features: dict[str, Feature],
    calibration: CalibrationParams,
    rng: random.Random,
) -> CustomerStage | None:
    """Attempt to advance the customer's pipeline stage. Returns new stage or None if no change."""
    # Check dealbreakers for closing
    if customer.stage == CustomerStage.in_deal:
        if not has_dealbreakers_met(customer, features):
            return None
        rubric_satisfaction = compute_rubric_satisfaction(customer, features)
        threshold = customer.close_threshold if customer.close_threshold > 0 else calibration.min_rubric_for_close
        if rubric_satisfaction < threshold:
            return None

    # Roll against probability
    if rng.random() < probability:
        return _next_stage(customer.stage, sell_action_type)
    return None


def compute_health_delta(
    customer: Customer,
    active_bugs: list[Bug],
    cs_capacity_allocated: int,
    calibration: CalibrationParams,
    unmet_emergent_needs: list[EmergentNeed] | None = None,
) -> float:
    """Compute the health change for an active customer this turn.

    `unmet_emergent_needs` is the per-customer slice of emergent needs that are
    actively bleeding this turn (not met, and not paused by active build). It is
    passed as an argument — rather than read off global state — to keep this
    function pure, mirroring `active_bugs`.
    """
    delta = 0.0

    # Bug impact (negative)
    for bug in active_bugs:
        if customer.id in bug.affected_customers and not bug.is_resolved:
            if bug.severity == BugSeverity.critical:
                delta += calibration.health_bug_critical_delta
            elif bug.severity == BugSeverity.major:
                delta += calibration.health_bug_major_delta
            else:
                delta += calibration.health_bug_minor_delta

    # CS attention (positive, diminishing returns) — or event-driven decay if neglected.
    # Curve: delta * (1 + factor * ln(capacity)). No hard ceiling, but spreading capacity
    # across customers beats concentrating it. Replaces the old min(capacity, 3) hard cap.
    if cs_capacity_allocated > 0:
        delta += calibration.health_cs_attention_delta * (
            1.0 + calibration.cs_attention_log_factor * math.log(cs_capacity_allocated)
        )
    else:
        delta -= calibration.health_neglect_base_decay
        for bug in active_bugs:
            if customer.id in bug.affected_customers and not bug.is_resolved:
                delta -= calibration.health_bug_fester_rate * bug.turns_unresolved
        if customer.onboarding_turns_remaining > 0:
            delta -= calibration.health_onboarding_neglect_penalty

    # Emergent-need bleed (negative) — applies regardless of CS attention. Not running
    # health_checks doesn't stop the bleed; it only keeps CS blind to its cause.
    # turns_unmet already counts only turns past the grace window.
    if unmet_emergent_needs:
        for need in unmet_emergent_needs:
            delta -= calibration.emergent_need_bleed_rate * need.turns_unmet

    # Competitive pressure (negative)
    delta -= customer.competitive_pressure * 0.5

    # Natural regression toward 7.0 (mild)
    if customer.health > 7.0:
        delta -= 0.1
    elif customer.health < 7.0:
        delta += 0.1

    return delta


def check_churn(customer: Customer, calibration: CalibrationParams) -> bool:
    """Check if a customer should churn based on health history."""
    return (
        customer.stage == CustomerStage.customer
        and customer.health < calibration.churn_health_threshold
        and customer.turns_below_churn_threshold >= calibration.churn_consecutive_turns
    )


def check_expansion(customer: Customer, calibration: CalibrationParams) -> bool:
    """Check if a customer qualifies for expansion (deal_value increase)."""
    return (
        customer.stage == CustomerStage.customer
        and customer.health > calibration.expansion_health_threshold
        and customer.turns_above_expansion_threshold >= calibration.expansion_consecutive_turns
    )


def check_timeline_expiry(customer: Customer) -> bool:
    """Check if an active countdown has reached zero."""
    return customer.timeline_active and customer.timeline <= 0


def update_engagement(
    customer: Customer,
    sell_capacity: int,
    calibration: CalibrationParams | None = None,
) -> Engagement:
    """Compute new engagement level based on recent sales attention, scaled by customer size."""
    if calibration is not None:
        hot_threshold = max(1, int(calibration.engagement_hot_threshold * customer.size))
        warm_threshold = max(1, int(calibration.engagement_warm_threshold * customer.size))
    else:
        hot_threshold = 3
        warm_threshold = 1

    if sell_capacity >= hot_threshold:
        return Engagement.hot
    elif sell_capacity >= warm_threshold:
        return Engagement.warm if customer.engagement != Engagement.hot else Engagement.hot
    else:
        # Decay: hot -> warm -> cold
        if customer.engagement == Engagement.hot:
            return Engagement.warm
        elif customer.engagement == Engagement.warm:
            return Engagement.cold
        return Engagement.cold


def compute_sell_minimum_capacity(
    customer: Customer,
    sell_action_type: str,
    calibration: CalibrationParams,
) -> int:
    """Compute the minimum capacity required for a sell action on this customer."""
    base_costs = {
        "outbound": calibration.sell_base_cost_outbound,
        "demo": calibration.sell_base_cost_demo,
        "proposal": calibration.sell_base_cost_proposal,
        "negotiate": calibration.sell_base_cost_negotiate,
    }
    return base_costs.get(sell_action_type, 1) * customer.size


def compute_sales_momentum_update(
    current_momentum: float,
    deals_closed_this_turn: int,
    active_customer_count: int,
    shipped_feature_count: int,
    marketing_investment_lagged: int,
    calibration: CalibrationParams,
) -> float:
    """Compute new sales momentum value after this turn's events.

    Momentum grows from closing deals (social proof), shipping features (product
    credibility), and lagged marketing investment (brand awareness). Decays
    naturally each turn. Capped at sales_momentum_max.
    """
    momentum = current_momentum

    # Growth from deal closings
    momentum += deals_closed_this_turn * calibration.sales_momentum_per_close

    # Growth from shipped features (diminishing returns)
    momentum += calibration.sales_momentum_feature_factor * math.log1p(shipped_feature_count)

    # Growth from lagged marketing investment
    momentum += calibration.sales_momentum_marketing_factor * marketing_investment_lagged

    # Natural decay
    momentum -= calibration.sales_momentum_decay

    return max(0.0, min(momentum, calibration.sales_momentum_max))


# --- Private helpers ---

_SHIPPED_STATUSES = {FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished}


def _get_shipped_quality(feature: Feature) -> str | None:
    if feature.status == FeatureStatus.shipped_mvp:
        return QualityLevel.mvp.value
    elif feature.status == FeatureStatus.shipped_solid:
        return QualityLevel.solid.value
    elif feature.status == FeatureStatus.shipped_polished:
        return QualityLevel.polished.value
    return None


def _get_base_rate_for_stage(
    stage: CustomerStage,
    sell_action_type: str,
    calibration: CalibrationParams,
) -> float:
    """Get the base conversion rate for advancing from the current stage."""
    if stage == CustomerStage.lead and sell_action_type == "outbound":
        return calibration.lead_to_prospect_rate
    elif stage == CustomerStage.prospect and sell_action_type in ("outbound", "demo"):
        return calibration.prospect_to_qualified_rate
    elif stage == CustomerStage.qualified and sell_action_type == "demo":
        return calibration.qualified_to_in_deal_rate
    elif stage == CustomerStage.in_deal and sell_action_type in ("proposal", "negotiate"):
        return calibration.in_deal_to_closed_rate
    return 0.0


def _next_stage(stage: CustomerStage, sell_action_type: str) -> CustomerStage | None:
    """Get the next pipeline stage."""
    transitions = {
        CustomerStage.lead: CustomerStage.prospect,
        CustomerStage.prospect: CustomerStage.qualified,
        CustomerStage.qualified: CustomerStage.in_deal,
        CustomerStage.in_deal: CustomerStage.customer,
    }
    return transitions.get(stage)
