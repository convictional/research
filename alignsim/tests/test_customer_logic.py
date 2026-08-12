"""Tests for alignsim.src.engine.customer_logic — pure customer mechanics."""

import math
import random

import pytest

from alignsim.src.engine.customer_logic import (
    advance_pipeline_stage,
    check_churn,
    check_expansion,
    check_timeline_expiry,
    compute_conversion_probability,
    compute_health_delta,
    compute_pricing_modifier,
    compute_rubric_satisfaction,
    compute_sales_momentum_update,
    compute_sandbagged_price,
    compute_sell_minimum_capacity,
    has_dealbreakers_met,
    update_engagement,
)
from alignsim.src.models.entities import (
    BugSeverity,
    CustomerRubric,
    CustomerStage,
    Engagement,
    FeatureStatus,
)
from alignsim.src.models.scenario import CalibrationParams

from .factories import make_bug, make_customer, make_emergent_need, make_feature


# --- compute_rubric_satisfaction ---

def _balanced_rubric() -> CustomerRubric:
    return CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2)


def test_rubric_satisfaction_no_shipped_features():
    """Without any shipped features, only price and support contribute."""
    customer = make_customer(
        size=3, health=10.0, rubric=_balanced_rubric(),
        feature_needs={"F01": {"mvp": 0.5, "solid": 0.8, "polished": 1.0}},
    )
    feature = make_feature(id="F01", status=FeatureStatus.not_started)
    score = compute_rubric_satisfaction(customer, {"F01": feature})
    # feature_coverage = 0, maturity = 0
    # price = 0.5 + 3*0.1 = 0.8 → contributes 0.2*0.8 = 0.16
    # support = 10/10 = 1.0 → contributes 0.2*1.0 = 0.20
    assert score == pytest.approx(0.36)


def test_rubric_satisfaction_all_needs_met():
    """Polished shipped features with high need scores produce high overall satisfaction."""
    customer = make_customer(
        size=3, health=10.0, rubric=_balanced_rubric(),
        feature_needs={"F01": {"mvp": 0.5, "solid": 0.8, "polished": 1.0}},
    )
    feature = make_feature(id="F01", status=FeatureStatus.shipped_polished)
    score = compute_rubric_satisfaction(customer, {"F01": feature})
    # feature_coverage: breadth=1, depth=1.0 → 0.4*1+0.6*1=1.0 → contributes 0.4
    # maturity: 1 polished / 1 shipped = 1.0 → contributes 0.2
    # price 0.8 → 0.16. support 1.0 → 0.20.
    assert score == pytest.approx(0.96)


def test_rubric_satisfaction_breadth_depth_formula():
    """Coverage = 0.4 * breadth + 0.6 * depth."""
    customer = make_customer(
        size=1, health=0.0,  # zero out price/support contributions
        rubric=CustomerRubric(feature_coverage=1.0, price=0.0, maturity=0.0, support=0.0),
        feature_needs={
            "F01": {"mvp": 0.5},  # met at 0.5 satisfaction
            "F02": {"mvp": 0.9},  # not met (not shipped)
        },
    )
    features = {
        "F01": make_feature(id="F01", status=FeatureStatus.shipped_mvp),
        "F02": make_feature(id="F02", status=FeatureStatus.not_started),
    }
    score = compute_rubric_satisfaction(customer, features)
    # breadth = 1/2 = 0.5, depth = 0.5/1 = 0.5
    # coverage = 0.4*0.5 + 0.6*0.5 = 0.5
    assert score == pytest.approx(0.5)


def test_rubric_satisfaction_timeline_reset_penalty():
    """Each timeline reset reduces satisfaction by 30%, floor at 30% of base."""
    customer_no_resets = make_customer(
        size=3, health=10.0, rubric=_balanced_rubric(),
        feature_needs={"F01": {"polished": 1.0}}, timeline_resets=0,
    )
    customer_one_reset = customer_no_resets.model_copy(update={"timeline_resets": 1})
    customer_three_resets = customer_no_resets.model_copy(update={"timeline_resets": 3})

    feature = make_feature(id="F01", status=FeatureStatus.shipped_polished)
    base = compute_rubric_satisfaction(customer_no_resets, {"F01": feature})
    one = compute_rubric_satisfaction(customer_one_reset, {"F01": feature})
    three = compute_rubric_satisfaction(customer_three_resets, {"F01": feature})

    assert one == pytest.approx(base * 0.70, rel=1e-3)
    # 3 resets would give factor 1 - 0.9 = 0.1 but floor is 0.30
    assert three == pytest.approx(base * 0.30, rel=1e-3)


def test_rubric_satisfaction_maturity():
    """Maturity score weights polished=1.0, solid=0.6, mvp=0."""
    customer = make_customer(
        size=1, health=0.0,
        rubric=CustomerRubric(feature_coverage=0.0, price=0.0, maturity=1.0, support=0.0),
        feature_needs={},
    )
    features = {
        "A": make_feature(id="A", status=FeatureStatus.shipped_polished),
        "B": make_feature(id="B", status=FeatureStatus.shipped_solid),
        "C": make_feature(id="C", status=FeatureStatus.shipped_mvp),
    }
    # (1.0 + 0.6 + 0.0) / 3 = 0.5333
    score = compute_rubric_satisfaction(customer, features)
    assert score == pytest.approx(0.5333, abs=0.001)


# --- has_dealbreakers_met ---

def test_dealbreakers_all_met():
    customer = make_customer(dealbreakers=["F01", "F02"])
    features = {
        "F01": make_feature(id="F01", status=FeatureStatus.shipped_mvp),
        "F02": make_feature(id="F02", status=FeatureStatus.shipped_polished),
    }
    assert has_dealbreakers_met(customer, features) is True


def test_dealbreakers_unmet():
    customer = make_customer(dealbreakers=["F01", "F02"])
    features = {
        "F01": make_feature(id="F01", status=FeatureStatus.shipped_mvp),
        "F02": make_feature(id="F02", status=FeatureStatus.in_progress),
    }
    assert has_dealbreakers_met(customer, features) is False


def test_dealbreakers_empty_list():
    customer = make_customer(dealbreakers=[])
    assert has_dealbreakers_met(customer, {}) is True


# --- compute_conversion_probability ---

def test_conversion_probability_basic():
    """At lead/outbound with full satisfaction, prob = base_rate (0.20)."""
    cal = CalibrationParams()
    customer = make_customer(stage=CustomerStage.lead, engagement=Engagement.warm)
    p = compute_conversion_probability(customer, "outbound", satisfaction=1.0,
                                       calibration=cal)
    assert p == pytest.approx(cal.lead_to_prospect_rate)


def test_conversion_probability_invalid_action_returns_zero():
    """Wrong action for the stage returns 0."""
    cal = CalibrationParams()
    customer = make_customer(stage=CustomerStage.lead)
    assert compute_conversion_probability(customer, "demo", 1.0, cal) == 0.0


def test_conversion_probability_cap_at_max_close():
    """Probability is capped at calibration.max_close_probability (default 0.70)."""
    cal = CalibrationParams()
    customer = make_customer(stage=CustomerStage.qualified, engagement=Engagement.hot)
    # Push above cap with momentum + bonus
    p = compute_conversion_probability(customer, "demo", satisfaction=1.0,
                                       calibration=cal, sales_momentum=0.40,
                                       process_bonus=0.5)
    assert p == cal.max_close_probability


def test_conversion_probability_engagement_modifiers():
    """Hot=1.3x, warm=1.0x (no modifier), cold=0.4x."""
    cal = CalibrationParams()
    base_cust = make_customer(stage=CustomerStage.lead)
    p_warm = compute_conversion_probability(
        base_cust.model_copy(update={"engagement": Engagement.warm}),
        "outbound", 1.0, cal,
    )
    p_hot = compute_conversion_probability(
        base_cust.model_copy(update={"engagement": Engagement.hot}),
        "outbound", 1.0, cal,
    )
    p_cold = compute_conversion_probability(
        base_cust.model_copy(update={"engagement": Engagement.cold}),
        "outbound", 1.0, cal,
    )
    assert p_hot == pytest.approx(p_warm * 1.3)
    assert p_cold == pytest.approx(p_warm * 0.4)


def test_conversion_probability_competitive_pressure():
    """Pressure factor = max(0.3, 1 - pressure*0.3); floors at 0.3."""
    cal = CalibrationParams()
    no_pressure = make_customer(stage=CustomerStage.lead, competitive_pressure=0.0)
    mid_pressure = make_customer(stage=CustomerStage.lead, competitive_pressure=1.0)
    extreme_pressure = make_customer(stage=CustomerStage.lead, competitive_pressure=10.0)

    p_none = compute_conversion_probability(no_pressure, "outbound", 1.0, cal)
    p_mid = compute_conversion_probability(mid_pressure, "outbound", 1.0, cal)
    p_extreme = compute_conversion_probability(extreme_pressure, "outbound", 1.0, cal)

    assert p_mid == pytest.approx(p_none * 0.7)  # 1 - 1.0*0.3 = 0.7
    assert p_extreme == pytest.approx(p_none * 0.3)  # floored


def test_conversion_probability_demo_capacity_bonus():
    """Demo with extra capacity above min adds 0.08*log1p(extra)."""
    cal = CalibrationParams()
    # qualified→in_deal demo, size=1 → min_cap=1
    customer = make_customer(stage=CustomerStage.qualified, size=1, engagement=Engagement.warm)
    base_p = compute_conversion_probability(customer, "demo", 1.0, cal,
                                            capacity_allocated=1)
    bonus_p = compute_conversion_probability(customer, "demo", 1.0, cal,
                                             capacity_allocated=10)
    # Extra above min = 10 - 1 = 9. Bonus = 0.08 * ln(10) ≈ 0.184
    expected_bonus = cal.demo_extra_capacity_bonus * math.log1p(9)
    assert bonus_p == pytest.approx(base_p + expected_bonus, rel=1e-3)


def test_conversion_probability_momentum_and_bonus():
    """Momentum and process bonus both multiply probability."""
    cal = CalibrationParams()
    customer = make_customer(stage=CustomerStage.lead, engagement=Engagement.warm)
    base = compute_conversion_probability(customer, "outbound", 1.0, cal)
    with_mom = compute_conversion_probability(customer, "outbound", 1.0, cal,
                                              sales_momentum=0.10)
    with_proc = compute_conversion_probability(customer, "outbound", 1.0, cal,
                                               process_bonus=0.10)
    assert with_mom == pytest.approx(base * 1.10)
    assert with_proc == pytest.approx(base * 1.10)


# --- advance_pipeline_stage ---

def test_advance_pipeline_rng_roll():
    """Probability 1.0 always advances; 0.0 never advances."""
    cal = CalibrationParams()
    customer = make_customer(stage=CustomerStage.lead)
    rng = random.Random(0)
    # Prob 1.0 — random() is in [0,1), so always less than 1
    for _ in range(100):
        new_stage = advance_pipeline_stage(
            customer, "outbound", 1.0, {}, cal, rng,
        )
        assert new_stage == CustomerStage.prospect
    # Prob 0.0 — never advances
    for _ in range(100):
        assert advance_pipeline_stage(customer, "outbound", 0.0, {}, cal, rng) is None


def test_advance_pipeline_in_deal_dealbreaker_gate():
    """In-deal customer with unmet dealbreakers stays at in_deal regardless of roll."""
    cal = CalibrationParams()
    customer = make_customer(
        stage=CustomerStage.in_deal,
        dealbreakers=["F01"],
        feature_needs={"F01": {"mvp": 1.0}},
    )
    features = {"F01": make_feature(id="F01", status=FeatureStatus.in_progress)}
    rng = random.Random(0)
    new_stage = advance_pipeline_stage(customer, "negotiate", 1.0, features, cal, rng)
    assert new_stage is None


def test_advance_pipeline_in_deal_min_rubric_gate():
    """In-deal customer below min_rubric_for_close cannot close."""
    cal = CalibrationParams()
    # Customer with no shipped features, low support → low satisfaction
    customer = make_customer(
        stage=CustomerStage.in_deal,
        size=1, health=0.0,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        dealbreakers=[],
    )
    rng = random.Random(0)
    new_stage = advance_pipeline_stage(customer, "negotiate", 1.0, {}, cal, rng)
    assert new_stage is None


def test_advance_pipeline_in_deal_closes_when_gates_pass():
    """In-deal customer with met dealbreakers and high satisfaction can advance."""
    cal = CalibrationParams()
    customer = make_customer(
        stage=CustomerStage.in_deal,
        size=5, health=10.0,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        dealbreakers=["F01"],
        feature_needs={"F01": {"polished": 1.0}},
    )
    features = {"F01": make_feature(id="F01", status=FeatureStatus.shipped_polished)}
    rng = random.Random(0)
    new_stage = advance_pipeline_stage(customer, "negotiate", 1.0, features, cal, rng)
    assert new_stage == CustomerStage.customer


# --- compute_health_delta ---

def test_health_delta_bug_penalties():
    """Critical=-2.0, major=-1.0, minor=-0.5 per affecting bug."""
    cal = CalibrationParams()
    customer = make_customer(id="C01", stage=CustomerStage.customer, health=7.0)
    bugs = [
        make_bug(id="B1", severity=BugSeverity.critical, affected_customers=["C01"]),
        make_bug(id="B2", severity=BugSeverity.major, affected_customers=["C01"]),
        make_bug(id="B3", severity=BugSeverity.minor, affected_customers=["C01"]),
    ]
    delta = compute_health_delta(customer, bugs, cs_capacity_allocated=0, calibration=cal)
    # bugs: -2 -1 -0.5 = -3.5; base neglect -0.1; fester=0 (turns_unresolved=0); regression=0
    assert delta == pytest.approx(-3.6)


def test_health_delta_cs_attention_diminishing_returns():
    """CS attention follows a log curve: delta * (1 + factor * ln(capacity)). No hard cap,
    but each extra unit on the same customer is worth less than the last."""
    cal = CalibrationParams()  # delta=1.0, factor=0.8
    customer = make_customer(id="C01", stage=CustomerStage.customer, health=7.0)
    f = cal.cs_attention_log_factor

    def expected(cap: int) -> float:
        return cal.health_cs_attention_delta * (1.0 + f * math.log(cap))

    # health==7.0 → regression term is 0, so delta is purely the attention curve.
    assert compute_health_delta(customer, [], 1, cal) == pytest.approx(expected(1))  # ln1=0 → 1.0
    assert compute_health_delta(customer, [], 2, cal) == pytest.approx(expected(2))  # ~1.55
    assert compute_health_delta(customer, [], 3, cal) == pytest.approx(expected(3))  # ~1.88
    assert compute_health_delta(customer, [], 6, cal) == pytest.approx(expected(6))  # ~2.43

    # Strictly increasing but concave: each marginal unit adds less.
    d1 = compute_health_delta(customer, [], 1, cal)
    d2 = compute_health_delta(customer, [], 2, cal)
    d3 = compute_health_delta(customer, [], 3, cal)
    assert d1 < d2 < d3
    assert (d2 - d1) > (d3 - d2)


def test_health_delta_emergent_need_bleed():
    """Unmet emergent needs bleed health at bleed_rate * turns_unmet, regardless of CS attention."""
    cal = CalibrationParams()
    customer = make_customer(id="C01", stage=CustomerStage.customer, health=7.0)
    # turns_unmet=2 → -bleed_rate*2; with cs=1 attention (+1.0), regression 0.
    need = make_emergent_need(customer_id="C01", feature_id="F02", turns_unmet=2)
    delta = compute_health_delta(customer, [], 1, cal, unmet_emergent_needs=[need])
    assert delta == pytest.approx(1.0 - cal.emergent_need_bleed_rate * 2)


def test_health_delta_emergent_need_no_bleed_within_grace():
    """A need still in its grace window has turns_unmet=0 → no bleed contribution."""
    cal = CalibrationParams()
    customer = make_customer(id="C01", stage=CustomerStage.customer, health=7.0)
    need = make_emergent_need(customer_id="C01", feature_id="F02", turns_unmet=0)
    delta = compute_health_delta(customer, [], 1, cal, unmet_emergent_needs=[need])
    assert delta == pytest.approx(1.0)  # pure attention, no bleed


def test_health_delta_no_cs_base_neglect():
    """Zero CS capacity, no bugs, past onboarding → base neglect only."""
    cal = CalibrationParams()
    customer = make_customer(id="C01", stage=CustomerStage.customer, health=7.0)
    delta = compute_health_delta(customer, [], 0, cal)
    assert delta == pytest.approx(-cal.health_neglect_base_decay)


def test_health_delta_regression_toward_7():
    """Health > 7 regresses by -0.1, health < 7 regresses by +0.1."""
    cal = CalibrationParams()
    above = make_customer(id="C01", stage=CustomerStage.customer, health=9.0)
    below = make_customer(id="C02", stage=CustomerStage.customer, health=5.0)
    # Use cs=1 to neutralize the -0.3 decay (gives +1.0 cs)
    d_above = compute_health_delta(above, [], 1, cal)
    d_below = compute_health_delta(below, [], 1, cal)
    # above: cs=+1.0, regression=-0.1 → 0.9
    assert d_above == pytest.approx(0.9)
    # below: cs=+1.0, regression=+0.1 → 1.1
    assert d_below == pytest.approx(1.1)


def test_health_delta_neglect_with_festering_bugs():
    """Unresolved bugs compound health loss when CS is absent."""
    cal = CalibrationParams()
    customer = make_customer(id="C01", stage=CustomerStage.customer, health=7.0)
    bugs = [
        make_bug(id="B1", severity=BugSeverity.major, affected_customers=["C01"],
                 turns_unresolved=4),
    ]
    delta = compute_health_delta(customer, bugs, cs_capacity_allocated=0, calibration=cal)
    # severity: -1.0; base neglect: -0.1; fester: -0.05*4 = -0.2; regression=0
    assert delta == pytest.approx(-1.3)


def test_health_delta_onboarding_neglect():
    """Onboarding customers without CS attention get an extra penalty."""
    cal = CalibrationParams()
    customer = make_customer(id="C01", stage=CustomerStage.customer, health=7.0,
                             onboarding_turns_remaining=3)
    delta = compute_health_delta(customer, [], 0, cal)
    # base neglect: -0.1; onboarding penalty: -0.3; regression=0
    assert delta == pytest.approx(-0.4)


def test_health_delta_no_fester_when_cs_present():
    """Bug fester and onboarding penalty only apply when CS is absent."""
    cal = CalibrationParams()
    customer = make_customer(id="C01", stage=CustomerStage.customer, health=7.0,
                             onboarding_turns_remaining=3)
    bugs = [
        make_bug(id="B1", severity=BugSeverity.major, affected_customers=["C01"],
                 turns_unresolved=4),
    ]
    delta = compute_health_delta(customer, bugs, cs_capacity_allocated=1, calibration=cal)
    # severity: -1.0; cs: +1.0; no fester/onboarding penalty; regression=0
    assert delta == pytest.approx(0.0)


# --- close_threshold ---

def test_close_threshold_used_when_set():
    """Per-customer close_threshold gates the deal instead of global default."""
    cal = CalibrationParams()
    # satisfaction ~0.82: passes global 0.75 but fails per-customer 0.95
    customer = make_customer(
        stage=CustomerStage.in_deal, size=1, health=5.0,
        close_threshold=0.95,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        dealbreakers=["F01"],
        feature_needs={"F01": {"polished": 1.0}},
    )
    features = {"F01": make_feature(id="F01", status=FeatureStatus.shipped_polished)}
    rng = random.Random(0)
    new_stage = advance_pipeline_stage(customer, "negotiate", 1.0, features, cal, rng)
    assert new_stage is None  # blocked by high close_threshold


def test_close_threshold_fallback_to_calibration():
    """close_threshold=0 falls back to calibration.min_rubric_for_close."""
    cal = CalibrationParams()
    customer = make_customer(
        stage=CustomerStage.in_deal, size=5, health=10.0,
        close_threshold=0.0,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        dealbreakers=["F01"],
        feature_needs={"F01": {"polished": 1.0}},
    )
    features = {"F01": make_feature(id="F01", status=FeatureStatus.shipped_polished)}
    rng = random.Random(0)
    new_stage = advance_pipeline_stage(customer, "negotiate", 1.0, features, cal, rng)
    assert new_stage == CustomerStage.customer


# --- check_churn / check_expansion / check_timeline_expiry ---

def test_churn_check():
    cal = CalibrationParams()  # threshold 3.0, consecutive 2
    bad_long = make_customer(stage=CustomerStage.customer, health=2.0,
                             turns_below_churn_threshold=2)
    bad_short = make_customer(stage=CustomerStage.customer, health=2.0,
                              turns_below_churn_threshold=1)
    fine = make_customer(stage=CustomerStage.customer, health=5.0,
                         turns_below_churn_threshold=10)
    not_active = make_customer(stage=CustomerStage.lead, health=1.0,
                               turns_below_churn_threshold=5)
    assert check_churn(bad_long, cal) is True
    assert check_churn(bad_short, cal) is False
    assert check_churn(fine, cal) is False
    assert check_churn(not_active, cal) is False


def test_expansion_check():
    cal = CalibrationParams()  # threshold 8.0, consecutive 4
    happy_long = make_customer(stage=CustomerStage.customer, health=9.0,
                               turns_above_expansion_threshold=4)
    happy_short = make_customer(stage=CustomerStage.customer, health=9.0,
                                turns_above_expansion_threshold=3)
    just_ok = make_customer(stage=CustomerStage.customer, health=7.5,
                            turns_above_expansion_threshold=10)
    assert check_expansion(happy_long, cal) is True
    assert check_expansion(happy_short, cal) is False
    assert check_expansion(just_ok, cal) is False


def test_timeline_expiry():
    """Active timeline at 0 expires; inactive or positive timeline does not."""
    expired = make_customer(timeline_active=True, timeline=0)
    still_running = make_customer(timeline_active=True, timeline=3)
    inactive = make_customer(timeline_active=False, timeline=0)
    assert check_timeline_expiry(expired) is True
    assert check_timeline_expiry(still_running) is False
    assert check_timeline_expiry(inactive) is False


# --- update_engagement ---

def test_engagement_thresholds_and_decay():
    """Hot/warm/cold based on capacity vs size; decay one tier per turn without attention."""
    cal = CalibrationParams()  # hot threshold 1.0*size, warm 0.4*size
    cold_cust = make_customer(size=2, engagement=Engagement.cold)
    warm_cust = make_customer(size=2, engagement=Engagement.warm)
    hot_cust = make_customer(size=2, engagement=Engagement.hot)

    # Capacity 2 = hot threshold for size 2 → hot
    assert update_engagement(cold_cust, sell_capacity=2, calibration=cal) == Engagement.hot
    # Capacity 1 = warm threshold for size 2 → warm
    assert update_engagement(cold_cust, sell_capacity=1, calibration=cal) == Engagement.warm
    # Capacity 0 → decay
    assert update_engagement(hot_cust, 0, cal) == Engagement.warm
    assert update_engagement(warm_cust, 0, cal) == Engagement.cold
    assert update_engagement(cold_cust, 0, cal) == Engagement.cold


# --- compute_sell_minimum_capacity ---

def test_sell_minimum_capacity():
    """Minimum capacity = base_cost × customer.size."""
    cal = CalibrationParams()
    small = make_customer(size=1)
    big = make_customer(size=5)
    assert compute_sell_minimum_capacity(small, "outbound", cal) == cal.sell_base_cost_outbound * 1
    assert compute_sell_minimum_capacity(big, "demo", cal) == cal.sell_base_cost_demo * 5
    assert compute_sell_minimum_capacity(big, "proposal", cal) == cal.sell_base_cost_proposal * 5
    assert compute_sell_minimum_capacity(big, "negotiate", cal) == cal.sell_base_cost_negotiate * 5
    # Unknown action defaults to base cost 1
    assert compute_sell_minimum_capacity(big, "unknown", cal) == 5


# --- compute_sales_momentum_update ---

def test_sales_momentum_update_growth():
    """Momentum grows from deals, features, and lagged marketing."""
    cal = CalibrationParams()
    new_mom = compute_sales_momentum_update(
        current_momentum=0.0,
        deals_closed_this_turn=2,
        active_customer_count=5,
        shipped_feature_count=3,
        marketing_investment_lagged=10,
        calibration=cal,
    )
    # 2*0.08 + log1p(3)*0.01 + 10*0.005 - 0.01
    expected = 2 * cal.sales_momentum_per_close \
        + cal.sales_momentum_feature_factor * math.log1p(3) \
        + cal.sales_momentum_marketing_factor * 10 \
        - cal.sales_momentum_decay
    assert new_mom == pytest.approx(expected)


def test_sales_momentum_decay_floor():
    """Idle momentum decays but never goes below 0."""
    cal = CalibrationParams()
    decayed = compute_sales_momentum_update(0.005, 0, 0, 0, 0, cal)
    assert decayed == 0.0


def test_sales_momentum_capped():
    """Momentum is capped at sales_momentum_max."""
    cal = CalibrationParams()
    capped = compute_sales_momentum_update(0.40, 100, 0, 0, 0, cal)
    assert capped == cal.sales_momentum_max


# --- compute_pricing_modifier ---

def test_pricing_modifier_at_desired_price():
    """Proposed == desired → delta=0 → modifier=1.0."""
    cal = CalibrationParams()
    assert compute_pricing_modifier(1000, 1000, cal) == 1.0


def test_pricing_modifier_within_dead_zone():
    """Small delta within dead zone returns 1.0."""
    cal = CalibrationParams(pricing_dead_zone=0.05)
    # 2% above desired → within 5% dead zone
    assert compute_pricing_modifier(1020, 1000, cal) == 1.0
    # 3% below desired → within dead zone
    assert compute_pricing_modifier(970, 1000, cal) == 1.0


def test_pricing_modifier_too_expensive():
    """Positive delta (too expensive) → penalty approaching floor."""
    cal = CalibrationParams()
    # 50% above desired
    modifier = compute_pricing_modifier(1500, 1000, cal)
    assert modifier < 1.0
    assert modifier >= cal.pricing_penalty_floor
    # Very expensive → closer to floor
    modifier_extreme = compute_pricing_modifier(3000, 1000, cal)
    assert modifier_extreme < modifier
    assert modifier_extreme >= cal.pricing_penalty_floor


def test_pricing_modifier_discount():
    """Negative delta (discount) → bonus approaching cap."""
    cal = CalibrationParams()
    # 30% below desired
    modifier = compute_pricing_modifier(700, 1000, cal)
    assert modifier > 1.0
    assert modifier <= cal.pricing_bonus_cap


def test_pricing_modifier_free_product():
    """Delta=-1.0 (free) → modifier near cap."""
    cal = CalibrationParams()
    modifier = compute_pricing_modifier(0, 1000, cal)
    assert modifier > 1.3
    assert modifier <= cal.pricing_bonus_cap


def test_pricing_modifier_no_desired_price():
    """Desired=0 → 1.0 (no pricing mechanic)."""
    cal = CalibrationParams()
    assert compute_pricing_modifier(500, 0, cal) == 1.0


# --- compute_sandbagged_price ---

def test_sandbagged_price_below_desired():
    """Sandbagged price is always less than or equal to desired."""
    cal = CalibrationParams()
    rng = random.Random(42)
    for _ in range(100):
        price = compute_sandbagged_price(1000, cal, rng)
        assert price <= 1000


def test_sandbagged_price_floor_at_zero():
    """Sandbag factor is floored at 0, so sandbagged price <= desired."""
    cal = CalibrationParams(pricing_sandbag_factor=0.0, pricing_sandbag_jitter=0.05)
    rng = random.Random(42)
    for _ in range(100):
        price = compute_sandbagged_price(1000, cal, rng)
        assert price <= 1000


def test_sandbagged_price_deterministic():
    """Same seed produces same sandbagged price."""
    cal = CalibrationParams()
    p1 = compute_sandbagged_price(1000, cal, random.Random(42))
    p2 = compute_sandbagged_price(1000, cal, random.Random(42))
    assert p1 == p2


# --- conversion with pricing modifier ---

def test_conversion_with_pricing_modifier():
    """Pricing modifier multiplies conversion probability."""
    cal = CalibrationParams()
    customer = make_customer(stage=CustomerStage.lead, engagement=Engagement.warm)
    base = compute_conversion_probability(customer, "outbound", 1.0, cal)
    boosted = compute_conversion_probability(customer, "outbound", 1.0, cal,
                                             pricing_modifier=1.2)
    assert boosted == pytest.approx(base * 1.2)


def test_max_close_probability_from_calibration():
    """Cap uses calibration.max_close_probability, not hardcoded 0.60."""
    cal = CalibrationParams(max_close_probability=0.50)
    customer = make_customer(stage=CustomerStage.qualified, engagement=Engagement.hot)
    p = compute_conversion_probability(customer, "demo", satisfaction=1.0,
                                       calibration=cal, sales_momentum=0.40,
                                       process_bonus=0.5)
    assert p == 0.50
