"""Tests for alignsim.src.engine.market_logic — marketing, discovery, competitor mechanics."""

import random

import pytest

from alignsim.src.engine.market_logic import (
    compute_awareness_reveal,
    apply_competitive_pressure,
    awareness_lead_weight,
    check_competitor_deal_win,
    compute_awareness_score,
    compute_inbound_leads,
    decay_awareness,
    discover_customers,
    effective_awareness_targets,
    fire_competitive_events,
    mature_pending_awareness,
    next_pipeline_stage_capped,
    roll_pipeline_progression,
    scan_competitor_radar,
    schedule_awareness,
)
from alignsim.src.models.entities import CustomerStage, Engagement, FeatureStatus, PendingAwareness, Segment
from alignsim.src.models.scenario import CalibrationParams, ChannelProfile

from .factories import (
    make_competitor,
    make_competitor_event,
    make_customer,
    make_feature,
)


# --- compute_inbound_leads ---

def test_inbound_leads_no_history():
    """With short marketing history, lagged_investment is 0."""
    cal = CalibrationParams()  # base 0.5, lag 10
    leads = compute_inbound_leads([], cal)
    # 0.5 + 0 = 0.5 → int(0.5) = 0
    assert leads == 0


def test_inbound_leads_lagged_10_turns():
    """Investment from 10 turns ago drives current leads."""
    cal = CalibrationParams()  # default lag 10
    history = [10, 0, 0, 0, 0, 0, 0, 0, 0, 5]  # invest of 10 was 10 turns ago (index -10)
    leads = compute_inbound_leads(history, cal)
    # base 0.5 + 10 * 0.3 = 3.5 → int(3.5) = 3
    assert leads == 3


def test_inbound_leads_formula():
    """Verifies base + lagged * effectiveness * (1 + bonus)."""
    cal = CalibrationParams(marketing_lag_turns=2, base_inbound_rate=1.0,
                            marketing_effectiveness=0.5)
    history = [4, 0]  # lagged_investment = history[-2] = 4
    leads = compute_inbound_leads(history, cal, process_bonus=0.5)
    # 1.0 + 4 * 0.5 * 1.5 = 1.0 + 3.0 = 4.0
    assert leads == 4


# --- discover_customers ---

def test_discover_customers_probability_high():
    """High capacity and low difficulty → almost always discover."""
    hidden = [make_customer(id=f"H{i}", discovery_difficulty=1.0) for i in range(3)]
    discovered = discover_customers(20, hidden, segment_filter=None,
                                    rng=random.Random(0))
    # Probability capped at 0.95; with 3 candidates expect typically all
    assert len(discovered) == 3


def test_discover_customers_cost_per_attempt():
    """Each attempt consumes 1 capacity unit."""
    hidden = [make_customer(id=f"H{i}", discovery_difficulty=1.0) for i in range(5)]
    # Capacity 2 → at most 2 attempts → at most 2 discoveries
    discovered = discover_customers(2, hidden, segment_filter=None,
                                    rng=random.Random(0))
    assert len(discovered) <= 2


def test_discover_customers_segment_filter():
    """Segment filter restricts the candidate pool."""
    hidden = [
        make_customer(id="S1", segment=Segment.startup),
        make_customer(id="G1", segment=Segment.growth),
        make_customer(id="E1", segment=Segment.enterprise),
    ]
    discovered = discover_customers(20, hidden, segment_filter="enterprise",
                                    rng=random.Random(0))
    assert all(d == "E1" for d in discovered)


def test_discover_customers_invalid_segment_falls_back():
    """Invalid segment string ignores filter (uses all candidates)."""
    hidden = [
        make_customer(id="S1", segment=Segment.startup, discovery_difficulty=0.1),
    ]
    discovered = discover_customers(5, hidden, segment_filter="not_a_segment",
                                    rng=random.Random(0))
    assert "S1" in discovered


def test_discover_customers_empty_pool():
    """No hidden customers → empty result."""
    assert discover_customers(10, [], segment_filter=None,
                              rng=random.Random(0)) == []


def test_discover_customers_segment_filter_no_match():
    """Filter matches no candidates → empty."""
    hidden = [make_customer(id="S1", segment=Segment.startup)]
    assert discover_customers(10, hidden, segment_filter="enterprise",
                              rng=random.Random(0)) == []


# --- fire_competitive_events ---

def test_fire_competitive_events_matches_turn():
    """Returns events whose turn matches the current turn."""
    e1 = make_competitor_event(turn=3, description="e1")
    e2 = make_competitor_event(turn=5, description="e2")
    e3 = make_competitor_event(turn=3, description="e3")
    competitors = {
        "C1": make_competitor(id="C1", events=[e1, e2]),
        "C2": make_competitor(id="C2", events=[e3]),
    }
    fired_turn3 = fire_competitive_events(competitors, 3)
    fired_turn5 = fire_competitive_events(competitors, 5)
    fired_turn4 = fire_competitive_events(competitors, 4)

    assert {e.description for e in fired_turn3} == {"e1", "e3"}
    assert [e.description for e in fired_turn5] == ["e2"]
    assert fired_turn4 == []


# --- apply_competitive_pressure ---

def test_apply_competitive_pressure_increases_then_decays():
    """Pressure rises by impact*0.3 - 0.05 decay; capped at 1.0."""
    customer = make_customer(id="C01", competitive_pressure=0.0)
    event = make_competitor_event(
        affected_customers=["C01"],
        rubric_impact={"feature_coverage": 0.5, "price": 0.5},  # avg 0.5
    )
    new_pressure = apply_competitive_pressure(customer, [event])
    # 0 + 0.5*0.3 - 0.05 = 0.10
    assert new_pressure == pytest.approx(0.10)


def test_apply_competitive_pressure_decay_only():
    """No matching events → pressure decays by 0.05 (floored at 0)."""
    customer_high = make_customer(id="C01", competitive_pressure=0.5)
    customer_zero = make_customer(id="C02", competitive_pressure=0.0)
    irrelevant = make_competitor_event(affected_customers=["other"])

    assert apply_competitive_pressure(customer_high, [irrelevant]) == pytest.approx(0.45)
    assert apply_competitive_pressure(customer_zero, []) == 0.0


def test_apply_competitive_pressure_capped_at_one():
    """Pressure cannot exceed 1.0."""
    customer = make_customer(id="C01", competitive_pressure=0.95)
    event = make_competitor_event(
        affected_customers=["C01"],
        rubric_impact={"x": 1.0, "y": 1.0},  # avg 1.0 → +0.30 - 0.05 = +0.25
    )
    pressure = apply_competitive_pressure(customer, [event])
    assert pressure == 1.0


# --- check_competitor_deal_win ---

def test_competitor_deal_win():
    customer = make_customer()
    assert check_competitor_deal_win(customer, 0.8, 0.6) is True
    assert check_competitor_deal_win(customer, 0.5, 0.7) is False
    # Tie goes to player (strict greater)
    assert check_competitor_deal_win(customer, 0.5, 0.5) is False


# --- effective_awareness_targets ---

def test_effective_targets_explicit_filters_to_existing():
    features = {"F01": make_feature(id="F01"), "F02": make_feature(id="F02")}
    assert effective_awareness_targets(["F01", "F99"], features) == ["F01"]


def test_effective_targets_empty_includes_shipped_and_in_progress():
    """Empty targets = broad across shipped + in-progress (not not_started)."""
    features = {
        "F01": make_feature(id="F01", status=FeatureStatus.shipped_mvp),
        "F02": make_feature(id="F02", status=FeatureStatus.in_progress),
        "F03": make_feature(id="F03", status=FeatureStatus.not_started),
    }
    targets = set(effective_awareness_targets([], features))
    assert targets == {"F01", "F02"}


# --- schedule_awareness (per-channel lag/spread/efficiency) ---

def test_schedule_awareness_events_burst():
    """events: lag 2, spread 1 (single burst), efficiency 0.8."""
    profile = ChannelProfile(lag=2, spread=1, efficiency=0.8, budget_cost_per_capacity=8000)
    pending = schedule_awareness(profile, ["F14"], capacity=5, current_turn=10)
    assert len(pending) == 1
    assert pending[0].land_turn == 12  # 10 + lag 2
    assert pending[0].feature_id == "F14"
    assert pending[0].amount == pytest.approx(5 * 0.8)  # all in one burst


def test_schedule_awareness_content_spread():
    """content: lag 8, spread 6 — total split evenly across 6 consecutive turns."""
    profile = ChannelProfile(lag=8, spread=6, efficiency=0.5, budget_cost_per_capacity=3000)
    pending = schedule_awareness(profile, ["F02"], capacity=6, current_turn=1)
    assert len(pending) == 6
    land_turns = sorted(p.land_turn for p in pending)
    assert land_turns == [9, 10, 11, 12, 13, 14]  # turn 1 + lag 8, spread over 6
    total = sum(p.amount for p in pending)
    assert total == pytest.approx(6 * 0.5)
    # Each increment equal
    assert all(p.amount == pytest.approx((6 * 0.5) / 6) for p in pending)


def test_schedule_awareness_splits_across_features():
    profile = ChannelProfile(lag=0, spread=1, efficiency=1.0, budget_cost_per_capacity=0)
    pending = schedule_awareness(profile, ["F01", "F02"], capacity=4, current_turn=1)
    assert len(pending) == 2
    # 4 * 1.0 = 4 total, split across 2 features = 2 each
    assert all(p.amount == pytest.approx(2.0) for p in pending)


def test_schedule_awareness_no_targets_empty():
    profile = ChannelProfile(lag=2, spread=1, efficiency=0.8, budget_cost_per_capacity=0)
    assert schedule_awareness(profile, [], capacity=5, current_turn=1) == []


# --- mature_pending_awareness ---

def test_mature_pending_awareness_partitions_by_turn():
    pending = [
        PendingAwareness(land_turn=5, feature_id="F01", amount=1.0),
        PendingAwareness(land_turn=6, feature_id="F01", amount=1.0),
        PendingAwareness(land_turn=4, feature_id="F02", amount=2.0),  # straggler <= turn
    ]
    matured, remaining = mature_pending_awareness(pending, current_turn=5)
    assert {p.feature_id for p in matured} == {"F01", "F02"}
    assert len(matured) == 2  # land_turn 5 and 4
    assert len(remaining) == 1 and remaining[0].land_turn == 6


# --- decay_awareness ---

def test_decay_awareness_applies_and_drops_epsilon():
    awareness = {"F01": 2.0, "F02": 0.005}
    decayed = decay_awareness(awareness, decay=0.10, epsilon=0.01)
    assert decayed["F01"] == pytest.approx(1.8)
    # F02 decays to 0.0045 < epsilon → dropped
    assert "F02" not in decayed


# --- compute_awareness_score ---

def test_compute_awareness_score_max_of_needed_features():
    customer = make_customer(feature_needs={
        "F01": {"mvp": 0.5}, "F02": {"mvp": 0.5},
    })
    awareness = {"F01": 1.0, "F02": 3.5, "F03": 9.0}
    # max over needed features (F03 not needed, ignored)
    assert compute_awareness_score(customer, awareness) == 3.5


def test_compute_awareness_score_no_needs_zero():
    customer = make_customer(feature_needs={})
    assert compute_awareness_score(customer, {"F01": 5.0}) == 0.0


# --- compute_awareness_reveal (engagement + timeline) ---

def test_compute_awareness_reveal_cold_below_warm():
    cal = CalibrationParams(awareness_warm_threshold=1.5, awareness_hot_threshold=4.0)
    engagement, timeline_bonus = compute_awareness_reveal(
        awareness_score=0.0, calibration=cal, rng=random.Random(0))
    assert engagement == Engagement.cold
    assert timeline_bonus == 0  # no extension at 0 awareness


def test_compute_awareness_reveal_warm():
    cal = CalibrationParams(awareness_warm_threshold=1.5, awareness_hot_threshold=4.0,
                            awareness_hot_prob=0.0, awareness_timeline_bonus_max=6)
    engagement, timeline_bonus = compute_awareness_reveal(
        awareness_score=2.0, calibration=cal, rng=random.Random(0))
    assert engagement == Engagement.warm  # hot_prob 0 → stays warm
    # round(6 * min(1, 2.0/4.0)) = round(3.0) = 3
    assert timeline_bonus == 3


def test_compute_awareness_reveal_hot_requires_threshold_and_roll():
    cal = CalibrationParams(awareness_warm_threshold=1.5, awareness_hot_threshold=4.0,
                            awareness_hot_prob=1.0, awareness_timeline_bonus_max=6)
    # Above hot threshold + prob 1.0 → hot
    engagement, timeline_bonus = compute_awareness_reveal(
        awareness_score=5.0, calibration=cal, rng=random.Random(0))
    assert engagement == Engagement.hot
    # round(6 * min(1, 5.0/4.0)) = round(6 * 1.0) = 6
    assert timeline_bonus == 6


def test_compute_awareness_reveal_hot_roll_deterministic():
    """Hot roll uses rng deterministically; prob 0.2 with a controlled rng."""
    cal = CalibrationParams(awareness_warm_threshold=1.5, awareness_hot_threshold=4.0,
                            awareness_hot_prob=0.2)
    # rng.random() first draw for Random(0) ~ 0.844 > 0.2 → NOT hot
    eng0, _ = compute_awareness_reveal(awareness_score=5.0, calibration=cal, rng=random.Random(0))
    assert eng0 == Engagement.warm
    # Random(1) first draw ~ 0.134 < 0.2 → hot
    eng1, _ = compute_awareness_reveal(awareness_score=5.0, calibration=cal, rng=random.Random(1))
    assert eng1 == Engagement.hot


def test_compute_awareness_reveal_never_shortens_timeline():
    cal = CalibrationParams(awareness_timeline_bonus_max=6, awareness_hot_threshold=4.0)
    _, timeline_bonus = compute_awareness_reveal(
        awareness_score=4.0, calibration=cal, rng=random.Random(0))
    assert timeline_bonus >= 0  # only ever a non-negative extension


# --- awareness_lead_weight (inbound feature-bias) ---

def test_awareness_lead_weight_scales_with_score():
    awareness = {"F01": 3.0}
    needy = make_customer(id="N", feature_needs={"F01": {"mvp": 0.5}})
    indifferent = make_customer(id="I", feature_needs={"F09": {"mvp": 0.5}})
    w_needy = awareness_lead_weight(needy, awareness, bias=2.0)
    w_indiff = awareness_lead_weight(indifferent, awareness, bias=2.0)
    assert w_needy == pytest.approx(1.0 + 2.0 * 3.0)  # 7.0
    assert w_indiff == pytest.approx(1.0)  # no awareness on its needs
    assert w_needy > w_indiff


# --- scan_competitor_radar ---

def _radar_cal(**kw):
    base = dict(radar_lookahead_turns=5, radar_base_prob=1.0,
                radar_uncertainty_jitter=0.0, awareness_hot_threshold=4.0)
    base.update(kw)
    return CalibrationParams(**base)


def test_radar_only_within_lookahead():
    """Events outside the lookahead window are never sensed."""
    cal = _radar_cal()
    cust = make_customer(id="C01", feature_needs={"F01": {"mvp": 0.5}})
    competitors = {
        "X": make_competitor(events=[
            make_competitor_event(turn=4, affected_customers=["C01"]),   # delta 3 (in window)
            make_competitor_event(turn=20, affected_customers=["C01"]),  # delta 19 (out)
        ]),
    }
    signals = scan_competitor_radar(
        competitors, {"C01": cust}, awareness={"F01": 4.0},
        current_turn=1, calibration=cal, rng=random.Random(0),
    )
    # Only the in-window event surfaces (base_prob 1.0, jitter 0 → always detected)
    assert len(signals) == 1
    assert signals[0].startswith("F01:")


def test_radar_requires_awareness_on_affected_features():
    """No awareness on the affected customers' features → no signal, no rng draw."""
    cal = _radar_cal()
    cust = make_customer(id="C01", feature_needs={"F01": {"mvp": 0.5}})
    competitors = {"X": make_competitor(events=[
        make_competitor_event(turn=3, affected_customers=["C01"]),
    ])}
    signals = scan_competitor_radar(
        competitors, {"C01": cust}, awareness={"F09": 5.0},  # awareness on unrelated feature
        current_turn=1, calibration=cal, rng=random.Random(0),
    )
    assert signals == []


def test_radar_fuzzy_timing_soon_vs_upcoming():
    cal = _radar_cal(radar_lookahead_turns=5)  # imminent_cutoff = 2
    cust = make_customer(id="C01", feature_needs={"F01": {"mvp": 0.5}})
    competitors = {"X": make_competitor(events=[
        make_competitor_event(turn=2, affected_customers=["C01"]),  # delta 1 <= 2 → soon
        make_competitor_event(turn=5, affected_customers=["C01"]),  # delta 4 > 2 → upcoming
    ])}
    signals = scan_competitor_radar(
        competitors, {"C01": cust}, awareness={"F01": 4.0},
        current_turn=1, calibration=cal, rng=random.Random(0),
    )
    assert "F01:soon" in signals
    assert "F01:upcoming" in signals


def test_radar_deterministic_per_seed():
    cal = _radar_cal(radar_base_prob=0.5, radar_uncertainty_jitter=0.15)
    cust = make_customer(id="C01", feature_needs={"F01": {"mvp": 0.5}})
    competitors = {"X": make_competitor(events=[
        make_competitor_event(turn=3, affected_customers=["C01"]),
    ])}
    args = (competitors, {"C01": cust}, {"F01": 2.0}, 1, cal)
    s1 = scan_competitor_radar(*args, rng=random.Random(7))
    s2 = scan_competitor_radar(*args, rng=random.Random(7))
    assert s1 == s2


# --- next_pipeline_stage_capped ---

def test_next_pipeline_stage_ladder():
    assert next_pipeline_stage_capped(CustomerStage.lead) == CustomerStage.prospect
    assert next_pipeline_stage_capped(CustomerStage.prospect) == CustomerStage.qualified
    assert next_pipeline_stage_capped(CustomerStage.qualified) == CustomerStage.in_deal


def test_next_pipeline_stage_capped_at_in_deal():
    """Closing (in_deal -> customer) is NOT reachable via progression."""
    assert next_pipeline_stage_capped(CustomerStage.in_deal) is None
    assert next_pipeline_stage_capped(CustomerStage.customer) is None
    assert next_pipeline_stage_capped(CustomerStage.churned) is None
    assert next_pipeline_stage_capped(CustomerStage.lost) is None


# --- roll_pipeline_progression ---

class _FixedRng:
    """rng stub returning a constant from random() to probe the computed probability."""
    def __init__(self, value: float):
        self.value = value
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.value


def _progression_cal(**kw):
    base = dict(
        progression_base_prob={"content": 0.20, "events": 0.40},
        progression_collab_scale=0.35,
        progression_budget_scale=0.05,
        progression_max_prob=0.75,
    )
    base.update(kw)
    return CalibrationParams(**base)


def test_roll_progression_events_probability_table():
    """events m=s=1 → p≈0.541 (channel_profiles events budget 8000/cap)."""
    cal = _progression_cal()
    # Just below p → success; just above → failure (locks the worked table value).
    assert roll_pipeline_progression("events", 1, 1, cal, _FixedRng(0.53)) is True
    assert roll_pipeline_progression("events", 1, 1, cal, _FixedRng(0.55)) is False


def test_roll_progression_content_probability_table():
    """content m=s=1 → p≈0.262 (content budget 3000/cap)."""
    cal = _progression_cal()
    assert roll_pipeline_progression("content", 1, 1, cal, _FixedRng(0.25)) is True
    assert roll_pipeline_progression("content", 1, 1, cal, _FixedRng(0.27)) is False


def test_roll_progression_clamped_at_max():
    cal = _progression_cal(progression_base_prob={"content": 0.2, "events": 2.0}, progression_max_prob=0.75)
    # Huge base would exceed 1.0 but is clamped to 0.75.
    assert roll_pipeline_progression("events", 5, 5, cal, _FixedRng(0.74)) is True
    assert roll_pipeline_progression("events", 5, 5, cal, _FixedRng(0.76)) is False


def test_roll_progression_joint_commitment_uses_min():
    """min(m, s) gates the collab term — a huge marketing cap with s=1 still uses joint=1."""
    cal = _progression_cal()
    big_m = _FixedRng(0.53)
    # events m=100, s=1 → joint=min=1; budget term uses m=100 though. p > events m=s=1 (more budget),
    # but verify it still draws and is a valid bool.
    assert isinstance(roll_pipeline_progression("events", 100, 1, cal, big_m), bool)


def test_roll_progression_no_draw_without_commitment():
    """Unfunded channel or zero collab → False with no RNG draw."""
    cal = _progression_cal()
    rng = _FixedRng(0.0)  # would always succeed if drawn
    assert roll_pipeline_progression("events", 0, 5, cal, rng) is False
    assert roll_pipeline_progression("events", 5, 0, cal, rng) is False
    assert roll_pipeline_progression("nonexistent", 5, 5, cal, rng) is False
    assert rng.calls == 0
