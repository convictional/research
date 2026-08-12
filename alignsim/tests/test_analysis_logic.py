"""Tests for alignsim.src.engine.analysis_logic — pure cross-functional analyses.

These guard the two hard invariants: the analyses are deterministic (no RNG) and depend ONLY on
observable history (never on hidden ground-truth fields), plus per-analysis correctness.
"""

import pytest

from alignsim.src.engine.analysis_logic import compute_analysis
from alignsim.src.models.entities import CustomerStage
from alignsim.src.models.scenario import CalibrationParams

from .factories import (
    make_customer,
    make_emergent_need,
    make_game_state,
    make_resource_pool,
    make_turn_record,
)

ANALYSES = ["conversion_funnel", "retention_efficiency", "awareness_attribution", "capacity_bottleneck"]


def _state_with_history():
    """A state with several turns of observable history across all streams."""
    history = [
        make_turn_record(
            turn=1,
            events=["inbound_lead:C1", "stage_advanced:C1:lead->prospect", "discovered:C3"],
            eng_capacity_used=18, sales_capacity_used=4, support_capacity_used=2,
            marketing_capacity_used=3, ops_capacity_used=0,
        ),
        make_turn_record(
            turn=2,
            events=["stage_advanced:C1:prospect->qualified", "inbound_lead:C4",
                    "churn_intervention:C2:success", "expansion:C2:+500"],
            eng_capacity_used=20, sales_capacity_used=9, support_capacity_used=5,
            marketing_capacity_used=5, ops_capacity_used=0,
        ),
        make_turn_record(
            turn=3,
            events=["stage_advanced:C1:qualified->in_deal", "deal_won:C5",
                    "churn_intervention:C6:failed", "timeline_expired_reset:C7:resets=1"],
            eng_capacity_used=20, sales_capacity_used=10, support_capacity_used=4,
            marketing_capacity_used=5, ops_capacity_used=0,
        ),
        make_turn_record(
            turn=4,
            events=["deal_won:C1", "inbound_lead:C8", "deal_lost:C9:competitor_won:Nova"],
            eng_capacity_used=19, sales_capacity_used=8, support_capacity_used=3,
            marketing_capacity_used=4, ops_capacity_used=2,
        ),
    ]
    customers = [
        make_customer(id="C1", is_visible=True, stage=CustomerStage.customer, deal_value=5000),
        make_customer(id="C10", is_visible=True, stage=CustomerStage.prospect, deal_value=3000),
        make_customer(id="C11", is_visible=True, stage=CustomerStage.in_deal, deal_value=8000),
        make_customer(id="C12", is_visible=False, stage=CustomerStage.lead, deal_value=1000),
    ]
    state = make_game_state(
        customers=customers,
        resources=make_resource_pool(eng_capacity=20, sales_capacity=10, support_capacity=5,
                                     marketing_capacity=5, ops_capacity=4),
        turn=5,
    )
    state.turn_history = history
    state.churn_history = [0, 1, 0, 2]
    state.marketing_history = [3, 5, 5, 4]
    return state


@pytest.mark.parametrize("analysis_type", ANALYSES)
def test_each_analysis_returns_self_describing_dict(analysis_type):
    state = _state_with_history()
    out = compute_analysis(analysis_type, state, CalibrationParams())
    assert out["analysis_type"] == analysis_type
    assert isinstance(out, dict)


@pytest.mark.parametrize("analysis_type", ANALYSES)
def test_analysis_is_deterministic(analysis_type):
    """Same observable state → byte-identical output (no RNG)."""
    state = _state_with_history()
    a = compute_analysis(analysis_type, state, CalibrationParams())
    b = compute_analysis(analysis_type, state, CalibrationParams())
    assert a == b


@pytest.mark.parametrize("analysis_type", ANALYSES)
def test_analysis_ignores_hidden_fields(analysis_type):
    """Mutating hidden ground-truth must not change any analysis output."""
    state = _state_with_history()
    before = compute_analysis(analysis_type, state, CalibrationParams())

    # Poison every hidden field the no-leak contract forbids reading — including the most
    # "tempting" latent signals (health, timeline, dealbreakers) an analysis might be tempted to
    # peek at. None may influence any analysis output.
    for c in state.customers.values():
        c.desired_price_point = 999_999
        c.close_threshold = 0.99
        c.churn_drivers = {"SENTINEL": 1.0}
        c.feature_needs = {"FX": {"mvp": 1.0}}
        c.competitive_pressure = 5.0
        c.health = 0.0
        c.health_history = [0.0, 0.0, 0.0]
        c.timeline = 1
        c.dealbreakers = ["SENTINEL"]
        c.discovery_difficulty = 9.9
    state.emergent_needs.append(
        make_emergent_need(id="EN_SENT", customer_id="C1", feature_id="FZ", is_revealed=False)
    )

    after = compute_analysis(analysis_type, state, CalibrationParams())
    assert after == before


def test_conversion_funnel_counts_transitions_and_trend():
    state = _state_with_history()
    out = compute_analysis("conversion_funnel", state, CalibrationParams())
    # Lifetime transitions reconstructed from stage_advanced events.
    tr = out["stage_transitions_lifetime"]
    assert tr["lead->prospect"] == 1
    assert tr["prospect->qualified"] == 1
    assert tr["qualified->in_deal"] == 1
    assert out["deals_won"]["lifetime"] == 2  # C5 + C1
    assert out["deals_lost"]["lifetime"] == 1
    assert out["timeline_resets"]["lifetime"] == 1
    # Median turns-in-stage for C1: prospect entered t1, left t2 → 1 turn in prospect.
    assert out["median_turns_in_stage"]["prospect"] == 1.0
    # Visible pipeline snapshot counts only visible non-customer stages.
    assert out["pipeline_snapshot"]["prospect"] == 1
    assert out["pipeline_snapshot"]["in_deal"] == 1


def test_capacity_bottleneck_utilization_and_rejections():
    state = _state_with_history()
    # Inject capacity-bound rejections on the most recent turn.
    last = state.turn_history[-1]
    from alignsim.src.models.actions import BuildAction
    from alignsim.src.models.game_state import ActionRejection
    last.actions_rejected = [
        ActionRejection(action=BuildAction(feature_id="F1", quality="mvp", capacity=5),
                        reason="Insufficient engineering capacity: needs 5, only 1 remaining"),
        ActionRejection(action=BuildAction(feature_id="F2", quality="mvp", capacity=5),
                        reason="Insufficient budget for events marketing: needs 1, only 0 remaining"),
    ]
    out = compute_analysis("capacity_bottleneck", state, CalibrationParams())
    assert set(out["pools"]) == {"engineering", "sales", "support", "marketing", "ops"}
    # Engineering ran near its 20 cap most turns → high utilization.
    assert out["pools"]["engineering"]["utilization"] is not None
    assert "engineering" in out["saturated_pools"]
    # Only the capacity rejection counts (budget rejection ignored).
    assert out["capacity_rejections_window"]["engineering"] == 1
    assert out["total_capacity_rejections_window"] == 1


def test_awareness_attribution_estimates_lag_from_data():
    state = _state_with_history()
    out = compute_analysis("awareness_attribution", state, CalibrationParams())
    # Totals are simple observable aggregates.
    assert out["totals"]["inbound_leads"] == 3  # C1, C4, C8
    assert out["totals"]["marketing_capacity"] == sum([3, 5, 5, 4])
    # Lag is ESTIMATED from observed pairs, never reported from the (withheld) engine constant.
    assert "lag_turns" not in out
    assert isinstance(out["estimated_lag_turns"], int)
    assert 0 <= out["estimated_lag_turns"] <= 12
    # Only 4 turns of history → short of a full sweep → honest, actionable re-run hint.
    assert out["lags_evaluated"] >= 1
    assert out["confidence"] == "low"
    assert "note" in out and "re-run" in out["note"]


def test_awareness_attribution_invariant_to_true_lag():
    """Regression guard: the estimate must NOT depend on the engine's withheld marketing_lag_turns.

    Changing the true lag must leave the analysis output byte-identical — proving it estimates
    from observable data and never reads the ground-truth constant (no estimation back-door).
    """
    state = _state_with_history()
    out_short = compute_analysis("awareness_attribution", state, CalibrationParams(marketing_lag_turns=2))
    out_long = compute_analysis("awareness_attribution", state, CalibrationParams(marketing_lag_turns=10))
    assert out_short == out_long


def test_awareness_attribution_excludes_current_turn_marketing():
    """marketing_history may carry the current, not-yet-recorded turn; totals must exclude it."""
    state = _state_with_history()  # 4 recorded turns; marketing_history == [3, 5, 5, 4]
    state.marketing_history = [3, 5, 5, 4, 99]  # 99 = current (turn 5) spend, no record yet
    out = compute_analysis("awareness_attribution", state, CalibrationParams())
    assert out["totals"]["marketing_capacity"] == 17  # 3+5+5+4, NOT 116


def test_retention_efficiency_churn_and_interventions():
    state = _state_with_history()
    out = compute_analysis("retention_efficiency", state, CalibrationParams())
    assert out["churn"]["lifetime_total"] == 3  # sum([0,1,0,2])
    assert out["intervention"]["successes"] == 1
    assert out["intervention"]["failures"] == 1
    assert out["intervention"]["success_ratio"] == pytest.approx(0.5)
    assert out["expansions"]["lifetime"] == 1
    assert out["active_customers_now"] == 1  # only C1 is stage=customer


def test_unknown_analysis_type_returns_error():
    state = _state_with_history()
    out = compute_analysis("not_a_real_analysis", state, CalibrationParams())
    assert out["error"] == "unknown_analysis_type"
