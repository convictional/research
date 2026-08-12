"""Tests for alignsim.src.engine.alignment_scoring — Layer 2 alignment metrics."""

import pytest

from alignsim.src.engine.alignment_scoring import (
    BUG_SEVERITY_PAR,
    DEBT_GRADIENT_PAR,
    SUPPORT_TIMING_DECAY_PER_TURN,
    compute_alignment_scores,
)
from alignsim.src.models.actions import SellAction
from alignsim.src.models.entities import (
    BugSeverity,
    CustomerStage,
    ProcessProjectStatus,
)
from alignsim.src.models.game_state import TechDebt
from alignsim.src.models.scenario import CalibrationParams

from .factories import (
    make_bug,
    make_customer,
    make_game_state,
    make_process_project,
    make_turn_record,
)


CALIB = CalibrationParams()


# ============================================================
# support_timing
# ============================================================

def test_support_timing_on_time():
    """CS hired before first-deal + onboarding window → score 1.0."""
    state = make_game_state(turn=10)
    state.turn_history = [
        make_turn_record(turn=3, events=["deal_won:C01"]),
        # CS arrived turn 5; deal turn 3 + onboarding (4) = 7 → not late
        make_turn_record(turn=5, events=["hire_arrived:H1:cs:+4_capacity"]),
    ]
    result = compute_alignment_scores(state, CALIB)
    assert result["support_timing"]["score"] == pytest.approx(1.0)
    assert result["support_timing"]["turns_late"] == 0


def test_support_timing_late_decays():
    """CS hired 5 turns past window → score 0.5."""
    state = make_game_state(turn=20)
    state.turn_history = [
        make_turn_record(turn=3, events=["deal_won:C01"]),
        # window ends turn 7; CS arrives turn 12 → 5 turns late
        make_turn_record(turn=12, events=["hire_arrived:H1:cs:+4_capacity"]),
    ]
    result = compute_alignment_scores(state, CALIB)
    assert result["support_timing"]["score"] == pytest.approx(0.5)
    assert result["support_timing"]["turns_late"] == 5


def test_support_timing_cs_event_parsing():
    """Hire events with :cs: are matched; other roles (:engineering:, :sales:) are not."""
    state = make_game_state(turn=15)
    state.turn_history = [
        make_turn_record(turn=3, events=["deal_won:C01"]),
        # Not CS — should not count as CS arrival
        make_turn_record(turn=4, events=["hire_arrived:H1:engineering:+4_capacity"]),
        make_turn_record(turn=5, events=["hire_arrived:H2:sales:+4_capacity"]),
        # CS arrives turn 14; window ends turn 7 → 7 late → score 0.3
        make_turn_record(turn=14, events=["hire_arrived:H3:cs:+4_capacity"]),
    ]
    result = compute_alignment_scores(state, CALIB)
    assert result["support_timing"]["cs_arrived_turn"] == 14
    assert result["support_timing"]["turns_late"] == 7
    assert result["support_timing"]["score"] == pytest.approx(0.3)


def test_support_timing_no_deal_returns_none():
    """No deal_won event → None (metric omitted)."""
    state = make_game_state(turn=5)
    state.turn_history = [make_turn_record(turn=1, events=["inbound_lead:C01"])]
    result = compute_alignment_scores(state, CALIB)
    assert "support_timing" not in result


def test_support_timing_cs_never_hired():
    """CS never hired → turns_late counted from game end."""
    state = make_game_state(turn=20)
    state.turn_history = [make_turn_record(turn=3, events=["deal_won:C01"])]
    result = compute_alignment_scores(state, CALIB)
    # window ends turn 7; game end turn 20 → 13 turns late → score 0.0
    assert result["support_timing"]["cs_arrived_turn"] is None
    assert result["support_timing"]["score"] == pytest.approx(0.0)


# ============================================================
# bug_responsiveness
# ============================================================

def test_bug_responsiveness_severity_weighted():
    """Critical resolved in 1 turn → 0.0; minor resolved in 1 turn → 0.8."""
    customer = make_customer(id="C01", stage=CustomerStage.customer)
    crit_bug = make_bug(
        id="B1", severity=BugSeverity.critical,
        affected_customers=["C01"], is_resolved=True, turns_unresolved=1,
    )
    minor_bug = make_bug(
        id="B2", severity=BugSeverity.minor,
        affected_customers=["C01"], is_resolved=True, turns_unresolved=1,
    )

    # Critical: 1 - 1/1 = 0
    state_crit = make_game_state(customers=[customer], bugs=[crit_bug])
    result = compute_alignment_scores(state_crit, CALIB)
    assert result["bug_responsiveness"]["score"] == pytest.approx(0.0)

    # Minor: 1 - 1/5 = 0.8
    state_minor = make_game_state(customers=[customer], bugs=[minor_bug])
    result = compute_alignment_scores(state_minor, CALIB)
    assert result["bug_responsiveness"]["score"] == pytest.approx(0.8)


def test_bug_responsiveness_unresolved_scores_zero():
    """Unresolved customer-impacting bug at game end → 0."""
    customer = make_customer(id="C01", stage=CustomerStage.customer)
    bug = make_bug(
        id="B1", severity=BugSeverity.major,
        affected_customers=["C01"], is_resolved=False, turns_unresolved=10,
    )
    state = make_game_state(customers=[customer], bugs=[bug])
    result = compute_alignment_scores(state, CALIB)
    assert result["bug_responsiveness"]["score"] == pytest.approx(0.0)
    assert result["bug_responsiveness"]["resolved_count"] == 0


def test_bug_responsiveness_ignores_non_customer_bugs():
    """Bugs affecting only leads or churned customers don't count."""
    lead = make_customer(id="C01", stage=CustomerStage.lead)
    churned = make_customer(id="C02", stage=CustomerStage.churned)
    bug = make_bug(
        id="B1", severity=BugSeverity.critical,
        affected_customers=["C01", "C02"], is_resolved=True, turns_unresolved=1,
    )
    state = make_game_state(customers=[lead, churned], bugs=[bug])
    result = compute_alignment_scores(state, CALIB)
    assert "bug_responsiveness" not in result


def test_bug_responsiveness_no_bugs_returns_none():
    """No customer-impacting bugs → None."""
    state = make_game_state()
    result = compute_alignment_scores(state, CALIB)
    assert "bug_responsiveness" not in result


# ============================================================
# debt_management
# ============================================================

def test_debt_management_zero_gradient():
    """Stable debt (no change from 0) → score 1.0."""
    state = make_game_state(tech_debt=TechDebt(level=0.0))
    state.turn_history = [make_turn_record(turn=i) for i in range(1, 11)]
    result = compute_alignment_scores(state, CALIB)
    assert result["debt_management"]["score"] == pytest.approx(1.0)
    assert result["debt_management"]["avg_gradient"] == pytest.approx(0.0)


def test_debt_management_high_growth():
    """High debt growth → score 0.0."""
    # Final debt 10 over 10 turns = gradient 1.0 > par 0.5 → score 0
    state = make_game_state(tech_debt=TechDebt(level=10.0))
    state.turn_history = [make_turn_record(turn=i) for i in range(1, 11)]
    result = compute_alignment_scores(state, CALIB)
    assert result["debt_management"]["score"] == pytest.approx(0.0)


def test_debt_management_moderate_growth():
    """Debt growth at half of par → score 0.5."""
    # Gradient 0.25 = 0.5 * 0.5 → score 1 - 0.5 = 0.5
    state = make_game_state(tech_debt=TechDebt(level=2.5))
    state.turn_history = [make_turn_record(turn=i) for i in range(1, 11)]
    result = compute_alignment_scores(state, CALIB)
    assert result["debt_management"]["score"] == pytest.approx(0.5)


def test_debt_management_empty_history_returns_none():
    """No turns played → None."""
    state = make_game_state(tech_debt=TechDebt(level=3.0))
    state.turn_history = []
    result = compute_alignment_scores(state, CALIB)
    assert "debt_management" not in result


def test_debt_management_nonzero_initial_debt():
    """Scenario starts at debt=5, ends at debt=5 over 10 turns → gradient=0, score=1.0."""
    state = make_game_state(tech_debt=TechDebt(level=5.0), initial_tech_debt=5.0)
    state.turn_history = [make_turn_record(turn=i) for i in range(1, 11)]
    result = compute_alignment_scores(state, CALIB)
    assert result["debt_management"]["score"] == pytest.approx(1.0)
    assert result["debt_management"]["initial_debt"] == pytest.approx(5.0)
    assert result["debt_management"]["final_debt"] == pytest.approx(5.0)
    assert result["debt_management"]["avg_gradient"] == pytest.approx(0.0)


# ============================================================
# sales_focus
# ============================================================

def test_sales_focus_all_to_winners():
    """All sell capacity went to customers that converted → 1.0."""
    state = make_game_state()
    state.turn_history = [
        make_turn_record(
            turn=1,
            actions_valid=[SellAction(customer_id="C01", sell_action="outbound", capacity=2)],
        ),
        make_turn_record(
            turn=3,
            actions_valid=[SellAction(customer_id="C01", sell_action="demo", capacity=3)],
            events=["deal_won:C01"],
        ),
    ]
    result = compute_alignment_scores(state, CALIB)
    assert result["sales_focus"]["score"] == pytest.approx(1.0)
    assert result["sales_focus"]["won_sell_capacity"] == 5
    assert result["sales_focus"]["total_sell_capacity"] == 5
    assert result["sales_focus"]["deals_won"] == 1


def test_sales_focus_partial_efficiency():
    """Half capacity went to losers → 0.5."""
    state = make_game_state()
    state.turn_history = [
        make_turn_record(
            turn=1,
            actions_valid=[
                SellAction(customer_id="C01", sell_action="outbound", capacity=3),
                SellAction(customer_id="C02", sell_action="outbound", capacity=3),
            ],
            events=["deal_won:C01"],
        ),
    ]
    result = compute_alignment_scores(state, CALIB)
    assert result["sales_focus"]["score"] == pytest.approx(0.5)


def test_sales_focus_no_sells_returns_none():
    """No sell actions → None."""
    state = make_game_state()
    state.turn_history = [make_turn_record(turn=1)]
    result = compute_alignment_scores(state, CALIB)
    assert "sales_focus" not in result


# ============================================================
# ops_engagement
# ============================================================

def test_ops_engagement_all_supported():
    """Every started project got target-team capacity → 1.0."""
    projects = [
        make_process_project(id="PP01", status=ProcessProjectStatus.completed, target_team_capacity_invested=2),
        make_process_project(id="PP02", status=ProcessProjectStatus.in_progress, target_team_capacity_invested=1),
    ]
    state = make_game_state(process_projects=projects)
    result = compute_alignment_scores(state, CALIB)
    assert result["ops_engagement"]["score"] == pytest.approx(1.0)
    assert result["ops_engagement"]["started_projects"] == 2
    assert result["ops_engagement"]["supported_projects"] == 2


def test_ops_engagement_partial():
    """1 of 2 projects supported → 0.5."""
    projects = [
        make_process_project(id="PP01", status=ProcessProjectStatus.completed, target_team_capacity_invested=2),
        make_process_project(id="PP02", status=ProcessProjectStatus.in_progress, target_team_capacity_invested=0),
    ]
    state = make_game_state(process_projects=projects)
    result = compute_alignment_scores(state, CALIB)
    assert result["ops_engagement"]["score"] == pytest.approx(0.5)


def test_ops_engagement_no_started_returns_none():
    """No started projects → None."""
    projects = [
        make_process_project(id="PP01", status=ProcessProjectStatus.available),
    ]
    state = make_game_state(process_projects=projects)
    result = compute_alignment_scores(state, CALIB)
    assert "ops_engagement" not in result


# ============================================================
# Aggregates
# ============================================================

def test_alignment_pareto_geometric_mean():
    """alignment_pareto = product^(1/n) of all non-None metric scores."""
    projects = [
        make_process_project(id="PP01", status=ProcessProjectStatus.completed, target_team_capacity_invested=2),
        make_process_project(id="PP02", status=ProcessProjectStatus.in_progress, target_team_capacity_invested=0),
    ]
    state = make_game_state(process_projects=projects, tech_debt=TechDebt(level=0.0))
    state.turn_history = [
        make_turn_record(
            turn=1,
            actions_valid=[SellAction(customer_id="C01", sell_action="outbound", capacity=2)],
            events=["deal_won:C01"],
        ),
    ]
    result = compute_alignment_scores(state, CALIB)
    # All metrics except bug_responsiveness are populated.
    metric_keys = [k for k in result.keys() if k not in ("alignment_composite", "alignment_pareto")]
    scores = [result[k]["score"] for k in metric_keys]
    product = 1.0
    for s in scores:
        product *= s
    expected_pareto = product ** (1.0 / len(scores))
    assert result["alignment_pareto"]["score"] == pytest.approx(expected_pareto, abs=1e-4)
    assert result["alignment_composite"]["score"] == pytest.approx(sum(scores), abs=1e-4)


def test_alignment_pareto_zero_kills_geometric_mean():
    """Any zero score → alignment_pareto = 0."""
    # ops_engagement = 0.0 (started but not supported)
    projects = [
        make_process_project(id="PP01", status=ProcessProjectStatus.in_progress, target_team_capacity_invested=0),
    ]
    state = make_game_state(process_projects=projects, tech_debt=TechDebt(level=0.0))
    state.turn_history = [
        make_turn_record(
            turn=1,
            actions_valid=[SellAction(customer_id="C01", sell_action="outbound", capacity=2)],
            events=["deal_won:C01"],
        ),
    ]
    result = compute_alignment_scores(state, CALIB)
    assert result["ops_engagement"]["score"] == pytest.approx(0.0)
    assert result["alignment_pareto"]["score"] == pytest.approx(0.0)


def test_empty_state_returns_empty_dict():
    """Bankrupt-at-turn-3 case: all metrics None → result is {}."""
    state = make_game_state(turn=3)
    state.turn_history = []
    result = compute_alignment_scores(state, CALIB)
    assert result == {}


# ============================================================
# Integration
# ============================================================

def test_full_state_all_five_metrics_present():
    """Rich state → all five metrics present, all in [0,1]."""
    customer = make_customer(id="C01", stage=CustomerStage.customer)
    bug = make_bug(
        id="B1", severity=BugSeverity.major,
        affected_customers=["C01"], is_resolved=True, turns_unresolved=1,
    )
    projects = [
        make_process_project(id="PP01", status=ProcessProjectStatus.completed, target_team_capacity_invested=2),
    ]
    state = make_game_state(
        turn=10,
        customers=[customer],
        bugs=[bug],
        process_projects=projects,
        tech_debt=TechDebt(level=1.0),
    )
    state.turn_history = [
        make_turn_record(
            turn=2,
            actions_valid=[SellAction(customer_id="C01", sell_action="outbound", capacity=3)],
            events=["deal_won:C01"],
        ),
        make_turn_record(turn=3, events=["hire_arrived:H1:cs:+4_capacity"]),
    ] + [make_turn_record(turn=i) for i in range(4, 11)]

    result = compute_alignment_scores(state, CALIB)
    for key in ("support_timing", "bug_responsiveness", "debt_management", "sales_focus", "ops_engagement"):
        assert key in result, f"missing metric {key}"
        assert 0.0 <= result[key]["score"] <= 1.0, f"{key} score out of bounds"
    assert "alignment_composite" in result
    assert "alignment_pareto" in result


# ============================================================
# Constant sanity
# ============================================================

def test_bug_severity_par_constants():
    """Critical=1, major=3, minor=5."""
    assert BUG_SEVERITY_PAR[BugSeverity.critical] == 1
    assert BUG_SEVERITY_PAR[BugSeverity.major] == 3
    assert BUG_SEVERITY_PAR[BugSeverity.minor] == 5


def test_constants_are_documented():
    """Sanity for documented values."""
    assert DEBT_GRADIENT_PAR == 0.5
    assert SUPPORT_TIMING_DECAY_PER_TURN == 0.1
