"""Tests for alignsim.src.engine.observer.ObservationGenerator."""

import pytest

from alignsim.src.engine.observer import ObservationGenerator
from alignsim.src.models.entities import (
    BugSeverity,
    CustomerStage,
    FeatureStatus,
    ProcessProjectStatus,
    QualityLevel,
)
from alignsim.src.models.scenario import CalibrationParams

from .factories import (
    make_active_bonus,
    make_bug,
    make_customer,
    make_emergent_need,
    make_feature,
    make_game_state,
    make_pending_hire,
    make_process_project,
    make_resource_pool,
)


def _generator(state):
    return ObservationGenerator(state, CalibrationParams())


def test_global_dashboard_fields():
    customer = make_customer(id="C1", stage=CustomerStage.customer, is_visible=True)
    state = make_game_state(
        customers=[customer],
        resources=make_resource_pool(mrr=12_000, runway_turns=8.5,
                                     eng_capacity=10, sales_capacity=8),
    )
    obs = _generator(state).generate()
    d = obs.global_dashboard
    assert d.turn == 1
    assert d.mrr == 12_000
    assert d.active_customers == 1
    assert d.runway_turns == 8.5
    assert d.eng_capacity == 10
    assert d.sales_capacity == 8


def test_sales_observation_filters_to_visible_pipeline():
    """Sales pipeline excludes hidden customers and customers/churned/lost."""
    customers = [
        make_customer(id="L1", stage=CustomerStage.lead, is_visible=True),  # included
        make_customer(id="H1", stage=CustomerStage.lead, is_visible=False),  # hidden
        make_customer(id="A1", stage=CustomerStage.customer, is_visible=True),  # active
        make_customer(id="X1", stage=CustomerStage.churned, is_visible=True),
    ]
    state = make_game_state(customers=customers)
    obs = _generator(state).generate()
    pipeline_ids = {p.customer_id for p in obs.sales.pipeline}
    assert pipeline_ids == {"L1"}


def test_product_eng_features_listed_with_status():
    features = [
        make_feature(id="F01", name="Core", status=FeatureStatus.shipped_solid,
                     progress=100.0),
        make_feature(id="F02", name="Reports", status=FeatureStatus.in_progress,
                     progress=50.0, current_target=QualityLevel.mvp),
    ]
    state = make_game_state(features=features)
    obs = _generator(state).generate()
    by_id = {f.feature_id: f for f in obs.product_eng.features}
    assert by_id["F01"].status == "shipped_solid"
    assert by_id["F02"].status == "in_progress"
    # F02 in_progress with current_target=mvp gets an est_completion_turns
    assert by_id["F02"].est_completion_turns is not None


def test_cs_observation_only_active_customers():
    customers = [
        make_customer(id="A1", stage=CustomerStage.customer, health=8.0),
        make_customer(id="A2", stage=CustomerStage.customer, health=4.0),  # at-risk
        make_customer(id="L1", stage=CustomerStage.lead),
    ]
    state = make_game_state(customers=customers)
    obs = _generator(state).generate()
    health_ids = {h.customer_id for h in obs.cs.customer_health}
    assert health_ids == {"A1", "A2"}
    assert "A2" in obs.cs.at_risk
    assert obs.cs.avg_customer_health == pytest.approx(6.0)


def test_cs_churned_this_turn_excludes_prior_churns():
    """churned_this_turn lists only customers that churned THIS turn (turns_in_current_stage == 0),
    not every customer that ever churned — mirrors the GlobalDashboard gating."""
    customers = [
        make_customer(id="A1", stage=CustomerStage.customer, health=7.0),
        make_customer(id="NOW", stage=CustomerStage.churned, turns_in_current_stage=0),
        make_customer(id="OLD", stage=CustomerStage.churned, turns_in_current_stage=5),
    ]
    state = make_game_state(customers=customers)
    obs = _generator(state).generate()
    assert obs.cs.churned_this_turn == ["NOW"]


def test_ops_observation_categorizes_projects():
    """Available, in_progress, and completed projects land in their respective lists."""
    projects = [
        make_process_project(id="A", status=ProcessProjectStatus.available),
        make_process_project(id="B", status=ProcessProjectStatus.in_progress,
                             progress_turns=1, target_team_capacity_invested=4),
        make_process_project(id="C", status=ProcessProjectStatus.completed,
                             completed_turn=5, target_team_capacity_invested=8),
    ]
    state = make_game_state(process_projects=projects)
    obs = _generator(state).generate()
    avail_ids = {p["id"] for p in obs.ops.available_projects}
    active_ids = {p["id"] for p in obs.ops.active_projects}
    completed_ids = {p["id"] for p in obs.ops.completed_projects}
    assert avail_ids == {"A"}
    assert active_ids == {"B"}
    assert completed_ids == {"C"}
    # The completed project (no active bonus) reports re_run_available
    c_info = next(p for p in obs.ops.completed_projects if p["id"] == "C")
    assert c_info["re_run_available"] is True


def test_ops_observation_active_bonuses():
    """Active bonuses are reported with maintenance cost and degradation."""
    project = make_process_project(id="PP01", ops_capacity_cost=4)
    bonus = make_active_bonus(project_id="PP01", turns_remaining=6,
                              bonus_duration_turns=12, bonus_value=0.20,
                              original_ops_capacity_cost=4)
    state = make_game_state(process_projects=[project])
    state.active_process_bonuses = [bonus]
    obs = _generator(state).generate()
    assert len(obs.ops.active_bonuses) == 1
    info = obs.ops.active_bonuses[0]
    assert info["project_id"] == "PP01"
    assert info["degradation_pct"] == 50  # half-degraded
    assert info["maintenance_cost"] == 2  # round(0.5 * 4)


def test_ops_observation_surfaces_permanent_floor():
    """A floored bonus reports permanent_floor + is_permanent so floored-at-0 isn't read as dead."""
    project = make_process_project(id="PP03", ops_capacity_cost=2)
    bonus = make_active_bonus(project_id="PP03", bonus_value=0.50,
                              permanent_floor_fraction=0.20,
                              turns_remaining=0, bonus_duration_turns=12)
    state = make_game_state(process_projects=[project])
    state.active_process_bonuses = [bonus]
    info = _generator(state).generate().ops.active_bonuses[0]
    assert info["permanent_floor"] == pytest.approx(0.10)   # 0.50 * 0.20
    assert info["is_permanent"] is True
    assert info["degradation_pct"] == 100                   # spike fully spent
    assert info["effective_bonus"] == pytest.approx(0.10)   # but the floor persists


def test_ops_observation_locked_project_visible_with_locked_by():
    """A project with an incomplete prerequisite is VISIBLE-but-locked; it clears once met."""
    prereq = make_process_project(id="PP01", status=ProcessProjectStatus.available)
    gated = make_process_project(id="PP07", status=ProcessProjectStatus.available,
                                 prerequisites=["PP01"])
    state = make_game_state(process_projects=[prereq, gated])
    obs = _generator(state).generate()

    g = next(p for p in obs.ops.available_projects if p["id"] == "PP07")
    assert g["prerequisites"] == ["PP01"]
    assert g["locked"] is True
    assert g["locked_by"] == ["PP01"]
    # The tier-0 prereq itself is unlocked.
    base = next(p for p in obs.ops.available_projects if p["id"] == "PP01")
    assert base["locked"] is False
    assert base["locked_by"] == []

    # Completing the prereq unlocks the gated project.
    state.process_projects["PP01"].status = ProcessProjectStatus.completed
    obs2 = _generator(state).generate()
    g2 = next(p for p in obs2.ops.available_projects if p["id"] == "PP07")
    assert g2["locked"] is False
    assert g2["locked_by"] == []


def test_ops_observation_multi_parent_locked_by_lists_all_incomplete():
    """A multi-parent capstone lists every incomplete prerequisite in locked_by."""
    p1 = make_process_project(id="PP07", status=ProcessProjectStatus.completed)
    p2 = make_process_project(id="PP08", status=ProcessProjectStatus.available)
    cap = make_process_project(id="PP10", status=ProcessProjectStatus.available,
                               prerequisites=["PP08", "PP07"])
    state = make_game_state(process_projects=[p1, p2, cap])
    obs = _generator(state).generate()
    c = next(p for p in obs.ops.available_projects if p["id"] == "PP10")
    # Only the incomplete prereq (PP08) appears in locked_by.
    assert c["locked"] is True
    assert c["locked_by"] == ["PP08"]


def test_pipeline_includes_last_proposed_price():
    """in_deal customer with last_proposed_price surfaces it in the observation."""
    customer = make_customer(
        id="D1", stage=CustomerStage.in_deal, is_visible=True,
        last_proposed_price=5_000, has_received_proposal=True,
    )
    state = make_game_state(customers=[customer])
    obs = _generator(state).generate()
    entry = obs.sales.pipeline[0]
    assert entry.last_proposed_price == 5_000


def test_pipeline_includes_pricing_feedback():
    """Pricing feedback events from turn_record appear on the pipeline entry."""
    from alignsim.src.models.game_state import TurnRecord

    customer = make_customer(
        id="D2", stage=CustomerStage.in_deal, is_visible=True,
        last_proposed_price=6_000, has_received_proposal=True,
    )
    state = make_game_state(customers=[customer])
    turn_record = TurnRecord(
        turn=3,
        events=["pricing_feedback:D2:Your price seems high — consider ~4800"],
    )
    obs = _generator(state).generate(turn_record=turn_record)
    entry = obs.sales.pipeline[0]
    assert entry.pricing_feedback is not None
    assert "D2" in entry.pricing_feedback
    assert "4800" in entry.pricing_feedback


def test_competitor_pricing_in_observation():
    """Competitor pricing steal events show up as deal losses."""
    from alignsim.src.models.game_state import TurnRecord

    customer = make_customer(
        id="D3", stage=CustomerStage.lost, is_visible=True,
        has_received_proposal=True,
    )
    state = make_game_state(customers=[customer])
    turn_record = TurnRecord(
        turn=4,
        events=["deal_lost:D3:competitor_pricing_event:Comp_Alpha"],
    )
    obs = _generator(state).generate(turn_record=turn_record)
    losses = [d for d in obs.sales.deals_this_turn if d.event_type == "loss"]
    assert len(losses) == 1
    assert losses[0].customer_id == "D3"
    assert losses[0].reason == "competitor_pricing_event"
    assert losses[0].lost_to == "Comp_Alpha"


def test_capacity_subtracts_sustain_commitments():
    """Active-phase pending hires reduce displayed capacity by hire_capacity_cost each."""
    hire_eng = make_pending_hire(id="H1", hiring_function="engineering",
                                active_turns_required=3, active_turns_completed=1)
    hire_sales = make_pending_hire(id="H2", hiring_function="sales",
                                  target_function="cs",
                                  active_turns_required=6, active_turns_completed=0,
                                  is_cross_function=True)
    state = make_game_state(
        resources=make_resource_pool(eng_capacity=20, sales_capacity=10, capacity_per_turn=40),
        pending_hires=[hire_eng, hire_sales],
    )
    obs = _generator(state).generate()
    d = obs.global_dashboard
    assert d.eng_capacity == 17  # 20 - 3 (H1 sustain)
    assert d.sales_capacity == 7  # 10 - 3 (H2 sustain)
    assert d.capacity_available == 34  # 40 - 3 - 3


def test_capacity_no_sustain_for_auto_phase_hires():
    """Hires past the active phase (auto) don't reduce displayed capacity."""
    hire = make_pending_hire(id="H1", hiring_function="engineering",
                            active_turns_required=3, active_turns_completed=3)
    state = make_game_state(
        resources=make_resource_pool(eng_capacity=20, capacity_per_turn=40),
        pending_hires=[hire],
    )
    obs = _generator(state).generate()
    d = obs.global_dashboard
    assert d.eng_capacity == 20
    assert d.capacity_available == 40


def test_global_dashboard_bug_backlog_groups_by_severity():
    bugs = [
        make_bug(id="B1", severity=BugSeverity.critical),
        make_bug(id="B2", severity=BugSeverity.major),
        make_bug(id="B3", severity=BugSeverity.major),
        make_bug(id="B4", severity=BugSeverity.minor, is_resolved=True),  # excluded
    ]
    state = make_game_state(bugs=bugs)
    obs = _generator(state).generate()
    assert obs.global_dashboard.bug_backlog == {"critical": 1, "major": 2}


# --- CS observation: emergent needs / churn drivers / undiagnosed decline ---

def _cs_report(obs, cid):
    return next(h for h in obs.cs.customer_health if h.customer_id == cid)


def test_emergent_need_absent_until_revealed():
    """An unrevealed emergent need never appears in the CS observation."""
    customer = make_customer(id="C1", stage=CustomerStage.customer, health=8.0)
    need = make_emergent_need(customer_id="C1", feature_id="F02", is_revealed=False)
    state = make_game_state(customers=[customer], emergent_needs=[need])
    obs = _generator(state).generate()
    assert _cs_report(obs, "C1").emergent_needs == []


def test_emergent_need_present_once_revealed():
    customer = make_customer(id="C1", stage=CustomerStage.customer, health=8.0)
    need = make_emergent_need(customer_id="C1", feature_id="F02", is_revealed=True)
    state = make_game_state(customers=[customer], emergent_needs=[need])
    obs = _generator(state).generate()
    assert _cs_report(obs, "C1").emergent_needs == ["F02"]


def test_emergent_need_hidden_once_met_or_expired():
    customer = make_customer(id="C1", stage=CustomerStage.customer, health=8.0)
    needs = [
        make_emergent_need(id="EN_001", customer_id="C1", feature_id="F02",
                           is_revealed=True, is_met=True),
        make_emergent_need(id="EN_002", customer_id="C1", feature_id="F03",
                           is_revealed=True, is_expired=True),
    ]
    state = make_game_state(customers=[customer], emergent_needs=needs)
    obs = _generator(state).generate()
    assert _cs_report(obs, "C1").emergent_needs == []


def test_churn_drivers_absent_until_revealed():
    customer = make_customer(id="C1", stage=CustomerStage.customer, health=8.0,
                             churn_drivers={"F02": 0.5}, churn_drivers_revealed=False)
    state = make_game_state(customers=[customer])
    obs = _generator(state).generate()
    assert _cs_report(obs, "C1").churn_drivers is None


def test_churn_drivers_present_once_revealed():
    customer = make_customer(id="C1", stage=CustomerStage.customer, health=8.0,
                             churn_drivers={"F02": 0.5}, churn_drivers_revealed=True)
    state = make_game_state(customers=[customer])
    obs = _generator(state).generate()
    assert _cs_report(obs, "C1").churn_drivers == {"F02": 0.5}


def test_undiagnosed_decline_for_unrevealed_bleeding_need():
    """A bleeding (turns_unmet>0) but unrevealed need shows as 'undiagnosed_decline'."""
    customer = make_customer(id="C1", stage=CustomerStage.customer, health=3.0)
    need = make_emergent_need(customer_id="C1", feature_id="F02",
                              is_revealed=False, turns_unmet=2)
    state = make_game_state(customers=[customer], emergent_needs=[need])
    obs = _generator(state).generate()
    assert _cs_report(obs, "C1").cause == "undiagnosed_decline"


def test_revealed_need_gives_specific_cause():
    customer = make_customer(id="C1", stage=CustomerStage.customer, health=3.0)
    need = make_emergent_need(customer_id="C1", feature_id="F02",
                              is_revealed=True, turns_unmet=2)
    state = make_game_state(customers=[customer], emergent_needs=[need])
    obs = _generator(state).generate()
    report = _cs_report(obs, "C1")
    assert report.cause == "unmet_feature_need"
    assert report.emergent_needs == ["F02"]


def test_bug_cause_takes_priority_over_emergent_need():
    """Coarse causes already shared elsewhere (bugs) stay auto-visible and take priority."""
    customer = make_customer(id="C1", stage=CustomerStage.customer, health=3.0)
    bug = make_bug(id="B1", feature_id="F09", affected_customers=["C1"])
    need = make_emergent_need(customer_id="C1", feature_id="F02",
                              is_revealed=False, turns_unmet=2)
    state = make_game_state(customers=[customer], bugs=[bug], emergent_needs=[need])
    obs = _generator(state).generate()
    assert _cs_report(obs, "C1").cause == "bug_in_F09"
