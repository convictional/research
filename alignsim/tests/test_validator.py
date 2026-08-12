"""Tests for alignsim.src.engine.validator — action legality and capacity allocation."""

import pytest

from alignsim.src.engine.validator import ActionValidator
from alignsim.src.models.actions import (
    AnalysisScopeAction,
    BuildAction,
    DiscoverAction,
    FireAction,
    FixBugsAction,
    HireAction,
    InfrastructureAction,
    MarketAction,
    MarketSupportAction,
    OpsAnalysisAction,
    OpsProjectAction,
    OpsProjectSupportAction,
    SellAction,
    SupportAction,
    SustainHireAction,
    TurnActions,
)
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
    make_feature,
    make_game_state,
    make_generator_config,
    make_pending_hire,
    make_process_project,
    make_resource_pool,
)


def _validate(state, actions, calibration=None, generator_config=None):
    cal = calibration or CalibrationParams()
    validator = ActionValidator(state, cal, generator_config=generator_config)
    return validator.validate(TurnActions(turn=state.turn, actions=actions))


# --- Capacity / pool mechanics ---

def test_pool_overflow_rejects_later_actions():
    """Once a pool is exhausted, further actions on that pool are rejected."""
    feature = make_feature(id="F01", cost={"mvp": 100})
    state = make_game_state(features=[feature], resources=make_resource_pool(eng_capacity=10))
    actions = [
        BuildAction(feature_id="F01", quality=QualityLevel.mvp, capacity=8),
        BuildAction(feature_id="F01", quality=QualityLevel.mvp, capacity=5),  # 8+5 > 10
    ]
    result = _validate(state, actions)
    assert len(result.valid_actions) == 1
    assert len(result.rejected_actions) == 1
    assert "Insufficient engineering capacity" in result.rejected_actions[0].reason


def test_pools_are_independent():
    """Engineering actions don't reduce the sales pool."""
    feature = make_feature(id="F01", cost={"mvp": 100})
    customer = make_customer(id="C1", stage=CustomerStage.lead, is_visible=True, size=1)
    state = make_game_state(
        features=[feature], customers=[customer],
        resources=make_resource_pool(eng_capacity=20, sales_capacity=2),
    )
    actions = [
        BuildAction(feature_id="F01", quality=QualityLevel.mvp, capacity=20),
        SellAction(customer_id="C1", sell_action="outbound", capacity=2),
    ]
    result = _validate(state, actions)
    assert len(result.valid_actions) == 2
    assert result.rejected_actions == []


def test_sustain_hire_capacity_pre_committed():
    """Sustain_hire is allocated before other actions, even if they consume the pool first."""
    hire = make_pending_hire(
        id="H1", hiring_function="engineering", target_function="engineering",
        active_turns_required=3, active_turns_completed=1,
    )
    feature = make_feature(id="F01", cost={"mvp": 100})
    state = make_game_state(
        features=[feature],
        resources=make_resource_pool(eng_capacity=5),
    )
    state.pending_hires.append(hire)

    cal = CalibrationParams()  # hire_capacity_cost=3
    actions = [
        # build wants 5, but sustain pre-commits 3, leaving 2 → build rejected
        BuildAction(feature_id="F01", quality=QualityLevel.mvp, capacity=5),
        SustainHireAction(hire_id="H1"),
    ]
    result = _validate(state, actions, calibration=cal)
    valid_types = {type(a).__name__ for a in result.valid_actions}
    rejected_types = {type(r.action).__name__ for r in result.rejected_actions}
    assert "SustainHireAction" in valid_types
    assert "BuildAction" in rejected_types


# --- Build validation ---

def test_build_nonexistent_feature_rejected():
    state = make_game_state(resources=make_resource_pool(eng_capacity=20))
    result = _validate(state, [BuildAction(feature_id="MISSING", quality=QualityLevel.mvp, capacity=5)])
    assert len(result.rejected_actions) == 1
    assert "MISSING" in result.rejected_actions[0].reason


def test_build_dependency_unmet_rejected():
    """Building a feature whose dependency isn't shipped is rejected."""
    parent = make_feature(id="F01", status=FeatureStatus.in_progress)
    child = make_feature(id="F02", depends_on=["F01"])
    state = make_game_state(features=[parent, child],
                            resources=make_resource_pool(eng_capacity=20))
    result = _validate(state, [BuildAction(feature_id="F02", quality=QualityLevel.mvp, capacity=5)])
    assert len(result.rejected_actions) == 1
    assert "Dependency F01" in result.rejected_actions[0].reason


def test_build_already_at_or_above_quality_rejected():
    """Cannot build to a quality at or below current shipped status."""
    feature = make_feature(id="F01", status=FeatureStatus.shipped_solid)
    state = make_game_state(features=[feature], resources=make_resource_pool(eng_capacity=20))
    # Trying to ship MVP when already shipped_solid
    result = _validate(state, [BuildAction(feature_id="F01", quality=QualityLevel.mvp, capacity=5)])
    assert len(result.rejected_actions) == 1
    assert "already at or above" in result.rejected_actions[0].reason


def test_build_with_dependency_met_accepted():
    parent = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    child = make_feature(id="F02", depends_on=["F01"])
    state = make_game_state(features=[parent, child],
                            resources=make_resource_pool(eng_capacity=20))
    result = _validate(state, [BuildAction(feature_id="F02", quality=QualityLevel.mvp, capacity=5)])
    assert len(result.valid_actions) == 1


# --- Sell validation ---

def test_sell_nonexistent_customer_rejected():
    state = make_game_state(resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [SellAction(customer_id="MISSING",
                                          sell_action="outbound", capacity=1)])
    assert len(result.rejected_actions) == 1
    assert "MISSING" in result.rejected_actions[0].reason


def test_sell_invisible_customer_rejected():
    customer = make_customer(id="C1", stage=CustomerStage.lead, is_visible=False)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [SellAction(customer_id="C1",
                                          sell_action="outbound", capacity=1)])
    assert len(result.rejected_actions) == 1
    assert "not been discovered" in result.rejected_actions[0].reason


def test_sell_wrong_stage_action_rejected():
    """A demo on a lead, or outbound on an in_deal, is rejected."""
    leadc = make_customer(id="C1", stage=CustomerStage.lead, is_visible=True, size=1)
    indealc = make_customer(id="C2", stage=CustomerStage.in_deal, is_visible=True, size=1)
    state = make_game_state(customers=[leadc, indealc],
                            resources=make_resource_pool(sales_capacity=10))
    actions = [
        SellAction(customer_id="C1", sell_action="demo", capacity=1),
        SellAction(customer_id="C2", sell_action="outbound", capacity=1),
    ]
    result = _validate(state, actions)
    assert len(result.rejected_actions) == 2


def test_sell_below_minimum_capacity_rejected():
    """Sell must be at least base_cost * customer.size capacity."""
    big_customer = make_customer(id="C1", stage=CustomerStage.qualified,
                                 is_visible=True, size=5)
    state = make_game_state(customers=[big_customer],
                            resources=make_resource_pool(sales_capacity=10))
    cal = CalibrationParams()  # demo base cost 1, so min for size 5 = 5
    result = _validate(state, [SellAction(customer_id="C1",
                                          sell_action="demo", capacity=4)],
                       calibration=cal)
    assert len(result.rejected_actions) == 1
    assert "minimum" in result.rejected_actions[0].reason


# --- Support validation ---

def test_support_non_customer_rejected():
    """Support only valid for active (stage=customer) customers."""
    leadc = make_customer(id="C1", stage=CustomerStage.lead, is_visible=True)
    state = make_game_state(customers=[leadc],
                            resources=make_resource_pool(support_capacity=5))
    result = _validate(state, [SupportAction(customer_id="C1",
                                             support_action="health_check", capacity=1)])
    assert len(result.rejected_actions) == 1
    assert "not an active customer" in result.rejected_actions[0].reason


def test_support_active_customer_accepted():
    active = make_customer(id="C1", stage=CustomerStage.customer, is_visible=True)
    state = make_game_state(customers=[active],
                            resources=make_resource_pool(support_capacity=5))
    result = _validate(state, [SupportAction(customer_id="C1",
                                             support_action="health_check", capacity=1)])
    assert len(result.valid_actions) == 1


# --- Fix bugs validation ---

def test_fix_bugs_nonexistent_id_rejected():
    state = make_game_state(resources=make_resource_pool(eng_capacity=10))
    result = _validate(state, [FixBugsAction(bug_id="BUG_999", capacity=4)])
    assert len(result.rejected_actions) == 1
    assert "BUG_999" in result.rejected_actions[0].reason


def test_fix_bugs_no_bugs_rejected():
    """Auto-target with no unresolved bugs is rejected."""
    state = make_game_state(resources=make_resource_pool(eng_capacity=10))
    result = _validate(state, [FixBugsAction(bug_id=None, capacity=4)])
    assert len(result.rejected_actions) == 1
    assert "No unresolved bugs" in result.rejected_actions[0].reason


def test_fix_bugs_with_unresolved_accepted():
    bug = make_bug(id="BUG_001", severity=BugSeverity.minor)
    state = make_game_state(bugs=[bug], resources=make_resource_pool(eng_capacity=10))
    result = _validate(state, [FixBugsAction(bug_id="BUG_001", capacity=2)])
    assert len(result.valid_actions) == 1


# --- Discover validation ---

def test_discover_no_hidden_rejected():
    visible = make_customer(id="C1", is_visible=True)
    shipped = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[visible], features=[shipped],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [DiscoverAction(capacity=3)])
    assert len(result.rejected_actions) == 1
    assert "No hidden customers" in result.rejected_actions[0].reason


def test_discover_target_features_must_be_shipped():
    """If target_features is non-empty, at least one must be shipped."""
    hidden = make_customer(id="H1", is_visible=False, feature_needs={"F02": {"mvp": 0.5}})
    unshipped = make_feature(id="F02", status=FeatureStatus.not_started)
    state = make_game_state(customers=[hidden], features=[unshipped],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [DiscoverAction(target_features=["F02"], capacity=3)])
    assert len(result.rejected_actions) == 1
    assert "shipped" in result.rejected_actions[0].reason


def test_discover_with_generator_config_always_valid():
    """With generator_config present, discover is valid even without hidden customers."""
    visible = make_customer(id="C1", is_visible=True)
    shipped = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[visible], features=[shipped],
                            resources=make_resource_pool(sales_capacity=10))
    config = make_generator_config()
    result = _validate(state, [DiscoverAction(capacity=3)], generator_config=config)
    assert len(result.valid_actions) == 1
    assert len(result.rejected_actions) == 0


def test_no_generator_config_backward_compat():
    """Without generator_config, behavior matches pre-change: hidden pool filtering."""
    hidden = make_customer(id="H1", is_visible=False, feature_needs={"F01": {"mvp": 0.6}})
    shipped = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[hidden], features=[shipped],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [DiscoverAction(target_features=["F01"], capacity=3)])
    assert len(result.valid_actions) == 1


# --- Hire validation ---

def test_hire_insufficient_budget_rejected():
    state = make_game_state(resources=make_resource_pool(eng_capacity=10, budget=10))
    cal = CalibrationParams()  # hire_budget_cost = 40 * 2 = 80 > 10
    result = _validate(state, [HireAction(hiring_function="engineering",
                                          target_function="engineering")],
                       calibration=cal)
    assert len(result.rejected_actions) == 1
    assert "Insufficient budget" in result.rejected_actions[0].reason


def test_hire_sufficient_budget_accepted():
    state = make_game_state(resources=make_resource_pool(eng_capacity=10, budget=200_000))
    cal = CalibrationParams()
    result = _validate(state, [HireAction(hiring_function="engineering",
                                          target_function="engineering")],
                       calibration=cal)
    assert len(result.valid_actions) == 1


# --- Fire validation ---

def test_fire_zero_capacity_rejected():
    state = make_game_state(resources=make_resource_pool(ops_capacity=0))
    result = _validate(state, [FireAction(function="ops")])
    assert len(result.rejected_actions) == 1
    assert "no capacity remaining" in result.rejected_actions[0].reason


def test_fire_with_capacity_accepted():
    state = make_game_state(resources=make_resource_pool(ops_capacity=4))
    result = _validate(state, [FireAction(function="ops")])
    assert len(result.valid_actions) == 1


# --- Ops project validation ---

def test_ops_project_nonexistent_rejected():
    state = make_game_state(resources=make_resource_pool(ops_capacity=10))
    result = _validate(state, [OpsProjectAction(project_id="MISSING", capacity=2)])
    assert len(result.rejected_actions) == 1


def test_ops_project_insufficient_capacity_rejected():
    project = make_process_project(id="PP01", ops_capacity_cost=4)
    state = make_game_state(process_projects=[project],
                            resources=make_resource_pool(ops_capacity=10))
    # 3 capacity < project's required 4 — rejected even though the pool has 10
    result = _validate(state, [OpsProjectAction(project_id="PP01", capacity=3)])
    assert len(result.rejected_actions) == 1
    assert "requires" in result.rejected_actions[0].reason


def test_ops_project_re_run_lapsed_full_capacity_required():
    """Re-run of completed project with no active bonus requires full ops cost."""
    project = make_process_project(id="PP01", ops_capacity_cost=6,
                                   status=ProcessProjectStatus.completed)
    state = make_game_state(process_projects=[project],
                            resources=make_resource_pool(ops_capacity=10))
    result = _validate(state, [OpsProjectAction(project_id="PP01", capacity=4)])
    assert len(result.rejected_actions) == 1
    assert "lapsed project" in result.rejected_actions[0].reason


def test_ops_project_maintenance_cost_check():
    """Maintenance refresh requires scaled capacity based on degradation."""
    project = make_process_project(id="PP01", ops_capacity_cost=6,
                                   status=ProcessProjectStatus.completed,
                                   bonus_duration_turns=12)
    bonus = make_active_bonus(project_id="PP01", original_ops_capacity_cost=6,
                              turns_remaining=6, bonus_duration_turns=12)
    state = make_game_state(process_projects=[project],
                            resources=make_resource_pool(ops_capacity=10))
    state.active_process_bonuses.append(bonus)
    # Half-degraded: round(0.5 * 6) = 3 capacity required
    result_low = _validate(state, [OpsProjectAction(project_id="PP01", capacity=2)])
    assert len(result_low.rejected_actions) == 1
    assert "Maintenance refresh" in result_low.rejected_actions[0].reason

    # 3 capacity passes
    result_ok = _validate(state, [OpsProjectAction(project_id="PP01", capacity=3)])
    assert len(result_ok.valid_actions) == 1


# --- Ops project support validation ---

def test_ops_support_not_in_progress_rejected():
    """Support actions only valid on in-progress projects."""
    project = make_process_project(id="PP01", target_function="sales",
                                   status=ProcessProjectStatus.available)
    state = make_game_state(process_projects=[project],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [OpsProjectSupportAction(project_id="PP01", capacity=2)])
    assert len(result.rejected_actions) == 1
    assert "not in progress" in result.rejected_actions[0].reason


def test_ops_support_in_progress_accepted():
    project = make_process_project(id="PP01", target_function="sales",
                                   status=ProcessProjectStatus.in_progress)
    state = make_game_state(process_projects=[project],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [OpsProjectSupportAction(project_id="PP01", capacity=2)])
    assert len(result.valid_actions) == 1


def test_ops_support_nonexistent_rejected():
    state = make_game_state(resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [OpsProjectSupportAction(project_id="MISSING", capacity=2)])
    assert len(result.rejected_actions) == 1


# --- Sustain hire validation ---

def test_sustain_unknown_id_rejected():
    state = make_game_state(resources=make_resource_pool(eng_capacity=10))
    result = _validate(state, [SustainHireAction(hire_id="H99")])
    assert len(result.rejected_actions) == 1
    assert "No pending hire" in result.rejected_actions[0].reason


def test_sustain_already_in_auto_phase_rejected():
    hire = make_pending_hire(id="H1", active_turns_required=3, active_turns_completed=3)
    state = make_game_state(resources=make_resource_pool(eng_capacity=10))
    state.pending_hires.append(hire)
    result = _validate(state, [SustainHireAction(hire_id="H1")])
    assert len(result.rejected_actions) == 1
    assert "auto-phase" in result.rejected_actions[0].reason


# --- Mixed actions ---

def test_mixed_valid_and_invalid_actions():
    """Mix of valid and invalid actions split correctly."""
    feature = make_feature(id="F01", cost={"mvp": 50})
    customer = make_customer(id="C1", stage=CustomerStage.lead, is_visible=True, size=1)
    state = make_game_state(
        features=[feature], customers=[customer],
        resources=make_resource_pool(eng_capacity=20, sales_capacity=10, marketing_capacity=0),
    )
    actions = [
        BuildAction(feature_id="F01", quality=QualityLevel.mvp, capacity=10),  # valid
        BuildAction(feature_id="MISSING", quality=QualityLevel.mvp, capacity=5),  # invalid
        SellAction(customer_id="C1", sell_action="outbound", capacity=1),  # valid
        SellAction(customer_id="C1", sell_action="demo", capacity=1),  # invalid (lead)
        InfrastructureAction(capacity=5),  # valid
        MarketAction(channel="content", capacity=3),  # rejected (marketing pool empty)
    ]
    result = _validate(state, actions)
    valid_count = len(result.valid_actions)
    rejected_count = len(result.rejected_actions)
    assert valid_count == 3
    assert rejected_count == 3


# --- Pricing validation ---

def test_proposed_deal_value_rejected_on_outbound():
    """proposed_deal_value is only valid for proposal/negotiate."""
    customer = make_customer(id="C1", stage=CustomerStage.lead, is_visible=True, size=1)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [SellAction(customer_id="C1", sell_action="outbound",
                                          capacity=1, proposed_deal_value=500)])
    assert len(result.rejected_actions) == 1
    assert "proposal/negotiate" in result.rejected_actions[0].reason


def test_proposed_deal_value_rejected_on_demo():
    """proposed_deal_value rejected on demo actions."""
    customer = make_customer(id="C1", stage=CustomerStage.qualified, is_visible=True, size=1)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [SellAction(customer_id="C1", sell_action="demo",
                                          capacity=1, proposed_deal_value=500)])
    assert len(result.rejected_actions) == 1
    assert "proposal/negotiate" in result.rejected_actions[0].reason


# --- Marketing budget gate ---

def test_budgeted_market_action_rejected_when_budget_too_low():
    """A budgeted (events) market action that would drive budget < 0 is rejected."""
    cal = CalibrationParams()  # events cost 8000/cap
    state = make_game_state(
        features=[make_feature(id="F01", status=FeatureStatus.shipped_mvp)],
        resources=make_resource_pool(marketing_capacity=10, budget=10_000),
    )
    # 3 cap * 8000 = 24000 > 10000 budget → rejected
    result = _validate(state, [MarketAction(channel="events", target_features=["F01"], capacity=3)],
                       calibration=cal)
    assert len(result.valid_actions) == 0
    assert len(result.rejected_actions) == 1
    assert "budget" in result.rejected_actions[0].reason.lower()


def test_outbound_market_action_never_budget_gated():
    """outbound_campaign is capacity-only — passes even at zero budget."""
    cal = CalibrationParams()
    state = make_game_state(
        features=[make_feature(id="F01", status=FeatureStatus.shipped_mvp)],
        resources=make_resource_pool(marketing_capacity=10, budget=0),
    )
    result = _validate(state, [MarketAction(channel="outbound_campaign", target_features=["F01"], capacity=3)],
                       calibration=cal)
    assert len(result.valid_actions) == 1
    assert len(result.rejected_actions) == 0


def test_budgeted_market_action_passes_with_sufficient_budget():
    cal = CalibrationParams()
    state = make_game_state(
        features=[make_feature(id="F01", status=FeatureStatus.shipped_mvp)],
        resources=make_resource_pool(marketing_capacity=10, budget=1_000_000),
    )
    result = _validate(state, [MarketAction(channel="events", target_features=["F01"], capacity=3)],
                       calibration=cal)
    assert len(result.valid_actions) == 1


def test_market_support_draws_from_sales_pool():
    """market_support consumes Sales capacity (it's Sales co-investing in Marketing's campaign)."""
    state = make_game_state(resources=make_resource_pool(sales_capacity=5, marketing_capacity=5))
    result = _validate(state, [MarketSupportAction(channel="events", capacity=3)])
    assert len(result.valid_actions) == 1
    assert result.total_capacity_used == 3  # from the sales pool


def test_market_support_over_sales_pool_rejected():
    state = make_game_state(resources=make_resource_pool(sales_capacity=2, marketing_capacity=10))
    result = _validate(state, [MarketSupportAction(channel="events", capacity=4)])
    assert len(result.valid_actions) == 0
    assert len(result.rejected_actions) == 1
    assert "sales" in result.rejected_actions[0].reason.lower()


def test_market_support_competes_with_sells_for_sales_pool():
    """market_support and sell draw from the same sales pool; the later one over-budget is dropped."""
    customer = make_customer(id="C1", stage=CustomerStage.lead, is_visible=True, size=1)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=3))
    result = _validate(state, [
        SellAction(customer_id="C1", sell_action="outbound", capacity=2),  # uses 2 of 3
        MarketSupportAction(channel="events", capacity=2),                 # needs 2, only 1 left → rejected
    ])
    assert len(result.valid_actions) == 1
    assert len(result.rejected_actions) == 1


def test_market_budget_gate_accumulates_across_actions():
    """Running budget is consumed across multiple budgeted actions in one turn."""
    cal = CalibrationParams()  # content cost 3000/cap
    state = make_game_state(
        features=[make_feature(id="F01", status=FeatureStatus.shipped_mvp)],
        resources=make_resource_pool(marketing_capacity=20, budget=15_000),
    )
    # First content action: 3 * 3000 = 9000 (ok, 6000 left). Second: 3 * 3000 = 9000 > 6000 → rejected.
    result = _validate(state, [
        MarketAction(channel="content", target_features=["F01"], capacity=3),
        MarketAction(channel="content", target_features=["F01"], capacity=3),
    ], calibration=cal)
    assert len(result.valid_actions) == 1
    assert len(result.rejected_actions) == 1


def test_proposed_deal_value_zero_rejected():
    """proposed_deal_value <= 0 is rejected."""
    customer = make_customer(id="C1", stage=CustomerStage.in_deal, is_visible=True, size=1)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [SellAction(customer_id="C1", sell_action="proposal",
                                          capacity=1, proposed_deal_value=0)])
    assert len(result.rejected_actions) == 1
    assert "positive" in result.rejected_actions[0].reason


def test_negotiate_without_prior_proposal_rejected():
    """Negotiate before any proposal is rejected."""
    customer = make_customer(id="C1", stage=CustomerStage.in_deal, is_visible=True,
                             size=1, has_received_proposal=False)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [SellAction(customer_id="C1", sell_action="negotiate",
                                          capacity=1)])
    assert len(result.rejected_actions) == 1
    assert "proposal" in result.rejected_actions[0].reason


def test_proposal_with_price_accepted():
    """Proposal with valid proposed_deal_value is accepted."""
    customer = make_customer(id="C1", stage=CustomerStage.in_deal, is_visible=True, size=1)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [SellAction(customer_id="C1", sell_action="proposal",
                                          capacity=1, proposed_deal_value=800)])
    assert len(result.valid_actions) == 1
    assert len(result.rejected_actions) == 0


def test_negotiate_after_proposal_accepted():
    """Negotiate is valid when has_received_proposal is True."""
    customer = make_customer(id="C1", stage=CustomerStage.in_deal, is_visible=True,
                             size=1, has_received_proposal=True)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(sales_capacity=10))
    result = _validate(state, [SellAction(customer_id="C1", sell_action="negotiate",
                                          capacity=1, proposed_deal_value=700)])
    assert len(result.valid_actions) == 1
    assert len(result.rejected_actions) == 0


# --- Ops analysis handshake validation (Stage B) ---

def test_ops_analysis_charges_ops_pool():
    state = make_game_state(resources=make_resource_pool(ops_capacity=4))
    result = _validate(state, [
        OpsAnalysisAction(target_function="sales", analysis_type="conversion_funnel", capacity=4),
    ])
    assert len(result.valid_actions) == 1
    # Drawn from the ops pool (4 of 4 used).
    assert result.total_capacity_used == 4


def test_ops_analysis_rejected_below_cost():
    state = make_game_state(resources=make_resource_pool(ops_capacity=10))
    # Default analysis_ops_capacity_cost is 2; capacity 1 is below it.
    result = _validate(state, [
        OpsAnalysisAction(target_function="sales", analysis_type="conversion_funnel", capacity=1),
    ])
    assert len(result.rejected_actions) == 1
    assert "requires" in result.rejected_actions[0].reason


def test_ops_analysis_rejected_when_ops_pool_short():
    state = make_game_state(resources=make_resource_pool(ops_capacity=2))
    # Meets the cost (2) but requests 4, and the ops pool only has 2.
    result = _validate(state, [
        OpsAnalysisAction(target_function="sales", analysis_type="conversion_funnel", capacity=4),
    ])
    assert len(result.rejected_actions) == 1
    assert "Insufficient ops capacity" in result.rejected_actions[0].reason


@pytest.mark.parametrize("tf,pool_name,zero_kwarg", [
    ("engineering", "engineering", "eng_capacity"),
    ("sales", "sales", "sales_capacity"),
    ("cs", "support", "support_capacity"),
    ("marketing", "marketing", "marketing_capacity"),
])
def test_analysis_scope_draws_from_target_pool(tf, pool_name, zero_kwarg):
    """Scope co-investment draws from the requesting team's pool (cs -> support)."""
    # Zero out the target pool only → the 1-cap scope is rejected for THAT pool.
    state = make_game_state(resources=make_resource_pool(**{zero_kwarg: 0}))
    result = _validate(state, [
        AnalysisScopeAction(target_function=tf, analysis_type="capacity_bottleneck", capacity=1),
    ])
    assert len(result.rejected_actions) == 1
    assert f"Insufficient {pool_name} capacity" in result.rejected_actions[0].reason


@pytest.mark.parametrize("tf,set_kwarg", [
    ("engineering", "eng_capacity"),
    ("sales", "sales_capacity"),
    ("cs", "support_capacity"),
    ("marketing", "marketing_capacity"),
])
def test_analysis_scope_accepted_with_target_pool_capacity(tf, set_kwarg):
    pools = dict(eng_capacity=0, sales_capacity=0, support_capacity=0,
                 marketing_capacity=0, ops_capacity=0)
    pools[set_kwarg] = 3  # only the target pool has capacity
    state = make_game_state(resources=make_resource_pool(**pools))
    result = _validate(state, [
        AnalysisScopeAction(target_function=tf, analysis_type="capacity_bottleneck", capacity=1),
    ])
    assert len(result.valid_actions) == 1


# --- Ops project prerequisites / tech-tree DAG (Stage C) ---

def test_ops_project_locked_prereq_incomplete_rejected():
    prereq = make_process_project(id="PP01", status=ProcessProjectStatus.available)
    gated = make_process_project(id="PP07", ops_capacity_cost=6, prerequisites=["PP01"])
    state = make_game_state(process_projects=[prereq, gated],
                            resources=make_resource_pool(ops_capacity=10))
    result = _validate(state, [OpsProjectAction(project_id="PP07", capacity=6)])
    assert len(result.rejected_actions) == 1
    assert "locked" in result.rejected_actions[0].reason
    assert "PP01" in result.rejected_actions[0].reason


def test_ops_project_unlocked_once_prereq_completed():
    prereq = make_process_project(id="PP01", status=ProcessProjectStatus.completed)
    gated = make_process_project(id="PP07", ops_capacity_cost=6, prerequisites=["PP01"])
    state = make_game_state(process_projects=[prereq, gated],
                            resources=make_resource_pool(ops_capacity=10))
    result = _validate(state, [OpsProjectAction(project_id="PP07", capacity=6)])
    assert len(result.valid_actions) == 1


def test_ops_project_missing_prereq_id_rejected():
    gated = make_process_project(id="PP07", ops_capacity_cost=6, prerequisites=["GHOST"])
    state = make_game_state(process_projects=[gated],
                            resources=make_resource_pool(ops_capacity=10))
    result = _validate(state, [OpsProjectAction(project_id="PP07", capacity=6)])
    assert len(result.rejected_actions) == 1
    assert "locked" in result.rejected_actions[0].reason


def test_ops_tier2_requires_both_prereqs():
    pp07 = make_process_project(id="PP07", status=ProcessProjectStatus.completed)
    pp08 = make_process_project(id="PP08", status=ProcessProjectStatus.available)  # not done
    cap = make_process_project(id="PP10", ops_capacity_cost=6, prerequisites=["PP08", "PP07"])
    state = make_game_state(process_projects=[pp07, pp08, cap],
                            resources=make_resource_pool(ops_capacity=10))
    rej = _validate(state, [OpsProjectAction(project_id="PP10", capacity=6)])
    assert len(rej.rejected_actions) == 1
    assert "PP08" in rej.rejected_actions[0].reason  # the incomplete one is named
    # Completing the second prereq makes it legal.
    state.process_projects["PP08"].status = ProcessProjectStatus.completed
    ok = _validate(state, [OpsProjectAction(project_id="PP10", capacity=6)])
    assert len(ok.valid_actions) == 1


def test_ops_project_in_progress_not_re_gated_by_prereq():
    """An already in-progress project is not re-blocked, even if a prereq is now incomplete."""
    prereq = make_process_project(id="PP01", status=ProcessProjectStatus.available)
    gated = make_process_project(id="PP07", ops_capacity_cost=6, prerequisites=["PP01"],
                                 status=ProcessProjectStatus.in_progress)
    state = make_game_state(process_projects=[prereq, gated],
                            resources=make_resource_pool(ops_capacity=10))
    result = _validate(state, [OpsProjectAction(project_id="PP07", capacity=6)])
    assert len(result.valid_actions) == 1
