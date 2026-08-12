"""Tests for alignsim.src.engine.scoring — goal attainment and metric extraction."""

import math

import pytest

from alignsim.src.engine.scoring import _compute_avg_churn_rate, _extract_metric, compute_goal_attainment
from alignsim.src.models.entities import (
    CustomerStage,
    FeatureStatus,
    ProcessProjectStatus,
)
from alignsim.src.models.goals import PrimaryGoal, RoleSubGoal

from alignsim.src.models.game_state import TurnRecord

from .factories import (
    make_customer,
    make_feature,
    make_game_state,
    make_process_project,
    make_resource_pool,
)


# --- Primary scores ---

def test_mrr_score_ratio():
    """MRR score = final_mrr / target."""
    state = make_game_state(resources=make_resource_pool(mrr=30_000, runway_turns=100))
    goal = PrimaryGoal(mrr_target=60_000, max_churn_rate=0.02, min_runway_turns=10)
    score = compute_goal_attainment(state, goal)
    assert score.mrr_score == pytest.approx(0.5)


def test_runway_score_log_scale():
    """Runway log scale: par=1.0 at target, log2(1 + ratio) shape, capped at 4×."""
    goal = PrimaryGoal(mrr_target=1, min_runway_turns=10)

    # At target (10 turns): log2(1 + 10/10) = log2(2) = 1.0
    state_par = make_game_state(resources=make_resource_pool(runway_turns=10))
    assert compute_goal_attainment(state_par, goal).runway_score == pytest.approx(1.0)

    # At 2× target (20 turns): log2(1 + 20/10) = log2(3) ≈ 1.585
    state_2x = make_game_state(resources=make_resource_pool(runway_turns=20))
    assert compute_goal_attainment(state_2x, goal).runway_score == pytest.approx(math.log2(3), abs=1e-4)

    # At 4× target (40 turns): log2(1 + 40/10) = log2(5) ≈ 2.322
    state_4x = make_game_state(resources=make_resource_pool(runway_turns=40))
    assert compute_goal_attainment(state_4x, goal).runway_score == pytest.approx(math.log2(5), abs=1e-4)

    # Beyond 4× target (e.g. profitable runway_turns=999): cap at log2(5) ≈ 2.322
    state_cap = make_game_state(resources=make_resource_pool(runway_turns=999))
    assert compute_goal_attainment(state_cap, goal).runway_score == pytest.approx(math.log2(5), abs=1e-4)


def test_runway_score_negative_clamped():
    """Negative runway is clamped to 0 in the score."""
    state = make_game_state(resources=make_resource_pool(runway_turns=-5))
    goal = PrimaryGoal(mrr_target=1, min_runway_turns=10)
    score = compute_goal_attainment(state, goal)
    assert score.runway_score == 0.0


def test_churn_score_retention_rate():
    """Churn score = retention rate = 1 - avg_churn_rate, bounded [0,1]."""
    # 1 churned, 1 active → ever_active=2, rate=0.5, retention=0.5
    churned = make_customer(id="C1", stage=CustomerStage.churned)
    active = make_customer(id="C2", stage=CustomerStage.customer)
    state = make_game_state(customers=[churned, active])
    state.churn_history = [1]  # 1 churn this period

    goal = PrimaryGoal(mrr_target=1, max_churn_rate=0.5, min_runway_turns=1)
    score = compute_goal_attainment(state, goal)
    assert score.churn_score == pytest.approx(0.5)


def test_churn_score_perfect_retention():
    """Zero churn → retention=1.0."""
    active = make_customer(id="C1", stage=CustomerStage.customer)
    state = make_game_state(customers=[active])
    state.churn_history = [0]

    goal = PrimaryGoal(mrr_target=1, max_churn_rate=0.1, min_runway_turns=1)
    score = compute_goal_attainment(state, goal)
    assert score.churn_score == pytest.approx(1.0)


def test_churn_score_full_churn():
    """100% churn → retention=0.0."""
    state = make_game_state(customers=[make_customer(id="C1", stage=CustomerStage.churned)])
    state.churn_history = [1]
    goal = PrimaryGoal(mrr_target=1, max_churn_rate=0.1, min_runway_turns=1)
    score = compute_goal_attainment(state, goal)
    assert score.churn_score == pytest.approx(0.0)


def test_composite_is_geomean_pareto_is_min():
    """composite = geometric mean of the three primary scores; pareto = min (retained)."""
    state = make_game_state(
        resources=make_resource_pool(mrr=30_000, runway_turns=20),
    )
    goal = PrimaryGoal(mrr_target=60_000, max_churn_rate=0.02, min_runway_turns=10)
    score = compute_goal_attainment(state, goal)
    assert score.composite == pytest.approx(
        (score.mrr_score * score.churn_score * score.runway_score) ** (1 / 3), abs=1e-4
    )
    assert score.pareto_score == min(score.mrr_score, score.churn_score, score.runway_score)


def test_composite_is_zero_if_any_primary_is_zero():
    """A single 0 sub-score zeroes the geometric-mean composite (no goal can be ignored)."""
    state = make_game_state(
        resources=make_resource_pool(mrr=0, runway_turns=20),
    )
    goal = PrimaryGoal(mrr_target=60_000, max_churn_rate=0.02, min_runway_turns=10)
    score = compute_goal_attainment(state, goal)
    assert score.mrr_score == 0.0
    assert score.composite == 0.0


# --- Function sub-goals ---

def test_function_sub_goal_metric_features_shipped_solid_plus():
    state = make_game_state(features=[
        make_feature(id="A", status=FeatureStatus.shipped_polished),
        make_feature(id="B", status=FeatureStatus.shipped_solid),
        make_feature(id="C", status=FeatureStatus.shipped_mvp),
        make_feature(id="D", status=FeatureStatus.in_progress),
    ])
    assert _extract_metric(state, "features_shipped_solid_plus") == 2.0


def test_function_sub_goal_pipeline_velocity():
    state = make_game_state(turn=11, total_customers_closed=2)
    # turns = max(11-1, 1) = 10. velocity = 0.2
    assert _extract_metric(state, "pipeline_velocity") == pytest.approx(0.2)


def test_function_sub_goal_avg_customer_health():
    state = make_game_state(customers=[
        make_customer(id="A", stage=CustomerStage.customer, health=8.0),
        make_customer(id="B", stage=CustomerStage.customer, health=6.0),
        make_customer(id="C", stage=CustomerStage.lead, health=10.0),  # not active
    ])
    assert _extract_metric(state, "avg_customer_health") == pytest.approx(7.0)


def test_function_sub_goal_avg_customer_health_no_active():
    state = make_game_state(customers=[make_customer(id="A", stage=CustomerStage.lead)])
    assert _extract_metric(state, "avg_customer_health") == 0.0


def test_function_sub_goal_marketing_leads():
    """Counts inbound_lead events across turn history."""
    state = make_game_state()
    state.turn_history = [
        TurnRecord(turn=1, events=["inbound_lead:C1", "inbound_lead:C2", "deal_won:C3"]),
        TurnRecord(turn=2, events=["inbound_lead:C4"]),
    ]
    assert _extract_metric(state, "marketing_leads_generated") == 3.0


def test_marketing_leads_metric_ignores_awareness_and_radar_events():
    """The metric is unchanged by the awareness rework — only inbound_lead counts."""
    state = make_game_state()
    state.turn_history = [
        TurnRecord(turn=1, events=[
            "inbound_lead:C1", "awareness_built:F14", "competitor_radar:F14:soon",
            "marketing_spend:24000",
        ]),
        TurnRecord(turn=2, events=["awareness_built:F02", "competitor_radar:F02:upcoming"]),
    ]
    assert _extract_metric(state, "marketing_leads_generated") == 1.0


def test_function_sub_goal_process_projects_completed():
    state = make_game_state(process_projects=[
        make_process_project(id="A", status=ProcessProjectStatus.completed),
        make_process_project(id="B", status=ProcessProjectStatus.in_progress),
        make_process_project(id="C", status=ProcessProjectStatus.completed),
    ])
    assert _extract_metric(state, "process_projects_completed") == 2.0


def test_function_sub_goal_tech_debt_control():
    """Score = max(0, 1 - debt/15)."""
    from alignsim.src.models.game_state import TechDebt
    state_low = make_game_state(tech_debt=TechDebt(level=0.0))
    state_high = make_game_state(tech_debt=TechDebt(level=15.0))
    state_over = make_game_state(tech_debt=TechDebt(level=30.0))
    assert _extract_metric(state_low, "tech_debt_control") == 1.0
    assert _extract_metric(state_high, "tech_debt_control") == 0.0
    assert _extract_metric(state_over, "tech_debt_control") == 0.0


def test_function_sub_goal_unknown_metric_raises():
    state = make_game_state()
    with pytest.raises(ValueError):
        _extract_metric(state, "not_a_real_metric")


def test_function_sub_goal_attainment_in_score():
    """Sub-goals get rolled into function_scores / function_composite / function_pareto."""
    state = make_game_state(
        resources=make_resource_pool(mrr=10_000, runway_turns=10),
        features=[
            make_feature(id="A", status=FeatureStatus.shipped_solid),
            make_feature(id="B", status=FeatureStatus.shipped_polished),
        ],
        customers=[
            make_customer(id="X", stage=CustomerStage.customer, health=8.0),
        ],
    )
    goal = PrimaryGoal(
        mrr_target=10_000, max_churn_rate=0.02, min_runway_turns=10,
        sub_goals=[
            RoleSubGoal(role="engineering", description="ship",
                        metric="features_shipped_solid_plus", target_value=2.0),
            RoleSubGoal(role="support", description="health",
                        metric="avg_customer_health", target_value=10.0),
        ],
    )
    score = compute_goal_attainment(state, goal)
    # Primary scores with new formulas: mrr=1.0, churn=1.0 (no churn), runway=log2(2)=1.0
    assert score.runway_score == pytest.approx(1.0)
    assert score.function_scores["engineering"] == pytest.approx(1.0)
    assert score.function_scores["support"] == pytest.approx(0.8)
    assert score.function_composite == pytest.approx((1.0 * 0.8) ** 0.5, abs=1e-4)  # geomean, not sum
    assert score.function_pareto == pytest.approx(0.8)


# --- Avg churn rate ---

def test_avg_churn_rate():
    """rate = total_churned / (current_active + total_churned)."""
    state = make_game_state(customers=[
        make_customer(id="A", stage=CustomerStage.customer),
        make_customer(id="B", stage=CustomerStage.customer),
        make_customer(id="C", stage=CustomerStage.churned),
    ])
    state.churn_history = [1, 0, 0]  # 1 ever churned
    # active=2, churned=1, ever_active=3, rate=1/3
    assert _compute_avg_churn_rate(state) == pytest.approx(1 / 3)


def test_avg_churn_rate_no_customers():
    """Empty state returns 0.0 (no division by zero)."""
    state = make_game_state()
    assert _compute_avg_churn_rate(state) == 0.0
