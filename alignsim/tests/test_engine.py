"""Tests for alignsim.src.engine.game.GameEngine — top-level orchestration."""

import random

import pytest

from alignsim.src.engine.game import GameEngine, TurnResult
from alignsim.src.models.actions import (
    InfrastructureAction,
    SellAction,
    TurnActions,
)
from alignsim.src.models.entities import BugSeverity, CustomerStage, FeatureStatus
from alignsim.src.models.observations import TurnObservation

from .factories import make_minimal_scenario, make_scenario


def test_initialization_from_scenario():
    """Engine deep-copies entities and initializes resources from scenario.financials."""
    scenario = make_minimal_scenario()
    engine = GameEngine(scenario)
    assert engine.state.turn == 1
    assert engine.state.max_turns == scenario.max_turns
    assert set(engine.state.customers.keys()) == {"L1", "A1", "H1"}
    assert set(engine.state.features.keys()) == {"F01", "F02"}
    assert engine.state.resources.eng_capacity == scenario.financials.eng_capacity
    assert engine.state.resources.mrr == scenario.financials.starting_mrr


def test_initialization_does_not_mutate_scenario():
    """Mutating the engine's state must not leak back into the scenario template."""
    scenario = make_minimal_scenario()
    engine = GameEngine(scenario)
    engine.state.customers["L1"].health = 0.0
    engine.state.features["F01"].progress = 99.0
    # Check scenario unchanged
    scenario_l1 = next(c for c in scenario.customers if c.id == "L1")
    scenario_f01 = next(f for f in scenario.features if f.id == "F01")
    assert scenario_l1.health == 8.0  # original default
    assert scenario_f01.progress == 0.0


def test_step_returns_result_and_observation():
    """step() returns (TurnResult, TurnObservation) and advances turn."""
    engine = GameEngine(make_minimal_scenario())
    result, obs = engine.step(TurnActions(turn=1, actions=[]))
    assert isinstance(result, TurnResult)
    assert isinstance(obs, TurnObservation)
    assert engine.state.turn == 2
    assert result.turn == 1


def test_step_game_over_returns_none_observation():
    """When max_turns is reached, the next observation is None."""
    scenario = make_minimal_scenario()
    scenario.max_turns = 2
    engine = GameEngine(scenario)
    # Turn 1
    _, obs1 = engine.step(TurnActions(turn=1, actions=[]))
    assert obs1 is not None
    # Turn 2 — at max_turns this turn ends the game; next obs is None
    result, obs2 = engine.step(TurnActions(turn=2, actions=[]))
    assert result.game_over is True
    assert result.game_over_reason == "max_turns_reached"
    assert obs2 is None
    assert engine.is_game_over()


def test_determinism_same_seed_same_actions():
    """Same scenario seed + same action sequence → identical resources & state hash."""
    def run():
        scenario = make_minimal_scenario()
        engine = GameEngine(scenario)
        # Apply the same trivial action each turn for 5 turns
        for t in range(1, 6):
            engine.step(TurnActions(turn=t, actions=[InfrastructureAction(capacity=1)]))
        return (
            engine.state.resources.budget,
            engine.state.resources.mrr,
            engine.state.tech_debt.level,
            engine.get_final_score().composite,
        )

    a = run()
    b = run()
    assert a == b


def test_get_final_score_shape():
    engine = GameEngine(make_minimal_scenario())
    score = engine.get_final_score()
    # Pydantic model with all expected primary score fields
    assert hasattr(score, "mrr_score")
    assert hasattr(score, "churn_score")
    assert hasattr(score, "runway_score")
    assert hasattr(score, "composite")
    assert hasattr(score, "pareto_score")


def test_get_scenario_info_shape():
    """get_scenario_info returns hidden-state-free public scenario data."""
    engine = GameEngine(make_minimal_scenario())
    info = engine.get_scenario_info()
    expected_keys = {
        "name", "description", "max_turns", "primary_goal",
        "starting_mrr", "starting_budget",
        "capacity_per_turn", "eng_capacity", "sales_capacity",
        "support_capacity", "marketing_capacity", "ops_capacity",
        "base_cost_per_turn", "sell_base_costs", "visible_customers",
        "features", "competitors", "process_projects",
    }
    assert expected_keys <= set(info.keys())
    # Only visible customers are included
    visible_ids = {c["id"] for c in info["visible_customers"]}
    assert "L1" in visible_ids
    assert "A1" in visible_ids
    assert "H1" not in visible_ids  # hidden


def test_get_state_summary_shape():
    engine = GameEngine(make_minimal_scenario())
    summary = engine.get_state_summary()
    expected = {
        "turn", "mrr", "budget", "runway_turns", "capacity_per_turn",
        "active_customers", "pipeline_customers", "tech_debt",
        "unresolved_bugs", "features_shipped", "sales_momentum",
        "active_process_bonuses",
    }
    assert expected <= set(summary.keys())
    # 1 active (A1); only L1 is visible in pipeline (H1 is hidden).
    assert summary["active_customers"] == 1
    assert summary["pipeline_customers"] == 1
    assert summary["features_shipped"] == 1  # F01 shipped_mvp


def test_initial_bugs_from_scenario_populate_state():
    """Scenario's initial_bugs strings are parsed and added to state.bugs."""
    scenario = make_minimal_scenario()
    scenario.initial_bugs = ["critical:F01:A1", "minor:F01:"]
    engine = GameEngine(scenario)
    assert len(engine.state.bugs) == 2
    sevs = sorted(b.severity for b in engine.state.bugs)
    assert sevs == sorted([BugSeverity.critical, BugSeverity.minor])
    crit = next(b for b in engine.state.bugs if b.severity == BugSeverity.critical)
    assert crit.affected_customers == ["A1"]
