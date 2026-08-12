"""Tests for alignsim.src.web.game_session — interactive web wrapper around GameEngine."""

import pytest

from alignsim.src.models.actions import (
    BuildAction,
    DiscoverAction,
    FireAction,
    FixBugsAction,
    HireAction,
    InfrastructureAction,
    MarketAction,
    OpsProjectAction,
    OpsProjectSupportAction,
    SellAction,
    SupportAction,
    SustainHireAction,
)
from alignsim.src.models.entities import QualityLevel
from alignsim.src.web.game_session import GameSession, parse_action


# --- parse_action ---

@pytest.mark.parametrize("form, cls", [
    ({"action_type": "build", "feature_id": "F01", "quality": "mvp", "capacity": 5}, BuildAction),
    ({"action_type": "fix_bugs", "bug_id": "BUG_001", "capacity": 4}, FixBugsAction),
    ({"action_type": "infrastructure", "capacity": 3}, InfrastructureAction),
    ({"action_type": "sell", "customer_id": "C1", "sell_action": "outbound", "capacity": 2}, SellAction),
    ({"action_type": "discover", "target_features": "F01,F02", "capacity": 3}, DiscoverAction),
    ({"action_type": "support", "customer_id": "C1", "support_action": "health_check", "capacity": 1}, SupportAction),
    ({"action_type": "market", "channel": "content", "capacity": 4}, MarketAction),
    ({"action_type": "hire", "hiring_function": "engineering", "target_function": "engineering", "capacity": 0}, HireAction),
    ({"action_type": "sustain_hire", "hire_id": "H1", "capacity": 0}, SustainHireAction),
    ({"action_type": "fire", "function": "engineering", "capacity": 0}, FireAction),
    ({"action_type": "ops_project", "project_id": "PP01", "capacity": 2}, OpsProjectAction),
    ({"action_type": "ops_project_support", "project_id": "PP01", "capacity": 2}, OpsProjectSupportAction),
])
def test_parse_action_each_type(form, cls):
    action = parse_action(form)
    assert isinstance(action, cls)


def test_parse_action_unknown_returns_none():
    assert parse_action({"action_type": "not_a_thing", "capacity": 0}) is None


def test_parse_action_build_quality_enum():
    action = parse_action({"action_type": "build", "feature_id": "F01",
                           "quality": "polished", "capacity": 5})
    assert action.quality == QualityLevel.polished


def test_parse_action_discover_empty_features_becomes_empty_list():
    action = parse_action({"action_type": "discover", "target_features": "", "capacity": 3})
    assert action.target_features == []


def test_parse_action_discover_target_features_as_list():
    action = parse_action({"action_type": "discover", "target_features": ["F01", "F02"], "capacity": 3})
    assert action.target_features == ["F01", "F02"]


def test_parse_action_discover_target_features_as_empty_list():
    action = parse_action({"action_type": "discover", "target_features": [], "capacity": 3})
    assert action.target_features == []


# --- GameSession ---

def test_game_session_unknown_scenario_raises():
    with pytest.raises(ValueError):
        GameSession(scenario="not_a_real_scenario")


def test_game_session_initializes_known_scenario():
    session = GameSession(scenario="seed_stage", seed=42, max_turns=8)
    assert session.scenario.name == "seed_stage"
    assert session.scenario.max_turns == 8
    assert session.turn == 1
    assert session.observation is not None


def test_game_session_submit_actions_returns_events():
    session = GameSession(scenario="seed_stage", seed=42, max_turns=8)
    result = session.submit_actions([InfrastructureAction(capacity=1)])
    assert "valid_count" in result
    assert "rejected" in result
    assert "events" in result
    # Turn should have advanced
    assert session.turn == 2


def test_game_session_get_context_keys():
    session = GameSession(scenario="seed_stage", seed=42, max_turns=8)
    ctx = session.get_context()
    expected_keys = {
        "turn", "mrr", "mrr_target", "mrr_pct", "runway", "min_runway",
        "capacity", "eng_capacity", "sales_capacity", "support_capacity",
        "marketing_capacity", "ops_capacity", "sales_momentum",
        "ops_observation", "pending_hires", "budget", "debt_level",
        "debt_value", "active_customers", "maturity", "max_turns",
        "turns_remaining", "pipeline", "features", "bugs", "events",
        "game_over", "score", "bug_backlog", "customer_details", "history",
        "min_rubric",
    }
    assert expected_keys <= set(ctx.keys())


def test_game_session_get_customer_detail_visible_customer():
    session = GameSession(scenario="seed_stage", seed=42, max_turns=8)
    # Find a visible customer to query
    visible = next(
        (c for c in session.state.customers.values() if c.is_visible),
        None,
    )
    assert visible is not None
    detail = session.get_customer_detail(visible.id)
    assert detail is not None
    assert detail["id"] == visible.id
    assert "satisfaction" in detail
    assert "satisfaction_breakdown" in detail
    assert "rubric_weights" in detail


def test_game_session_get_customer_detail_unknown_returns_none():
    session = GameSession(scenario="seed_stage", seed=42, max_turns=8)
    assert session.get_customer_detail("MISSING") is None


def test_game_session_get_customer_detail_hidden_returns_none():
    session = GameSession(scenario="seed_stage", seed=42, max_turns=8)
    hidden = next(
        (c for c in session.state.customers.values() if not c.is_visible),
        None,
    )
    assert hidden is not None, "seed_stage should have hidden customers"
    assert session.get_customer_detail(hidden.id) is None


def test_game_session_history_grows_with_turns():
    session = GameSession(scenario="seed_stage", seed=42, max_turns=8)
    session.submit_actions([InfrastructureAction(capacity=1)])
    session.submit_actions([InfrastructureAction(capacity=1)])
    history = session.get_history()
    assert len(history) == 2
    assert history[0]["turn"] == 1
    assert history[1]["turn"] == 2
