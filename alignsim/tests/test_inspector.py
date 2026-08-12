"""Tests for alignsim.src.harness.inspector.GameInspector.compute_capacity_cost."""

from alignsim.src.engine.game import GameEngine
from alignsim.src.harness.inspector import GameInspector

from .factories import make_pending_hire, make_scenario


def _inspector(**scenario_overrides) -> GameInspector:
    scenario = make_scenario(**scenario_overrides)
    engine = GameEngine(scenario)
    return GameInspector(engine)


def test_sustain_hire_deducts_from_hiring_pool():
    """sustain_hire should deduct hire_capacity_cost from the hiring function's pool."""
    insp = _inspector()
    insp._engine.state.pending_hires = [
        make_pending_hire(id="H1", hiring_function="engineering",
                         active_turns_required=3, active_turns_completed=1),
    ]
    result = insp.compute_capacity_cost([
        {"action_type": "sustain_hire", "hire_id": "H1"},
    ])
    assert result["engineering"]["used"] == 3


def test_sustain_hire_unknown_id_warns():
    """sustain_hire with a nonexistent hire_id should produce a warning, not crash."""
    insp = _inspector()
    result = insp.compute_capacity_cost([
        {"action_type": "sustain_hire", "hire_id": "H99"},
    ])
    assert any("H99" in w for w in result["warnings"])
    assert result["engineering"]["used"] == 0


def test_hire_deducts_from_hiring_function_pool():
    """hire action uses hiring_function (not 'function') to find the pool."""
    insp = _inspector()
    result = insp.compute_capacity_cost([
        {"action_type": "hire", "hiring_function": "sales", "target_function": "cs"},
    ])
    assert result["sales"]["used"] == 3
    assert result["support"]["used"] == 0


def test_malformed_action_string_ignored_with_warning():
    """A non-object action (bare string) must warn, not raise AttributeError -> 500.

    Regression: Haiku POSTed an `actions` list containing a string to
    /compute/capacity-cost; inspector called `action.get(...)` on it and 500'd.
    """
    insp = _inspector()
    result = insp.compute_capacity_cost([
        "build feature F02",  # malformed — a string, not an action object
        {"action_type": "build", "capacity": 4},
    ])
    assert any("malformed action" in w for w in result["warnings"])
    assert result["engineering"]["used"] == 4  # the valid action is still costed


def test_hire_and_sustain_combined():
    """A new hire + sustain of an existing hire both deduct correctly."""
    insp = _inspector()
    insp._engine.state.pending_hires = [
        make_pending_hire(id="H1", hiring_function="engineering",
                         active_turns_required=3, active_turns_completed=0),
    ]
    result = insp.compute_capacity_cost([
        {"action_type": "sustain_hire", "hire_id": "H1"},
        {"action_type": "hire", "hiring_function": "engineering", "target_function": "engineering"},
        {"action_type": "build", "feature_id": "F01", "quality": "mvp", "capacity": 5},
    ])
    # sustain=3 + hire=3 + build=5 = 11 eng capacity used
    assert result["engineering"]["used"] == 11
