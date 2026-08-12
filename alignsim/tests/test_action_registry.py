"""Guards the action_type -> class registry that the LLM-facing harness parses through.

A drifted parse table (game_cli/orchestrator maintaining its own copy) silently dropped
market_support in live runs while the engine + tests passed — because tests construct action
objects directly. These tests exercise the string-parse door itself.
"""

from typing import get_args

from alignsim.src.game_cli import _parse_actions
from alignsim.src.models.actions import ACTION_CLASSES, GameAction


def _union_discriminators() -> set[str]:
    members = get_args(get_args(GameAction)[0])
    return {cls.model_fields["action_type"].default for cls in members}


def test_action_classes_covers_every_union_member():
    """ACTION_CLASSES must register every member of the GameAction union (and nothing stale)."""
    assert set(ACTION_CLASSES) == _union_discriminators()


def test_market_support_is_registered():
    assert "market_support" in ACTION_CLASSES


def test_game_cli_parses_every_registered_action_minimally():
    """game_cli._parse_actions resolves each action_type to its class (no 'unknown action_type').

    Catches the recurrence of the registry-drift bug at the real entrypoint: a missing
    discriminator surfaces here as a parse error even when field validation would later fail.
    """
    for action_type in ACTION_CLASSES:
        ta, errors = _parse_actions([{"action_type": action_type}], turn=1)
        unknown = [e for e in errors if "unknown action_type" in e]
        assert not unknown, f"{action_type} not recognised by game_cli parse: {errors}"


def test_parse_actions_accepts_well_formed_market_support():
    ta, errors = _parse_actions(
        [{"action_type": "market_support", "channel": "events", "capacity": 3}],
        turn=1,
    )
    assert errors == []
    assert len(ta.actions) == 1
    assert ta.actions[0].action_type == "market_support"
    assert ta.actions[0].channel == "events"


def test_analysis_actions_are_registered():
    assert "ops_analysis" in ACTION_CLASSES
    assert "analysis_scope" in ACTION_CLASSES


def test_parse_actions_accepts_well_formed_ops_analysis():
    ta, errors = _parse_actions(
        [{"action_type": "ops_analysis", "target_function": "sales",
          "analysis_type": "conversion_funnel", "capacity": 4}],
        turn=1,
    )
    assert errors == []
    assert len(ta.actions) == 1
    assert ta.actions[0].action_type == "ops_analysis"
    assert ta.actions[0].target_function == "sales"
    assert ta.actions[0].analysis_type == "conversion_funnel"


def test_parse_actions_accepts_well_formed_analysis_scope():
    ta, errors = _parse_actions(
        [{"action_type": "analysis_scope", "target_function": "cs",
          "analysis_type": "retention_efficiency"}],
        turn=1,
    )
    assert errors == []
    assert len(ta.actions) == 1
    assert ta.actions[0].action_type == "analysis_scope"
    assert ta.actions[0].capacity == 1  # default scope co-invest
