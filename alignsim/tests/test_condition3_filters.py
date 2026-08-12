"""Tests for Condition 3 observation filtering, action validation, and permissions."""

from __future__ import annotations

import pytest

from alignsim.src.engine.game import GameEngine
from alignsim.src.harness.condition3_filters import (
    AGENT_TO_ENGINE_FUNCTION,
    ALL_FUNCTIONS,
    ENGINE_TO_AGENT_FUNCTION,
    FUNCTION_PERMISSIONS,
    SHARED_EVENT_PREFIXES,
    STARTING_FUNCTIONS,
    FunctionActionValidation,
    _parse_hire_started_owner,
    filter_events,
    filter_observation,
    is_compute_allowed,
    is_query_allowed,
    validate_function_actions,
)
from alignsim.src.models.game_state import PendingHire
from alignsim.src.scenarios.seed_stage import create_seed_stage_scenario


@pytest.fixture
def engine() -> GameEngine:
    scenario = create_seed_stage_scenario(seed=42)
    scenario.max_turns = 12
    return GameEngine(scenario)


# ---------------------------------------------------------------------------
# Constants / config tests
# ---------------------------------------------------------------------------


class TestConstants:
    def test_all_functions(self):
        assert ALL_FUNCTIONS == {"engineering", "sales", "support", "marketing", "ops"}

    def test_starting_functions(self):
        assert STARTING_FUNCTIONS == {"engineering", "sales", "marketing"}

    def test_agent_engine_function_roundtrip(self):
        for agent_fn, engine_fn in AGENT_TO_ENGINE_FUNCTION.items():
            assert ENGINE_TO_AGENT_FUNCTION[engine_fn] == agent_fn

    def test_support_maps_to_cs(self):
        assert AGENT_TO_ENGINE_FUNCTION["support"] == "cs"
        assert ENGINE_TO_AGENT_FUNCTION["cs"] == "support"

    def test_every_function_has_permissions(self):
        for fn in ALL_FUNCTIONS:
            assert fn in FUNCTION_PERMISSIONS

    def test_all_functions_see_global(self):
        for fn in ALL_FUNCTIONS:
            assert "global" in FUNCTION_PERMISSIONS[fn].obs_sections

    def test_all_functions_can_query_rejections(self):
        for fn in ALL_FUNCTIONS:
            assert "rejections" in FUNCTION_PERMISSIONS[fn].allowed_queries


# ---------------------------------------------------------------------------
# Observation filtering
# ---------------------------------------------------------------------------


class TestFilterObservation:
    def test_engineering_sees_global_and_product_eng(self, engine: GameEngine):
        obs = engine.get_initial_observation()
        filtered = filter_observation(obs, "engineering", engine)

        assert "global" in filtered
        assert "product_eng" in filtered
        assert "sales" not in filtered
        assert "cs" not in filtered
        assert "ops" not in filtered
        assert "marketing_history" not in filtered

    def test_sales_sees_global_and_sales(self, engine: GameEngine):
        obs = engine.get_initial_observation()
        filtered = filter_observation(obs, "sales", engine)

        assert "global" in filtered
        assert "sales" in filtered
        assert "product_eng" not in filtered
        assert "cs" not in filtered

    def test_support_sees_global_and_cs(self, engine: GameEngine):
        obs = engine.get_initial_observation()
        filtered = filter_observation(obs, "support", engine)

        assert "global" in filtered
        assert "cs" in filtered
        assert "sales" not in filtered
        assert "product_eng" not in filtered

    def test_marketing_sees_global_and_marketing_history(self, engine: GameEngine):
        obs = engine.get_initial_observation()
        filtered = filter_observation(obs, "marketing", engine)

        assert "global" in filtered
        assert "marketing_history" in filtered
        assert "sales" not in filtered
        assert "product_eng" not in filtered
        assert "cs" not in filtered

        mh = filtered["marketing_history"]
        assert "capacity_invested_per_turn" in mh
        assert "lag_turns" in mh
        assert mh["lag_turns"] == 10
        assert "leads_generated_this_turn" in mh
        assert "total_leads_generated" in mh
        assert "marketing_bonus_active" in mh
        assert "sales_momentum" in mh
        # Awareness keystone fields
        assert "awareness_by_feature" in mh
        assert "pending_awareness_summary" in mh
        assert "competitor_radar" in mh
        # Co-investment feedback fields
        assert "collab_received_this_turn" in mh
        assert "pipeline_progressions_this_turn" in mh

    def test_marketing_obs_surfaces_awareness_and_radar(self, engine: GameEngine):
        """Awareness stock, pending maturation schedule, and radar surface in the obs."""
        from alignsim.src.models.entities import PendingAwareness
        from alignsim.src.models.game_state import TurnRecord

        engine.state.awareness = {"F14": 3.456, "F02": 1.2}
        engine.state.pending_awareness = [
            PendingAwareness(land_turn=12, feature_id="F14", amount=0.5),
            PendingAwareness(land_turn=12, feature_id="F14", amount=0.25),  # same key → summed
            PendingAwareness(land_turn=15, feature_id="F02", amount=1.0),
        ]
        # Radar/co-investment events are read from the most-recently-resolved turn (state.turn - 1).
        engine.state.turn = 6
        engine.state.turn_history.append(TurnRecord(turn=5, events=[
            "competitor_radar:F14:soon", "inbound_lead:C09",
            "market_support:events:capacity=3:matched=2",
            "pipeline_progression:C09:lead->prospect",
        ]))

        obs = engine.get_initial_observation()
        mh = filter_observation(obs, "marketing", engine)["marketing_history"]

        assert mh["awareness_by_feature"] == {"F02": 1.2, "F14": 3.46}  # rounded
        # pending aggregated by (feature, land_turn), sorted by land_turn then feature
        assert mh["pending_awareness_summary"] == [
            {"feature": "F14", "matures_turn": 12, "amount": 0.75},
            {"feature": "F02", "matures_turn": 15, "amount": 1.0},
        ]
        assert mh["competitor_radar"] == ["F14:soon"]
        assert mh["collab_received_this_turn"] == [
            {"channel": "events", "sales_capacity": 3, "marketing_capacity": 2},
        ]
        assert mh["pipeline_progressions_this_turn"] == ["C09:lead->prospect"]

    def test_marketing_obs_never_exposes_hidden_customers(self, engine: GameEngine):
        """The marketing obs has no customer/pipeline section — only aggregate signals."""
        obs = engine.get_initial_observation()
        filtered = filter_observation(obs, "marketing", engine)
        assert "sales" not in filtered
        assert "cs" not in filtered
        # marketing_history is the only function section, and it carries no per-customer state
        mh = filtered["marketing_history"]
        assert "customers" not in mh
        assert "pipeline" not in mh

    def test_ops_sees_global_and_ops(self, engine: GameEngine):
        obs = engine.get_initial_observation()
        filtered = filter_observation(obs, "ops", engine)

        assert "global" in filtered
        assert "ops" in filtered
        assert "sales" not in filtered
        assert "product_eng" not in filtered
        assert "cs" not in filtered

    def test_global_section_has_capacity(self, engine: GameEngine):
        obs = engine.get_initial_observation()
        filtered = filter_observation(obs, "engineering", engine)
        g = filtered["global"]

        assert "turn" in g
        assert "mrr" in g
        assert "capacity" in g
        assert "engineering" in g["capacity"]
        assert "sales" in g["capacity"]

    def test_no_sections_leak_across_functions(self, engine: GameEngine):
        obs = engine.get_initial_observation()
        all_visible: dict[str, set[str]] = {}

        for fn in ALL_FUNCTIONS:
            filtered = filter_observation(obs, fn, engine)
            all_visible[fn] = set(filtered.keys())

        assert "sales" not in all_visible["engineering"]
        assert "product_eng" not in all_visible["sales"]
        assert "cs" not in all_visible["marketing"]
        assert "marketing_history" not in all_visible["engineering"]
        assert "ops" not in all_visible["sales"]


# ---------------------------------------------------------------------------
# Event filtering
# ---------------------------------------------------------------------------


class TestFilterEvents:
    SAMPLE_EVENTS = [
        "feature_shipped:F01:shipped_mvp",
        "feature_upgraded:F01:shipped_solid",
        "bug_fixed:BUG001:major:F01",
        "bug_injected:BUG002:critical:F01:C01,C02",
        "infrastructure_work:capacity=5",
        "deal_won:C01",
        "stage_advanced:C01:lead->prospect",
        "timeline_started:C01",
        "timeline_expired_reset:C01:resets=1",
        "deal_lost:C02:competitor_won:Nova pricing",
        "discovered:C09",
        "churn:C05",
        "expansion:C05:+500",
        "churn_intervention:C05:success",
        "emergent_need_injected:EN_001:C05:F06",
        "emergent_need_revealed:EN_002:C05:F07",
        "emergent_need_met:EN_003:C05:F08",
        "emergent_need_expired:EN_004:C05",
        "inbound_lead:C09",
        "awareness_built:F14",
        "competitor_radar:F14:soon",
        "pipeline_progression:C09:lead->prospect",
        "market_support:events:capacity=3:matched=3",
        "market_support_unmatched:content",
        "ops_project_started:PP01:Sales Process Optimization",
        "ops_project_completed:PP02:CI/CD:bonus=0.100:bug_rate_reduction",
        "ops_project_support:PP02:capacity=3:total=6",
        "hire_started:H1:engineering:arrives_in_6_turns:capacity_4:active_phase_3_turns",
        "hire_started:H2:engineering:cross_hire_from_sales:arrives_in_12_turns:capacity_3:active_phase_6_turns",
        "hire_sustained:H1:engineering:active_2/3",
        "hire_cancelled:H3:marketing:missed_sustain:was_1/3",
        "hire_arrived:H4:cs:+4_capacity",
        "fire:engineering:-4_capacity:severance_50000",
        "fire:cs:-4_capacity:severance_50000",
        "game_over:bankruptcy",
        "competitive:feature_launch:Nova launches integration",
    ]

    HIRE_OWNERS = {
        "H1": "engineering",
        "H2": "sales",
        "H3": "marketing",
        "H4": "cs",
    }

    def test_engineering_sees_own_events(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "engineering", self.HIRE_OWNERS)

        assert "feature_shipped:F01:shipped_mvp" in filtered
        assert "feature_upgraded:F01:shipped_solid" in filtered
        assert "bug_fixed:BUG001:major:F01" in filtered
        assert "bug_injected:BUG002:critical:F01:C01,C02" in filtered
        assert "infrastructure_work:capacity=5" in filtered

    def test_engineering_does_not_see_sales_events(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "engineering", self.HIRE_OWNERS)

        assert not any(e.startswith("deal_won") for e in filtered)
        assert not any(e.startswith("stage_advanced") for e in filtered)
        assert not any(e.startswith("discovered") for e in filtered)

    def test_sales_sees_own_events(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "sales", self.HIRE_OWNERS)

        assert "deal_won:C01" in filtered
        assert any(e.startswith("stage_advanced") for e in filtered)
        assert any(e.startswith("timeline_started") for e in filtered)
        assert any(e.startswith("deal_lost") for e in filtered)
        assert "discovered:C09" in filtered

    def test_support_sees_churn_and_expansion(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "support", self.HIRE_OWNERS)

        assert "churn:C05" in filtered
        assert "expansion:C05:+500" in filtered
        assert not any(e.startswith("deal_won") for e in filtered)
        assert not any(e.startswith("feature_shipped") for e in filtered)

    def test_support_sees_emergent_lifecycle_and_intervention(self):
        """Reveal/met/expired (post-discovery) and the CS-owned churn_intervention reach CS."""
        filtered = filter_events(self.SAMPLE_EVENTS, "support", self.HIRE_OWNERS)

        assert "emergent_need_revealed:EN_002:C05:F07" in filtered
        assert "emergent_need_met:EN_003:C05:F08" in filtered
        assert "emergent_need_expired:EN_004:C05" in filtered
        assert "churn_intervention:C05:success" in filtered

    def test_support_does_not_see_emergent_injected(self):
        """The injection event must NOT reach CS — that would bypass the health_check gate."""
        filtered = filter_events(self.SAMPLE_EVENTS, "support", self.HIRE_OWNERS)
        assert not any(e.startswith("emergent_need_injected") for e in filtered)

    def test_emergent_injected_filtered_for_every_role(self):
        """emergent_need_injected is hidden ground truth — invisible to ALL roles."""
        assert "emergent_need_injected" not in SHARED_EVENT_PREFIXES
        for fn in ALL_FUNCTIONS:
            assert fn != "support" or True  # support covered above; assert for all anyway
            filtered = filter_events(self.SAMPLE_EVENTS, fn, self.HIRE_OWNERS)
            assert not any(e.startswith("emergent_need_injected") for e in filtered), fn

    def test_non_support_roles_do_not_see_emergent_lifecycle(self):
        """Reveal/met/expired are CS-owned; no other role sees them."""
        for fn in ALL_FUNCTIONS - {"support"}:
            filtered = filter_events(self.SAMPLE_EVENTS, fn, self.HIRE_OWNERS)
            assert not any(e.startswith("emergent_need_") for e in filtered), fn
            assert not any(e.startswith("churn_intervention") for e in filtered), fn

    def test_marketing_sees_inbound_leads(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "marketing", self.HIRE_OWNERS)

        assert "inbound_lead:C09" in filtered
        assert not any(e.startswith("deal_won") for e in filtered)
        assert not any(e.startswith("churn") for e in filtered)

    def test_marketing_sees_radar_and_awareness(self):
        """Radar + awareness_built are marketing-only information sources."""
        filtered = filter_events(self.SAMPLE_EVENTS, "marketing", self.HIRE_OWNERS)
        assert "competitor_radar:F14:soon" in filtered
        assert "awareness_built:F14" in filtered

    def test_radar_and_awareness_not_shared(self):
        """Radar/awareness must NOT be in the shared prefixes (else they'd leak to all)."""
        assert "competitor_radar" not in SHARED_EVENT_PREFIXES
        assert "awareness_built" not in SHARED_EVENT_PREFIXES

    def test_radar_filtered_for_every_non_marketing_role(self):
        """No-leak: competitor_radar + awareness_built are invisible to every non-marketing role.

        Sales/Product see only the EFFECT (warm leads via engagement), never the radar or
        the awareness values themselves.
        """
        for fn in ALL_FUNCTIONS - {"marketing"}:
            filtered = filter_events(self.SAMPLE_EVENTS, fn, self.HIRE_OWNERS)
            assert not any(e.startswith("competitor_radar") for e in filtered), fn
            assert not any(e.startswith("awareness_built") for e in filtered), fn

    def test_sales_sees_progression_and_co_investment(self):
        """The Marketing<->Sales handshake reaches Sales (it's Sales' pipeline + its collab)."""
        filtered = filter_events(self.SAMPLE_EVENTS, "sales", self.HIRE_OWNERS)
        assert "pipeline_progression:C09:lead->prospect" in filtered
        assert "market_support:events:capacity=3:matched=3" in filtered
        assert "market_support_unmatched:content" in filtered

    def test_marketing_sees_progression_and_co_investment(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "marketing", self.HIRE_OWNERS)
        assert "pipeline_progression:C09:lead->prospect" in filtered
        assert "market_support:events:capacity=3:matched=3" in filtered
        assert "market_support_unmatched:content" in filtered

    def test_progression_and_co_investment_not_shared(self):
        for prefix in ("pipeline_progression", "market_support", "market_support_unmatched"):
            assert prefix not in SHARED_EVENT_PREFIXES

    def test_progression_co_investment_filtered_for_eng_cs_ops(self):
        """No-leak: pipeline_progression / market_support[_unmatched] hidden from eng/cs/ops."""
        for fn in ALL_FUNCTIONS - {"sales", "marketing"}:
            filtered = filter_events(self.SAMPLE_EVENTS, fn, self.HIRE_OWNERS)
            assert not any(e.startswith("pipeline_progression") for e in filtered), fn
            assert not any(e.startswith("market_support") for e in filtered), fn

    def test_ops_sees_project_events(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "ops", self.HIRE_OWNERS)

        assert any(e.startswith("ops_project_started") for e in filtered)
        assert any(e.startswith("ops_project_completed") for e in filtered)
        assert any(e.startswith("ops_project_support") for e in filtered)

    def test_all_see_game_over(self):
        for fn in ALL_FUNCTIONS:
            filtered = filter_events(self.SAMPLE_EVENTS, fn, self.HIRE_OWNERS)
            assert "game_over:bankruptcy" in filtered

    def test_all_see_competitive(self):
        for fn in ALL_FUNCTIONS:
            filtered = filter_events(self.SAMPLE_EVENTS, fn, self.HIRE_OWNERS)
            assert any(e.startswith("competitive:") for e in filtered)

    def test_hire_started_native_visible_to_owner(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "engineering", self.HIRE_OWNERS)
        assert any("hire_started:H1:" in e for e in filtered)

    def test_hire_started_cross_visible_to_hiring_function(self):
        # H2 is a cross-hire from sales → engineering
        filtered = filter_events(self.SAMPLE_EVENTS, "sales", self.HIRE_OWNERS)
        assert any("hire_started:H2:" in e for e in filtered)

    def test_hire_started_cross_not_visible_to_target(self):
        # H2 targets engineering but was initiated by sales
        filtered = filter_events(self.SAMPLE_EVENTS, "engineering", self.HIRE_OWNERS)
        assert not any("hire_started:H2:" in e for e in filtered)

    def test_hire_sustained_visible_to_owner(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "engineering", self.HIRE_OWNERS)
        assert any("hire_sustained:H1:" in e for e in filtered)

    def test_hire_sustained_not_visible_to_other(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "sales", self.HIRE_OWNERS)
        assert not any("hire_sustained:H1:" in e for e in filtered)

    def test_hire_cancelled_visible_to_owner(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "marketing", self.HIRE_OWNERS)
        assert any("hire_cancelled:H3:" in e for e in filtered)

    def test_hire_cancelled_not_visible_to_other(self):
        filtered = filter_events(self.SAMPLE_EVENTS, "engineering", self.HIRE_OWNERS)
        assert not any("hire_cancelled:H3:" in e for e in filtered)

    def test_hire_arrived_visible_to_owner(self):
        # H4 was initiated by cs (support agent)
        filtered = filter_events(self.SAMPLE_EVENTS, "support", self.HIRE_OWNERS)
        assert any("hire_arrived:H4:" in e for e in filtered)

    def test_fire_visible_only_to_fired_function(self):
        eng_events = filter_events(self.SAMPLE_EVENTS, "engineering", self.HIRE_OWNERS)
        assert any("fire:engineering:" in e for e in eng_events)
        assert not any("fire:cs:" in e for e in eng_events)

        support_events = filter_events(self.SAMPLE_EVENTS, "support", self.HIRE_OWNERS)
        assert any("fire:cs:" in e for e in support_events)
        assert not any("fire:engineering:" in e for e in support_events)

    def test_empty_events(self):
        for fn in ALL_FUNCTIONS:
            assert filter_events([], fn) == []


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------


class TestValidateFunctionActions:
    def test_engineering_core_actions_accepted(self, engine: GameEngine):
        actions = [
            {"action_type": "build", "feature_id": "F02", "quality": "mvp", "capacity": 5},
            {"action_type": "fix_bugs", "bug_id": None, "capacity": 3},
            {"action_type": "infrastructure", "capacity": 2},
        ]
        result = validate_function_actions(actions, "engineering", engine)

        assert len(result.valid_actions) == 3
        assert len(result.rejected_actions) == 0

    def test_non_dict_action_rejected_not_raised(self, engine: GameEngine):
        actions = [
            "build",
            {"action_type": "build", "feature_id": "F02", "quality": "mvp", "capacity": 5},
            42,
        ]
        result = validate_function_actions(actions, "engineering", engine)

        assert len(result.valid_actions) == 1
        assert len(result.rejected_actions) == 2
        assert "must be a JSON object" in result.rejected_actions[0]["reason"]
        assert "str" in result.rejected_actions[0]["reason"]
        assert "int" in result.rejected_actions[1]["reason"]

    def test_engineering_cannot_sell(self, engine: GameEngine):
        actions = [
            {"action_type": "sell", "customer_id": "C01", "sell_action": "outbound", "capacity": 3},
        ]
        result = validate_function_actions(actions, "engineering", engine)

        assert len(result.valid_actions) == 0
        assert len(result.rejected_actions) == 1
        assert "not allowed" in result.rejected_actions[0]["reason"]

    def test_sales_core_actions_accepted(self, engine: GameEngine):
        actions = [
            {"action_type": "sell", "customer_id": "C01", "sell_action": "outbound", "capacity": 3},
            {"action_type": "discover", "segment": None, "capacity": 2},
        ]
        result = validate_function_actions(actions, "sales", engine)

        assert len(result.valid_actions) == 2
        assert len(result.rejected_actions) == 0

    def test_sales_cannot_build(self, engine: GameEngine):
        actions = [
            {"action_type": "build", "feature_id": "F02", "quality": "mvp", "capacity": 5},
        ]
        result = validate_function_actions(actions, "sales", engine)

        assert len(result.rejected_actions) == 1

    def test_support_core_action(self, engine: GameEngine):
        actions = [
            {"action_type": "support", "customer_id": "C01", "support_action": "health_check", "capacity": 3},
        ]
        result = validate_function_actions(actions, "support", engine)
        assert len(result.valid_actions) == 1

    def test_marketing_core_action(self, engine: GameEngine):
        actions = [
            {"action_type": "market", "channel": "content", "capacity": 3},
        ]
        result = validate_function_actions(actions, "marketing", engine)
        assert len(result.valid_actions) == 1

    def test_ops_core_action(self, engine: GameEngine):
        actions = [
            {"action_type": "ops_project", "project_id": "PP01", "capacity": 4},
        ]
        result = validate_function_actions(actions, "ops", engine)
        assert len(result.valid_actions) == 1

    # -- Hire validation --

    def test_hire_from_own_pool_accepted(self, engine: GameEngine):
        actions = [
            {"action_type": "hire", "hiring_function": "engineering", "target_function": "engineering"},
        ]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.valid_actions) == 1

    def test_support_hire_uses_cs_engine_name(self, engine: GameEngine):
        actions = [
            {"action_type": "hire", "hiring_function": "cs", "target_function": "cs"},
        ]
        result = validate_function_actions(actions, "support", engine)
        assert len(result.valid_actions) == 1

    def test_hire_from_wrong_pool_rejected(self, engine: GameEngine):
        actions = [
            {"action_type": "hire", "hiring_function": "sales", "target_function": "engineering"},
        ]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.rejected_actions) == 1
        assert "engineering" in result.rejected_actions[0]["reason"]

    def test_cross_hire_accepted_if_from_own_pool(self, engine: GameEngine):
        actions = [
            {"action_type": "hire", "hiring_function": "sales", "target_function": "cs"},
        ]
        result = validate_function_actions(actions, "sales", engine)
        assert len(result.valid_actions) == 1

    # -- Sustain hire validation --

    def test_sustain_own_hire_accepted(self, engine: GameEngine):
        engine.state.pending_hires.append(PendingHire(
            id="H1", target_function="engineering", hiring_function="engineering",
            turns_remaining=5, active_turns_required=3, active_turns_completed=1,
        ))
        actions = [{"action_type": "sustain_hire", "hire_id": "H1"}]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.valid_actions) == 1

    def test_sustain_others_hire_rejected(self, engine: GameEngine):
        engine.state.pending_hires.append(PendingHire(
            id="H1", target_function="engineering", hiring_function="sales",
            turns_remaining=5, active_turns_required=3, active_turns_completed=1,
        ))
        actions = [{"action_type": "sustain_hire", "hire_id": "H1"}]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.rejected_actions) == 1
        assert "owned by sales" in result.rejected_actions[0]["reason"]

    def test_sustain_support_hire_by_support_agent(self, engine: GameEngine):
        engine.state.pending_hires.append(PendingHire(
            id="H2", target_function="cs", hiring_function="cs",
            turns_remaining=5, active_turns_required=3, active_turns_completed=1,
        ))
        actions = [{"action_type": "sustain_hire", "hire_id": "H2"}]
        result = validate_function_actions(actions, "support", engine)
        assert len(result.valid_actions) == 1

    def test_sustain_nonexistent_hire_passes_through(self, engine: GameEngine):
        actions = [{"action_type": "sustain_hire", "hire_id": "H99"}]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.valid_actions) == 1

    # -- Fire validation --

    def test_fire_own_function_accepted(self, engine: GameEngine):
        actions = [{"action_type": "fire", "function": "engineering"}]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.valid_actions) == 1

    def test_fire_other_function_rejected(self, engine: GameEngine):
        actions = [{"action_type": "fire", "function": "sales"}]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.rejected_actions) == 1

    def test_support_fires_cs(self, engine: GameEngine):
        actions = [{"action_type": "fire", "function": "cs"}]
        result = validate_function_actions(actions, "support", engine)
        assert len(result.valid_actions) == 1

    # -- ops_project_support validation --

    def test_ops_project_support_own_function_accepted(self, engine: GameEngine):
        # PP02 targets engineering
        actions = [{"action_type": "ops_project_support", "project_id": "PP02", "capacity": 3}]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.valid_actions) == 1

    def test_ops_project_support_wrong_function_rejected(self, engine: GameEngine):
        # PP02 targets engineering, not sales
        actions = [{"action_type": "ops_project_support", "project_id": "PP02", "capacity": 3}]
        result = validate_function_actions(actions, "sales", engine)
        assert len(result.rejected_actions) == 1
        assert "targets engineering" in result.rejected_actions[0]["reason"]

    def test_ops_project_support_for_support(self, engine: GameEngine):
        # PP03 targets support
        actions = [{"action_type": "ops_project_support", "project_id": "PP03", "capacity": 2}]
        result = validate_function_actions(actions, "support", engine)
        assert len(result.valid_actions) == 1

    def test_ops_project_support_sales_project(self, engine: GameEngine):
        # PP01 targets sales
        actions = [{"action_type": "ops_project_support", "project_id": "PP01", "capacity": 2}]
        result = validate_function_actions(actions, "sales", engine)
        assert len(result.valid_actions) == 1

    def test_ops_project_support_unknown_project_passes_through(self, engine: GameEngine):
        actions = [{"action_type": "ops_project_support", "project_id": "PP99", "capacity": 2}]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.valid_actions) == 1

    # -- market_support validation (sales-only co-investment) --

    def test_market_support_accepted_for_sales(self, engine: GameEngine):
        actions = [{"action_type": "market_support", "channel": "events", "capacity": 3}]
        result = validate_function_actions(actions, "sales", engine)
        assert len(result.valid_actions) == 1
        assert len(result.rejected_actions) == 0

    def test_market_support_rejected_for_marketing(self, engine: GameEngine):
        """Marketing runs the campaign; only Sales co-invests."""
        actions = [{"action_type": "market_support", "channel": "events", "capacity": 3}]
        result = validate_function_actions(actions, "marketing", engine)
        assert len(result.valid_actions) == 0
        assert len(result.rejected_actions) == 1
        assert "sales" in result.rejected_actions[0]["reason"].lower()

    def test_market_support_rejected_for_other_roles(self, engine: GameEngine):
        actions = [{"action_type": "market_support", "channel": "content", "capacity": 2}]
        for fn in ("engineering", "support", "ops"):
            result = validate_function_actions(actions, fn, engine)
            assert len(result.valid_actions) == 0, fn
            assert len(result.rejected_actions) == 1, fn

    # -- Unknown action --

    def test_unknown_action_type_rejected(self, engine: GameEngine):
        actions = [{"action_type": "teleport", "capacity": 99}]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.rejected_actions) == 1
        assert "not allowed" in result.rejected_actions[0]["reason"]

    # -- Mixed actions --

    def test_mixed_valid_and_invalid(self, engine: GameEngine):
        actions = [
            {"action_type": "build", "feature_id": "F02", "quality": "mvp", "capacity": 5},
            {"action_type": "sell", "customer_id": "C01", "sell_action": "outbound", "capacity": 3},
            {"action_type": "hire", "hiring_function": "engineering", "target_function": "engineering"},
        ]
        result = validate_function_actions(actions, "engineering", engine)
        assert len(result.valid_actions) == 2
        assert len(result.rejected_actions) == 1

    def test_empty_actions(self, engine: GameEngine):
        result = validate_function_actions([], "engineering", engine)
        assert len(result.valid_actions) == 0
        assert len(result.rejected_actions) == 0


# ---------------------------------------------------------------------------
# Query / compute permissions
# ---------------------------------------------------------------------------


class TestQueryPermissions:
    def test_engineering_queries(self):
        assert is_query_allowed("engineering", "feature")
        assert is_query_allowed("engineering", "bugs")
        assert is_query_allowed("engineering", "rejections")
        assert not is_query_allowed("engineering", "customer")

    def test_sales_queries(self):
        assert is_query_allowed("sales", "customer")
        assert is_query_allowed("sales", "rejections")
        assert not is_query_allowed("sales", "feature")
        assert not is_query_allowed("sales", "bugs")

    def test_support_queries(self):
        assert is_query_allowed("support", "customer")
        assert is_query_allowed("support", "rejections")
        assert not is_query_allowed("support", "feature")

    def test_marketing_queries(self):
        assert is_query_allowed("marketing", "rejections")
        assert not is_query_allowed("marketing", "customer")
        assert not is_query_allowed("marketing", "feature")
        assert not is_query_allowed("marketing", "bugs")

    def test_ops_queries(self):
        assert is_query_allowed("ops", "rejections")
        assert not is_query_allowed("ops", "customer")


class TestComputePermissions:
    def test_engineering_computes(self):
        assert is_compute_allowed("engineering", "maturity")
        assert is_compute_allowed("engineering", "maturity-if")
        assert is_compute_allowed("engineering", "capacity-cost")
        assert not is_compute_allowed("engineering", "satisfaction")

    def test_sales_computes(self):
        assert is_compute_allowed("sales", "satisfaction")
        assert is_compute_allowed("sales", "capacity-cost")
        assert not is_compute_allowed("sales", "maturity")

    def test_support_computes(self):
        assert is_compute_allowed("support", "satisfaction")
        assert not is_compute_allowed("support", "maturity")
        assert not is_compute_allowed("support", "capacity-cost")

    def test_marketing_no_computes(self):
        assert not is_compute_allowed("marketing", "maturity")
        assert not is_compute_allowed("marketing", "satisfaction")
        assert not is_compute_allowed("marketing", "capacity-cost")

    def test_ops_no_computes(self):
        assert not is_compute_allowed("ops", "maturity")
        assert not is_compute_allowed("ops", "satisfaction")


# ---------------------------------------------------------------------------
# Hire event parsing (regression tests for event format coupling)
# ---------------------------------------------------------------------------


class TestParseHireStartedOwner:
    def test_native_hire(self):
        event = "hire_started:H1:engineering:arrives_in_6_turns:capacity_4:active_phase_3_turns"
        assert _parse_hire_started_owner(event) == "engineering"

    def test_cross_hire(self):
        event = "hire_started:H2:engineering:cross_hire_from_sales:arrives_in_12_turns:capacity_3:active_phase_6_turns"
        assert _parse_hire_started_owner(event) == "sales"

    def test_native_cs_hire(self):
        event = "hire_started:H3:cs:arrives_in_6_turns:capacity_4:active_phase_3_turns"
        assert _parse_hire_started_owner(event) == "cs"

    def test_cross_hire_to_cs(self):
        event = "hire_started:H4:cs:cross_hire_from_sales:arrives_in_12_turns:capacity_3:active_phase_6_turns"
        assert _parse_hire_started_owner(event) == "sales"

    def test_malformed_short_event(self):
        assert _parse_hire_started_owner("hire_started:H1") is None


# ---------------------------------------------------------------------------
# Ops cross-functional analysis handshake (Stage B)
# ---------------------------------------------------------------------------


class TestAnalysisResultDelivery:
    """The analysis result reaches the REQUESTER only — never another role, never Ops."""

    def _seed_pending(self, engine: GameEngine) -> None:
        engine.state.pending_analyses = {
            "sales": [{"analysis_type": "conversion_funnel", "target_function": "sales"}],
            "support": [{"analysis_type": "retention_efficiency", "target_function": "cs"}],
        }

    def test_requester_receives_its_own_result(self, engine: GameEngine):
        obs = engine.get_initial_observation()
        self._seed_pending(engine)
        sales = filter_observation(obs, "sales", engine)
        assert sales["analyses_received_this_turn"] == [
            {"analysis_type": "conversion_funnel", "target_function": "sales"}
        ]
        support = filter_observation(obs, "support", engine)
        assert support["analyses_received_this_turn"][0]["target_function"] == "cs"

    def test_no_other_role_sees_a_result(self, engine: GameEngine):
        obs = engine.get_initial_observation()
        self._seed_pending(engine)
        # sales' result must not surface for anyone but sales; cs' not for anyone but support.
        for fn in ALL_FUNCTIONS - {"sales"}:
            got = filter_observation(obs, fn, engine)["analyses_received_this_turn"]
            assert all(a.get("target_function") != "sales" for a in got), fn
        for fn in ALL_FUNCTIONS - {"support"}:
            got = filter_observation(obs, fn, engine)["analyses_received_this_turn"]
            assert all(a.get("target_function") != "cs" for a in got), fn

    def test_ops_never_receives_a_payload(self, engine: GameEngine):
        """Ops is the provider — it is never a requester, so its delivery slice is always empty."""
        obs = engine.get_initial_observation()
        self._seed_pending(engine)
        ops = filter_observation(obs, "ops", engine)
        assert ops["analyses_received_this_turn"] == []

    def test_empty_when_nothing_pending(self, engine: GameEngine):
        obs = engine.get_initial_observation()
        for fn in ALL_FUNCTIONS:
            assert filter_observation(obs, fn, engine)["analyses_received_this_turn"] == []


class TestAnalysisSubmissionFilter:
    def test_only_ops_may_submit_ops_analysis(self, engine: GameEngine):
        action = {"action_type": "ops_analysis", "target_function": "sales",
                  "analysis_type": "conversion_funnel", "capacity": 4}
        assert len(validate_function_actions([action], "ops", engine).valid_actions) == 1
        for fn in ALL_FUNCTIONS - {"ops"}:
            res = validate_function_actions([action], fn, engine)
            assert len(res.rejected_actions) == 1, fn
            assert "ops_analysis" in res.rejected_actions[0]["reason"]

    def test_team_may_scope_only_its_own_function(self, engine: GameEngine):
        a_sales = {"action_type": "analysis_scope", "target_function": "sales",
                   "analysis_type": "conversion_funnel", "capacity": 1}
        a_cs = {"action_type": "analysis_scope", "target_function": "cs",
                "analysis_type": "retention_efficiency", "capacity": 1}
        # Sales scopes for itself → ok; for cs → rejected.
        assert len(validate_function_actions([a_sales], "sales", engine).valid_actions) == 1
        rej = validate_function_actions([a_cs], "sales", engine)
        assert len(rej.rejected_actions) == 1
        assert "own function" in rej.rejected_actions[0]["reason"]
        # The support agent (engine fn cs) is the legitimate owner of analysis_scope{cs}.
        assert len(validate_function_actions([a_cs], "support", engine).valid_actions) == 1

    def test_ops_cannot_scope(self, engine: GameEngine):
        """Ops provides analyses; it does not scope (its engine fn is never a tf it owns here)."""
        a_sales = {"action_type": "analysis_scope", "target_function": "sales",
                   "analysis_type": "conversion_funnel", "capacity": 1}
        res = validate_function_actions([a_sales], "ops", engine)
        assert len(res.rejected_actions) == 1


class TestAnalysisEventRouting:
    EVENTS = [
        "ops_analysis:sales:conversion_funnel",
        "ops_analysis:cs:retention_efficiency",
        "analysis_unmatched:marketing:awareness_attribution",
    ]

    def test_ops_sees_all_analysis_events(self):
        assert set(filter_events(self.EVENTS, "ops")) == set(self.EVENTS)

    def test_each_requester_sees_only_its_own(self):
        assert filter_events(self.EVENTS, "sales") == ["ops_analysis:sales:conversion_funnel"]
        assert filter_events(self.EVENTS, "support") == ["ops_analysis:cs:retention_efficiency"]
        assert filter_events(self.EVENTS, "marketing") == ["analysis_unmatched:marketing:awareness_attribution"]

    def test_non_participant_sees_no_analysis_events(self):
        # Engineering is not a target in any of these events.
        assert filter_events(self.EVENTS, "engineering") == []

    def test_analysis_prefixes_are_tf_gated_not_broadcast(self):
        """They live in SHARED_EVENT_PREFIXES but are tf-gated, so they never broadcast."""
        assert "ops_analysis" in SHARED_EVENT_PREFIXES
        assert "analysis_unmatched" in SHARED_EVENT_PREFIXES
        # No role outside {ops, sales} sees the sales event.
        for fn in ALL_FUNCTIONS - {"ops", "sales"}:
            assert "ops_analysis:sales:conversion_funnel" not in filter_events(self.EVENTS, fn), fn


def test_analyses_received_is_never_an_obs_section():
    """Load-bearing no-leak invariant for analysis-result delivery.

    filter_observation copies each role's obs_sections verbatim from the shared (god-view)
    TurnObservation, then UNCONDITIONALLY overwrites analyses_received_this_turn with that role's
    own slice of pending_analyses. The whole defense rests on 'analyses_received_this_turn' NOT
    being an obs_section: if it ever were, the god-view (every function's analyses) would be copied
    into a role's view before the per-function overwrite — leaking all functions' results. Pin it
    shut so a future refactor that adds the key to an obs_sections set fails here.
    """
    for fn, perms in FUNCTION_PERMISSIONS.items():
        assert "analyses_received_this_turn" not in perms.obs_sections, (
            f"{fn}.obs_sections must never contain 'analyses_received_this_turn' (god-view leak)"
        )
