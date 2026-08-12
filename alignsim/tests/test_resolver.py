"""Tests for alignsim.src.engine.resolver — turn resolution against full game state.

Each test creates a fresh GameState (via factories), runs a single TurnResolver.resolve()
with hand-crafted actions, and asserts on state mutations and emitted events.
"""

import math
import random

import pytest

from alignsim.src.engine.ops_logic import compute_effective_bonus
from alignsim.src.engine.resolver import TurnResolver
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
)
from alignsim.src.models.entities import (
    BugSeverity,
    CompetitorEvent,
    CustomerRubric,
    CustomerStage,
    Engagement,
    FeatureStatus,
    ProcessProjectStatus,
    QualityLevel,
    Segment,
)
from alignsim.src.models.scenario import CalibrationParams

from .factories import (
    make_active_bonus,
    make_bug,
    make_competitor,
    make_competitor_event,
    make_customer,
    make_emergent_need,
    make_feature,
    make_game_state,
    make_generator_config,
    make_pending_awareness,
    make_process_project,
    make_resource_pool,
    make_turn_record,
)


def _resolve(state, actions, calibration=None, seed=1, generator_config=None, features_dict=None):
    """Resolve actions against state. Default seed=1 produces a low first random
    (~0.13), which pairs with high-probability scenarios so deterministic asserts
    on advancement work."""
    cal = calibration or CalibrationParams()
    rng = random.Random(seed)
    resolver = TurnResolver(state, cal, rng, generator_config=generator_config, features_dict=features_dict)
    return resolver.resolve(actions)


# --- Engineering: build ---

def test_build_progress_updates_feature():
    feature = make_feature(id="F01", cost={"mvp": 20})
    state = make_game_state(features=[feature], resources=make_resource_pool(eng_capacity=20))
    actions = [BuildAction(feature_id="F01", quality=QualityLevel.mvp, capacity=10)]
    _resolve(state, actions)
    f = state.features["F01"]
    assert f.progress > 0
    assert f.status == FeatureStatus.in_progress


def test_build_ships_feature_emits_event():
    """A feature that completes is shipped, emits feature_shipped event, turns_worked resets."""
    cal = CalibrationParams(build_min_turns_factor=0.0)
    # In-progress feature near completion. progress=35 + 65% cap of remaining
    # = 35 + 65 = 100 → ships this turn. current_target=mvp prevents reset.
    feature = make_feature(
        id="F01", cost={"mvp": 8}, turns_worked=5, progress=35.0,
        status=FeatureStatus.in_progress, current_target=QualityLevel.mvp,
    )
    state = make_game_state(features=[feature], resources=make_resource_pool(eng_capacity=20))
    actions = [BuildAction(feature_id="F01", quality=QualityLevel.mvp, capacity=8)]
    record = _resolve(state, actions, calibration=cal)
    f = state.features["F01"]
    assert f.status == FeatureStatus.shipped_mvp
    assert any(e.startswith("feature_shipped:F01") for e in record.events)
    assert f.turns_worked == 0


# --- Engineering: bugs ---

def test_fix_bug_targeted():
    bug = make_bug(id="BUG_001", severity=BugSeverity.major)
    state = make_game_state(bugs=[bug], resources=make_resource_pool(eng_capacity=10))
    actions = [FixBugsAction(bug_id="BUG_001", capacity=2)]
    record = _resolve(state, actions)
    assert state.bugs[0].is_resolved
    assert record.bugs_fixed == 1


def test_fix_bug_auto_target_priority():
    """Auto-target prioritizes critical, then major, then minor."""
    bugs = [
        make_bug(id="MIN", severity=BugSeverity.minor),
        make_bug(id="CRIT", severity=BugSeverity.critical),
        make_bug(id="MAJ", severity=BugSeverity.major),
    ]
    state = make_game_state(bugs=bugs, resources=make_resource_pool(eng_capacity=20))
    # 7 capacity: critical(4) → critical fixed; remaining 3 ≥ major(2) → major fixed; remaining 1 ≥ minor(1) → minor fixed
    record = _resolve(state, [FixBugsAction(bug_id=None, capacity=7)])
    by_id = {b.id: b for b in state.bugs}
    assert by_id["CRIT"].is_resolved
    assert by_id["MAJ"].is_resolved
    assert by_id["MIN"].is_resolved
    assert record.bugs_fixed == 3


def test_tech_debt_changes_from_build_and_infra():
    feature = make_feature(id="F01", cost={"mvp": 100})
    state = make_game_state(features=[feature], resources=make_resource_pool(eng_capacity=20))
    actions = [
        BuildAction(feature_id="F01", quality=QualityLevel.mvp, capacity=10),
        InfrastructureAction(capacity=5),
    ]
    _resolve(state, actions)
    # +1.0 from 10 mvp, -1.0 from 5 infra → net 0.0
    assert state.tech_debt.level == pytest.approx(0.0)


# --- Sales: pipeline advancement ---

def test_sell_advances_pipeline():
    """High satisfaction + 100% base rate guarantees advance."""
    cal = CalibrationParams(lead_to_prospect_rate=1.0)
    # Maximize satisfaction so probability stays at 1.0:
    # rubric weighted entirely on price; price = 0.5 + size*0.1 = 1.0 at size=5.
    customer = make_customer(
        id="C1", stage=CustomerStage.lead, is_visible=True,
        size=5, engagement=Engagement.warm,
        rubric=CustomerRubric(feature_coverage=0.0, price=1.0, maturity=0.0, support=0.0),
    )
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=20))
    # Min outbound capacity for size 5 is 5
    actions = [SellAction(customer_id="C1", sell_action="outbound", capacity=5)]
    record = _resolve(state, actions, calibration=cal)
    assert state.customers["C1"].stage == CustomerStage.prospect
    assert any(e.startswith("stage_advanced:C1") for e in record.events)


def test_sell_activates_timeline_on_first_action():
    customer = make_customer(id="C1", stage=CustomerStage.lead, is_visible=True,
                             size=1, timeline=10, timeline_active=False)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    record = _resolve(state, [SellAction(customer_id="C1", sell_action="outbound", capacity=1)])
    c = state.customers["C1"]
    # Timeline activated then ticks once → ends turn at original-1
    assert c.timeline == 9
    assert any(e.startswith("timeline_started:C1") for e in record.events)


def test_sell_timeline_expiry_resets_and_increments_resets():
    """When timeline reaches 0, customer goes back to lead and resets count up."""
    customer = make_customer(id="C1", stage=CustomerStage.qualified, is_visible=True,
                             size=1, timeline=1, timeline_active=True,
                             timeline_original=8, timeline_resets=0)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    cal = CalibrationParams(qualified_to_in_deal_rate=0.0)  # don't advance
    record = _resolve(state, [SellAction(customer_id="C1", sell_action="demo", capacity=1)],
                      calibration=cal)
    c = state.customers["C1"]
    assert c.stage == CustomerStage.lead
    assert c.timeline_resets == 1
    assert c.timeline == 8  # reset to original
    assert c.timeline_active is False
    assert c.engagement == Engagement.cold
    assert any("timeline_expired_reset:C1" in e for e in record.events)


def test_sell_close_increases_mrr():
    """A closed deal increases MRR by deal_value."""
    cal = CalibrationParams(in_deal_to_closed_rate=1.0, min_rubric_for_close=0.0)
    # Maximize satisfaction with size=5 + price-only rubric so prob stays at 1.0
    customer = make_customer(
        id="C1", stage=CustomerStage.in_deal, is_visible=True, size=5,
        engagement=Engagement.warm, deal_value=5_000, timeline=8,
        rubric=CustomerRubric(feature_coverage=0.0, price=1.0, maturity=0.0, support=0.0),
    )
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=20, mrr=0))
    # Min negotiate capacity for size 5 is 5
    record = _resolve(state, [SellAction(customer_id="C1", sell_action="negotiate", capacity=5)],
                      calibration=cal)
    assert state.customers["C1"].stage == CustomerStage.customer
    assert state.resources.mrr == 5_000
    assert any(e.startswith("deal_won:C1") for e in record.events)


def test_sell_engagement_updates_for_visible_pipeline():
    """All visible non-customer entities get engagement updated based on capacity per customer."""
    cal = CalibrationParams()
    c1 = make_customer(id="C1", stage=CustomerStage.lead, is_visible=True,
                       size=1, engagement=Engagement.cold)
    c2 = make_customer(id="C2", stage=CustomerStage.prospect, is_visible=True,
                       size=1, engagement=Engagement.hot)
    state = make_game_state(customers=[c1, c2], resources=make_resource_pool(sales_capacity=10))
    actions = [SellAction(customer_id="C1", sell_action="outbound", capacity=2)]
    _resolve(state, actions, calibration=cal)
    # C1 received attention → hot. C2 received none → decay hot→warm.
    assert state.customers["C1"].engagement == Engagement.hot
    assert state.customers["C2"].engagement == Engagement.warm


# --- CS: health, churn, expansion ---

def test_cs_attention_updates_health():
    cal = CalibrationParams()
    customer = make_customer(id="C1", stage=CustomerStage.customer, health=5.0)
    state = make_game_state(customers=[customer], resources=make_resource_pool(support_capacity=5))
    _resolve(state, [SupportAction(customer_id="C1", support_action="health_check", capacity=2)],
             calibration=cal)
    # Log curve at cap 2: 1.0 * (1 + 0.8*ln2) ≈ 1.5545; +0.1 regression-toward-7 (health<7).
    attention = cal.health_cs_attention_delta * (1.0 + cal.cs_attention_log_factor * math.log(2))
    assert state.customers["C1"].health == pytest.approx(5.0 + attention + 0.1)


def test_cs_churn_cascade_removes_mrr():
    """Customer below churn threshold for required turns → churns, MRR drops by deal_value."""
    cal = CalibrationParams()
    customer = make_customer(id="C1", stage=CustomerStage.customer,
                             health=2.0, deal_value=5_000,
                             turns_below_churn_threshold=2)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(support_capacity=5, mrr=5_000))
    record = _resolve(state, [], calibration=cal)
    assert state.customers["C1"].stage == CustomerStage.churned
    assert state.resources.mrr == 0
    assert record.churn_count == 1


def test_cs_expansion_increases_deal_value():
    """Customer with sustained high health gets expansion bump."""
    cal = CalibrationParams()
    customer = make_customer(id="C1", stage=CustomerStage.customer,
                             health=9.0, deal_value=10_000,
                             turns_above_expansion_threshold=4)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(support_capacity=5, mrr=10_000))
    _resolve(state, [], calibration=cal)
    # 20% increase = +2000
    assert state.customers["C1"].deal_value == 12_000
    assert state.resources.mrr == 12_000


# --- Marketing ---

def test_marketing_logs_investment():
    """Marketing capacity is appended to marketing_history."""
    state = make_game_state(resources=make_resource_pool(marketing_capacity=5))
    _resolve(state, [MarketAction(channel="content", capacity=4)])
    assert state.marketing_history == [4]


def test_marketing_lagged_inbound_reveals_hidden_customer():
    """After enough lagged investment, a hidden customer is revealed as an inbound lead."""
    cal = CalibrationParams(marketing_lag_turns=2, base_inbound_rate=1.0,
                            marketing_effectiveness=2.0)
    hidden = make_customer(id="H1", is_visible=False)
    state = make_game_state(customers=[hidden],
                            resources=make_resource_pool(marketing_capacity=10))
    # Pre-seed: one prior turn of investment; this turn appends 0 → history = [5, 0],
    # then lag=2 looks at history[-2] = 5 → leads = 1.0 + 5*2.0 = 11.
    state.marketing_history = [5]
    record = _resolve(state, [], calibration=cal)
    assert state.customers["H1"].is_visible
    assert any(e.startswith("inbound_lead:H1") for e in record.events)


# --- Marketing: awareness accrual / channels / decay ---

def test_market_action_schedules_pending_awareness_events_burst():
    """events channel (lag 2, spread 1) schedules a single burst at turn+lag; not yet matured."""
    cal = CalibrationParams()
    state = make_game_state(
        features=[make_feature(id="F14", status=FeatureStatus.in_progress)],
        resources=make_resource_pool(marketing_capacity=5), turn=10,
    )
    _resolve(state, [MarketAction(channel="events", target_features=["F14"], capacity=3)],
             calibration=cal)
    assert len(state.pending_awareness) == 1
    assert state.pending_awareness[0].land_turn == 12  # turn 10 + lag 2
    assert state.pending_awareness[0].feature_id == "F14"
    assert "F14" not in state.awareness  # nothing matured this turn


def test_market_action_content_spreads_pending_awareness():
    """content channel (lag 8, spread 6) spreads increments over 6 consecutive turns."""
    cal = CalibrationParams()
    state = make_game_state(
        features=[make_feature(id="F02", status=FeatureStatus.shipped_mvp)],
        resources=make_resource_pool(marketing_capacity=10), turn=1,
    )
    _resolve(state, [MarketAction(channel="content", target_features=["F02"], capacity=6)],
             calibration=cal)
    assert len(state.pending_awareness) == 6
    assert sorted(p.land_turn for p in state.pending_awareness) == [9, 10, 11, 12, 13, 14]


def test_pending_awareness_matures_then_decays():
    """An increment landing this turn matures into the stock, then decays the same turn."""
    cal = CalibrationParams(awareness_decay=0.10)
    state = make_game_state(
        turn=5,
        pending_awareness=[make_pending_awareness(land_turn=5, feature_id="F01", amount=2.0)],
        resources=make_resource_pool(marketing_capacity=5),
    )
    record = _resolve(state, [], calibration=cal)
    assert state.awareness["F01"] == pytest.approx(1.8)  # 2.0 matured, then *0.9
    assert state.pending_awareness == []
    assert any(e == "awareness_built:F01" for e in record.events)


def test_marketing_budget_deducts_for_events_not_outbound():
    """events/content spend shared runway budget; outbound is capacity-only (free)."""
    cal = CalibrationParams()
    feature = make_feature(id="F01", status=FeatureStatus.shipped_mvp)

    def run(channel: str) -> tuple[int, list[str]]:
        state = make_game_state(
            features=[make_feature(id="F01", status=FeatureStatus.shipped_mvp)],
            resources=make_resource_pool(marketing_capacity=5, budget=1_000_000, mrr=0),
        )
        record = _resolve(
            state, [MarketAction(channel=channel, target_features=["F01"], capacity=3)],
            calibration=cal,
        )
        return state.resources.budget, record.events

    budget_events, events_evts = run("events")
    budget_outbound, outbound_evts = run("outbound_campaign")

    # events costs 3 * 8000 = 24000 more than outbound (which is free).
    assert budget_outbound - budget_events == 24000
    assert "marketing_spend:24000" in events_evts
    assert not any(e.startswith("marketing_spend:") for e in outbound_evts)


def test_turn_spend_fully_accounted_and_runway_excludes_one_time():
    """Conservation: the whole budget delta for a turn == recurring cost + one-time spend (with
    zero revenue), nothing unaccounted. Runway then projects recurring burn ONLY — the one-time
    marketing spend hits cash but not the per-turn denominator (regression for the phantom cliff).

    One-time spend is summed generically from the channel profiles, so the test is robust to
    cost tuning and doesn't hardcode dollar amounts.
    """
    cal = CalibrationParams()
    state = make_game_state(
        features=[make_feature(id="F01", status=FeatureStatus.shipped_solid)],
        resources=make_resource_pool(marketing_capacity=10, budget=1_000_000, mrr=0),
    )
    budget_before = state.resources.budget

    market_actions = [
        MarketAction(channel="events", target_features=["F01"], capacity=2),
        MarketAction(channel="content", target_features=["F01"], capacity=3),
        MarketAction(channel="outbound_campaign", target_features=["F01"], capacity=1),
    ]
    _resolve(state, market_actions, calibration=cal)

    # Recompute the two cost components from the same inputs the resolver uses.
    one_time = sum(
        a.capacity * cal.channel_profiles[a.channel].budget_cost_per_capacity
        for a in market_actions
    )
    recurring = (
        state.resources.capacity_per_turn * cal.team_cost_per_capacity
        + state.resources.base_cost_per_turn
        + sum(
            f.maintenance_cost for f in state.features.values()
            if f.status in (
                FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished
            )
        )
    )

    # Every dollar of budget change is accounted for — no leak, no double-count.
    assert budget_before - state.resources.budget == recurring + one_time
    # Runway uses the recurring denominator only (one-time spend excluded).
    assert state.resources.runway_turns == pytest.approx(state.resources.budget / recurring)


# --- Marketing/Discovery: reveal-state modulation from awareness ---

def test_inbound_reveal_applies_warm_engagement_and_timeline():
    """A hidden customer needing a high-awareness feature arrives warm + with longer timeline."""
    cal = CalibrationParams(
        marketing_lag_turns=2, base_inbound_rate=1.0, marketing_effectiveness=2.0,
        awareness_warm_threshold=1.5, awareness_hot_threshold=4.0, awareness_hot_prob=0.0,
        awareness_decay=0.0, awareness_timeline_bonus_max=6,
    )
    hidden = make_customer(
        id="H1", is_visible=False, engagement=Engagement.cold,
        timeline=10, timeline_original=10, feature_needs={"F01": {"mvp": 0.5}},
    )
    state = make_game_state(customers=[hidden], awareness={"F01": 4.0},
                            resources=make_resource_pool(marketing_capacity=10))
    state.marketing_history = [5]
    _resolve(state, [], calibration=cal)
    h = state.customers["H1"]
    assert h.is_visible
    assert h.engagement == Engagement.warm  # hot_prob 0 → warm not hot
    assert h.timeline == 16  # +round(6 * min(1, 4.0/4.0)) = +6
    assert h.timeline_original == 16


def test_discovery_reveal_applies_awareness_engagement():
    """Discovered customers also pick up reveal-state from awareness (uniform at reveal)."""
    cal = CalibrationParams(
        awareness_warm_threshold=1.5, awareness_hot_threshold=4.0,
        awareness_hot_prob=0.0, awareness_decay=0.0,
    )
    hidden = make_customer(
        id="H1", is_visible=False, discovery_difficulty=0.1, engagement=Engagement.cold,
        feature_needs={"F01": {"mvp": 0.6}},
    )
    shipped = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[hidden], features=[shipped], awareness={"F01": 3.0},
                            resources=make_resource_pool(sales_capacity=10))
    _resolve(state, [DiscoverAction(capacity=5, target_features=["F01"])], calibration=cal)
    h = state.customers["H1"]
    assert h.is_visible
    assert h.engagement == Engagement.warm


def test_low_awareness_reveal_stays_cold():
    cal = CalibrationParams(
        marketing_lag_turns=2, base_inbound_rate=1.0, marketing_effectiveness=2.0,
        awareness_warm_threshold=1.5, awareness_decay=0.0,
    )
    hidden = make_customer(id="H1", is_visible=False, engagement=Engagement.cold,
                           feature_needs={"F01": {"mvp": 0.5}})
    state = make_game_state(customers=[hidden], awareness={"F01": 0.5},  # below warm threshold
                            resources=make_resource_pool(marketing_capacity=10))
    state.marketing_history = [5]
    _resolve(state, [], calibration=cal)
    assert state.customers["H1"].engagement == Engagement.cold


# --- Marketing: competitive radar ---

def test_radar_emits_event_within_lookahead_with_awareness():
    cal = CalibrationParams(
        radar_lookahead_turns=5, radar_base_prob=1.0, radar_uncertainty_jitter=0.0,
        awareness_hot_threshold=4.0, awareness_decay=0.0,
    )
    affected = make_customer(id="C50", is_visible=False, feature_needs={"F14": {"mvp": 0.5}})
    comp = make_competitor(events=[make_competitor_event(turn=4, affected_customers=["C50"])])
    state = make_game_state(customers=[affected], competitors=[comp], awareness={"F14": 4.0},
                            resources=make_resource_pool(marketing_capacity=5), turn=1)
    record = _resolve(state, [], calibration=cal)
    assert any(e.startswith("competitor_radar:F14:") for e in record.events)


def test_radar_silent_without_awareness():
    cal = CalibrationParams(
        radar_lookahead_turns=5, radar_base_prob=1.0, radar_uncertainty_jitter=0.0,
    )
    affected = make_customer(id="C50", is_visible=False, feature_needs={"F14": {"mvp": 0.5}})
    comp = make_competitor(events=[make_competitor_event(turn=4, affected_customers=["C50"])])
    state = make_game_state(customers=[affected], competitors=[comp],
                            resources=make_resource_pool(marketing_capacity=5), turn=1)
    record = _resolve(state, [], calibration=cal)
    assert not any(e.startswith("competitor_radar:") for e in record.events)


def test_radar_silent_outside_lookahead():
    cal = CalibrationParams(
        radar_lookahead_turns=5, radar_base_prob=1.0, radar_uncertainty_jitter=0.0,
        awareness_decay=0.0,
    )
    affected = make_customer(id="C50", is_visible=False, feature_needs={"F14": {"mvp": 0.5}})
    comp = make_competitor(events=[make_competitor_event(turn=20, affected_customers=["C50"])])
    state = make_game_state(customers=[affected], competitors=[comp], awareness={"F14": 4.0},
                            resources=make_resource_pool(marketing_capacity=5), turn=1)
    record = _resolve(state, [], calibration=cal)
    assert not any(e.startswith("competitor_radar:") for e in record.events)


# --- Marketing<->Sales pipeline progression (market_support co-investment) ---

def _progression_certain_cal(**kw):
    """Calibration where a matched roll always succeeds (p clamped to 1.0)."""
    base = dict(
        marketing_lag_turns=2, base_inbound_rate=1.0, marketing_effectiveness=2.0,
        awareness_decay=0.0,
        progression_base_prob={"content": 1.0, "events": 1.0},
        progression_max_prob=1.0,
    )
    base.update(kw)
    return CalibrationParams(**base)


def test_market_support_events_progresses_revealed_inbound():
    cal = _progression_certain_cal()
    hidden = make_customer(id="H1", is_visible=False, stage=CustomerStage.lead,
                           feature_needs={"F01": {"mvp": 0.5}})
    feat = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[hidden], features=[feat],
                            resources=make_resource_pool(marketing_capacity=10, sales_capacity=10))
    state.marketing_history = [5]  # drives inbound this turn
    actions = [
        MarketAction(channel="events", target_features=["F01"], capacity=3),
        MarketSupportAction(channel="events", capacity=3),
    ]
    record = _resolve(state, actions, calibration=cal)
    h = state.customers["H1"]
    assert h.is_visible
    assert h.stage == CustomerStage.prospect  # lead -> prospect (one stage)
    assert any(e == "pipeline_progression:H1:lead->prospect" for e in record.events)
    assert any(e.startswith("market_support:events:capacity=3:matched=3") for e in record.events)
    # Progression must NOT activate the timeline clock (avoids instant-expiry trap).
    assert h.timeline_active is False


def test_market_support_events_advances_named_existing_customer():
    cal = _progression_certain_cal(base_inbound_rate=0.0)  # no inbound, isolate existing-customer path
    existing = make_customer(id="C5", is_visible=True, stage=CustomerStage.prospect,
                             feature_needs={"F01": {"mvp": 0.5}})
    feat = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[existing], features=[feat],
                            resources=make_resource_pool(marketing_capacity=10, sales_capacity=10))
    actions = [
        MarketAction(channel="events", target_features=["F01"], capacity=3),
        MarketSupportAction(channel="events", capacity=3, target_customer_id="C5"),
    ]
    record = _resolve(state, actions, calibration=cal)
    assert state.customers["C5"].stage == CustomerStage.qualified  # prospect -> qualified
    assert any(e == "pipeline_progression:C5:prospect->qualified" for e in record.events)


def test_market_support_content_progresses_inbound_only():
    """content has no existing-customer push; it only advances newly-revealed inbound."""
    cal = _progression_certain_cal()
    hidden = make_customer(id="H1", is_visible=False, stage=CustomerStage.lead,
                           feature_needs={"F01": {"mvp": 0.5}})
    feat = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[hidden], features=[feat],
                            resources=make_resource_pool(marketing_capacity=10, sales_capacity=10))
    state.marketing_history = [5]
    actions = [
        MarketAction(channel="content", target_features=["F01"], capacity=3),
        MarketSupportAction(channel="content", capacity=3),
    ]
    record = _resolve(state, actions, calibration=cal)
    assert state.customers["H1"].stage == CustomerStage.prospect
    assert any(e.startswith("market_support:content:") for e in record.events)


def test_market_support_unmatched_wastes_sales_capacity():
    cal = _progression_certain_cal(base_inbound_rate=0.0)
    existing = make_customer(id="C5", is_visible=True, stage=CustomerStage.prospect)
    state = make_game_state(customers=[existing],
                            resources=make_resource_pool(marketing_capacity=10, sales_capacity=10))
    # content collab but NO matching content market action this turn → unmatched, wasted.
    record = _resolve(state, [MarketSupportAction(channel="content", capacity=4)], calibration=cal)
    assert any(e == "market_support_unmatched:content" for e in record.events)
    assert state.customers["C5"].stage == CustomerStage.prospect  # unchanged
    assert not any(e.startswith("pipeline_progression") for e in record.events)
    assert record.sales_capacity_used == 4  # capacity consumed despite the no-op


def test_market_support_progression_capped_at_in_deal():
    cal = _progression_certain_cal(base_inbound_rate=0.0)
    existing = make_customer(id="C5", is_visible=True, stage=CustomerStage.in_deal,
                             feature_needs={"F01": {"mvp": 0.5}})
    feat = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[existing], features=[feat],
                            resources=make_resource_pool(marketing_capacity=10, sales_capacity=10))
    actions = [
        MarketAction(channel="events", target_features=["F01"], capacity=3),
        MarketSupportAction(channel="events", capacity=3, target_customer_id="C5"),
    ]
    record = _resolve(state, actions, calibration=cal)
    assert state.customers["C5"].stage == CustomerStage.in_deal  # never advances past in_deal
    assert not any(e.startswith("pipeline_progression") for e in record.events)


def test_market_without_support_does_not_progress():
    """Solo budget campaign (no Sales co-investment) reveals but never progresses."""
    cal = _progression_certain_cal()
    hidden = make_customer(id="H1", is_visible=False, stage=CustomerStage.lead,
                           feature_needs={"F01": {"mvp": 0.5}})
    feat = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[hidden], features=[feat],
                            resources=make_resource_pool(marketing_capacity=10, sales_capacity=10))
    state.marketing_history = [5]
    record = _resolve(state, [MarketAction(channel="events", target_features=["F01"], capacity=3)],
                      calibration=cal)
    assert state.customers["H1"].is_visible
    assert state.customers["H1"].stage == CustomerStage.lead  # revealed, not progressed
    assert not any(e.startswith("pipeline_progression") for e in record.events)
    assert not any(e.startswith("market_support") for e in record.events)


def test_market_support_prefers_events_when_both_funded():
    """With both channels funded+matched, new-lead progression uses the events channel."""
    cal = _progression_certain_cal()
    hidden = make_customer(id="H1", is_visible=False, stage=CustomerStage.lead,
                           feature_needs={"F01": {"mvp": 0.5}})
    feat = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[hidden], features=[feat],
                            resources=make_resource_pool(marketing_capacity=20, sales_capacity=20))
    state.marketing_history = [5]
    actions = [
        MarketAction(channel="events", target_features=["F01"], capacity=3),
        MarketAction(channel="content", target_features=["F01"], capacity=3),
        MarketSupportAction(channel="events", capacity=3),
        MarketSupportAction(channel="content", capacity=3),
    ]
    record = _resolve(state, actions, calibration=cal)
    assert state.customers["H1"].stage == CustomerStage.prospect
    # both matched channels echo their handshake
    assert any(e.startswith("market_support:events:") for e in record.events)
    assert any(e.startswith("market_support:content:") for e in record.events)


# --- Discovery ---

def test_discovery_reveals_hidden_customer():
    cal = CalibrationParams()
    hidden = make_customer(id="H1", is_visible=False, discovery_difficulty=0.1,
                           segment=Segment.startup,
                           feature_needs={"F01": {"mvp": 0.6}})
    shipped = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[hidden], features=[shipped],
                            resources=make_resource_pool(sales_capacity=10))
    record = _resolve(state, [DiscoverAction(capacity=5, target_features=["F01"])],
                      calibration=cal)
    assert state.customers["H1"].is_visible
    assert any(e.startswith("discovered:H1") for e in record.events)


# --- Ops ---

def test_ops_project_lifecycle():
    """Project starts → progresses → completes → adds bonus."""
    project = make_process_project(
        id="PP01", ops_capacity_cost=2, duration_turns=1,
        bonus_base=0.10, bonus_scale_factor=0.0, bonus_max=0.10,  # deterministic narrow range
        target_function="sales", bonus_type="conversion_rate",
    )
    state = make_game_state(process_projects=[project],
                            resources=make_resource_pool(ops_capacity=5))
    record = _resolve(state, [OpsProjectAction(project_id="PP01", capacity=2)])
    p = state.process_projects["PP01"]
    assert p.status == ProcessProjectStatus.completed
    assert p.completed_turn == 1
    assert any(e.startswith("ops_project_started:PP01") for e in record.events)
    assert any(e.startswith("ops_project_completed:PP01") for e in record.events)
    assert len(state.active_process_bonuses) == 1


def test_ops_bonus_tick_and_expiry():
    """Each turn ticks down active bonuses; expired ones are removed."""
    bonus = make_active_bonus(project_id="PP01", turns_remaining=1, bonus_duration_turns=12)
    state = make_game_state(resources=make_resource_pool(ops_capacity=5))
    state.active_process_bonuses = [bonus]
    _resolve(state, [])
    # turns_remaining was 1 → after tick → removed
    assert state.active_process_bonuses == []


def test_ops_maintenance_refresh_resets_bonus():
    """Maintenance action on a completed project resets bonus to full duration."""
    project = make_process_project(id="PP01", ops_capacity_cost=4,
                                   status=ProcessProjectStatus.completed,
                                   bonus_duration_turns=12)
    bonus = make_active_bonus(project_id="PP01", turns_remaining=4,
                              bonus_duration_turns=12,
                              original_ops_capacity_cost=4)
    state = make_game_state(process_projects=[project],
                            resources=make_resource_pool(ops_capacity=10))
    state.active_process_bonuses = [bonus]
    record = _resolve(state, [OpsProjectAction(project_id="PP01", capacity=3)])
    # After refresh: turns_remaining set to bonus_duration_turns + 1, then tick removes 1 → 12
    assert state.active_process_bonuses[0].turns_remaining == 12
    assert any(e.startswith("ops_project_refresh:PP01") for e in record.events)


def _make_floored_project(**overrides):
    """A 1-turn project with a deterministic peak bonus of 0.20 and a 25% permanent floor."""
    defaults = dict(
        id="PP01", ops_capacity_cost=2, duration_turns=1,
        bonus_base=0.20, bonus_scale_factor=0.0, bonus_max=0.20,  # deterministic peak = 0.20
        bonus_duration_turns=4, permanent_floor_fraction=0.25,
        target_function="sales", bonus_type="conversion_rate",
    )
    defaults.update(overrides)
    return make_process_project(**defaults)


def test_ops_floored_bonus_persists_past_duration():
    """A floored project's bonus survives past bonus_duration_turns, pinned at the floor."""
    project = _make_floored_project()
    state = make_game_state(process_projects=[project],
                            resources=make_resource_pool(ops_capacity=5, budget=10_000_000))
    cal = CalibrationParams()
    _resolve(state, [OpsProjectAction(project_id="PP01", capacity=2)], calibration=cal)
    assert state.process_projects["PP01"].status == ProcessProjectStatus.completed
    assert state.active_process_bonuses[0].permanent_floor_fraction == pytest.approx(0.25)

    # Tick well past bonus_duration_turns (4): the spike fully decays but the floor persists.
    for _ in range(8):
        _resolve(state, [], calibration=cal)

    assert len(state.active_process_bonuses) == 1
    persisted = state.active_process_bonuses[0]
    assert persisted.turns_remaining == 0
    assert compute_effective_bonus(persisted) == pytest.approx(0.05)  # floor = 0.20 * 0.25


def test_floored_bonus_adds_no_financial_burn():
    """Key finding: a permanently-floored bonus is genuinely free — no per-turn cost.

    Two states identical except for the presence of a floored bonus burn identically.
    """
    cal = CalibrationParams()
    floored = make_active_bonus(project_id="PP01", bonus_value=0.20,
                                permanent_floor_fraction=0.25,
                                turns_remaining=0, bonus_duration_turns=4)
    with_bonus = make_game_state(resources=make_resource_pool(ops_capacity=5))
    with_bonus.active_process_bonuses = [floored]
    without_bonus = make_game_state(resources=make_resource_pool(ops_capacity=5))

    _resolve(with_bonus, [], calibration=cal)
    _resolve(without_bonus, [], calibration=cal)

    assert with_bonus.resources.budget == without_bonus.resources.budget
    # The floored bonus is untouched by the financial path and survives the tick at 0.
    assert len(with_bonus.active_process_bonuses) == 1
    assert with_bonus.active_process_bonuses[0].turns_remaining == 0


def test_permanent_floor_scale_kill_switch():
    """permanent_floor_scale=0.0 turns floors OFF — the bonus reverts to pure decay / removal."""
    project = _make_floored_project()
    state = make_game_state(process_projects=[project],
                            resources=make_resource_pool(ops_capacity=5, budget=10_000_000))
    cal = CalibrationParams(permanent_floor_scale=0.0)
    _resolve(state, [OpsProjectAction(project_id="PP01", capacity=2)], calibration=cal)
    assert state.active_process_bonuses[0].permanent_floor_fraction == 0.0

    # With no floor, the spike decays to zero and the bonus is removed (as pre-floor behaviour).
    for _ in range(8):
        _resolve(state, [], calibration=cal)
    assert state.active_process_bonuses == []


def test_permanent_floor_scale_doubles_stored_fraction():
    """permanent_floor_scale=2.0 doubles the stored fraction at completion (calibration knob)."""
    project = _make_floored_project()
    state = make_game_state(process_projects=[project],
                            resources=make_resource_pool(ops_capacity=5, budget=10_000_000))
    cal = CalibrationParams(permanent_floor_scale=2.0)
    _resolve(state, [OpsProjectAction(project_id="PP01", capacity=2)], calibration=cal)
    bonus = state.active_process_bonuses[0]
    assert bonus.permanent_floor_fraction == pytest.approx(0.50)  # 0.25 * 2.0
    # Effective floor doubles too: peak 0.20 * 0.50 = 0.10.
    for _ in range(8):
        _resolve(state, [], calibration=cal)
    assert compute_effective_bonus(state.active_process_bonuses[0]) == pytest.approx(0.10)


# --- Ops cross-functional analysis handshake (Stage B) ---

def _state_for_analysis():
    """A state with one turn of observable history + a couple of customers to poison."""
    history = [
        make_turn_record(
            turn=1,
            events=["inbound_lead:C1", "stage_advanced:C1:lead->prospect"],
            sales_capacity_used=4, eng_capacity_used=18,
        ),
    ]
    customers = [
        make_customer(id="C1", is_visible=True, stage=CustomerStage.prospect, deal_value=5000),
        make_customer(id="C2", is_visible=True, stage=CustomerStage.customer, deal_value=4000),
    ]
    state = make_game_state(
        customers=customers,
        resources=make_resource_pool(ops_capacity=10, sales_capacity=10),
        turn=2,
    )
    state.turn_history = history
    state.churn_history = [0]
    state.marketing_history = [3]
    return state


def test_analysis_matched_populates_pending_and_debits_both_pools():
    state = _state_for_analysis()
    actions = [
        OpsAnalysisAction(target_function="sales", analysis_type="conversion_funnel", capacity=4),
        AnalysisScopeAction(target_function="sales", analysis_type="conversion_funnel", capacity=1),
    ]
    rec = _resolve(state, actions)
    assert state.pending_analyses["sales"][0]["analysis_type"] == "conversion_funnel"
    assert state.pending_analyses["sales"][0]["target_function"] == "sales"
    assert "ops_analysis:sales:conversion_funnel" in rec.events
    assert not any(e.startswith("analysis_unmatched") for e in rec.events)
    assert rec.ops_capacity_used == 4       # ops side draws from ops pool
    assert rec.sales_capacity_used == 1     # scope side draws from the requester's pool


def test_analysis_cs_target_routes_to_support_pool_and_key():
    """target_function 'cs' routes to the 'support' pool and the 'support' pending key."""
    state = _state_for_analysis()
    state.resources.support_capacity = 10
    actions = [
        OpsAnalysisAction(target_function="cs", analysis_type="retention_efficiency", capacity=4),
        AnalysisScopeAction(target_function="cs", analysis_type="retention_efficiency", capacity=1),
    ]
    rec = _resolve(state, actions)
    assert "support" in state.pending_analyses
    assert state.pending_analyses["support"][0]["target_function"] == "cs"
    assert rec.support_capacity_used == 1
    assert rec.ops_capacity_used == 4


def test_analysis_unmatched_ops_side_alone():
    state = _state_for_analysis()
    rec = _resolve(state, [OpsAnalysisAction(target_function="sales", analysis_type="conversion_funnel", capacity=4)])
    assert "analysis_unmatched:sales:conversion_funnel" in rec.events
    assert not any(e.startswith("ops_analysis:") for e in rec.events)
    assert state.pending_analyses == {}
    assert rec.ops_capacity_used == 4  # capacity wasted but still debited


def test_analysis_unmatched_scope_side_alone():
    state = _state_for_analysis()
    rec = _resolve(state, [AnalysisScopeAction(target_function="sales", analysis_type="conversion_funnel", capacity=1)])
    assert "analysis_unmatched:sales:conversion_funnel" in rec.events
    assert state.pending_analyses == {}
    assert rec.sales_capacity_used == 1


def test_analysis_unmatched_mismatched_type():
    """Same target_function but different analysis_type → both sides wasted, no result."""
    state = _state_for_analysis()
    actions = [
        OpsAnalysisAction(target_function="sales", analysis_type="conversion_funnel", capacity=4),
        AnalysisScopeAction(target_function="sales", analysis_type="capacity_bottleneck", capacity=1),
    ]
    rec = _resolve(state, actions)
    assert "analysis_unmatched:sales:conversion_funnel" in rec.events
    assert "analysis_unmatched:sales:capacity_bottleneck" in rec.events
    assert not any(e.startswith("ops_analysis:") for e in rec.events)
    assert state.pending_analyses == {}


def test_analysis_resolution_is_seed_independent():
    """No RNG in analysis → identical output regardless of resolver seed."""
    actions = [
        OpsAnalysisAction(target_function="sales", analysis_type="conversion_funnel", capacity=4),
        AnalysisScopeAction(target_function="sales", analysis_type="conversion_funnel", capacity=1),
    ]
    s1 = _state_for_analysis()
    _resolve(s1, actions, seed=1)
    s2 = _state_for_analysis()
    _resolve(s2, actions, seed=98765)
    assert s1.pending_analyses == s2.pending_analyses


def test_analysis_invariant_to_hidden_fields_via_resolver():
    """Poisoning hidden customer fields does not change the stashed analysis."""
    actions = [
        OpsAnalysisAction(target_function="sales", analysis_type="conversion_funnel", capacity=4),
        AnalysisScopeAction(target_function="sales", analysis_type="conversion_funnel", capacity=1),
    ]
    base = _state_for_analysis()
    _resolve(base, actions)

    poisoned = _state_for_analysis()
    for c in poisoned.customers.values():
        c.desired_price_point = 999_999
        c.close_threshold = 0.99
        c.churn_drivers = {"SENTINEL": 1.0}
    poisoned.emergent_needs.append(
        make_emergent_need(id="EN_SENT", customer_id="C1", feature_id="FZ", is_revealed=False)
    )
    _resolve(poisoned, actions)

    assert base.pending_analyses == poisoned.pending_analyses


def test_pending_analyses_cleared_at_resolve_top():
    """Last turn's stashed results are cleared at the top of resolve() (1-turn buffer)."""
    state = _state_for_analysis()
    state.pending_analyses = {"sales": [{"stale": True}]}
    _resolve(state, [])  # empty turn — nothing new stashed
    assert state.pending_analyses == {}


# --- Financial ---

def test_financial_budget_update():
    """Budget = previous + (revenue - costs)."""
    cal = CalibrationParams(team_cost_per_capacity=1_000)
    state = make_game_state(resources=make_resource_pool(
        capacity_per_turn=10, mrr=20_000, budget=100_000, base_cost_per_turn=2_000,
        eng_capacity=10, sales_capacity=0, support_capacity=0, marketing_capacity=0,
    ))
    _resolve(state, [], calibration=cal)
    # team cost = 10 * 1000 = 10K. base = 2K. total = 12K. revenue = 20K. net = +8K.
    assert state.resources.budget == 108_000


def test_financial_bankruptcy_ends_game():
    """Negative budget triggers game over with reason 'bankruptcy'."""
    cal = CalibrationParams(team_cost_per_capacity=1_000)
    state = make_game_state(resources=make_resource_pool(
        capacity_per_turn=20, mrr=0, budget=5_000, base_cost_per_turn=10_000,
        eng_capacity=20, sales_capacity=0, support_capacity=0, marketing_capacity=0,
    ))
    record = _resolve(state, [], calibration=cal)
    # team cost = 20K, base = 10K, total = 30K, revenue = 0. budget: 5K - 30K = -25K.
    assert state.resources.budget < 0
    assert state.game_over is True
    assert state.game_over_reason == "bankruptcy"
    assert any("game_over:bankruptcy" in e for e in record.events)


# --- Metrics ---

def test_metrics_records_churn_history():
    """churn_count is appended to churn_history each turn."""
    cal = CalibrationParams()
    customer = make_customer(id="C1", stage=CustomerStage.customer,
                             health=2.0, deal_value=1_000,
                             turns_below_churn_threshold=2)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(support_capacity=5, mrr=1_000))
    _resolve(state, [], calibration=cal)
    assert state.churn_history == [1]


def test_metrics_max_turns_ends_game():
    """When current turn >= max_turns, game ends with 'max_turns_reached'."""
    state = make_game_state(turn=48, max_turns=48)
    _resolve(state, [])
    assert state.game_over is True
    assert state.game_over_reason == "max_turns_reached"


# --- Competitive events ---

def test_competitive_event_applies_pressure():
    competitor = make_competitor(id="X", events=[
        make_competitor_event(turn=1, affected_customers=["C1"],
                              rubric_impact={"price": 0.5, "feature_coverage": 0.5}),
    ])
    customer = make_customer(id="C1", competitive_pressure=0.0)
    state = make_game_state(customers=[customer], competitors=[competitor])
    state.turn = 1
    _resolve(state, [])
    # impact avg = 0.5 → +0.5*0.3=0.15 then -0.05 decay = 0.10
    assert state.customers["C1"].competitive_pressure == pytest.approx(0.10)


def test_sell_deactivates_timeline_when_customer_won():
    """When a deal is won, the customer's timeline should be deactivated."""
    cal = CalibrationParams(in_deal_to_closed_rate=1.0, min_rubric_for_close=0.0)
    customer = make_customer(
        id="C1", stage=CustomerStage.in_deal, is_visible=True, size=5,
        engagement=Engagement.warm, deal_value=5_000, timeline=10,
        timeline_active=True, timeline_original=10,
        rubric=CustomerRubric(feature_coverage=0.0, price=1.0, maturity=0.0, support=0.0),
    )
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=20, mrr=0))
    # Min negotiate capacity for size 5 is 5
    record = _resolve(state, [SellAction(customer_id="C1", sell_action="negotiate", capacity=5)],
                      calibration=cal)
    c = state.customers["C1"]
    assert c.stage == CustomerStage.customer
    assert c.timeline_active is False
    assert any(e.startswith("deal_won:C1") for e in record.events)


# --- Generator integration ---

def test_discovery_with_generator_adds_customers():
    """With generator_config, discovery adds generated customers to state."""
    config = make_generator_config()
    shipped = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    features_dict = {"F01": shipped}
    state = make_game_state(features=[shipped],
                            resources=make_resource_pool(sales_capacity=10))
    initial_count = len(state.customers)
    record = _resolve(
        state,
        [DiscoverAction(target_features=["F01"], capacity=5)],
        generator_config=config,
        features_dict=features_dict,
    )
    assert len(state.customers) > initial_count
    gen_ids = [cid for cid in state.customers if cid.startswith("G")]
    assert len(gen_ids) > 0
    assert any(e.startswith("discovered:G") for e in record.events)


def test_marketing_inbound_with_generator():
    """Marketing generates inbound candidates when hidden pool is empty."""
    config = make_generator_config()
    shipped = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    features_dict = {"F01": shipped}
    cal = CalibrationParams(marketing_lag_turns=2, base_inbound_rate=1.0,
                            marketing_effectiveness=2.0)
    state = make_game_state(features=[shipped],
                            resources=make_resource_pool(marketing_capacity=10))
    state.marketing_history = [5]
    initial_count = len(state.customers)
    record = _resolve(
        state, [],
        calibration=cal,
        generator_config=config,
        features_dict=features_dict,
    )
    assert len(state.customers) > initial_count
    assert any(e.startswith("inbound_lead:G") for e in record.events)


def test_no_generator_config_backward_compat_resolver():
    """Without generator_config, resolver behaves identically to pre-change."""
    cal = CalibrationParams()
    hidden = make_customer(id="H1", is_visible=False, discovery_difficulty=0.1,
                           feature_needs={"F01": {"mvp": 0.6}})
    shipped = make_feature(id="F01", status=FeatureStatus.shipped_mvp)
    state = make_game_state(customers=[hidden], features=[shipped],
                            resources=make_resource_pool(sales_capacity=10))
    record = _resolve(state, [DiscoverAction(target_features=["F01"], capacity=5)],
                      calibration=cal)
    assert state.customers["H1"].is_visible
    gen_ids = [cid for cid in state.customers if cid.startswith("G")]
    assert len(gen_ids) == 0


# --- Pricing negotiation ---

def _make_in_deal_customer(**overrides):
    """Helper for pricing tests: in_deal, visible, high-satisfaction customer."""
    defaults = dict(
        id="C1", stage=CustomerStage.in_deal, is_visible=True, size=1,
        engagement=Engagement.warm, deal_value=1000, timeline=20,
        desired_price_point=800,
        rubric=CustomerRubric(feature_coverage=0.0, price=1.0, maturity=0.0, support=0.0),
    )
    defaults.update(overrides)
    return make_customer(**defaults)


def test_proposal_sets_last_proposed_price():
    cal = CalibrationParams(in_deal_to_closed_rate=0.0, min_rubric_for_close=0.0)
    customer = _make_in_deal_customer()
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    _resolve(state, [SellAction(customer_id="C1", sell_action="proposal",
                                capacity=1, proposed_deal_value=900)], calibration=cal)
    assert state.customers["C1"].last_proposed_price == 900


def test_proposal_sets_has_received_proposal():
    cal = CalibrationParams(in_deal_to_closed_rate=0.0, min_rubric_for_close=0.0)
    customer = _make_in_deal_customer(has_received_proposal=False)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    _resolve(state, [SellAction(customer_id="C1", sell_action="proposal",
                                capacity=1, proposed_deal_value=900)], calibration=cal)
    assert state.customers["C1"].has_received_proposal is True


def test_proposal_failure_emits_pricing_feedback():
    """Failed proposal where price > desired emits pricing_feedback event."""
    cal = CalibrationParams(in_deal_to_closed_rate=0.0, min_rubric_for_close=0.0)
    customer = _make_in_deal_customer(desired_price_point=800)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    record = _resolve(state, [SellAction(customer_id="C1", sell_action="proposal",
                                         capacity=1, proposed_deal_value=1200)], calibration=cal)
    feedback_events = [e for e in record.events if e.startswith("pricing_feedback:")]
    assert len(feedback_events) == 1
    assert "proposed=1200" in feedback_events[0]
    assert "indicated=" in feedback_events[0]


def test_proposal_failure_no_feedback_when_price_below_desired():
    """No pricing feedback when proposed price is at or below desired."""
    cal = CalibrationParams(in_deal_to_closed_rate=0.0, min_rubric_for_close=0.0)
    customer = _make_in_deal_customer(desired_price_point=800)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    record = _resolve(state, [SellAction(customer_id="C1", sell_action="proposal",
                                         capacity=1, proposed_deal_value=700)], calibration=cal)
    feedback_events = [e for e in record.events if e.startswith("pricing_feedback:")]
    assert len(feedback_events) == 0


def test_negotiate_with_different_price():
    """Negotiate updates last_proposed_price to the new value."""
    cal = CalibrationParams(in_deal_to_closed_rate=0.0, min_rubric_for_close=0.0)
    customer = _make_in_deal_customer(has_received_proposal=True, last_proposed_price=900)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    _resolve(state, [SellAction(customer_id="C1", sell_action="negotiate",
                                capacity=1, proposed_deal_value=750)], calibration=cal)
    assert state.customers["C1"].last_proposed_price == 750


def test_deal_close_uses_proposed_price_as_mrr():
    """Closed deal MRR = proposed price, not sticker price."""
    cal = CalibrationParams(in_deal_to_closed_rate=1.0, min_rubric_for_close=0.0)
    customer = _make_in_deal_customer(size=5, deal_value=5000, desired_price_point=4000)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=20, mrr=0))
    _resolve(state, [SellAction(customer_id="C1", sell_action="proposal",
                                capacity=5, proposed_deal_value=3500)], calibration=cal)
    assert state.resources.mrr == 3500


def test_deal_close_updates_deal_value():
    """Customer's deal_value is updated to the closing price."""
    cal = CalibrationParams(in_deal_to_closed_rate=1.0, min_rubric_for_close=0.0)
    customer = _make_in_deal_customer(size=5, deal_value=5000, desired_price_point=4000)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=20, mrr=0))
    _resolve(state, [SellAction(customer_id="C1", sell_action="proposal",
                                capacity=5, proposed_deal_value=3500)], calibration=cal)
    assert state.customers["C1"].deal_value == 3500


def test_proposal_without_price_defaults_to_sticker():
    """Omitting proposed_deal_value defaults to customer.deal_value."""
    cal = CalibrationParams(in_deal_to_closed_rate=0.0, min_rubric_for_close=0.0)
    customer = _make_in_deal_customer(deal_value=2000)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    _resolve(state, [SellAction(customer_id="C1", sell_action="proposal",
                                capacity=1)], calibration=cal)
    assert state.customers["C1"].last_proposed_price == 2000


def test_no_pricing_when_desired_is_zero():
    """When desired_price_point=0, pricing modifier is not applied."""
    cal = CalibrationParams(in_deal_to_closed_rate=0.0, min_rubric_for_close=0.0)
    customer = _make_in_deal_customer(desired_price_point=0, deal_value=1000)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    # Propose an absurdly high price — should have no penalty since desired=0
    record = _resolve(state, [SellAction(customer_id="C1", sell_action="proposal",
                                         capacity=1, proposed_deal_value=99999)], calibration=cal)
    feedback = [e for e in record.events if e.startswith("pricing_feedback:")]
    assert len(feedback) == 0


# --- Competitor pricing events ---

def test_competitor_pricing_event_fires_on_eligible():
    """Competitor pricing events fire on in_deal customers with proposals."""
    cal = CalibrationParams(pricing_competitor_event_lambda=5.0)
    customer = _make_in_deal_customer(has_received_proposal=True, desired_price_point=800)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    record = _resolve(state, [], calibration=cal)
    pricing_events = [e for e in record.events
                      if e.startswith("competitor_pricing:") or e.startswith("deal_lost:")]
    assert len(pricing_events) > 0


def test_competitor_pricing_event_skips_no_proposal():
    """No competitor pricing events when no customers have proposals."""
    cal = CalibrationParams(pricing_competitor_event_lambda=5.0)
    customer = _make_in_deal_customer(has_received_proposal=False, desired_price_point=800)
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    record = _resolve(state, [], calibration=cal)
    pricing_events = [e for e in record.events if e.startswith("competitor_pricing:")]
    assert len(pricing_events) == 0


def test_competitor_pricing_can_steal_deal():
    """With high lambda, competitor can steal a deal (customer → lost)."""
    cal = CalibrationParams(
        pricing_competitor_event_lambda=10.0,
        pricing_competitor_assumed_satisfaction=1.0,
        in_deal_to_closed_rate=1.0,
    )
    customer = _make_in_deal_customer(
        has_received_proposal=True, desired_price_point=800,
        engagement=Engagement.hot,
    )
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    # Run many times to get at least one steal
    stolen = False
    for seed in range(100):
        state_copy = make_game_state(
            customers=[_make_in_deal_customer(
                has_received_proposal=True, desired_price_point=800,
                engagement=Engagement.hot,
            )],
            resources=make_resource_pool(sales_capacity=10),
        )
        record = _resolve(state_copy, [], calibration=cal, seed=seed)
        if any(e.startswith("deal_lost:") and "competitor_pricing" in e for e in record.events):
            stolen = True
            break
    assert stolen, "Expected competitor to steal at least one deal in 100 tries"


def test_competitor_pricing_boosts_pressure_on_miss():
    """When competitor doesn't steal, competitive_pressure increases."""
    cal = CalibrationParams(
        pricing_competitor_event_lambda=5.0,
        pricing_competitor_assumed_satisfaction=0.01,
        in_deal_to_closed_rate=0.01,
    )
    customer = _make_in_deal_customer(
        has_received_proposal=True, desired_price_point=800,
        competitive_pressure=0.0, engagement=Engagement.cold,
    )
    state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
    _resolve(state, [], calibration=cal, seed=1)
    # With lambda=5 and very low conversion, at least one event should fire and boost pressure
    assert state.customers["C1"].competitive_pressure > 0.0


def test_competitor_offer_has_jitter():
    """Same customer, different seeds → different competitor offers."""
    cal = CalibrationParams(pricing_competitor_event_lambda=5.0,
                            pricing_competitor_assumed_satisfaction=0.0,
                            in_deal_to_closed_rate=0.0)
    offers = set()
    for seed in range(50):
        customer = _make_in_deal_customer(has_received_proposal=True, desired_price_point=800)
        state = make_game_state(customers=[customer], resources=make_resource_pool(sales_capacity=10))
        record = _resolve(state, [], calibration=cal, seed=seed)
        for e in record.events:
            if e.startswith("competitor_pricing:") and "offer=" in e:
                offer_str = e.split("offer=")[1]
                offers.add(int(offer_str))
    assert len(offers) > 1, "Expected jitter in competitor offers"


# --- Emergent needs: lifecycle, verbs, discovery gate, no-leak ---

def _emergent_cal(**overrides):
    """Calibration with injection OFF by default so lifecycle can be tested in isolation."""
    base = dict(emergent_need_injection_rate=0.0, emergent_need_injection_floor=0.0)
    base.update(overrides)
    return CalibrationParams(**base)


def _active_customer_with_needs(**overrides):
    defaults = dict(
        id="C1", stage=CustomerStage.customer, is_visible=True,
        health=8.0, known_needs=["F01"], deal_value=4_000,
    )
    defaults.update(overrides)
    return make_customer(**defaults)


def test_emergent_need_injection_runs_after_cs():
    """REGRESSION GUARD: injection (step 8b) runs strictly AFTER _resolve_cs (step 3), so a
    need injected on turn T cannot be revealed by a same-turn health_check. Do not reorder."""
    cal = _emergent_cal(emergent_need_injection_floor=5.0)  # force several injections
    customer = _active_customer_with_needs()
    features = [
        make_feature(id="F01", status=FeatureStatus.shipped_mvp),
        make_feature(id="F02", status=FeatureStatus.not_started),
    ]
    state = make_game_state(customers=[customer], features=features,
                            resources=make_resource_pool(support_capacity=5))
    record = _resolve(
        state,
        [SupportAction(customer_id="C1", support_action="health_check", capacity=2)],
        calibration=cal, seed=1,
    )
    # Needs were injected this turn...
    assert any(e.startswith("emergent_need_injected:") for e in record.events)
    assert len(state.emergent_needs) > 0
    # ...but none are revealed, because they did not exist when the health_check resolved.
    assert all(not n.is_revealed for n in state.emergent_needs)
    assert not any(e.startswith("emergent_need_revealed:") for e in record.events)


def test_emergent_need_turns_unmet_ticks_past_grace():
    cal = _emergent_cal()  # grace=3
    customer = _active_customer_with_needs()
    features = [make_feature(id="F02", status=FeatureStatus.not_started)]
    # injected turn 1, now turn 5 → age 4 >= grace, not built, not met → ticks.
    need = make_emergent_need(id="EN_001", customer_id="C1", feature_id="F02",
                              turn_injected=1, turns_unmet=0)
    state = make_game_state(customers=[customer], features=features,
                            emergent_needs=[need], turn=5)
    _resolve(state, [], calibration=cal)
    assert state.emergent_needs[0].turns_unmet == 1


def test_emergent_need_no_tick_within_grace():
    cal = _emergent_cal()  # grace=3
    customer = _active_customer_with_needs()
    features = [make_feature(id="F02", status=FeatureStatus.not_started)]
    # injected turn 4, now turn 5 → age 1 < grace → no tick.
    need = make_emergent_need(customer_id="C1", feature_id="F02", turn_injected=4, turns_unmet=0)
    state = make_game_state(customers=[customer], features=features,
                            emergent_needs=[need], turn=5)
    _resolve(state, [], calibration=cal)
    assert state.emergent_needs[0].turns_unmet == 0


def test_emergent_need_pause_when_feature_built():
    """While Eng allocates build capacity to the need's feature, the clock halts (and so does
    the bleed — see compute_health_delta tests)."""
    cal = _emergent_cal()
    customer = _active_customer_with_needs()
    features = [make_feature(id="F02", status=FeatureStatus.not_started,
                             cost={"mvp": 30, "solid": 50, "polished": 80})]
    need = make_emergent_need(customer_id="C1", feature_id="F02",
                              turn_injected=1, turns_unmet=2)
    state = make_game_state(customers=[customer], features=features,
                            emergent_needs=[need], turn=8,
                            resources=make_resource_pool(eng_capacity=10))
    _resolve(state, [BuildAction(feature_id="F02", quality=QualityLevel.mvp, capacity=5)],
             calibration=cal)
    # Built this turn (large feature won't ship in one turn) → clock paused, no tick, not met.
    assert state.features["F02"].status == FeatureStatus.in_progress
    assert state.emergent_needs[0].turns_unmet == 2
    assert not state.emergent_needs[0].is_met


def test_emergent_need_met_when_feature_ships():
    cal = _emergent_cal()
    customer = _active_customer_with_needs(health=6.0)
    features = [make_feature(id="F02", status=FeatureStatus.shipped_mvp)]
    need = make_emergent_need(customer_id="C1", feature_id="F02",
                              turn_injected=1, turns_unmet=2)
    state = make_game_state(customers=[customer], features=features,
                            emergent_needs=[need], turn=8)
    record = _resolve(state, [], calibration=cal)
    assert state.emergent_needs[0].is_met
    assert any(e.startswith("emergent_need_met:") for e in record.events)


def test_emergent_need_expiry_writes_churn_driver():
    cal = _emergent_cal()  # expiry=5
    customer = _active_customer_with_needs()
    features = [make_feature(id="F02", status=FeatureStatus.not_started)]
    # turns_unmet=4 → ticks to 5 == expiry → expires.
    need = make_emergent_need(id="EN_007", customer_id="C1", feature_id="F02",
                              turn_injected=1, turns_unmet=4)
    state = make_game_state(customers=[customer], features=features,
                            emergent_needs=[need], turn=10)
    record = _resolve(state, [], calibration=cal)
    assert state.emergent_needs[0].is_expired
    assert state.customers["C1"].churn_drivers.get("F02") == pytest.approx(
        cal.emergent_need_churn_driver_weight
    )
    assert any(e.startswith("emergent_need_expired:") for e in record.events)


def test_health_check_reveals_needs_and_churn_drivers():
    cal = _emergent_cal()
    customer = _active_customer_with_needs()
    features = [make_feature(id="F02", status=FeatureStatus.not_started)]
    # within grace (age 0) so no same-turn tick muddies the reveal assertion.
    need = make_emergent_need(id="EN_003", customer_id="C1", feature_id="F02",
                              turn_injected=1, turns_unmet=0)
    state = make_game_state(customers=[customer], features=features,
                            emergent_needs=[need], turn=1,
                            resources=make_resource_pool(support_capacity=5))
    record = _resolve(state, [SupportAction(customer_id="C1", support_action="health_check", capacity=2)],
                      calibration=cal)
    assert state.emergent_needs[0].is_revealed
    assert state.customers["C1"].churn_drivers_revealed
    assert any(e.startswith("emergent_need_revealed:EN_003:C1:F02") for e in record.events)


def test_onboard_accelerates_onboarding_window():
    cal = _emergent_cal()  # onboard_acceleration=1
    customer = _active_customer_with_needs(onboarding_turns_remaining=4)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(support_capacity=5))
    _resolve(state, [SupportAction(customer_id="C1", support_action="onboard", capacity=2)],
             calibration=cal)
    # normal tick (1) + onboard_acceleration (1) = 2 decrement → 4 - 2 = 2.
    assert state.customers["C1"].onboarding_turns_remaining == 2


def test_churn_intervention_success_seed():
    cal = _emergent_cal()  # threshold=4.0, prob=0.6, recovery=3.0, min_cap=2
    customer = _active_customer_with_needs(health=2.0)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(support_capacity=5))
    record = _resolve(
        state,
        [SupportAction(customer_id="C1", support_action="churn_intervention", capacity=2)],
        calibration=cal, seed=1,  # first rng.random() ≈ 0.134 < 0.6 → success
    )
    assert any(e == "churn_intervention:C1:success" for e in record.events)


def test_churn_intervention_failure_seed():
    cal = _emergent_cal()
    customer = _active_customer_with_needs(health=2.0)
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(support_capacity=5))
    record = _resolve(
        state,
        [SupportAction(customer_id="C1", support_action="churn_intervention", capacity=2)],
        calibration=cal, seed=0,  # first rng.random() ≈ 0.844 >= 0.6 → failure
    )
    assert any(e == "churn_intervention:C1:failed" for e in record.events)


def test_churn_intervention_no_fire_above_threshold():
    """The save only fires below the health threshold; a healthy customer is untouched by it."""
    cal = _emergent_cal()
    customer = _active_customer_with_needs(health=8.0)  # above threshold 4.0
    state = make_game_state(customers=[customer],
                            resources=make_resource_pool(support_capacity=5))
    record = _resolve(
        state,
        [SupportAction(customer_id="C1", support_action="churn_intervention", capacity=2)],
        calibration=cal, seed=1,
    )
    assert not any(e.startswith("churn_intervention:") for e in record.events)


def test_emergent_need_injected_not_leaked_to_cs():
    """NO-FREE-LEAK: emergent_need_injected is recorded for analysis but never reaches CS."""
    from alignsim.src.harness.condition3_filters import filter_events
    cal = _emergent_cal(emergent_need_injection_floor=5.0)
    customer = _active_customer_with_needs()
    features = [
        make_feature(id="F01", status=FeatureStatus.shipped_mvp),
        make_feature(id="F02", status=FeatureStatus.not_started),
    ]
    state = make_game_state(customers=[customer], features=features)
    record = _resolve(state, [], calibration=cal, seed=1)
    assert any(e.startswith("emergent_need_injected:") for e in record.events)
    cs_events = filter_events(record.events, "support")
    assert not any(e.startswith("emergent_need_injected:") for e in cs_events)


def test_emergent_need_determinism_per_seed():
    """Same seed → identical injection (ids, customers, features)."""
    cal = _emergent_cal(emergent_need_injection_floor=3.0)

    def run():
        customer = _active_customer_with_needs()
        features = [
            make_feature(id="F01", status=FeatureStatus.shipped_mvp),
            make_feature(id="F02", status=FeatureStatus.not_started),
            make_feature(id="F03", status=FeatureStatus.not_started),
        ]
        state = make_game_state(customers=[customer], features=features)
        _resolve(state, [], calibration=cal, seed=7)
        return [(n.id, n.customer_id, n.feature_id) for n in state.emergent_needs]

    assert run() == run()
