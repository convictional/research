"""Factory functions for building test entities with sensible defaults.

All factories accept keyword overrides for any field of the underlying Pydantic
model. They return real Pydantic models that the engine code can consume directly.
"""

from typing import Any

from alignsim.src.models.actions import GameAction
from alignsim.src.models.entities import (
    Bug,
    BugSeverity,
    Competitor,
    CompetitorEvent,
    Customer,
    CustomerRubric,
    CustomerStage,
    EmergentNeed,
    Engagement,
    Feature,
    FeatureStatus,
    PendingAwareness,
    ProcessProject,
    ProcessProjectSize,
    ProcessProjectStatus,
    QualityLevel,
    Segment,
)
from alignsim.src.models.game_state import (
    ActiveProcessBonus,
    GameState,
    PendingHire,
    ResourcePool,
    TechDebt,
    TurnRecord,
)
from alignsim.src.models.goals import PrimaryGoal, RoleSubGoal
from alignsim.src.models.scenario import (
    CalibrationParams,
    CustomerGeneratorConfig,
    InitialFinancials,
    ScenarioDefinition,
)


def make_customer(**overrides: Any) -> Customer:
    defaults: dict[str, Any] = {
        "id": "C01",
        "size": 1,
        "segment": Segment.startup,
        "stage": CustomerStage.lead,
        "engagement": Engagement.cold,
        "known_needs": [],
        "deal_value": 1_000,
        "rubric": CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        "feature_needs": {},
        "dealbreakers": [],
        "timeline": 8,
        "churn_drivers": {},
        "discovery_difficulty": 1.0,
        "health": 8.0,
        "is_visible": False,
        "close_threshold": 0.0,
        "desired_price_point": 0,
        "last_proposed_price": None,
        "has_received_proposal": False,
    }
    defaults.update(overrides)
    return Customer(**defaults)


def make_feature(**overrides: Any) -> Feature:
    defaults: dict[str, Any] = {
        "id": "F01",
        "name": "Test Feature",
        "description": "A test feature",
        "cost": {"mvp": 8, "solid": 15, "polished": 25},
        "depends_on": [],
        "status": FeatureStatus.not_started,
        "progress": 0.0,
        "current_target": None,
        "turns_worked": 0,
        "customer_impact": {},
        "bug_rate_modifier": 1.0,
        "maintenance_cost": 0,
    }
    defaults.update(overrides)
    return Feature(**defaults)


def make_bug(**overrides: Any) -> Bug:
    defaults: dict[str, Any] = {
        "id": "BUG_001",
        "severity": BugSeverity.minor,
        "feature_id": "F01",
        "turn_injected": 1,
        "turns_unresolved": 0,
        "affected_customers": [],
        "is_resolved": False,
    }
    defaults.update(overrides)
    return Bug(**defaults)


def make_emergent_need(**overrides: Any) -> EmergentNeed:
    defaults: dict[str, Any] = {
        "id": "EN_001",
        "customer_id": "C01",
        "feature_id": "F02",
        "turn_injected": 1,
        "turns_unmet": 0,
        "is_revealed": False,
        "is_met": False,
        "is_expired": False,
    }
    defaults.update(overrides)
    return EmergentNeed(**defaults)


def make_competitor(**overrides: Any) -> Competitor:
    defaults: dict[str, Any] = {
        "id": "Comp_Alpha",
        "name": "Alpha Co",
        "events": [],
    }
    defaults.update(overrides)
    return Competitor(**defaults)


def make_competitor_event(**overrides: Any) -> CompetitorEvent:
    defaults: dict[str, Any] = {
        "turn": 5,
        "event_type": "feature_launch",
        "description": "Competitor launches",
        "affected_customers": [],
        "rubric_impact": {},
    }
    defaults.update(overrides)
    return CompetitorEvent(**defaults)


def make_process_project(**overrides: Any) -> ProcessProject:
    defaults: dict[str, Any] = {
        "id": "PP01",
        "name": "Process Optimization",
        "description": "Improve a process",
        "size": ProcessProjectSize.small,
        "ops_capacity_cost": 2,
        "target_function": "sales",
        "bonus_type": "conversion_rate",
        "bonus_base": 0.05,
        "bonus_scale_factor": 0.02,
        "bonus_max": 0.20,
        "duration_turns": 3,
        "bonus_duration_turns": 12,
        "permanent_floor_fraction": 0.0,
        "prerequisites": [],
        "status": ProcessProjectStatus.available,
        "progress_turns": 0,
        "target_team_capacity_invested": 0,
        "completed_turn": None,
    }
    defaults.update(overrides)
    return ProcessProject(**defaults)


def make_active_bonus(**overrides: Any) -> ActiveProcessBonus:
    defaults: dict[str, Any] = {
        "project_id": "PP01",
        "bonus_type": "conversion_rate",
        "bonus_value": 0.10,
        "target_function": "sales",
        "turns_remaining": 12,
        "bonus_duration_turns": 12,
        "original_ops_capacity_cost": 2,
        "permanent_floor_fraction": 0.0,
    }
    defaults.update(overrides)
    return ActiveProcessBonus(**defaults)


def make_resource_pool(**overrides: Any) -> ResourcePool:
    defaults: dict[str, Any] = {
        "capacity_per_turn": 40,
        "eng_capacity": 20,
        "sales_capacity": 10,
        "support_capacity": 5,
        "marketing_capacity": 5,
        "ops_capacity": 0,
        "budget": 500_000,
        "runway_turns": 10.0,
        "base_cost_per_turn": 5_000,
        "mrr": 0,
    }
    defaults.update(overrides)
    return ResourcePool(**defaults)


def make_turn_record(
    turn: int = 1,
    events: list[str] | None = None,
    actions_valid: list[GameAction] | None = None,
    **overrides: Any,
) -> TurnRecord:
    defaults: dict[str, Any] = {
        "turn": turn,
        "actions_submitted": [],
        "actions_valid": actions_valid or [],
        "actions_rejected": [],
        "events": events or [],
        "mrr": 0,
        "churn_count": 0,
        "bugs_injected": 0,
        "bugs_fixed": 0,
        "capacity_used": 0,
        "capacity_available": 0,
        "eng_capacity_used": 0,
        "sales_capacity_used": 0,
        "support_capacity_used": 0,
        "marketing_capacity_used": 0,
        "ops_capacity_used": 0,
        "runway_turns": 0.0,
        "budget": 0,
    }
    defaults.update(overrides)
    return TurnRecord(**defaults)


def make_pending_hire(**overrides: Any) -> PendingHire:
    defaults: dict[str, Any] = {
        "id": "H1",
        "target_function": "engineering",
        "hiring_function": "engineering",
        "turns_remaining": 6,
        "onboarding_turns_remaining": 4,
        "capacity_bonus": 4,
        "is_cross_function": False,
        "active_turns_required": 3,
        "active_turns_completed": 1,
    }
    defaults.update(overrides)
    return PendingHire(**defaults)


def make_pending_awareness(**overrides: Any) -> PendingAwareness:
    defaults: dict[str, Any] = {
        "land_turn": 3,
        "feature_id": "F01",
        "amount": 1.0,
    }
    defaults.update(overrides)
    return PendingAwareness(**defaults)


def make_game_state(
    customers: list[Customer] | None = None,
    features: list[Feature] | None = None,
    competitors: list[Competitor] | None = None,
    process_projects: list[ProcessProject] | None = None,
    bugs: list[Bug] | None = None,
    emergent_needs: list[EmergentNeed] | None = None,
    resources: ResourcePool | None = None,
    awareness: dict[str, float] | None = None,
    pending_awareness: list[PendingAwareness] | None = None,
    **overrides: Any,
) -> GameState:
    defaults: dict[str, Any] = {
        "turn": 1,
        "max_turns": 48,
        "seed": 42,
        "customers": {c.id: c for c in (customers or [])},
        "features": {f.id: f for f in (features or [])},
        "competitors": {c.id: c for c in (competitors or [])},
        "process_projects": {p.id: p for p in (process_projects or [])},
        "bugs": list(bugs or []),
        "emergent_needs": list(emergent_needs or []),
        "next_emergent_need_id": (len(emergent_needs) + 1) if emergent_needs else 1,
        "resources": resources or make_resource_pool(),
        "tech_debt": TechDebt(level=0.0),
        "awareness": dict(awareness or {}),
        "pending_awareness": list(pending_awareness or []),
    }
    defaults.update(overrides)
    return GameState(**defaults)


def make_initial_financials(**overrides: Any) -> InitialFinancials:
    defaults: dict[str, Any] = {
        "starting_budget": 500_000,
        "base_cost_per_turn": 5_000,
        "starting_mrr": 0,
        "capacity_per_turn": 40,
        "eng_capacity": 20,
        "sales_capacity": 10,
        "support_capacity": 5,
        "marketing_capacity": 5,
        "ops_capacity": 0,
    }
    defaults.update(overrides)
    return InitialFinancials(**defaults)


def make_primary_goal(**overrides: Any) -> PrimaryGoal:
    defaults: dict[str, Any] = {
        "mrr_target": 60_000,
        "max_churn_rate": 0.02,
        "min_runway_turns": 10.0,
        "target_turn": 48,
        "sub_goals": [],
    }
    defaults.update(overrides)
    return PrimaryGoal(**defaults)


def make_generator_config(**overrides: Any) -> CustomerGeneratorConfig:
    defaults: dict[str, Any] = {
        "feature_segment_affinity": {
            "F01": {"startup": 0.25, "growth": 0.25, "mid_market": 0.25, "enterprise": 0.25},
            "F02": {"growth": 0.45, "startup": 0.40, "mid_market": 0.10, "enterprise": 0.05},
        },
        "rubric_archetypes": {
            "startup": {"feature_coverage": 0.30, "price": 0.35, "maturity": 0.15, "support": 0.20},
            "growth": {"feature_coverage": 0.40, "price": 0.20, "maturity": 0.20, "support": 0.20},
            "mid_market": {"feature_coverage": 0.40, "price": 0.15, "maturity": 0.25, "support": 0.20},
            "enterprise": {"feature_coverage": 0.45, "price": 0.10, "maturity": 0.30, "support": 0.15},
        },
        "segment_weights": {"startup": 0.30, "growth": 0.30, "mid_market": 0.25, "enterprise": 0.15},
        "size_distributions": {
            "startup": {1: 0.40, 2: 0.40, 3: 0.20},
            "growth": {2: 0.30, 3: 0.40, 4: 0.30},
            "mid_market": {2: 0.20, 3: 0.40, 4: 0.30, 5: 0.10},
            "enterprise": {3: 0.15, 4: 0.40, 5: 0.45},
        },
        "deal_value_per_size": {"startup": 900, "growth": 1350, "mid_market": 1125, "enterprise": 1800},
        "discovery_difficulty_range": {"tier2": (1.5, 2.5), "tier3": (3.0, 4.5)},
        "timeline_range": {
            "startup": (24, 35), "growth": (22, 30),
            "mid_market": (20, 28), "enterprise": (16, 22),
        },
    }
    defaults.update(overrides)
    return CustomerGeneratorConfig(**defaults)


def make_scenario(
    customers: list[Customer] | None = None,
    features: list[Feature] | None = None,
    competitors: list[Competitor] | None = None,
    process_projects: list[ProcessProject] | None = None,
    financials: InitialFinancials | None = None,
    primary_goal: PrimaryGoal | None = None,
    calibration: CalibrationParams | None = None,
    initial_bugs: list[str] | None = None,
    generator_config: CustomerGeneratorConfig | None = None,
    **overrides: Any,
) -> ScenarioDefinition:
    defaults: dict[str, Any] = {
        "name": "test_scenario",
        "description": "A minimal test scenario",
        "seed": 42,
        "max_turns": 48,
        "customers": customers or [],
        "features": features or [],
        "competitors": competitors or [],
        "process_projects": process_projects or [],
        "financials": financials or make_initial_financials(),
        "initial_bugs": initial_bugs or [],
        "primary_goal": primary_goal or make_primary_goal(),
        "calibration": calibration or CalibrationParams(),
        "generator_config": generator_config,
    }
    defaults.update(overrides)
    return ScenarioDefinition(**defaults)


def make_minimal_scenario() -> ScenarioDefinition:
    """Tiny scenario: 3 customers (1 visible lead, 1 active customer, 1 hidden),
    2 features (1 shipped MVP, 1 not started), minimal financials."""
    customers = [
        make_customer(
            id="L1",
            stage=CustomerStage.lead,
            is_visible=True,
            size=1,
            deal_value=2_000,
            timeline=12,
            feature_needs={"F01": {"mvp": 0.6, "solid": 0.8, "polished": 1.0}},
        ),
        make_customer(
            id="A1",
            stage=CustomerStage.customer,
            is_visible=True,
            size=2,
            deal_value=4_000,
            health=8.0,
            timeline=0,
            feature_needs={"F01": {"mvp": 0.7, "solid": 0.9, "polished": 1.0}},
        ),
        make_customer(
            id="H1",
            stage=CustomerStage.lead,
            is_visible=False,
            size=1,
            deal_value=1_500,
            timeline=10,
            discovery_difficulty=2.0,
        ),
    ]
    features = [
        make_feature(
            id="F01",
            name="Core",
            cost={"mvp": 8, "solid": 15, "polished": 25},
            status=FeatureStatus.shipped_mvp,
        ),
        make_feature(
            id="F02",
            name="Reports",
            cost={"mvp": 6, "solid": 12, "polished": 20},
            status=FeatureStatus.not_started,
        ),
    ]
    return make_scenario(
        customers=customers,
        features=features,
        financials=make_initial_financials(starting_mrr=4_000),
    )
