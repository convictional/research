"""Seed-stage scenario: blank-slate startup with wide strategic space.

48 customers (8 visible leads, 40 hidden), 16 features (diamond DAG), 3 competitors.

Starting state: 1 shipped MVP product, 0 MRR, small team (15 capacity),
$3M budget (~81 turns runway). No bugs, no tech debt.

The agent chooses which segments to target, what to build, and how to grow.
Many viable paths: enterprise rush, SMB volume, segment specialization, balanced.

Target: $40K MRR in 48 turns. Growing past break-even, not just surviving.
"""

import random

from alignsim.src.models.entities import (
    Competitor,
    CompetitorEvent,
    Customer,
    CustomerRubric,
    CustomerStage,
    Engagement,
    Feature,
    FeatureStatus,
    ProcessProject,
    ProcessProjectSize,
    Segment,
)
from alignsim.src.models.goals import PrimaryGoal, RoleSubGoal
from alignsim.src.models.scenario import CalibrationParams, CustomerGeneratorConfig, InitialFinancials, ScenarioDefinition


_DESIRED_PRICE_DISCOUNT_RANGE: dict[str, tuple[float, float]] = {
    "startup": (0.20, 0.40),
    "growth": (0.15, 0.35),
    "mid_market": (0.10, 0.25),
    "enterprise": (0.05, 0.20),
}


def _assign_desired_prices(customers: list[Customer], seed: int) -> list[Customer]:
    """Set desired_price_point on handwritten customers using segment-based discount."""
    rng = random.Random(seed)
    for c in customers:
        discount_range = _DESIRED_PRICE_DISCOUNT_RANGE.get(c.segment.value)
        if discount_range is not None and c.deal_value > 0:
            discount = rng.uniform(discount_range[0], discount_range[1])
            c.desired_price_point = max(50, round(c.deal_value * (1 - discount) / 50) * 50)
    return customers


def _assign_close_thresholds(
    customers: list[Customer], seed: int, mean: float = 0.75, std: float = 0.05,
) -> list[Customer]:
    """Set per-customer close_threshold using gauss(mean, std), clamped to [0.50, 0.95]."""
    rng = random.Random(seed + 1)
    for c in customers:
        c.close_threshold = round(max(0.50, min(0.95, rng.gauss(mean, std))), 3)
    return customers


_DEAL_VALUE_SEGMENT_MULTIPLIER: dict[str, float] = {
    "startup": 1.05,
    "growth": 1.30,
    "mid_market": 1.80,
    "enterprise": 1.45,
}


def _reprice_by_segment(customers: list[Customer]) -> list[Customer]:
    """Scale handwritten anchor deal_values by segment (v2 economics).

    Fixes the mid_market-below-growth inversion and lifts the overall level ~1.4x so
    growth becomes reachable. Preserves within-segment hand-tuning (relative spread is
    kept; only the per-segment level shifts). Runs before _assign_desired_prices so
    desired_price_point auto-re-derives from the new deal_value.
    """
    for c in customers:
        mult = _DEAL_VALUE_SEGMENT_MULTIPLIER.get(c.segment.value)
        if mult is not None and c.deal_value > 0:
            c.deal_value = max(50, round(c.deal_value * mult / 50) * 50)
    return customers


def create_seed_stage_scenario(seed: int = 42) -> ScenarioDefinition:
    customers = _assign_desired_prices(_reprice_by_segment(_create_customers()), seed)
    customers = _assign_close_thresholds(customers, seed)
    return ScenarioDefinition(
        name="seed_stage",
        description="Seed-stage startup: 1 MVP, 0 MRR, anchor customers + procedural generation, 16 features, diamond DAG, 6 ops projects.",
        seed=seed,
        max_turns=48,
        customers=customers,
        features=_create_features(),
        competitors=_create_competitors(),
        process_projects=_create_process_projects(),
        financials=InitialFinancials(
            starting_budget=3_000_000,
            base_cost_per_turn=3_500,
            starting_mrr=0,
            capacity_per_turn=15,  # eng(6) + sales(6) + marketing(3); support/ops start at 0
            eng_capacity=6,
            sales_capacity=6,
            support_capacity=0,
            marketing_capacity=3,
            ops_capacity=0,
        ),
        primary_goal=PrimaryGoal(
            mrr_target=40_000,
            max_churn_rate=0.02,
            min_runway_turns=60,
            target_turn=48,
            sub_goals=[
                RoleSubGoal(
                    role="engineering",
                    description="Ship features at solid quality or better",
                    metric="features_shipped_solid_plus",
                    target_value=12.0,
                ),
                RoleSubGoal(
                    role="sales",
                    description="Maintain steady deal closure rate",
                    metric="pipeline_velocity",
                    target_value=0.2,
                ),
                RoleSubGoal(
                    role="support",
                    description="Keep average customer health above 7.0",
                    metric="avg_customer_health",
                    target_value=7.0,
                ),
                RoleSubGoal(
                    role="marketing",
                    description="Generate inbound leads over the game",
                    metric="marketing_leads_generated",
                    target_value=24.0,
                ),
                RoleSubGoal(
                    role="ops",
                    description="Complete all process improvement projects",
                    metric="process_projects_completed",
                    target_value=6.0,
                ),
            ],
        ),
        calibration=CalibrationParams(
            lead_to_prospect_rate=0.35,
            prospect_to_qualified_rate=0.55,
            qualified_to_in_deal_rate=0.48,
            in_deal_to_closed_rate=0.40,
            team_cost_per_capacity=2200,
        ),
        initial_bugs=[],
        generator_config=CustomerGeneratorConfig(
            feature_segment_affinity={
                "F01": {"startup": 0.25, "growth": 0.25, "mid_market": 0.25, "enterprise": 0.25},
                "F02": {"growth": 0.45, "startup": 0.40, "mid_market": 0.10, "enterprise": 0.05},
                "F03": {"startup": 0.60, "growth": 0.05, "mid_market": 0.05, "enterprise": 0.30},
                "F04": {"startup": 0.35, "mid_market": 0.45, "growth": 0.10, "enterprise": 0.10},
                "F05": {"mid_market": 0.40, "growth": 0.35, "enterprise": 0.20, "startup": 0.05},
                "F06": {"growth": 0.50, "enterprise": 0.40, "startup": 0.05, "mid_market": 0.05},
                "F07": {"startup": 0.50, "enterprise": 0.40, "growth": 0.05, "mid_market": 0.05},
                "F08": {"startup": 0.40, "enterprise": 0.40, "growth": 0.10, "mid_market": 0.10},
                "F09": {"startup": 0.50, "enterprise": 0.30, "mid_market": 0.10, "growth": 0.10},
                "F10": {"startup": 0.40, "enterprise": 0.30, "mid_market": 0.20, "growth": 0.10},
                "F11": {"mid_market": 0.55, "enterprise": 0.35, "startup": 0.05, "growth": 0.05},
                "F12": {"mid_market": 0.45, "enterprise": 0.35, "growth": 0.10, "startup": 0.10},
                "F13": {"growth": 0.40, "enterprise": 0.40, "mid_market": 0.10, "startup": 0.10},
                "F14": {"enterprise": 0.80, "growth": 0.15, "mid_market": 0.05, "startup": 0.00},
                "F15": {"enterprise": 0.80, "startup": 0.15, "mid_market": 0.05, "growth": 0.00},
                "F16": {"enterprise": 0.75, "mid_market": 0.15, "startup": 0.10, "growth": 0.00},
            },
            rubric_archetypes={
                "startup": {"feature_coverage": 0.35, "price": 0.35, "maturity": 0.15, "support": 0.15},
                "growth": {"feature_coverage": 0.40, "price": 0.20, "maturity": 0.20, "support": 0.20},
                "mid_market": {"feature_coverage": 0.35, "price": 0.15, "maturity": 0.25, "support": 0.25},
                "enterprise": {"feature_coverage": 0.35, "price": 0.10, "maturity": 0.25, "support": 0.30},
            },
            segment_weights={"startup": 0.30, "growth": 0.30, "mid_market": 0.25, "enterprise": 0.15},
            size_distributions={
                "startup": {1: 0.40, 2: 0.40, 3: 0.20},
                "growth": {2: 0.30, 3: 0.40, 4: 0.30},
                "mid_market": {2: 0.20, 3: 0.40, 4: 0.30, 5: 0.10},
                "enterprise": {3: 0.15, 4: 0.40, 5: 0.45},
            },
            deal_value_per_size={"startup": 1420, "growth": 1950, "mid_market": 2250, "enterprise": 2900},
            discovery_difficulty_range={"tier2": (1.5, 2.5), "tier3": (3.0, 4.5), "tier4": (4.5, 5.5)},
            timeline_range={
                "startup": (14, 22), "growth": (18, 26),
                "mid_market": (22, 30), "enterprise": (28, 38),
            },
            desired_price_discount_range=_DESIRED_PRICE_DISCOUNT_RANGE,
        ),
    )


# =============================================================================
# CUSTOMERS (48 total)
# =============================================================================

def _create_customers() -> list[Customer]:
    customers: list[Customer] = []

    # =========================================================================
    # VISIBLE LEADS (8) — 2 per segment, starting pipeline
    # Small/mid-market, simple needs: F01 + one tier-2 feature
    # =========================================================================

    # Segment A leads
    customers.append(Customer(
        id="C01", size=2, segment=Segment.growth, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F01", "F02"], deal_value=3000,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=[], timeline=30, timeline_original=30, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C02", size=3, segment=Segment.growth, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F01", "F02"], deal_value=5250,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.20, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=28, timeline_original=28, health=8.0, is_visible=True,
    ))

    # Segment B leads
    customers.append(Customer(
        id="C03", size=2, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F01", "F02"], deal_value=2700,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=30, timeline_original=30, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C04", size=1, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F01", "F03"], deal_value=1800,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.30, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F03": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=32, timeline_original=32, health=8.0, is_visible=True,
    ))

    # Segment C leads
    customers.append(Customer(
        id="C05", size=2, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F01", "F04"], deal_value=2700,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=30, timeline_original=30, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C06", size=1, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F01", "F03"], deal_value=1500,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.30, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F03": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=35, timeline_original=35, health=8.0, is_visible=True,
    ))

    # Segment D leads
    customers.append(Customer(
        id="C07", size=2, segment=Segment.mid_market, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F01", "F04"], deal_value=2250,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=30, timeline_original=30, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C08", size=1, segment=Segment.mid_market, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F01", "F05"], deal_value=1500,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.30, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F05": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=35, timeline_original=35, health=8.0, is_visible=True,
    ))

    # =========================================================================
    # HIDDEN — EASY TO DISCOVER (16) — difficulty 1.5-2.5
    # 4 per segment, sizes 1-3, need tier-2 features
    # =========================================================================

    # Segment A easy
    customers.append(Customer(
        id="C09", size=2, segment=Segment.growth, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F02"], deal_value=2700,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=25, timeline_original=25, discovery_difficulty=1.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C10", size=1, segment=Segment.growth, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F02"], deal_value=1620,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.30, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=28, timeline_original=28, discovery_difficulty=2.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C11", size=3, segment=Segment.growth, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F02", "F05"], deal_value=4050,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.20, maturity=0.15, support=0.20),
        feature_needs={
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F05": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=[], timeline=24, timeline_original=24, discovery_difficulty=2.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C12", size=2, segment=Segment.growth, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F05"], deal_value=2970,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F05": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=26, timeline_original=26, discovery_difficulty=2.0, health=8.0, is_visible=False,
    ))

    # Segment B easy
    customers.append(Customer(
        id="C13", size=2, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F02"], deal_value=2430,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=25, timeline_original=25, discovery_difficulty=1.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C14", size=1, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F03"], deal_value=1350,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.30, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F03": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=30, timeline_original=30, discovery_difficulty=2.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C15", size=3, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F02", "F03"], deal_value=4050,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.20, maturity=0.15, support=0.20),
        feature_needs={
            "F02": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F03": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=24, timeline_original=24, discovery_difficulty=2.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C16", size=2, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F03"], deal_value=2700,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F03": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=26, timeline_original=26, discovery_difficulty=2.0, health=8.0, is_visible=False,
    ))

    # Segment C easy
    customers.append(Customer(
        id="C17", size=2, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F04"], deal_value=2430,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=25, timeline_original=25, discovery_difficulty=1.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C18", size=1, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F03"], deal_value=1350,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.30, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F03": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=30, timeline_original=30, discovery_difficulty=2.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C19", size=3, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F03", "F04"], deal_value=4320,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.20, maturity=0.15, support=0.20),
        feature_needs={
            "F03": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=24, timeline_original=24, discovery_difficulty=2.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C20", size=2, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F04"], deal_value=2700,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=26, timeline_original=26, discovery_difficulty=2.0, health=8.0, is_visible=False,
    ))

    # Segment D easy
    customers.append(Customer(
        id="C21", size=2, segment=Segment.mid_market, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F04"], deal_value=2025,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=25, timeline_original=25, discovery_difficulty=1.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C22", size=1, segment=Segment.mid_market, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F05"], deal_value=1350,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.30, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F05": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=30, timeline_original=30, discovery_difficulty=2.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C23", size=3, segment=Segment.mid_market, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F04", "F05"], deal_value=3780,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.20, maturity=0.15, support=0.20),
        feature_needs={
            "F04": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F05": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=24, timeline_original=24, discovery_difficulty=2.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C24", size=2, segment=Segment.mid_market, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F05"], deal_value=2430,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.25, maturity=0.15, support=0.20),
        feature_needs={
            "F01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F05": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=26, timeline_original=26, discovery_difficulty=2.0, health=8.0, is_visible=False,
    ))

    # =========================================================================
    # HIDDEN — MODERATE DISCOVERY (16) — difficulty 3.0-4.0
    # 4 per segment, sizes 2-4, need tier-3 features, higher deal values
    # =========================================================================

    # Segment A moderate
    customers.append(Customer(
        id="C25", size=3, segment=Segment.growth, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F02", "F06"], deal_value=3500,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.20, support=0.20),
        feature_needs={
            "F02": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F06": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F06"], timeline=22, timeline_original=22, discovery_difficulty=3.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C26", size=4, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F06", "F13"], deal_value=5000,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.20, support=0.20),
        feature_needs={
            "F06": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F13": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=["F06"], timeline=20, timeline_original=20, discovery_difficulty=3.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C27", size=3, segment=Segment.growth, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F05", "F13"], deal_value=4000,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.20, support=0.20),
        feature_needs={
            "F05": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F13": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F13"], timeline=22, timeline_original=22, discovery_difficulty=4.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C28", size=2, segment=Segment.growth, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F06"], deal_value=2500,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.20, maturity=0.20, support=0.20),
        feature_needs={
            "F02": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F06": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=24, timeline_original=24, discovery_difficulty=3.0, health=8.0, is_visible=False,
    ))

    # Segment B moderate
    customers.append(Customer(
        id="C29", size=3, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F02", "F07"], deal_value=3500,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.20, support=0.20),
        feature_needs={
            "F02": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F07": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F07"], timeline=22, timeline_original=22, discovery_difficulty=3.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C30", size=4, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F07", "F08"], deal_value=5500,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.20, support=0.20),
        feature_needs={
            "F07": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F08": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=["F07"], timeline=20, timeline_original=20, discovery_difficulty=3.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C31", size=3, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F03", "F08"], deal_value=3800,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.20, support=0.20),
        feature_needs={
            "F03": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F08": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F08"], timeline=22, timeline_original=22, discovery_difficulty=4.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C32", size=2, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F07"], deal_value=2200,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.20, maturity=0.20, support=0.20),
        feature_needs={
            "F02": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F07": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=24, timeline_original=24, discovery_difficulty=3.0, health=8.0, is_visible=False,
    ))

    # Segment C moderate
    customers.append(Customer(
        id="C33", size=3, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F04", "F09"], deal_value=3200,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.20, support=0.20),
        feature_needs={
            "F04": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F09": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F09"], timeline=22, timeline_original=22, discovery_difficulty=3.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C34", size=4, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F09", "F10"], deal_value=5000,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.20, support=0.20),
        feature_needs={
            "F09": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F10": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=["F09"], timeline=20, timeline_original=20, discovery_difficulty=3.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C35", size=3, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F03", "F09"], deal_value=3500,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.20, support=0.20),
        feature_needs={
            "F03": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F09": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=22, timeline_original=22, discovery_difficulty=4.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C36", size=2, segment=Segment.startup, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F10"], deal_value=2200,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.20, maturity=0.20, support=0.20),
        feature_needs={
            "F04": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F10": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=24, timeline_original=24, discovery_difficulty=3.0, health=8.0, is_visible=False,
    ))

    # Segment D moderate
    customers.append(Customer(
        id="C37", size=3, segment=Segment.mid_market, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F04", "F11"], deal_value=3000,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.20, support=0.20),
        feature_needs={
            "F04": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F11": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F11"], timeline=22, timeline_original=22, discovery_difficulty=3.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C38", size=4, segment=Segment.mid_market, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F11", "F12"], deal_value=4500,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.20, support=0.20),
        feature_needs={
            "F11": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F12": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=["F11"], timeline=20, timeline_original=20, discovery_difficulty=3.5, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C39", size=3, segment=Segment.mid_market, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F05", "F12"], deal_value=3200,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.20, support=0.20),
        feature_needs={
            "F05": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F12": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=22, timeline_original=22, discovery_difficulty=4.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C40", size=2, segment=Segment.mid_market, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F11"], deal_value=2000,
        rubric=CustomerRubric(feature_coverage=0.40, price=0.20, maturity=0.20, support=0.20),
        feature_needs={
            "F04": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F11": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=[], timeline=24, timeline_original=24, discovery_difficulty=3.0, health=8.0, is_visible=False,
    ))

    # =========================================================================
    # HIDDEN — HARD TO DISCOVER (8) — difficulty 4.5-5.0
    # 2 per segment, sizes 4-5, enterprise, need tier-4 or polished tier-3
    # =========================================================================

    # Enterprise customers (hard to discover; 4 are visible at turn 1 as tech-tree anchors)
    customers.append(Customer(
        id="C41", size=5, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F06", "F14"], deal_value=9000,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.25, support=0.15),
        feature_needs={
            "F06": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F14": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=["F14"], timeline=18, timeline_original=18, discovery_difficulty=5.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C42", size=4, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F13", "F14"], deal_value=7000,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.25, support=0.15),
        feature_needs={
            "F13": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F14": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F14"], timeline=20, timeline_original=20, discovery_difficulty=4.5, health=8.0, is_visible=True,
    ))

    customers.append(Customer(
        id="C43", size=5, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F07", "F15"], deal_value=10000,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.25, support=0.15),
        feature_needs={
            "F07": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F15": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=["F15"], timeline=18, timeline_original=18, discovery_difficulty=5.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C44", size=4, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F08", "F15"], deal_value=7500,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.25, support=0.15),
        feature_needs={
            "F08": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F15": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F15"], timeline=20, timeline_original=20, discovery_difficulty=4.5, health=8.0, is_visible=True,
    ))

    customers.append(Customer(
        id="C45", size=5, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F09", "F16"], deal_value=8500,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.25, support=0.15),
        feature_needs={
            "F09": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F16": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=["F16"], timeline=18, timeline_original=18, discovery_difficulty=5.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C46", size=4, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F10", "F16"], deal_value=6500,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.25, support=0.15),
        feature_needs={
            "F10": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "F16": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F16"], timeline=20, timeline_original=20, discovery_difficulty=4.5, health=8.0, is_visible=True,
    ))

    customers.append(Customer(
        id="C47", size=5, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F11", "F12"], deal_value=7000,
        rubric=CustomerRubric(feature_coverage=0.50, price=0.10, maturity=0.25, support=0.15),
        feature_needs={
            "F11": {"mvp": 0.4, "solid": 0.6, "polished": 0.85},
            "F12": {"mvp": 0.4, "solid": 0.6, "polished": 0.85},
        },
        dealbreakers=["F11", "F12"], timeline=18, timeline_original=18, discovery_difficulty=5.0, health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C48", size=4, segment=Segment.enterprise, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F11"], deal_value=5500,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.25, support=0.15),
        feature_needs={
            "F04": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "F11": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=["F11"], timeline=20, timeline_original=20, discovery_difficulty=4.5, health=8.0, is_visible=True,
    ))

    return customers


# =============================================================================
# FEATURES (16 total)
# =============================================================================

def _create_features() -> list[Feature]:
    features: list[Feature] = []

    # ---- Tier 1: Shipped MVP (1 feature) ----

    features.append(Feature(
        id="F01", name="Nexus Core",
        description="Core platform. High-complexity foundation serving all segments.",
        cost={"mvp": 20, "solid": 35, "polished": 55},
        depends_on=[], status=FeatureStatus.shipped_mvp, progress=100.0,
        customer_impact={
            "C01": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C02": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C03": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C05": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C06": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C07": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C08": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        bug_rate_modifier=1.2, maintenance_cost=500,
    ))

    # ---- Tier 2: Segment bridges (4 features, each deps on F01) ----

    features.append(Feature(
        id="F02", name="Relay",
        description="Data sync and integration layer for segments A and B.",
        cost={"mvp": 10, "solid": 18, "polished": 30},
        depends_on=["F01"], status=FeatureStatus.not_started,
        customer_impact={
            "C01": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "C02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C03": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C09": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C10": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C11": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C13": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C15": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=1.0, maintenance_cost=300,
    ))

    features.append(Feature(
        id="F03", name="Conduit",
        description="Workflow engine and analytics pipeline for segments B and C.",
        cost={"mvp": 12, "solid": 22, "polished": 36},
        depends_on=["F01"], status=FeatureStatus.not_started,
        customer_impact={
            "C04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C06": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C14": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C15": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C16": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C18": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C19": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=1.1, maintenance_cost=350,
    ))

    features.append(Feature(
        id="F04", name="Lattice",
        description="Reporting and configuration engine for segments C and D.",
        cost={"mvp": 10, "solid": 18, "polished": 30},
        depends_on=["F01"], status=FeatureStatus.not_started,
        customer_impact={
            "C05": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C07": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C17": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C19": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C20": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C21": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C23": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=0.9, maintenance_cost=350,
    ))

    features.append(Feature(
        id="F05", name="Anchor",
        description="Compliance and enterprise auth for segments D and A.",
        cost={"mvp": 11, "solid": 20, "polished": 33},
        depends_on=["F01"], status=FeatureStatus.not_started,
        customer_impact={
            "C08": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C11": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C12": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C22": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C23": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C24": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C27": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=1.0, maintenance_cost=350,
    ))

    # ---- Tier 3: Segment-specific (8 features) ----

    features.append(Feature(
        id="F06", name="Flare",
        description="Advanced integrations for segment A.",
        cost={"mvp": 20, "solid": 35, "polished": 55},
        depends_on=["F02"], status=FeatureStatus.not_started,
        customer_impact={
            "C25": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C26": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C28": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C41": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=1.1, maintenance_cost=500,
    ))

    features.append(Feature(
        id="F07", name="Ripple",
        description="Automation suite for segment B.",
        cost={"mvp": 21, "solid": 37, "polished": 58},
        depends_on=["F02"], status=FeatureStatus.not_started,
        customer_impact={
            "C29": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C30": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C32": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C43": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=1.0, maintenance_cost=520,
    ))

    features.append(Feature(
        id="F08", name="Pulse",
        description="Analytics for segment B. Diamond with Ripple for enterprise B.",
        cost={"mvp": 23, "solid": 40, "polished": 62},
        depends_on=["F03"], status=FeatureStatus.not_started,
        customer_impact={
            "C30": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C31": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C44": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=1.1, maintenance_cost=560,
    ))

    features.append(Feature(
        id="F09", name="Ember",
        description="Dashboards for segment C.",
        cost={"mvp": 20, "solid": 35, "polished": 55},
        depends_on=["F03"], status=FeatureStatus.not_started,
        customer_impact={
            "C33": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C34": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C35": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C45": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=0.9, maintenance_cost=500,
    ))

    features.append(Feature(
        id="F10", name="Drift",
        description="Reporting for segment C. Diamond with Ember for enterprise C.",
        cost={"mvp": 21, "solid": 37, "polished": 58},
        depends_on=["F04"], status=FeatureStatus.not_started,
        customer_impact={
            "C34": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C36": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C46": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=0.9, maintenance_cost=520,
    ))

    features.append(Feature(
        id="F11", name="Vault",
        description="Security and configuration for segment D.",
        cost={"mvp": 23, "solid": 40, "polished": 62},
        depends_on=["F04"], status=FeatureStatus.not_started,
        customer_impact={
            "C37": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C38": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C40": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C47": {"mvp": 0.4, "solid": 0.6, "polished": 0.85},
            "C48": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        bug_rate_modifier=1.0, maintenance_cost=560,
    ))

    features.append(Feature(
        id="F12", name="Bastion",
        description="Compliance for segment D. Diamond with Vault for enterprise D.",
        cost={"mvp": 22, "solid": 38, "polished": 60},
        depends_on=["F05"], status=FeatureStatus.not_started,
        customer_impact={
            "C38": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C39": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C47": {"mvp": 0.4, "solid": 0.6, "polished": 0.85},
        },
        bug_rate_modifier=1.0, maintenance_cost=540,
    ))

    features.append(Feature(
        id="F13", name="Beacon",
        description="Enterprise auth for segment A. Diamond with Flare for enterprise A.",
        cost={"mvp": 24, "solid": 42, "polished": 65},
        depends_on=["F05"], status=FeatureStatus.not_started,
        customer_impact={
            "C26": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C27": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C42": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=1.1, maintenance_cost=580,
    ))

    # ---- Tier 4: Enterprise premium (3 features, diamond deps) ----

    features.append(Feature(
        id="F14", name="Zenith",
        description="Enterprise A platform. Requires both Flare (F06) and Beacon (F13).",
        cost={"mvp": 35, "solid": 58, "polished": 88},
        depends_on=["F06", "F13"], status=FeatureStatus.not_started,
        customer_impact={
            "C41": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "C42": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        bug_rate_modifier=1.3, maintenance_cost=800,
    ))

    features.append(Feature(
        id="F15", name="Summit",
        description="Enterprise B platform. Requires both Ripple (F07) and Pulse (F08).",
        cost={"mvp": 38, "solid": 63, "polished": 93},
        depends_on=["F07", "F08"], status=FeatureStatus.not_started,
        customer_impact={
            "C43": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "C44": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        bug_rate_modifier=1.3, maintenance_cost=850,
    ))

    features.append(Feature(
        id="F16", name="Pinnacle",
        description="Enterprise C platform. Requires both Ember (F09) and Drift (F10).",
        cost={"mvp": 32, "solid": 55, "polished": 82},
        depends_on=["F09", "F10"], status=FeatureStatus.not_started,
        customer_impact={
            "C45": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "C46": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        bug_rate_modifier=1.2, maintenance_cost=750,
    ))

    return features


# =============================================================================
# COMPETITORS (3)
# =============================================================================

def _create_competitors() -> list[Competitor]:
    return [
        Competitor(
            id="Comp_Nova",
            name="Nova Labs",
            events=[
                CompetitorEvent(
                    turn=10, event_type="feature_launch",
                    description="Nova launches segment A integration MVP",
                    affected_customers=["C25", "C26", "C28"],
                    rubric_impact={"feature_coverage": 0.5, "maturity": 0.4},
                ),
                CompetitorEvent(
                    turn=20, event_type="pricing_change",
                    description="Nova cuts segment A pricing by 25%",
                    affected_customers=["C25", "C26", "C27", "C41", "C42"],
                    rubric_impact={"price": 0.7},
                ),
                CompetitorEvent(
                    turn=35, event_type="feature_launch",
                    description="Nova launches enterprise A platform",
                    affected_customers=["C41", "C42"],
                    rubric_impact={"feature_coverage": 0.75, "maturity": 0.6},
                ),
            ],
        ),
        Competitor(
            id="Comp_Stratum",
            name="Stratum Inc",
            events=[
                CompetitorEvent(
                    turn=12, event_type="feature_launch",
                    description="Stratum launches B+C workflow tool",
                    affected_customers=["C29", "C30", "C33", "C34"],
                    rubric_impact={"feature_coverage": 0.5, "maturity": 0.4},
                ),
                CompetitorEvent(
                    turn=25, event_type="feature_launch",
                    description="Stratum ships advanced B+C analytics",
                    affected_customers=["C30", "C31", "C34", "C35", "C43", "C45"],
                    rubric_impact={"feature_coverage": 0.6, "maturity": 0.5},
                ),
                CompetitorEvent(
                    turn=40, event_type="deal_win",
                    description="Stratum rumored to be acquired — creates urgency",
                    affected_customers=["C43", "C44", "C45", "C46"],
                    rubric_impact={"feature_coverage": 0.7, "maturity": 0.55},
                ),
            ],
        ),
        Competitor(
            id="Comp_Bedrock",
            name="Bedrock Systems",
            events=[
                CompetitorEvent(
                    turn=15, event_type="feature_launch",
                    description="Bedrock launches segment D config tool",
                    affected_customers=["C37", "C38", "C40"],
                    rubric_impact={"feature_coverage": 0.5, "maturity": 0.4},
                ),
                CompetitorEvent(
                    turn=30, event_type="feature_launch",
                    description="Bedrock pushes into enterprise D",
                    affected_customers=["C38", "C47", "C48"],
                    rubric_impact={"feature_coverage": 0.65, "maturity": 0.5},
                ),
            ],
        ),
    ]


# =============================================================================
# PROCESS PROJECTS (tech-tree DAG, 10 total)
#
# Tier 0 (PP01-PP06): no prerequisites — the always-available base.
# Tier 1 (PP07-PP09): each gated behind one tier-0 project; higher bonus_max + floor.
# Tier 2 (PP10): multi-parent capstone (needs PP07 + PP08); highest bonus_max + floor.
# Hand-authored & deterministic (no RNG) so the tree can't confound cross-seed / C-condition
# comparisons. bonus_max and permanent_floor_fraction strictly increase by tier along each chain.
# =============================================================================

def _create_process_projects() -> list[ProcessProject]:
    return [
        ProcessProject(
            id="PP01",
            name="Sales Process Optimization",
            description="Streamline demo-to-close pipeline with better tooling and playbooks.",
            size=ProcessProjectSize.medium,
            ops_capacity_cost=4,
            target_function="sales",
            bonus_type="conversion_rate",
            bonus_base=0.05,
            bonus_scale_factor=0.03,
            bonus_max=0.15,
            duration_turns=2,
            bonus_duration_turns=16,
            permanent_floor_fraction=0.15,
        ),
        ProcessProject(
            id="PP02",
            name="Engineering CI/CD Pipeline",
            description="Automated testing and deployment reduces bugs from builds.",
            size=ProcessProjectSize.large,
            ops_capacity_cost=6,
            target_function="engineering",
            bonus_type="bug_rate_reduction",
            bonus_base=0.10,
            bonus_scale_factor=0.04,
            bonus_max=0.20,
            duration_turns=3,
            bonus_duration_turns=20,
        ),
        ProcessProject(
            id="PP03",
            name="Support Automation",
            description="Ticketing and health monitoring automation boosts CS effectiveness.",
            size=ProcessProjectSize.small,
            ops_capacity_cost=2,
            target_function="support",
            bonus_type="health_delta_bonus",
            bonus_base=0.2,
            bonus_scale_factor=0.1,
            bonus_max=0.5,
            duration_turns=1,
            bonus_duration_turns=12,
            permanent_floor_fraction=0.20,
        ),
        ProcessProject(
            id="PP04",
            name="Marketing Analytics Platform",
            description="Better attribution and targeting increases marketing ROI.",
            size=ProcessProjectSize.medium,
            ops_capacity_cost=4,
            target_function="marketing",
            bonus_type="marketing_effectiveness",
            bonus_base=0.05,
            bonus_scale_factor=0.03,
            bonus_max=0.12,
            duration_turns=2,
            bonus_duration_turns=16,
        ),
        ProcessProject(
            id="PP05",
            name="Engineering Code Review Process",
            description="Structured code review improves build efficiency.",
            size=ProcessProjectSize.small,
            ops_capacity_cost=2,
            target_function="engineering",
            bonus_type="build_efficiency",
            bonus_base=0.05,
            bonus_scale_factor=0.02,
            bonus_max=0.10,
            duration_turns=1,
            bonus_duration_turns=12,
        ),
        ProcessProject(
            id="PP06",
            name="Discovery Playbook",
            description="Structured outbound process improves customer discovery rates.",
            size=ProcessProjectSize.medium,
            ops_capacity_cost=4,
            target_function="sales",
            bonus_type="discovery_bonus",
            bonus_base=0.10,
            bonus_scale_factor=0.05,
            bonus_max=0.25,
            duration_turns=2,
            bonus_duration_turns=16,
        ),
        # --- Tier 1: gated behind one tier-0 project each ---
        ProcessProject(
            id="PP07",
            name="Revenue Operations Suite",
            description="Advanced RevOps tooling on top of a streamlined sales process — "
                        "forecasting, routing, and deal desk. Requires PP01.",
            size=ProcessProjectSize.large,
            ops_capacity_cost=6,
            target_function="sales",
            bonus_type="conversion_rate",
            bonus_base=0.10,
            bonus_scale_factor=0.04,
            bonus_max=0.22,
            duration_turns=3,
            bonus_duration_turns=16,
            permanent_floor_fraction=0.25,
            prerequisites=["PP01"],
        ),
        ProcessProject(
            id="PP08",
            name="Test Automation Platform",
            description="Full regression + integration test automation on a mature CI/CD pipeline, "
                        "sharply cutting build-introduced bugs. Requires PP02.",
            size=ProcessProjectSize.large,
            ops_capacity_cost=6,
            target_function="engineering",
            bonus_type="bug_rate_reduction",
            bonus_base=0.15,
            bonus_scale_factor=0.05,
            bonus_max=0.30,
            duration_turns=3,
            bonus_duration_turns=20,
            permanent_floor_fraction=0.30,
            prerequisites=["PP02"],
        ),
        ProcessProject(
            id="PP09",
            name="Customer Success Intelligence",
            description="Predictive health scoring + playbook automation on top of support "
                        "automation, lifting CS effectiveness. Requires PP03.",
            size=ProcessProjectSize.large,
            ops_capacity_cost=6,
            target_function="support",
            bonus_type="health_delta_bonus",
            bonus_base=0.35,
            bonus_scale_factor=0.10,
            bonus_max=0.70,
            duration_turns=2,
            bonus_duration_turns=14,
            permanent_floor_fraction=0.30,
            prerequisites=["PP03"],
        ),
        # --- Tier 2: multi-parent capstone ---
        ProcessProject(
            id="PP10",
            name="Operational Excellence Program",
            description="Company-wide engineering excellence program built on the RevOps suite "
                        "and test automation platform — durable bug-rate reduction. "
                        "Requires PP07 and PP08.",
            size=ProcessProjectSize.large,
            ops_capacity_cost=6,
            target_function="engineering",
            bonus_type="bug_rate_reduction",
            bonus_base=0.20,
            bonus_scale_factor=0.06,
            bonus_max=0.40,
            duration_turns=4,
            bonus_duration_turns=24,
            permanent_floor_fraction=0.40,
            prerequisites=["PP08", "PP07"],
        ),
    ]
