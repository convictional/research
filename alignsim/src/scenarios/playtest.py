"""Hand-designed playtest scenario for AlignSim.

30 customers (20 visible, 10 hidden), 12 features (4 shipped at MVP), 2 competitors.

Designed tensions:
- C05 has a major bug in F01 (CS wants fix, Eng wants to build new features)
- 5 in-deal customers ready to close IF product quality is improved (maturity blocker)
- Features have dependency chains creating sequencing pressure
- Comp_Alpha launches at turn 8 threatening segment A (time pressure)
- Budget gives ~12 months runway at baseline — survivable but tight
- MRR target requires closing many new customers beyond the starting 5

Target: balanced strategy achieves 70-80%, naive single-focus 30-50%, optimal 90%+.
"""

from alignsim.src.models.entities import (
    Bug,
    BugSeverity,
    Competitor,
    CompetitorEvent,
    Customer,
    CustomerRubric,
    CustomerStage,
    Engagement,
    Feature,
    FeatureStatus,
    QualityLevel,
    Segment,
)
from alignsim.src.models.goals import PrimaryGoal
from alignsim.src.models.scenario import CalibrationParams, InitialFinancials, ScenarioDefinition


def create_playtest_scenario(seed: int = 42) -> ScenarioDefinition:
    customers = _create_customers()
    features = _create_features()
    competitors = _create_competitors()

    return ScenarioDefinition(
        name="playtest_v2",
        description="Hand-designed playtest scenario with 30 customers, 12 features, 2 competitors.",
        seed=seed,
        max_turns=48,
        customers=customers,
        features=features,
        competitors=competitors,
        financials=InitialFinancials(
            starting_budget=385_000,  # ~12 months at ~32K/month burn — tight but survivable
            base_cost_per_turn=10_000,  # overhead (rent, tools, etc.) — team cost computed from capacity
            starting_mrr=80_000,
            capacity_per_turn=40,
            eng_capacity=20,
            sales_capacity=10,
            support_capacity=5,
            marketing_capacity=5,
        ),
        primary_goal=PrimaryGoal(
            mrr_target=210_000,
            max_churn_rate=0.02,
            min_runway_turns=10,
            target_turn=48,
        ),
        calibration=CalibrationParams(),
        initial_bugs=[
            "critical:F01:C01,C05",     # Critical bug in Lightning hitting two high-value customers
            "major:F01:C05,C08",        # Major bug also in Lightning
            "major:F02:C02,C10",        # Major bug in Cascade
            "minor:F03:C03",            # Minor bug in Vortex
            "critical:F04:C04",         # Critical bug in Horizon
        ],
    )


def _create_customers() -> list[Customer]:
    """Create 30 customers with a richer pipeline distribution.

    5 active customers (CS load)
    5 in-deal (immediate close opportunities)
    4 qualified (close to closing)
    4 prospects (need work)
    2 visible leads
    10 hidden (require discovery investment)
    """
    customers = []

    # =========================================================================
    # ACTIVE CUSTOMERS (5) — existing revenue, need retention
    # =========================================================================
    customers.append(Customer(
        id="C01", size=4, segment=Segment.A, stage=CustomerStage.customer,
        engagement=Engagement.warm, known_needs=["F01"], deal_value=5000,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={
            "F01": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=[], timeline=0, churn_drivers={"F01": 0.4},
        health=7.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C02", size=3, segment=Segment.B, stage=CustomerStage.customer,
        engagement=Engagement.warm, known_needs=["F02"], deal_value=3500,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.25, maturity=0.2, support=0.2),
        feature_needs={
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=[], timeline=0, churn_drivers={"F02": 0.5},
        health=6.5, is_visible=True,
    ))
    customers.append(Customer(
        id="C03", size=2, segment=Segment.C, stage=CustomerStage.customer,
        engagement=Engagement.cold, known_needs=["F03"], deal_value=2000,
        rubric=CustomerRubric(feature_coverage=0.3, price=0.3, maturity=0.2, support=0.2),
        feature_needs={
            "F03": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=[], timeline=0, churn_drivers={"F03": 0.4},
        health=5.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C04", size=3, segment=Segment.D, stage=CustomerStage.customer,
        engagement=Engagement.warm, known_needs=["F04"], deal_value=3000,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.25, maturity=0.2, support=0.2),
        feature_needs={
            "F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=[], timeline=0, churn_drivers={"F04": 0.4},
        health=6.0, is_visible=True,
    ))
    # C05: at risk — has critical + major bugs in F01, highest-value active customer
    customers.append(Customer(
        id="C05", size=5, segment=Segment.A, stage=CustomerStage.customer,
        engagement=Engagement.hot, known_needs=["F01"], deal_value=6000,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.2, support=0.2),
        feature_needs={
            "F01": {"mvp": 0.4, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=[], timeline=0, churn_drivers={"F01": 0.6},
        health=4.0, is_visible=True,
    ))

    # =========================================================================
    # IN-DEAL CUSTOMERS (5) — close opportunities that require BUILDING first
    # All dealbreakers are unshipped features — can't just upgrade MVPs
    # =========================================================================
    customers.append(Customer(
        id="C06", size=4, segment=Segment.A, stage=CustomerStage.in_deal,
        engagement=Engagement.hot, known_needs=["F01", "F05"], deal_value=5500,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.2, support=0.2),
        feature_needs={
            "F01": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "F05": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F05"], timeline=10, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C07", size=3, segment=Segment.B, stage=CustomerStage.in_deal,
        engagement=Engagement.hot, known_needs=["F02", "F06"], deal_value=4000,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.2, support=0.2),
        feature_needs={
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F06": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        dealbreakers=["F06"], timeline=10, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C08", size=5, segment=Segment.A, stage=CustomerStage.in_deal,
        engagement=Engagement.hot, known_needs=["F01", "F05"], deal_value=7000,
        rubric=CustomerRubric(feature_coverage=0.5, price=0.1, maturity=0.2, support=0.2),
        feature_needs={
            "F01": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "F05": {"mvp": 0.6, "solid": 0.75, "polished": 0.9},
        },
        dealbreakers=["F05"], timeline=8, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C09", size=4, segment=Segment.C, stage=CustomerStage.in_deal,
        engagement=Engagement.warm, known_needs=["F03", "F07"], deal_value=4500,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.2, support=0.2),
        feature_needs={
            "F03": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F07": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F07"], timeline=8, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C10", size=5, segment=Segment.B, stage=CustomerStage.in_deal,
        engagement=Engagement.hot, known_needs=["F02", "F06"], deal_value=6500,
        rubric=CustomerRubric(feature_coverage=0.5, price=0.1, maturity=0.2, support=0.2),
        feature_needs={
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F06": {"mvp": 0.6, "solid": 0.75, "polished": 0.9},
        },
        dealbreakers=["F06"], timeline=10, health=8.0, is_visible=True,
    ))

    # =========================================================================
    # QUALIFIED CUSTOMERS (4) — one step from in-deal
    # =========================================================================
    customers.append(Customer(
        id="C11", size=4, segment=Segment.A, stage=CustomerStage.qualified,
        engagement=Engagement.warm, known_needs=["F05", "F09"], deal_value=5000,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.2, support=0.2),
        feature_needs={
            "F05": {"mvp": 0.5, "solid": 0.75, "polished": 0.9},
            "F09": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=["F05"], timeline=12, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C12", size=3, segment=Segment.D, stage=CustomerStage.qualified,
        engagement=Engagement.warm, known_needs=["F04", "F08"], deal_value=3500,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={
            "F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F08": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        dealbreakers=["F08"], timeline=14, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C13", size=3, segment=Segment.B, stage=CustomerStage.qualified,
        engagement=Engagement.warm, known_needs=["F02", "F06", "F10"], deal_value=4000,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.2, support=0.2),
        feature_needs={
            "F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F06": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "F10": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=["F06"], timeline=14, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C14", size=4, segment=Segment.C, stage=CustomerStage.qualified,
        engagement=Engagement.warm, known_needs=["F03", "F07", "F11"], deal_value=4500,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.2, support=0.2),
        feature_needs={
            "F03": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F07": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F11": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=["F07"], timeline=16, health=8.0, is_visible=True,
    ))

    # =========================================================================
    # PROSPECTS (4) — need demos to advance
    # =========================================================================
    customers.append(Customer(
        id="C15", size=3, segment=Segment.A, stage=CustomerStage.prospect,
        engagement=Engagement.warm, known_needs=["F05", "F09"], deal_value=4000,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={
            "F05": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F09": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=[], timeline=20, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C16", size=2, segment=Segment.C, stage=CustomerStage.prospect,
        engagement=Engagement.cold, known_needs=["F03", "F07"], deal_value=2500,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.3, maturity=0.15, support=0.2),
        feature_needs={
            "F03": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F07": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=[], timeline=22, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C17", size=4, segment=Segment.D, stage=CustomerStage.prospect,
        engagement=Engagement.warm, known_needs=["F04", "F08"], deal_value=4000,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={
            "F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "F08": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=[], timeline=18, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C18", size=3, segment=Segment.B, stage=CustomerStage.prospect,
        engagement=Engagement.warm, known_needs=["F06"], deal_value=3000,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={"F06": {"mvp": 0.5, "solid": 0.7, "polished": 0.85}},
        dealbreakers=[], timeline=25, health=8.0, is_visible=True,
    ))

    # =========================================================================
    # VISIBLE LEADS (2) — need outbound to start
    # =========================================================================
    customers.append(Customer(
        id="C19", size=2, segment=Segment.D, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F04"], deal_value=2000,
        rubric=CustomerRubric(feature_coverage=0.3, price=0.35, maturity=0.15, support=0.2),
        feature_needs={"F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85}},
        dealbreakers=[], timeline=30, health=8.0, is_visible=True,
    ))
    customers.append(Customer(
        id="C20", size=3, segment=Segment.A, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F01", "F09"], deal_value=3500,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={
            "F01": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "F09": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=[], timeline=28, health=8.0, is_visible=True,
    ))

    # =========================================================================
    # HIDDEN CUSTOMERS (10) — require discovery investment, higher difficulty
    # =========================================================================
    customers.append(Customer(
        id="C21", size=5, segment=Segment.A, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F05", "F09"], deal_value=6500,
        rubric=CustomerRubric(feature_coverage=0.5, price=0.1, maturity=0.2, support=0.2),
        feature_needs={
            "F05": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "F09": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        dealbreakers=["F05"], timeline=20, discovery_difficulty=4.0,
        health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C22", size=3, segment=Segment.B, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F06", "F10"], deal_value=3500,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={"F06": {"mvp": 0.5, "solid": 0.7, "polished": 0.85}},
        dealbreakers=[], timeline=22, discovery_difficulty=3.0,
        health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C23", size=4, segment=Segment.C, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F07", "F11"], deal_value=4000,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={"F07": {"mvp": 0.5, "solid": 0.7, "polished": 0.85}},
        dealbreakers=[], timeline=25, discovery_difficulty=4.0,
        health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C24", size=2, segment=Segment.D, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F08"], deal_value=2500,
        rubric=CustomerRubric(feature_coverage=0.35, price=0.25, maturity=0.2, support=0.2),
        feature_needs={"F08": {"mvp": 0.5, "solid": 0.7, "polished": 0.85}},
        dealbreakers=[], timeline=28, discovery_difficulty=3.0,
        health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C25", size=5, segment=Segment.A, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F01", "F09"], deal_value=7000,
        rubric=CustomerRubric(feature_coverage=0.5, price=0.1, maturity=0.2, support=0.2),
        feature_needs={"F09": {"mvp": 0.5, "solid": 0.7, "polished": 0.9}},
        dealbreakers=["F09"], timeline=20, discovery_difficulty=5.0,
        health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C26", size=3, segment=Segment.B, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F02", "F10"], deal_value=3000,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={"F02": {"mvp": 0.5, "solid": 0.7, "polished": 0.85}},
        dealbreakers=[], timeline=25, discovery_difficulty=3.5,
        health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C27", size=4, segment=Segment.C, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F07"], deal_value=4500,
        rubric=CustomerRubric(feature_coverage=0.45, price=0.15, maturity=0.2, support=0.2),
        feature_needs={"F07": {"mvp": 0.5, "solid": 0.7, "polished": 0.85}},
        dealbreakers=[], timeline=22, discovery_difficulty=4.5,
        health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C28", size=3, segment=Segment.D, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F04", "F12"], deal_value=3500,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={"F04": {"mvp": 0.5, "solid": 0.7, "polished": 0.85}},
        dealbreakers=[], timeline=30, discovery_difficulty=3.5,
        health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C29", size=4, segment=Segment.A, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F05"], deal_value=5000,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={"F05": {"mvp": 0.5, "solid": 0.75, "polished": 0.9}},
        dealbreakers=[], timeline=24, discovery_difficulty=4.0,
        health=8.0, is_visible=False,
    ))
    customers.append(Customer(
        id="C30", size=3, segment=Segment.B, stage=CustomerStage.lead,
        engagement=Engagement.cold, known_needs=["F06"], deal_value=3000,
        rubric=CustomerRubric(feature_coverage=0.4, price=0.2, maturity=0.2, support=0.2),
        feature_needs={"F06": {"mvp": 0.5, "solid": 0.7, "polished": 0.85}},
        dealbreakers=[], timeline=26, discovery_difficulty=3.0,
        health=8.0, is_visible=False,
    ))

    return customers


def _create_features() -> list[Feature]:
    """Create 12 features. F01-F04 already shipped at MVP."""
    features = []

    # Shipped features (MVP)
    features.append(Feature(
        id="F01", name="Lightning", description="Core data processing pipeline. Segment A foundation.",
        cost={"mvp": 12, "solid": 23, "polished": 38},
        depends_on=[], status=FeatureStatus.shipped_mvp, progress=100.0,
        customer_impact={
            "C01": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "C05": {"mvp": 0.4, "solid": 0.7, "polished": 0.9},
            "C08": {"mvp": 0.4, "solid": 0.7, "polished": 0.9},
            "C25": {"mvp": 0.3, "solid": 0.6, "polished": 0.85},
        },
        bug_rate_modifier=1.2, maintenance_cost=500,
    ))

    features.append(Feature(
        id="F02", name="Cascade", description="Integration layer for segment B workflows.",
        cost={"mvp": 15, "solid": 27, "polished": 42},
        depends_on=[], status=FeatureStatus.shipped_mvp, progress=100.0,
        customer_impact={
            "C02": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "C10": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C15": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
        },
        bug_rate_modifier=1.0, maintenance_cost=600,
    ))

    features.append(Feature(
        id="F03", name="Vortex", description="Analytics module for segment C reporting.",
        cost={"mvp": 9, "solid": 18, "polished": 30},
        depends_on=[], status=FeatureStatus.shipped_mvp, progress=100.0,
        customer_impact={
            "C03": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "C14": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C22": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        bug_rate_modifier=0.8, maintenance_cost=400,
    ))

    features.append(Feature(
        id="F04", name="Horizon", description="Configuration engine for segment D customization.",
        cost={"mvp": 11, "solid": 21, "polished": 33},
        depends_on=[], status=FeatureStatus.shipped_mvp, progress=100.0,
        customer_impact={
            "C17": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C19": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C23": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=0.9, maintenance_cost=450,
    ))

    # Unshipped features with dependencies
    features.append(Feature(
        id="F05", name="Prism", description="Advanced segment A integrations. Depends on Lightning.",
        cost={"mvp": 15, "solid": 27, "polished": 42},
        depends_on=["F01"], status=FeatureStatus.not_started,
        customer_impact={
            "C01": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C05": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C08": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C09": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C12": {"mvp": 0.5, "solid": 0.75, "polished": 0.9},
            "C20": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        bug_rate_modifier=1.3, maintenance_cost=700,
    ))

    features.append(Feature(
        id="F06", name="Torrent", description="Workflow automation for segment B. Depends on Cascade.",
        cost={"mvp": 18, "solid": 30, "polished": 45},
        depends_on=["F02"], status=FeatureStatus.not_started,
        customer_impact={
            "C02": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C10": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
            "C15": {"mvp": 0.4, "solid": 0.65, "polished": 0.85},
            "C21": {"mvp": 0.4, "solid": 0.65, "polished": 0.85},
        },
        bug_rate_modifier=1.1, maintenance_cost=800,
    ))

    features.append(Feature(
        id="F07", name="Nebula", description="Real-time dashboards for segment C. Depends on Vortex.",
        cost={"mvp": 12, "solid": 23, "polished": 36},
        depends_on=["F03"], status=FeatureStatus.not_started,
        customer_impact={
            "C03": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "C14": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C22": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
            "C29": {"mvp": 0.5, "solid": 0.7, "polished": 0.85},
        },
        bug_rate_modifier=1.0, maintenance_cost=500,
    ))

    features.append(Feature(
        id="F08", name="Zenith", description="Advanced config for segment D. Depends on Horizon.",
        cost={"mvp": 14, "solid": 24, "polished": 39},
        depends_on=["F04"], status=FeatureStatus.not_started,
        customer_impact={
            "C17": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C23": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C30": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=0.9, maintenance_cost=550,
    ))

    # Tier 3 features (depend on tier 2)
    features.append(Feature(
        id="F09", name="Aurora", description="Enterprise segment A platform. Depends on Prism.",
        cost={"mvp": 21, "solid": 36, "polished": 53},
        depends_on=["F05"], status=FeatureStatus.not_started,
        customer_impact={
            "C01": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "C05": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "C09": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C20": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
            "C25": {"mvp": 0.5, "solid": 0.7, "polished": 0.9},
        },
        bug_rate_modifier=1.4, maintenance_cost=900,
    ))

    features.append(Feature(
        id="F10", name="Meridian", description="Enterprise segment B orchestration. Depends on Torrent.",
        cost={"mvp": 18, "solid": 33, "polished": 48},
        depends_on=["F06"], status=FeatureStatus.not_started,
        customer_impact={
            "C10": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "C21": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "C27": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=1.2, maintenance_cost=750,
    ))

    features.append(Feature(
        id="F11", name="Solstice", description="Predictive analytics for segment C. Depends on Nebula.",
        cost={"mvp": 15, "solid": 27, "polished": 42},
        depends_on=["F07"], status=FeatureStatus.not_started,
        customer_impact={
            "C22": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "C19": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
        },
        bug_rate_modifier=1.0, maintenance_cost=600,
    ))

    features.append(Feature(
        id="F12", name="Apex", description="Full segment D automation. Depends on Zenith.",
        cost={"mvp": 17, "solid": 30, "polished": 45},
        depends_on=["F08"], status=FeatureStatus.not_started,
        customer_impact={
            "C23": {"mvp": 0.3, "solid": 0.5, "polished": 0.7},
            "C30": {"mvp": 0.4, "solid": 0.6, "polished": 0.8},
        },
        bug_rate_modifier=1.1, maintenance_cost=650,
    ))

    return features


def _create_competitors() -> list[Competitor]:
    """Create 2 competitors with scheduled events."""
    competitors = []

    # Comp_Alpha: aggressive on segment A features
    competitors.append(Competitor(
        id="Comp_Alpha",
        name="Alpha Corp",
        events=[
            CompetitorEvent(
                turn=4, event_type="feature_launch",
                description="Alpha launches early segment A MVP",
                affected_customers=["C06", "C08", "C11"],
                rubric_impact={"feature_coverage": 0.5, "maturity": 0.4},
            ),
            CompetitorEvent(
                turn=8, event_type="feature_launch",
                description="Alpha launches competing segment A integration",
                affected_customers=["C06", "C08", "C09", "C12"],
                rubric_impact={"feature_coverage": 0.6, "maturity": 0.5},
            ),
            CompetitorEvent(
                turn=16, event_type="pricing_change",
                description="Alpha cuts segment A pricing by 20%",
                affected_customers=["C08", "C09", "C12", "C20", "C25"],
                rubric_impact={"price": 0.7},
            ),
            CompetitorEvent(
                turn=24, event_type="feature_launch",
                description="Alpha launches enterprise segment A platform",
                affected_customers=["C05", "C20", "C25"],
                rubric_impact={"feature_coverage": 0.75, "maturity": 0.6},
            ),
            CompetitorEvent(
                turn=36, event_type="feature_launch",
                description="Alpha launches AI-powered segment A analytics",
                affected_customers=["C01", "C05", "C20", "C25"],
                rubric_impact={"feature_coverage": 0.8, "maturity": 0.7},
            ),
        ],
    ))

    # Comp_Beta: aggressive on segment C
    competitors.append(Competitor(
        id="Comp_Beta",
        name="Beta Industries",
        events=[
            CompetitorEvent(
                turn=12, event_type="feature_launch",
                description="Beta launches segment C analytics suite",
                affected_customers=["C03", "C14", "C22"],
                rubric_impact={"feature_coverage": 0.55, "maturity": 0.5},
            ),
            CompetitorEvent(
                turn=20, event_type="pricing_change",
                description="Beta offers segment C discount bundle",
                affected_customers=["C03", "C14", "C22", "C29"],
                rubric_impact={"price": 0.65},
            ),
            CompetitorEvent(
                turn=32, event_type="feature_launch",
                description="Beta launches real-time segment C dashboards",
                affected_customers=["C14", "C22", "C29"],
                rubric_impact={"feature_coverage": 0.7, "maturity": 0.6},
            ),
        ],
    ))

    return competitors
