from pydantic import BaseModel, Field

from alignsim.src.models.entities import Competitor, Customer, Feature, ProcessProject
from alignsim.src.models.goals import PrimaryGoal


class CustomerGeneratorConfig(BaseModel):
    """Configuration for procedural customer generation. Scenario-specific data only."""

    feature_segment_affinity: dict[str, dict[str, float]]
    rubric_archetypes: dict[str, dict[str, float]]
    segment_weights: dict[str, float]
    size_distributions: dict[str, dict[int, float]]
    deal_value_per_size: dict[str, int]
    deal_value_jitter: float = 0.15
    discovery_difficulty_range: dict[str, tuple[float, float]]
    max_candidates_per_discover: int = 6
    max_candidates_per_inbound: int = 3
    feature_needs_count: tuple[int, int] = (1, 3)
    timeline_range: dict[str, tuple[int, int]]
    close_threshold_mean: float = 0.75
    close_threshold_std: float = 0.05
    desired_price_discount_range: dict[str, tuple[float, float]] = Field(default_factory=dict)


class ChannelProfile(BaseModel):
    """Awareness-accrual profile for a marketing channel.

    Each channel buys a different awareness shape. The total awareness contribution of a
    market action is `capacity * efficiency`, split across the targeted features, scheduled
    to begin maturing after `lag` turns and spread evenly over `spread` turns
    (spread=1 = a single burst). budget_cost_per_capacity is the shared runway-budget cost
    per capacity unit spent on this channel (0 = capacity-only, no budget bite).
    """

    lag: int = Field(ge=0)
    spread: int = Field(ge=1)
    efficiency: float = Field(gt=0)
    budget_cost_per_capacity: int = Field(ge=0)


def _default_channel_profiles() -> dict[str, ChannelProfile]:
    # Conservative starting table (tune in review). events: expensive timed burst before a
    # discovery sprint; content: budgeted long-game durable awareness; outbound_campaign:
    # free-but-slow, concentrated.
    return {
        "events": ChannelProfile(lag=2, spread=1, efficiency=0.8, budget_cost_per_capacity=8000),
        "content": ChannelProfile(lag=8, spread=6, efficiency=0.5, budget_cost_per_capacity=3000),
        "outbound_campaign": ChannelProfile(lag=5, spread=3, efficiency=0.6, budget_cost_per_capacity=0),
    }


class CalibrationParams(BaseModel):
    """Tunable parameters that control game difficulty and dynamics."""

    # Pipeline conversion base rates (modified by rubric satisfaction)
    lead_to_prospect_rate: float = 0.20
    prospect_to_qualified_rate: float = 0.50
    qualified_to_in_deal_rate: float = 0.40
    in_deal_to_closed_rate: float = 0.25
    min_rubric_for_close: float = 0.75

    # Engineering velocity
    base_capacity: int = 40

    # Tech debt and bugs
    debt_per_10_mvp_units: float = 1.0
    debt_per_10_solid_units: float = 0.5
    debt_per_10_polished_units: float = 0.2
    debt_reduction_per_5_infra: float = 1.0
    bug_injection_multiplier: float = 0.3  # lambda = debt_level * this

    # Bug severity distribution
    bug_critical_pct: float = 0.20
    bug_major_pct: float = 0.40
    bug_minor_pct: float = 0.40

    # Customer health
    health_bug_critical_delta: float = -2.0
    health_bug_major_delta: float = -1.0
    health_bug_minor_delta: float = -0.5
    health_cs_attention_delta: float = 1.0
    churn_health_threshold: float = 3.0
    churn_consecutive_turns: int = 2
    expansion_health_threshold: float = 8.0
    expansion_consecutive_turns: int = 4
    expansion_deal_value_increase: float = 0.20

    # Sell action capacity (minimum = base_cost * customer.size)
    sell_base_cost_outbound: int = 1
    sell_base_cost_demo: int = 1
    sell_base_cost_proposal: int = 1
    sell_base_cost_negotiate: int = 1
    demo_extra_capacity_bonus: float = 0.08  # bonus = factor * ln(1 + extra_capacity_above_min)
    engagement_hot_threshold: float = 1.0    # hot if sell_cap >= threshold * customer.size
    engagement_warm_threshold: float = 0.4   # warm if sell_cap >= threshold * customer.size

    # Hiring capacity cost (from hiring_function pool)
    hire_capacity_cost: int = 3
    cross_hire_delay_multiplier: float = 2.0   # cross-function hire takes 2x as long
    cross_hire_capacity_factor: float = 0.7    # cross-function hire yields 0.7x capacity (→ 3)
    fire_severance_turns: int = 4              # severance = 4 × per-turn capacity cost of that role

    # Mythical Man-Month: diminishing returns on build capacity
    build_optimal_capacity: int = 12         # ideal crew size per feature per turn
    build_overallocation_alpha: float = 0.5  # exponent for diminishing returns above optimal
    build_max_progress_pct: float = 65.0     # max % of remaining cost completable per turn
    build_min_turns_factor: float = 0.15     # min_turns = ceil(remaining_cost * factor)

    # Sales momentum
    sales_momentum_per_close: float = 0.08       # momentum gained per deal closed
    sales_momentum_decay: float = 0.01            # natural decay per turn
    sales_momentum_marketing_factor: float = 0.005  # from lagged marketing effect
    sales_momentum_feature_factor: float = 0.01   # from each shipped feature
    sales_momentum_max: float = 0.40              # cap on momentum multiplier

    # Marketing
    marketing_lag_turns: int = 10
    base_inbound_rate: float = 0.5  # leads per turn without marketing
    marketing_effectiveness: float = 0.3  # additional leads per capacity unit (lagged)

    # Marketing awareness (keystone mechanic) — conservative defaults, tuned during review.
    # Per-feature awareness is a decaying stock built via channels; it changes the QUALITY of
    # revealed leads (engagement + timeline) and biases inbound toward hyped features, but
    # never changes the lead COUNT (that stays governed by compute_inbound_leads above).
    channel_profiles: dict[str, ChannelProfile] = Field(default_factory=_default_channel_profiles)
    awareness_decay: float = 0.10                 # fraction of each feature's stock lost per turn
    awareness_epsilon: float = 0.01               # stocks below this are dropped after decay
    awareness_warm_threshold: float = 1.5         # stock at/above this reveals a lead warm (not cold)
    awareness_hot_threshold: float = 4.0          # stock at/above this gates the hot-reveal roll
    awareness_hot_prob: float = 0.20              # P(hot) when a warm lead is also above hot threshold
    awareness_timeline_bonus_max: int = 6         # max extra timeline turns at full (hot-threshold) awareness
    inbound_awareness_bias: float = 2.0           # weight multiplier pulling inbound toward hyped features

    # Competitive radar (marketing-only passive signal) — conservative defaults.
    radar_lookahead_turns: int = 5                # how far ahead radar can sense competitor events
    radar_base_prob: float = 0.5                  # base detection prob, scaled by awareness on affected features
    radar_uncertainty_jitter: float = 0.15        # stochastic jitter on the detection roll

    # Sales-gated pipeline progression (Marketing<->Sales co-investment) — conservative defaults.
    # When Sales co-invests (market_support) in a budget channel's same-turn campaign, leads can
    # roll to advance one pipeline stage (capped at in_deal; closing still needs a real proposal).
    # p = clamp(base[ch] * (1 + collab_scale*log1p(min(m,s)) + budget_scale*log1p(budget_$K)), 0, max)
    progression_base_prob: dict[str, float] = Field(
        default_factory=lambda: {"content": 0.20, "events": 0.40}
    )
    progression_collab_scale: float = 0.35   # a: log1p(min(m, s)) multiplier (joint-commitment term)
    progression_budget_scale: float = 0.05   # b: log1p(budget_$K) multiplier (diminishing budget term)
    progression_max_prob: float = 0.75       # clamp on the per-stage roll probability

    # Ops process projects — permanent floor regime. The per-project permanent_floor_fraction
    # sets the structural shape (like bonus_max); this single global scale multiplies the whole
    # regime (0.0 = floors off / pure decay kill-switch, 1.0 = as-authored, >1 = stronger).
    permanent_floor_scale: float = 1.0

    # Ops cross-functional analysis (same-turn co-invest handshake). The Ops side pays a flat
    # capacity cost; the requesting team's scope side pays a small fixed co-invest.
    analysis_ops_capacity_cost: int = 2
    analysis_scope_capacity: int = 1

    # Team cost
    team_cost_per_capacity: int = 2500  # cost per capacity unit per turn (40 units = 100K/turn)

    # Hiring
    hire_budget_cost_multiplier: int = 2  # multiplier on capacity for budget cost
    hire_arrival_delay: int = 6  # turns until new hire arrives
    hire_onboarding_turns: int = 4  # turns at 50% capacity
    hire_capacity_bonus: int = 4  # capacity units once fully onboarded

    # Customer health (neglect, event-driven decay)
    health_neglect_base_decay: float = 0.1
    health_bug_fester_rate: float = 0.05
    health_onboarding_neglect_penalty: float = 0.3

    # Onboarding
    new_customer_onboarding_turns: int = 4
    new_customer_starting_health: float = 8.0

    # Emergent needs (CS keystone mechanic) — conservative defaults, tuned during review
    emergent_need_injection_rate: float = 0.10   # expected new needs per active customer per turn
    emergent_need_injection_floor: float = 0.3   # lambda floor so needs appear even with few customers
    emergent_need_grace_turns: int = 3           # turns after injection with no health impact
    emergent_need_expiry_turns: int = 5          # turns_unmet (past grace) before churn-driver conversion
    emergent_need_bleed_rate: float = 0.4        # health lost per turn per turns_unmet while unmet
    emergent_need_met_health_bonus: float = 1.0  # one-time health bonus when the need's feature ships
    emergent_need_churn_driver_weight: float = 0.5  # weight written to churn_drivers on expiry (informational)

    # CS verbs (baseline + specialty)
    onboard_health_bonus: float = 1.0            # extra health during onboarding window
    onboard_acceleration: int = 1                # extra onboarding_turns_remaining decrement
    churn_intervention_health_threshold: float = 4.0  # only fires below this health
    churn_intervention_success_prob: float = 0.6      # stochastic success chance
    churn_intervention_health_recovery: float = 3.0   # health restored on success
    churn_intervention_min_capacity: int = 2          # minimum capacity for the verb to fire
    cs_attention_log_factor: float = 0.8         # diminishing-returns curve factor (replaces min(cap,3))

    # Pricing negotiation
    max_close_probability: float = 0.70
    pricing_dead_zone: float = 0.05
    pricing_penalty_steepness: float = 4.0
    pricing_penalty_floor: float = 0.15
    pricing_bonus_steepness: float = 5.0
    pricing_bonus_cap: float = 1.35
    pricing_sandbag_factor: float = 0.08
    pricing_sandbag_jitter: float = 0.02
    pricing_competitor_event_lambda: float = 0.3
    pricing_competitor_offer_discount: float = 0.10
    pricing_competitor_offer_jitter: float = 0.05
    pricing_competitor_pressure_boost: float = 0.20
    pricing_competitor_assumed_satisfaction: float = 0.70


class InitialFinancials(BaseModel):
    """Starting financial conditions."""

    starting_budget: int
    base_cost_per_turn: int
    starting_mrr: int
    capacity_per_turn: int = 40
    eng_capacity: int = 20
    sales_capacity: int = 10
    support_capacity: int = 5
    marketing_capacity: int = 5
    ops_capacity: int = 0


class ScenarioDefinition(BaseModel):
    """Complete definition of a game scenario. Pure data, no logic."""

    name: str
    description: str = ""
    seed: int = 42
    max_turns: int = 48

    # Entities
    customers: list[Customer] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    competitors: list[Competitor] = Field(default_factory=list)
    process_projects: list[ProcessProject] = Field(default_factory=list)

    # Initial conditions
    financials: InitialFinancials
    initial_bugs: list[str] = Field(default_factory=list)  # bug descriptions for starting bugs

    # Goals
    primary_goal: PrimaryGoal = Field(default_factory=PrimaryGoal)

    # Calibration
    calibration: CalibrationParams = Field(default_factory=CalibrationParams)

    # Procedural customer generation (None = pure handwritten pool)
    generator_config: CustomerGeneratorConfig | None = None
