from enum import Enum

from pydantic import BaseModel, Field


class CustomerStage(str, Enum):
    lead = "lead"
    prospect = "prospect"
    qualified = "qualified"
    in_deal = "in_deal"
    customer = "customer"
    churned = "churned"
    lost = "lost"


class Engagement(str, Enum):
    cold = "cold"
    warm = "warm"
    hot = "hot"


class Segment(str, Enum):
    # NOTE: renamed from A/B/C/D in v3.5. This is a hard break — any pickled GameState,
    # JSONL turn records, or DB rows written before this change cannot be deserialized.
    # Old condition 2 run data in results/ should be treated as invalidated.
    startup    = "startup"
    growth     = "growth"
    mid_market = "mid_market"
    enterprise = "enterprise"


class QualityLevel(str, Enum):
    mvp = "mvp"
    solid = "solid"
    polished = "polished"


class FeatureStatus(str, Enum):
    not_started = "not_started"
    in_progress = "in_progress"
    shipped_mvp = "shipped_mvp"
    shipped_solid = "shipped_solid"
    shipped_polished = "shipped_polished"


class BugSeverity(str, Enum):
    critical = "critical"
    major = "major"
    minor = "minor"


class ProcessProjectSize(str, Enum):
    small = "small"      # 2 ops capacity
    medium = "medium"    # 4 ops capacity
    large = "large"      # 6 ops capacity


class ProcessProjectStatus(str, Enum):
    available = "available"
    in_progress = "in_progress"
    completed = "completed"


# --- Sub-models ---


class CustomerRubric(BaseModel):
    """Weighted scorecard for customer evaluation. Weights sum to 1.0."""

    feature_coverage: float = Field(ge=0, le=1)
    price: float = Field(ge=0, le=1)
    maturity: float = Field(ge=0, le=1)
    support: float = Field(ge=0, le=1)


class CompetitorEvent(BaseModel):
    """A scheduled competitive event that fires at a specific turn."""

    turn: int = Field(ge=1)
    event_type: str  # e.g. "feature_launch", "pricing_change", "deal_win"
    description: str
    affected_customers: list[str] = Field(default_factory=list)
    rubric_impact: dict[str, float] = Field(default_factory=dict)


# --- Core Entities ---


class Customer(BaseModel):
    """A customer entity with visible stats, hidden stats, and mutable runtime state."""

    # Identity
    id: str  # e.g. "C01"

    # Visible stats (revealed when customer becomes known)
    size: int = Field(ge=1, le=5)
    segment: Segment
    stage: CustomerStage = CustomerStage.lead
    engagement: Engagement = Engagement.cold
    known_needs: list[str] = Field(default_factory=list)  # partial view of feature needs
    deal_value: int = Field(ge=0)  # revenue per turn if closed

    # Hidden stats (drive game outcomes, never directly revealed)
    rubric: CustomerRubric
    feature_needs: dict[str, dict[str, float]] = Field(default_factory=dict)
    # ^ feature_id -> {quality_level -> satisfaction_score}
    dealbreakers: list[str] = Field(default_factory=list)  # feature IDs
    timeline: int = Field(ge=0)  # turns until decision window closes
    churn_drivers: dict[str, float] = Field(default_factory=dict)
    # ^ feature_id -> weight (how much this feature's quality affects retention)
    discovery_difficulty: float = Field(ge=0.1, default=1.0)

    # Runtime mutable state
    health: float = Field(ge=0, le=10, default=8.0)
    health_history: list[float] = Field(default_factory=list)
    turns_below_churn_threshold: int = Field(default=0)
    turns_above_expansion_threshold: int = Field(default=0)
    onboarding_turns_remaining: int = Field(default=0)
    is_visible: bool = Field(default=False)
    turns_in_current_stage: int = Field(default=0)
    competitive_pressure: float = Field(ge=0, default=0.0)
    churn_drivers_revealed: bool = Field(default=False)  # flipped only by a CS health_check

    # Timeline mechanic — clock starts on first sell action
    timeline_active: bool = Field(default=False)
    timeline_original: int = Field(ge=0, default=0)
    timeline_resets: int = Field(default=0)

    # Per-customer close threshold (0 = use global calibration.min_rubric_for_close)
    close_threshold: float = Field(default=0.0)

    # Pricing negotiation
    desired_price_point: int = Field(default=0)
    last_proposed_price: int | None = Field(default=None)
    has_received_proposal: bool = Field(default=False)


class Feature(BaseModel):
    """A feature entity with visible stats and hidden stats."""

    # Visible stats
    id: str  # e.g. "F01"
    name: str  # e.g. "Lightning"
    description: str
    cost: dict[str, int]  # quality_level -> capacity units (e.g. {"mvp": 8, "solid": 15, "polished": 25})
    depends_on: list[str] = Field(default_factory=list)  # prerequisite feature IDs
    status: FeatureStatus = FeatureStatus.not_started
    progress: float = Field(ge=0, le=100, default=0.0)  # percentage toward current target
    current_target: QualityLevel | None = None
    turns_worked: int = Field(default=0)  # turns capacity applied toward current target

    # Hidden stats
    customer_impact: dict[str, dict[str, float]] = Field(default_factory=dict)
    # ^ customer_id -> {quality_level -> rubric_satisfaction_score}
    bug_rate_modifier: float = Field(ge=0, default=1.0)
    maintenance_cost: int = Field(ge=0, default=0)  # capacity cost per turn once shipped


class Competitor(BaseModel):
    """A competitor with a pre-defined event schedule."""

    id: str  # e.g. "Comp_Alpha"
    name: str
    events: list[CompetitorEvent] = Field(default_factory=list)


class Bug(BaseModel):
    """A bug affecting a shipped feature."""

    id: str  # e.g. "BUG_001"
    severity: BugSeverity
    feature_id: str  # which feature this bug is in
    turn_injected: int
    turns_unresolved: int = Field(default=0)
    affected_customers: list[str] = Field(default_factory=list)  # customer IDs impacted
    is_resolved: bool = Field(default=False)


class EmergentNeed(BaseModel):
    """A new feature need that develops on an active customer over time.

    Mirrors Bug, but is per-customer: the save/sacrifice decision and the
    churn-driver conversion on expiry are both per-customer. Hidden ground truth —
    revealed to CS only through a health_check action (the CS discovery gate).
    """

    id: str                       # e.g. "EN_001"
    customer_id: str              # single owning customer
    feature_id: str               # drawn from features NOT in customer.known_needs
    turn_injected: int
    turns_unmet: int = Field(default=0)   # ticks while past grace and not actively worked
    is_revealed: bool = Field(default=False)  # flipped only by health_check
    is_met: bool = Field(default=False)
    is_expired: bool = Field(default=False)   # converted to churn driver


class PendingAwareness(BaseModel):
    """A scheduled marketing-awareness increment awaiting maturation.

    Mirrors PendingHire: marketing spend on a channel is lagged (channel.lag turns) and
    spread (channel.spread turns), so each market action schedules several of these. When
    the game turn reaches land_turn, `amount` is added to awareness[feature_id].
    """

    land_turn: int       # turn at which this increment matures into the awareness stock
    feature_id: str      # which feature's awareness this builds
    amount: float        # awareness units added on maturation


class ProcessProject(BaseModel):
    """A process improvement project that Ops can execute to boost another team."""

    id: str                                     # e.g. "PP01"
    name: str                                   # e.g. "Sales Process Optimization"
    description: str
    size: ProcessProjectSize
    ops_capacity_cost: int                      # 2, 4, or 6 matching size
    target_function: str                        # "engineering", "sales", "support", "marketing"
    bonus_type: str                             # key mapping to modifier in logic modules
    bonus_base: float                           # base bonus at zero target team investment
    bonus_scale_factor: float                   # how target team capacity amplifies the bonus
    bonus_max: float                            # hard cap on the bonus
    duration_turns: int                         # turns of ops work to complete
    bonus_duration_turns: int = 12              # how long bonus lasts after completion
    permanent_floor_fraction: float = 0.0       # fraction of peak bonus that never decays (0.0 = pure decay)
    prerequisites: list[str] = Field(default_factory=list)  # project ids that must be completed first (tech-tree DAG)
    status: ProcessProjectStatus = ProcessProjectStatus.available
    progress_turns: int = 0                     # turns of ops work completed
    target_team_capacity_invested: int = 0      # cumulative capacity from target team
    completed_turn: int | None = None           # turn when completed
