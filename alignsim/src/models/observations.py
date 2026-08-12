from pydantic import BaseModel, Field


class CustomerPipelineStatus(BaseModel):
    """Status of a single customer in the pipeline, as seen by Sales."""

    customer_id: str
    size: int = 1
    stage: str
    engagement: str
    interest: str | None = None
    known_needs: list[str] = Field(default_factory=list)
    deal_value: int = 0
    timeline_remaining: int | None = None
    timeline_resets: int | None = None
    competitor_bidding: str | None = None
    min_sell_capacity: dict[str, int] = Field(default_factory=dict)  # sell_action -> min capacity
    last_proposed_price: int | None = None
    pricing_feedback: str | None = None


class DealEvent(BaseModel):
    """A deal win or loss event for this turn."""

    customer_id: str
    event_type: str  # "win" or "loss"
    deal_value: int | None = None
    reason: str | None = None
    lost_to: str | None = None


class FeatureProgressReport(BaseModel):
    """Status of a single feature, as seen by Product/Engineering."""

    feature_id: str
    name: str
    status: str
    progress: float
    capacity_invested: int = 0   # cumulative capacity applied toward current quality level
    capacity_needed: int = 0     # total capacity cost for current target quality level
    est_completion_turns: int | None = None
    blocked_by: str | None = None


class BugReport(BaseModel):
    """A bug event (injected or fixed) for this turn."""

    bug_id: str
    severity: str
    feature_id: str
    event_type: str  # "injected" or "fixed"
    affected_customers: list[str] = Field(default_factory=list)


class CustomerHealthReport(BaseModel):
    """Health status of a single customer, as seen by CS."""

    customer_id: str
    health: float
    health_trend: str  # "improving", "stable", "declining"
    cause: str | None = None
    onboarding_remaining: int | None = None
    expansion_signal: bool = False
    emergent_needs: list[str] = Field(default_factory=list)  # feature IDs, only once revealed by health_check
    churn_drivers: dict[str, float] | None = None  # only populated once churn_drivers_revealed


# --- Role-Specific Observations ---


class GlobalDashboard(BaseModel):
    """Information visible to all roles every turn."""

    turn: int
    mrr: int
    pipeline_value: int
    active_customers: int
    churn_this_turn: list[str] = Field(default_factory=list)  # customer IDs that churned
    churn_reasons: dict[str, str] = Field(default_factory=dict)  # customer_id -> reason
    new_leads_this_turn: list[str] = Field(default_factory=list)  # newly discovered customer IDs
    debt_level: str  # "low", "medium", "high", "critical"
    bug_backlog: dict[str, int] = Field(default_factory=dict)  # severity -> count
    runway_turns: float
    capacity_available: int
    eng_capacity: int = 0
    sales_capacity: int = 0
    support_capacity: int = 0
    marketing_capacity: int = 0
    ops_capacity: int = 0
    sales_momentum: float = 0.0
    capacity_used_last_turn: int = 0
    pending_hires: list[dict] = Field(default_factory=list)  # {target_function, hiring_function, turns_remaining, capacity_on_arrival, is_cross_function}


class SalesObservation(BaseModel):
    """Information visible only to the Head of Sales role."""

    pipeline: list[CustomerPipelineStatus] = Field(default_factory=list)
    deals_this_turn: list[DealEvent] = Field(default_factory=list)
    competitor_pricing_events: list[str] = Field(default_factory=list)
    pipeline_summary: str = ""  # e.g. "8 prospects, 3 in-deal, est_close_value=280K"


class ProductEngObservation(BaseModel):
    """Information visible only to Product/Engineering roles."""

    features: list[FeatureProgressReport] = Field(default_factory=list)
    bugs_this_turn: list[BugReport] = Field(default_factory=list)
    debt_delta: float = 0.0
    infrastructure_impact: str | None = None
    feature_requests_from_pipeline: dict[str, int] = Field(default_factory=dict)
    # ^ feature_id -> number of prospects requesting it


class CSObservation(BaseModel):
    """Information visible only to the Head of CS role."""

    customer_health: list[CustomerHealthReport] = Field(default_factory=list)
    churned_this_turn: list[str] = Field(default_factory=list)
    at_risk: list[str] = Field(default_factory=list)  # customer IDs with health < 5
    avg_customer_health: float = 0.0
    onboarding_in_progress: list[str] = Field(default_factory=list)


class OpsObservation(BaseModel):
    """Information visible about the Ops function."""

    available_projects: list[dict] = Field(default_factory=list)
    active_projects: list[dict] = Field(default_factory=list)
    completed_projects: list[dict] = Field(default_factory=list)
    active_bonuses: list[dict] = Field(default_factory=list)


# --- Combined Observation ---


class TurnObservation(BaseModel):
    """Complete observation for a turn. In Condition 1, the LLM sees all components."""

    global_dashboard: GlobalDashboard
    sales: SalesObservation
    product_eng: ProductEngObservation
    cs: CSObservation
    ops: OpsObservation = Field(default_factory=OpsObservation)
    # Cross-functional analysis results delivered this turn, keyed by the REQUESTING agent-function
    # name (e.g. "sales", "support"). Full god-view map here; C3 routes per-function so a role only
    # ever sees its own results. Each result dict self-describes via its "target_function" key.
    analyses_received: dict[str, list[dict]] = Field(default_factory=dict)
