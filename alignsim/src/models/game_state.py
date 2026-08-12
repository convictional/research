from pydantic import BaseModel, Field

from alignsim.src.models.actions import GameAction
from alignsim.src.models.entities import (
    Bug,
    Competitor,
    Customer,
    EmergentNeed,
    Feature,
    PendingAwareness,
    ProcessProject,
)


class ResourcePool(BaseModel):
    """Financial and capacity resources with function-specific pools."""

    capacity_per_turn: int = 40  # total (sum of pools), kept for financial calculations
    eng_capacity: int = 20       # build, fix_bugs, infrastructure
    sales_capacity: int = 10     # sell, discover
    support_capacity: int = 5    # support
    marketing_capacity: int = 5  # market
    ops_capacity: int = 0        # ops_project (default 0; only set by scenarios with ops)
    budget: int = 0  # abstract currency
    runway_turns: float = 0.0
    base_cost_per_turn: int = 0  # fixed operating costs
    mrr: int = 0  # monthly recurring revenue


class TechDebt(BaseModel):
    """Technical debt state."""

    level: float = Field(ge=0, default=0.0)

    @property
    def category(self) -> str:
        if self.level < 5:
            return "low"
        elif self.level < 10:
            return "medium"
        elif self.level < 15:
            return "high"
        else:
            return "critical"


class ActiveProcessBonus(BaseModel):
    """A completed process project's bonus currently in effect.

    The effective bonus degrades toward a permanent floor:
    effective = floor + (peak - floor) * (turns_remaining / bonus_duration_turns),
    where floor = bonus_value * permanent_floor_fraction. With permanent_floor_fraction == 0
    this collapses to the original linear-to-zero decay. Use compute_effective_bonus() from
    ops_logic rather than bonus_value directly.
    """

    project_id: str
    bonus_type: str
    bonus_value: float           # peak value at full turns_remaining; spike decays toward the floor
    target_function: str
    turns_remaining: int         # countdown to expiry; drives spike degradation (pinned at 0 when floored)
    bonus_duration_turns: int    # total duration; needed for degradation fraction
    original_ops_capacity_cost: int  # ops cost per turn of the original project; used for maintenance cost
    permanent_floor_fraction: float = 0.0  # fraction of peak that never decays (already scaled by calibration)


class PendingHire(BaseModel):
    """A hire in progress, not yet active."""

    id: str                # short ID like "H1", "H2" for player reference
    target_function: str   # engineering, sales, cs, marketing, ops — who receives the headcount
    hiring_function: str   # who spent capacity to initiate the hire
    turns_remaining: int   # turns until the hire arrives
    onboarding_turns_remaining: int = 4  # turns at 50% capacity after arrival
    capacity_bonus: int = 4  # capacity units added once fully onboarded
    is_cross_function: bool = False  # True when hiring_function != target_function
    active_turns_required: int = 0   # first half of total delay — must sustain for this many turns
    active_turns_completed: int = 0  # how many active-phase sustain turns completed so far


class ActionRejection(BaseModel):
    """An action that was rejected by the validator."""

    action: GameAction
    reason: str


class TurnRecord(BaseModel):
    """Record of what happened during a single turn, for analysis."""

    turn: int
    actions_submitted: list[GameAction] = Field(default_factory=list)
    actions_valid: list[GameAction] = Field(default_factory=list)
    actions_rejected: list[ActionRejection] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)  # narrative log of what happened
    mrr: int = 0
    churn_count: int = 0
    bugs_injected: int = 0
    bugs_fixed: int = 0
    capacity_used: int = 0
    capacity_available: int = 0
    eng_capacity_used: int = 0
    sales_capacity_used: int = 0
    support_capacity_used: int = 0
    marketing_capacity_used: int = 0
    ops_capacity_used: int = 0
    runway_turns: float = 0.0
    budget: int = 0


class GameState(BaseModel):
    """Full mutable state of the game."""

    turn: int = 1
    max_turns: int = 48
    seed: int = 42
    game_over: bool = False
    game_over_reason: str | None = None

    # Entity dicts keyed by ID
    customers: dict[str, Customer] = Field(default_factory=dict)
    features: dict[str, Feature] = Field(default_factory=dict)
    competitors: dict[str, Competitor] = Field(default_factory=dict)

    # Resources
    resources: ResourcePool = Field(default_factory=ResourcePool)
    tech_debt: TechDebt = Field(default_factory=TechDebt)
    initial_tech_debt: float = 0.0  # debt level at game start; used by alignment scoring

    # Bugs
    bugs: list[Bug] = Field(default_factory=list)

    # Emergent customer needs (CS keystone mechanic)
    emergent_needs: list[EmergentNeed] = Field(default_factory=list)
    next_emergent_need_id: int = 1

    # Hiring
    pending_hires: list[PendingHire] = Field(default_factory=list)
    next_hire_id: int = 1

    # Ops system
    process_projects: dict[str, ProcessProject] = Field(default_factory=dict)
    active_process_bonuses: list[ActiveProcessBonus] = Field(default_factory=list)

    # Cross-functional analysis results: a 1-turn scratch buffer keyed by agent-function name
    # (e.g. "sales", "support"), cleared at the top of each resolve(). Each value is a list of
    # rich result dicts delivered to that function's NEXT-turn observation (requester only).
    pending_analyses: dict[str, list[dict]] = Field(default_factory=dict)

    # Sales momentum
    sales_momentum: float = Field(ge=0, default=0.0)
    total_customers_closed: int = 0

    # Marketing history (capacity invested per turn, for lagged effect)
    marketing_history: list[int] = Field(default_factory=list)

    # Marketing awareness (keystone mechanic): per-feature awareness stock + lagged pipeline.
    # awareness[feature_id] is a decaying stock; pending_awareness holds channel-lagged
    # increments awaiting maturation (mirrors pending_hires). Awareness changes the quality
    # of revealed customers (engagement + timeline), never the lead count.
    awareness: dict[str, float] = Field(default_factory=dict)
    pending_awareness: list[PendingAwareness] = Field(default_factory=list)

    # Churn tracking
    churn_history: list[int] = Field(default_factory=list)  # churned count per turn

    # Turn records
    turn_history: list[TurnRecord] = Field(default_factory=list)

    # Bug ID counter
    next_bug_id: int = 1

    # Generated customer ID counter
    next_generated_customer_id: int = 1
