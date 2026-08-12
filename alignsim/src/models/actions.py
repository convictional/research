from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field

from alignsim.src.models.entities import QualityLevel


class BuildAction(BaseModel):
    """Allocate engineering capacity to build a feature toward a quality level."""

    action_type: Literal["build"] = "build"
    feature_id: str
    quality: QualityLevel
    capacity: int = Field(gt=0)


class FixBugsAction(BaseModel):
    """Allocate engineering capacity to fix bugs. If bug_id is None, targets highest severity first."""

    action_type: Literal["fix_bugs"] = "fix_bugs"
    bug_id: str | None = None
    capacity: int = Field(gt=0)


class InfrastructureAction(BaseModel):
    """Allocate engineering capacity to infrastructure work (reduces tech debt)."""

    action_type: Literal["infrastructure"] = "infrastructure"
    capacity: int = Field(gt=0)


class SellAction(BaseModel):
    """Allocate sales capacity to advance a customer through the pipeline."""

    action_type: Literal["sell"] = "sell"
    customer_id: str
    sell_action: Literal["outbound", "demo", "proposal", "negotiate"]
    capacity: int = Field(gt=0)
    proposed_deal_value: int | None = None


class DiscoverAction(BaseModel):
    """Allocate capacity to discover new customers by targeting shipped features."""

    action_type: Literal["discover"] = "discover"
    target_features: list[str] = Field(default_factory=list)
    capacity: int = Field(gt=0)


class SupportAction(BaseModel):
    """Allocate CS capacity to support an existing customer."""

    action_type: Literal["support"] = "support"
    customer_id: str
    support_action: Literal["onboard", "churn_intervention", "health_check"]
    capacity: int = Field(gt=0)


class MarketAction(BaseModel):
    """Build per-feature awareness via a channel (lagged, decaying stock).

    channel shapes the awareness-accrual profile (lag / spread / efficiency / budget cost).
    target_features lists which features to build awareness for; empty = broad across all
    shipped + in-progress features (mirrors discovery's default-to-shipped behaviour, but
    marketing may also build awareness for not-yet-shipped features). Awareness changes the
    QUALITY of revealed leads (warmer engagement + longer timeline), not the lead count.
    """

    action_type: Literal["market"] = "market"
    channel: Literal["content", "events", "outbound_campaign"]
    target_features: list[str] = Field(default_factory=list)
    capacity: int = Field(gt=0)


class HireAction(BaseModel):
    """Hire for a function. hiring_function pays the capacity cost; target_function receives
    the new headcount. Cross-function hires (hiring_function != target_function) take 2x
    longer and deliver 0.7x capacity (rounded to 3). Costs budget regardless."""

    action_type: Literal["hire"] = "hire"
    hiring_function: Literal["engineering", "sales", "cs", "marketing", "ops"]
    target_function: Literal["engineering", "sales", "cs", "marketing", "ops"]


class FireAction(BaseModel):
    """Release one headcount from a function. Costs severance (budget); capacity drops next
    turn. Cannot reduce a function below 0 capacity."""

    action_type: Literal["fire"] = "fire"
    function: Literal["engineering", "sales", "cs", "marketing", "ops"]


class SustainHireAction(BaseModel):
    """Continue an active hiring process. Must be submitted each turn during
    the active phase or the hire is cancelled."""

    action_type: Literal["sustain_hire"] = "sustain_hire"
    hire_id: str


class OpsProjectAction(BaseModel):
    """Allocate ops capacity to advance a process improvement project."""

    action_type: Literal["ops_project"] = "ops_project"
    project_id: str
    capacity: int = Field(gt=0)


class OpsProjectSupportAction(BaseModel):
    """Allocate target team capacity to support an in-progress ops project (change management)."""

    action_type: Literal["ops_project_support"] = "ops_project_support"
    project_id: str
    capacity: int = Field(gt=0)


AnalysisType = Literal[
    "conversion_funnel", "retention_efficiency", "awareness_attribution", "capacity_bottleneck",
]


class OpsAnalysisAction(BaseModel):
    """Ops runs a cross-functional analysis for a requesting team. Draws from the OPS pool.

    Requires a matching AnalysisScopeAction (same target_function + analysis_type) the SAME turn,
    else wasted (analysis_unmatched). Engine-computed from observable history only; the result is
    delivered to the requesting team's NEXT-turn observation. Ops itself is not a valid
    target_function (cross-functional only — no Ops self-dealing)."""

    action_type: Literal["ops_analysis"] = "ops_analysis"
    target_function: Literal["engineering", "sales", "cs", "marketing"]
    analysis_type: AnalysisType
    capacity: int = Field(gt=0)


class AnalysisScopeAction(BaseModel):
    """Requesting team co-invests capacity to scope an analysis. Draws from THAT team's pool.

    Must be matched same-turn by a matching OpsAnalysisAction (same target_function +
    analysis_type), else wasted (analysis_unmatched). Co-presence is the gate: one agent does
    both in C2; two agents coordinate via chat in C3 (that coordination is the benchmark signal)."""

    action_type: Literal["analysis_scope"] = "analysis_scope"
    target_function: Literal["engineering", "sales", "cs", "marketing"]
    analysis_type: AnalysisType
    capacity: int = Field(default=1, gt=0)


class MarketSupportAction(BaseModel):
    """Sales co-invests capacity in this turn's budget-channel marketing campaign to buy
    pipeline progression (same-turn, channel-matched). Draws from the SALES pool.

    Requires a matching MarketAction on the same channel to run the same turn (coordinated
    out-of-band — chat in C3, self in C2). If unmatched, the capacity is wasted (a
    market_support_unmatched event fires). content/events only; outbound is not a co-invest
    channel."""

    action_type: Literal["market_support"] = "market_support"
    channel: Literal["content", "events"]
    capacity: int = Field(gt=0)
    target_customer_id: str | None = None  # events-only: push one existing pipeline customer a stage


GameAction = Annotated[
    BuildAction | FixBugsAction | InfrastructureAction | SellAction | DiscoverAction | SupportAction | MarketAction | HireAction | FireAction | SustainHireAction | OpsProjectAction | OpsProjectSupportAction | MarketSupportAction | OpsAnalysisAction | AnalysisScopeAction,
    Field(discriminator="action_type"),
]


# Single source of truth (action_type -> class), derived from the GameAction union above so a new
# member auto-registers here. Import this anywhere action classes are needed; never redefine it.
ACTION_CLASSES: dict[str, type[BaseModel]] = {
    cls.model_fields["action_type"].default: cls for cls in get_args(get_args(GameAction)[0])
}


class TurnActions(BaseModel):
    """All actions submitted by the player(s) for a single turn."""

    turn: int = Field(ge=1)
    actions: list[GameAction] = Field(default_factory=list)
