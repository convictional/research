from datetime import datetime
from typing import Literal

from pydantic import BaseModel, model_validator


class ActivityEvent(BaseModel):
    event_id: str
    event_type: Literal["task", "meeting", "discussion", "decision", "comment"]
    title: str
    body: str
    created_at: datetime
    author: str | None
    parent_ids: list[str] = []


class StatedGoal(BaseModel):
    id: str
    title: str
    description: str
    source: Literal["convictional_seed"]


class Decision(BaseModel):
    id: str
    title: str
    description: str
    author_stated_goals: str | None     # free-text goals as the decision author wrote them
    options: list[dict]                 # {title, description}
    criteria: list[dict]                # {title, description}
    comments: list[dict]                # {user, created_at, text}
    created_at: datetime


class CandidateGoal(BaseModel):
    title: str
    description: str
    supporting_evidence: list[str]      # short quoted snippets
    source_event_ids: list[str]


class StatedGoalEvidence(BaseModel):
    goal_id: str
    supporting_event_ids: list[str]
    contradicting_event_ids: list[str]
    activity_support_score: float       # 0-1
    notes: str


# Fixed sentinel ID for the synthetic orphan goal — injected at render/viz time only,
# never persisted into step5_final_goal_set.pkl. All-zeros is not a valid uuid4 output
# so collision with any real CanonicalGoal is impossible.
ORPHAN_GOAL_ID = "00000000-0000-0000-0000-000000000000"


class CanonicalGoal(BaseModel):
    id: str                             # uuid4, assigned at consolidation; STABLE downstream
    title: str
    description: str
    is_stated: bool
    is_unstated: bool
    is_orphan: bool = False             # True only for the synthetic orphan goal (render/viz only)
    origin_stated_goal_ids: list[str]
    origin_unstated_candidate_ids: list[str]
    activity_support_score: float

    @model_validator(mode="after")
    def check_origin_exclusive(self) -> "CanonicalGoal":
        if self.is_orphan:
            if self.is_stated or self.is_unstated:
                raise ValueError("orphan goal must have is_stated=False and is_unstated=False")
            return self
        if self.is_stated == self.is_unstated:
            raise ValueError(
                "is_stated and is_unstated must be mutually exclusive — exactly one must be True"
            )
        return self


def make_orphan_goal() -> CanonicalGoal:
    """Return the synthetic orphan goal used to collect decisions with no goal connection.

    Injected at render and viz time only — never saved to step5_final_goal_set.pkl.
    """
    return CanonicalGoal(
        id=ORPHAN_GOAL_ID,
        title="Unattached / Miscellaneous Decisions",
        description=(
            "Decisions that the mapping did not connect to any organizational goal. "
            "Grouped here so they remain visible and countable rather than disappearing "
            "from the mapping. A large collection here signals coverage gaps in the goal set "
            "or the mapping approach."
        ),
        is_stated=False,
        is_unstated=False,
        is_orphan=True,
        origin_stated_goal_ids=[],
        origin_unstated_candidate_ids=[],
        activity_support_score=0.0,
    )


class GoalRelation(BaseModel):
    goal_a_id: str
    goal_b_id: str
    relation: Literal["synergy", "tension", "neutral"]
    label: str
    confidence: float


class AlignmentReport(BaseModel):
    relations: list[GoalRelation]
    summary: str


class FinalizedGoalSet(BaseModel):
    condition_name: Literal["unstated", "stated", "mixed"]
    goals: list[CanonicalGoal]
    alignment_report: AlignmentReport           # always produced (step 4 always runs)
    summary_markdown: str
    run_metadata: dict                          # {"steps_run": [1,2,3,4,5], "model_ids": {...}, ...}
