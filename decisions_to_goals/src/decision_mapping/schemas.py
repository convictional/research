from typing import Literal

from pydantic import BaseModel


# ── Shared analysis types ─────────────────────────────────────────────────────

class MappingAssumption(BaseModel):
    text: str
    confidence: Literal["low", "medium", "high"]


class MappingAnalysis(BaseModel):
    decision_id: str
    analysis: str
    assumptions: list[MappingAssumption]


# ── DM: Direct Mapping ────────────────────────────────────────────────────────

class DMEntry(BaseModel):
    decision_id: str
    goal_id: str | None              # None = "no goal applies" (do not force a link)
    confidence: Literal["low", "medium", "high"]
    reasoning: str


class DMMapping(BaseModel):
    condition_name: Literal["unstated", "stated", "mixed"]
    entries: list[DMEntry]
    model_ids: dict                  # {"analysis": ..., "judgement": ...}


# ── DSM: Direct Score Mapping ─────────────────────────────────────────────────

class DSMScore(BaseModel):
    goal_id: str
    score: float                     # 0.0–1.0
    reasoning: str


class DSMEntry(BaseModel):
    decision_id: str
    scored_goals: list[DSMScore]     # may be empty, may be many; ONLY emit score >= 0.20


class DSMMapping(BaseModel):
    condition_name: Literal["unstated", "stated", "mixed"]
    score_threshold: float = 0.20    # cached in artifact for auditability
    entries: list[DSMEntry]
    model_ids: dict


# ── GM: Graph Map ─────────────────────────────────────────────────────────────

NodeKind = Literal["decision", "goal"]
RelationKind = Literal[
    "advances", "blocks", "informs", "depends_on",
    "synergizes_with", "tensions_with", "supersedes", "is_evidence_of",
]


class GMEdge(BaseModel):
    source_id: str
    source_kind: NodeKind
    target_id: str
    target_kind: NodeKind
    relation: RelationKind           # CLOSED vocabulary — do not allow open strings
    label: str                       # short human-readable
    confidence: Literal["low", "medium", "high"]
    reasoning: str


class GMMapping(BaseModel):
    condition_name: Literal["unstated", "stated", "mixed"]
    # Allowed edge directions: decision→goal, goal→decision, goal→goal.
    # NOTE: decision↔decision edges are NOT produced — the mapper analyzes one decision
    # per call and never has another decision's ID available. See gm_mapper._judge_one.
    edges: list[GMEdge]
    model_ids: dict
