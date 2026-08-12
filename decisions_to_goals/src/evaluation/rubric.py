"""Rubric definition and Pydantic schemas for Phase 3 evaluation.

Five dimensions × 1 point = 5 max. Each dimension is a binary 0/1 pass-fail
judgement; the overall is their sum (0–5).

NOTE: The judge scores a fixed-length research summary (the obfuscation layer
output), NOT the raw mapping artifact. The rubric dimensions are calibrated for
neutral prose, not for the structural properties of the underlying mapping schema.
"""
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


# ── Rubric anchor text ────────────────────────────────────────────────────────

RUBRIC_DIMENSIONS: dict[str, dict] = {
    "coverage": {
        "label": "Coverage",
        "question": "What percentage of the decisions in the corpus have a defensible goal connection?",
        "anchors": {
            0: "Fewer than ~60% of decisions have a defensible goal connection (or the mapping is largely empty/null).",
            1: "A clear majority (~60%+) of decisions have defensible goal connections, and stated connections are defensible.",
        },
    },
    "fidelity": {
        "label": "Fidelity",
        "question": "For the connections that ARE made, are they semantically correct?",
        "anchors": {
            0: "A substantial share of connections are semantically wrong — wrong goal, wrong direction, or hallucinated.",
            1: "The large majority of connections are semantically correct and specific; only minor errors at most.",
        },
    },
    "synthesis_quality": {
        "label": "Synthesis Quality",
        "question": (
            "Does the briefing demonstrate genuine synthesis — distinguishing strong from weak "
            "decision-to-goal connections, surfacing real tensions and gaps — rather than a flat, "
            "undifferentiated list? Insight and prioritization count; restating everything at the "
            "same weight does not."
        ),
        "anchors": {
            0: (
                "Flat or undifferentiated: treats all connections as equal, misses tensions and gaps, "
                "or offers no prioritization or insight beyond enumeration."
            ),
            1: (
                "Genuine synthesis: clearly separates strong from weak connections, surfaces meaningful "
                "tensions, synergies, and gaps, and conveys a coherent overall picture."
            ),
        },
    },
    "interpretability": {
        "label": "Interpretability",
        "question": (
            "Can an organizational goal owner (non-technical) read this briefing and understand "
            "which decisions relate to their goal and why? Length and repetition do NOT help — a "
            "padded, restated, or repetitive document is harder to navigate, not easier."
        ),
        "anchors": {
            0: "A non-technical goal owner could not readily understand which decisions relate to their goal and why.",
            1: "A non-technical goal owner could read it and understand the relevant decisions with minimal guidance.",
        },
    },
    "information_density": {
        "label": "Information Density",
        "question": (
            "Does each word earn its place? Penalizes BOTH excessive sparsity (starvation) "
            "AND excessive length (bloat). Length is NOT quality — repeated, restated, or filler "
            "content is a defect, not a strength."
        ),
        "anchors": {
            0: (
                "Density failure — entries are mostly empty/null, OR the text is padded: it repeats, "
                "restates, or pads content without adding new information. Score 0 if cutting words "
                "would lose no information."
            ),
            1: (
                "Tight and complete: every entry adds distinct information, with no repetition, "
                "restatement, or filler. Any length is justified purely by content."
            ),
        },
    },
}


def cell_id(condition: str, schema: str) -> str:
    return f"{condition}__{schema}"


def format_rubric_text() -> str:
    lines = ["## Evaluation Rubric (5 dimensions × 1 point = 5 max)\n"]
    for dim, info in RUBRIC_DIMENSIONS.items():
        lines.append(f"### {info['label']}")
        lines.append(f"*{info['question']}*\n")
        for score, anchor in sorted(info["anchors"].items()):
            lines.append(f"- **{score}/1**: {anchor}")
        lines.append("")
    lines.append("**Scoring reminder:** Assign a binary score of 0 or 1 for each dimension "
                 "(1 = the artifact clearly meets that dimension's bar, 0 = it does not).")
    lines.append("Your `self_reported_overall` should equal the sum of the five scores (0–5).")
    lines.append("**Length is not quality.** A longer briefing is not a better one. Repeated, restated, "
                 "or filler content adds nothing: it must lower Information Density and must NOT raise "
                 "any other dimension. Judge only the unique, substantive content.")
    return "\n".join(lines)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class JudgeScore(BaseModel):
    reasoning: str = Field(
        description="Concise justification — ~1–2 sentences per rubric dimension, under ~250 words total."
    )
    coverage: Annotated[int, Field(ge=0, le=1)]
    fidelity: Annotated[int, Field(ge=0, le=1)]
    synthesis_quality: Annotated[int, Field(ge=0, le=1)]
    interpretability: Annotated[int, Field(ge=0, le=1)]
    information_density: Annotated[int, Field(ge=0, le=1)]
    self_reported_overall: Annotated[int, Field(ge=0, le=5)]


class JudgeRun(BaseModel):
    cell_id: str                        # e.g. "mixed__gm" — bookkeeping only, never sent to LLM
    model_id: str
    role: Literal["strategy_analyst", "ops_reviewer", "skeptic"]
    temperature: float
    score: JudgeScore
    duration_seconds: float
    rendered_word_count: int            # word count of the summary passed to the judge


class CellAggregate(BaseModel):
    cell_id: str
    condition_name: Literal["unstated", "stated", "mixed"]
    schema_name: Literal["dm", "dsm", "gm"]    # internal — NOT shown to judges
    judge_runs: list[JudgeRun]
    trimmed_mean_overall: float         # drop high+low across 9 runs
    per_dimension_mean: dict            # {coverage: float, ...}
    inter_judge_variance: float
    model_decomposition: dict           # mean overall by model_id
    role_decomposition: dict            # mean overall by role


class CalibrationResult(BaseModel):
    ran_at: datetime
    # Check A: cross-schema length normalization (the core obfuscation-layer guard)
    summary_word_counts: dict[str, int] = {}     # {"dm": .., "dsm": .., "gm": ..}
    length_band_ok: bool = True
    max_pairwise_word_ratio: float = 1.0
    # Check B: padding-bias guard (retained from prior design, now run on summaries)
    real_trimmed_mean: float
    padded_trimmed_mean: float
    delta: float
    length_bias_ok: bool = True
    passed: bool
    threshold: float = 0.2
    warning_message: str | None = None
