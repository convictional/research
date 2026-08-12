from pydantic import BaseModel, Field


class RatedReport(BaseModel):
    id: str
    question: str
    community_relation: str
    variant: str
    research_output: str
    quality_score: int = Field(..., ge=0, le=3)


class ScaleAnchor(BaseModel):
    score: int = Field(..., ge=0, le=3)
    description: str


class RubricDimension(BaseModel):
    name: str
    description: str
    weight: float = Field(..., ge=0, le=1)
    anchors: list[ScaleAnchor]


class DiscoveredRubric(BaseModel):
    dimensions: list[RubricDimension]
    version: int = 1
    general_notes: str = ""


# LLM response models for rubric discovery

class ScoringPattern(BaseModel):
    dimension_name: str = Field(..., description="Name of the quality dimension identified")
    description: str = Field(..., description="What this dimension measures")
    high_score_indicators: list[str] = Field(..., description="What the expert looks for in high-scoring reports")
    low_score_indicators: list[str] = Field(..., description="What characterizes low-scoring reports")


class BatchAnalysis(BaseModel):
    patterns: list[ScoringPattern] = Field(..., description="Quality dimensions identified from this batch")
    observations: str = Field(..., description="General observations about the expert's scoring approach")


class SynthesizedRubric(BaseModel):
    dimensions: list[RubricDimension] = Field(..., description="Consolidated quality dimensions with 0-3 scale anchors")
    general_notes: str = Field(..., description="Overall notes about the scoring approach")


# LLM response model for Q-A alignment gate

class QAAlignmentScore(BaseModel):
    alignment_score: int = Field(..., ge=0, le=3, description="How directly the report answers the question (0-3)")
    justification: str = Field(..., description="Why this alignment score was assigned")


# LLM response model for decoupled critic pass

class ReportCritique(BaseModel):
    question_asks_for: str = Field(..., description="One sentence: what specific information does the question request?")
    report_actually_provides: str = Field(..., description="One sentence: what does the report actually deliver?")
    alignment_gap: str = Field(..., description="What is missing or mismatched between what was asked and what was provided? Say 'None' only if perfect alignment.")
    weaknesses: list[str] = Field(..., description="List every weakness you can find: missing information, wrong focus, weak evidence, generic advice, tangential content, lack of depth, etc.")
    strengths: list[str] = Field(..., description="List the report's genuine strengths.")
    severity: str = Field(..., description="Overall severity: 'critical' (fundamentally fails), 'significant' (major gaps), 'moderate' (adequate with issues), or 'minor' (strong with small issues)")


# LLM response model for claim analysis (Trial 9)

class ClaimDetail(BaseModel):
    claim: str = Field(..., description="The extracted claim from the report")
    is_specific: bool = Field(..., description="True if the claim includes concrete details (names, dates, numbers, examples)")
    is_hedged: bool = Field(..., description="True if the claim uses hedging language (may, might, could, possibly, generally)")
    has_citation: bool = Field(..., description="True if the claim references a specific source or evidence")
    is_relevant: bool = Field(..., description="True if the claim directly addresses the research question")


class ClaimAnalysis(BaseModel):
    claims: list[ClaimDetail] = Field(..., description="10-15 key claims extracted from the report")
    total_claims: int = Field(..., description="Total number of substantive claims in the report")
    internal_inconsistencies: list[str] = Field(default_factory=list, description="Any claims that contradict each other")
    summary: str = Field(..., description="Brief assessment of overall claim quality: specificity, evidence support, and relevance")


# LLM response models for pointwise scoring

class DimensionScore(BaseModel):
    dimension: str = Field(..., description="Name of the rubric dimension")
    score: int = Field(..., ge=0, le=3, description="Score for this dimension (0-3)")
    justification: str = Field(..., description="Why this score was assigned")


class PointwiseScore(BaseModel):
    dimension_scores: list[DimensionScore] = Field(..., description="Scores per rubric dimension")
    quality_score: int = Field(..., ge=0, le=3, description="Overall quality score (0-3). Must be consistent with the critique provided.")
    overall_justification: str = Field(..., description="Overall justification referencing the critique findings")


# Evaluation models

class EvaluationResult(BaseModel):
    spearman_correlation: float
    spearman_continuous: float | None = None
    mae: float
    exact_match_rate: float
    adjacent_match_rate: float
    per_score_accuracy: dict[str, float]
    n_samples: int
    split: str


class ScoredReport(BaseModel):
    report: RatedReport
    predicted_score: PointwiseScore
    continuous_score: float | None = None


# RAG verification models (Trial 11)

class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    score: float


class ClaimVerification(BaseModel):
    claim: str
    verdict: str = Field(..., description="One of: supported, partially_supported, unsupported, no_evidence_found")
    evidence_summary: str = Field(..., description="Brief explanation of what evidence was found")
    confidence: float = Field(..., ge=0.0, le=1.0)


class ClaimVerificationRollup(BaseModel):
    total_claims: int
    supported: int
    partially_supported: int
    unsupported: int
    no_evidence_found: int
    avg_confidence: float
    details: list[ClaimVerification]


class FormatAssessment(BaseModel):
    structure_score: int = Field(..., ge=0, le=3, description="Quality of structure and organization")
    length_adequacy: str = Field(..., description="One of: too_short, appropriate, padded")
    tone_score: int = Field(..., ge=0, le=3, description="Professionalism and tone quality")
    qa_alignment_score: int = Field(..., ge=0, le=3, description="How well the report addresses the question")
    notes: str = Field(..., description="Brief notes on format quality")


class RAGFinalScore(BaseModel):
    quality_score: int = Field(..., ge=0, le=3, description="Final quality score 0-3")
    justification: str = Field(..., description="Justification for the final score")


# LLM response models for disagreement analysis

class DisagreementInsight(BaseModel):
    case_index: int = Field(..., description="Index of the disagreement case")
    analysis: str = Field(..., description="Why the scorer likely got this wrong")
    root_cause: str = Field(..., description="Category of error: rubric_gap, ambiguity, edge_case, or calibration")


class DisagreementAnalysis(BaseModel):
    insights: list[DisagreementInsight] = Field(..., description="Analysis of each disagreement case")
    rubric_changes: list[str] = Field(..., description="Specific rubric changes to improve scoring")
    prompt_changes: list[str] = Field(..., description="Specific prompt changes to improve scoring")


# LLM response model for rubric refinement

class RefinedRubric(BaseModel):
    dimensions: list[RubricDimension] = Field(..., description="Updated rubric dimensions")
    change_summary: str = Field(..., description="Summary of what changed and why")
