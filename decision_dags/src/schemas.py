"""Pydantic schemas for structured LLM outputs in decision DAG system."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from .models import NodeType, DecisionType, DecisionReasoningType, MutationProposal


class SimplifiedDecisionNode(BaseModel):
    """Simplified DecisionNode schema for LLM generation.

    Excludes fields that should be generated programmatically:
    - id: Generated automatically with uuid4()
    - embedding: Generated later in enrichment
    - people_impacted: Added during enrichment pass
    - resource_requirements: Added during enrichment pass
    """

    layer: int = Field(..., description="Layer depth in the DAG")
    type: NodeType = Field(..., description="Whether this is a decision or option node")
    title: str = Field(..., description="Short descriptive title")
    description: str = Field(..., description="Detailed description")

    # Only include essential fields for initial generation
    decision_type: Optional[DecisionType] = Field(default=None, description="Type of decision (null for option nodes)")
    reasoning_type: Optional[DecisionReasoningType] = Field(default=None, description="How this decision is reached (reactive/proactive/logical/strategic/intuitive/practical) - for decision nodes only")
    goal_impacts: Dict[str, str] = Field(default_factory=dict, description="Impact on organizational goals")

    # Basic metadata
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    confidence_score: Optional[float] = Field(default=None, description="Generation confidence")

    @field_validator("decision_type")
    @classmethod
    def validate_decision_type(cls, v, info):
        """Ensure decision_type is set for decision nodes."""
        node_type = info.data.get("type")
        if node_type == NodeType.DECISION and v is None:
            # Default to STRATEGIC for decision nodes if not specified
            return DecisionType.STRATEGIC
        if node_type == NodeType.OPTION and v is not None:
            # Clear decision_type for option nodes
            return None
        return v


class SimplifiedChildNodesSchema(BaseModel):
    """Schema for LLM-generated child nodes using simplified structure."""

    child_nodes: List[SimplifiedDecisionNode] = Field(..., description="Generated child nodes")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Edges to child nodes")
    reasoning: str = Field(..., description="Reasoning for these children")
    should_continue: bool = Field(default=True, description="Whether to continue building")
    confidence: float = Field(default=0.8, ge=0, le=1, description="Confidence in generation")


class NodeEnrichmentSchema(BaseModel):
    """Schema for structured node enrichment with people and resource details."""

    people_impacted: List[Dict[str, str]] = Field(
        ...,
        description="List of people/roles affected and how they're impacted"
    )
    resource_requirements: Dict[str, str] = Field(
        ...,
        description="Required resources: personnel, budget, timeline, tools, etc."
    )
    reasoning: str = Field(
        ...,
        description="Detailed reasoning behind the people and resource assessments"
    )


class EdgeEnrichmentSchema(BaseModel):
    """Schema for structured edge enrichment with cost/timeline/risk estimates."""

    cost_estimate: str = Field(
        ...,
        description="Cost category: very-low, low, low-medium, medium, medium-high, high, very-high"
    )
    estimated_cost_dollars: Optional[float] = Field(
        default=None,
        description="Numeric cost estimate in USD (optional, can be None if not quantifiable)"
    )
    timeline_estimate: str = Field(
        ...,
        description="Timeline estimate in human-readable format (e.g., '2-4 weeks', '3-6 months')"
    )
    implementation_risks: List[str] = Field(
        ...,
        description="List of specific implementation risks and challenges"
    )
    success_conditions: List[str] = Field(
        ...,
        description="Conditions required for successful transition"
    )
    likelihood: str = Field(
        ...,
        description="Likelihood of success: very-low, low, medium, high, very-high"
    )
    resource_requirements: List[str] = Field(
        default_factory=list,
        description="Specific resource requirements (skills, tools, people)"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Dependencies that must be satisfied"
    )
    reasoning: str = Field(
        ...,
        description="Detailed reasoning behind the estimates and assessments"
    )


class NodeModificationSchema(BaseModel):
    """Schema for HITL node modification requests."""

    modified_title: Optional[str] = Field(default=None, description="Updated node title")
    modified_description: Optional[str] = Field(default=None, description="Updated node description")
    modified_tags: Optional[List[str]] = Field(default=None, description="Updated tags")
    modification_reason: str = Field(..., description="Reason for the modification")


class DAGValidationSchema(BaseModel):
    """Schema for comprehensive DAG validation results."""

    is_valid: bool = Field(..., description="Whether the DAG passes all validations")
    structural_validity: bool = Field(..., description="Structural integrity check")
    alternating_pattern_valid: bool = Field(..., description="Alternating decision-option pattern")
    connectivity_valid: bool = Field(..., description="Proper connectivity between layers")
    mece_compliance: bool = Field(..., description="MECE principles compliance")

    errors: List[str] = Field(default_factory=list, description="Critical validation errors")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")

    quality_score: float = Field(..., description="Overall quality score (0.0-1.0)")
    completeness_score: float = Field(..., description="Completeness score (0.0-1.0)")
    coherence_score: float = Field(..., description="Coherence score (0.0-1.0)")

    detailed_analysis: str = Field(..., description="Detailed analysis and recommendations")


class PathEvaluationSchema(BaseModel):
    """Schema for strategic path fitness evaluation."""

    overall_fitness: float = Field(..., description="Overall fitness score (0.0-1.0)")

    goal_alignment_score: float = Field(..., description="Alignment with organizational goals")
    feasibility_score: float = Field(..., description="Implementation feasibility")
    cost_efficiency_score: float = Field(..., description="Cost efficiency assessment")
    timeline_efficiency_score: float = Field(..., description="Timeline efficiency assessment")
    risk_mitigation_score: float = Field(..., description="Risk mitigation effectiveness")
    innovation_score: float = Field(..., description="Innovation and strategic value")

    strengths: List[str] = Field(..., description="Key strengths of this path")
    weaknesses: List[str] = Field(..., description="Key weaknesses and concerns")
    recommendations: List[str] = Field(..., description="Specific improvement recommendations")

    addressed_goals: List[str] = Field(..., description="Organizational goals addressed by this path")
    missing_elements: List[str] = Field(..., description="Important elements missing from this path")

    detailed_reasoning: str = Field(..., description="Comprehensive evaluation reasoning")


class DAGLayerValidationSchema(BaseModel):
    """Schema for context-aware layer validation in DAGs."""

    mece_compliant: bool = Field(..., description="Whether the layer is MECE compliant")
    coverage_gaps: List[str] = Field(..., description="Areas not adequately covered by nodes in this layer")
    redundancies: List[str] = Field(..., description="Overlapping or redundant nodes in this layer")
    alignment_issues: List[str] = Field(..., description="Nodes not aligned with organizational goals")
    feasibility_concerns: List[str] = Field(..., description="Implementation feasibility concerns for nodes")
    quality_score: float = Field(..., description="Overall quality score for this layer (0.0-1.0)")
    recommendations: List[str] = Field(..., description="Specific recommendations for improving this layer")
    detailed_analysis: str = Field(..., description="Detailed analysis of the layer considering organizational context")


class DAGCoherenceAssessmentSchema(BaseModel):
    """Schema for strategic coherence assessment of the entire DAG."""

    coherence_score: float = Field(..., description="Overall strategic coherence score (0.0-1.0)")
    narrative_strength: float = Field(..., description="Strength of strategic narrative (0.0-1.0)")
    logical_flow: float = Field(..., description="Logical flow between layers and paths (0.0-1.0)")
    strategic_gaps: List[str] = Field(..., description="Missing strategic elements or connections")
    coherence_issues: List[str] = Field(..., description="Issues with strategic coherence")
    recommendations: List[str] = Field(..., description="Recommendations for improving coherence")
    detailed_analysis: str = Field(..., description="Detailed coherence analysis")


class DAGFeasibilityAssessmentSchema(BaseModel):
    """Schema for implementation feasibility assessment of the DAG."""

    feasibility_score: float = Field(..., description="Overall implementation feasibility (0.0-1.0)")
    resource_adequacy: float = Field(..., description="Adequacy of organizational resources (0.0-1.0)")
    timeline_realism: float = Field(..., description="Realism of implied timelines (0.0-1.0)")
    capability_gaps: List[str] = Field(..., description="Missing capabilities needed for implementation")
    resource_constraints: List[str] = Field(..., description="Resource constraints that may impact implementation")
    recommendations: List[str] = Field(..., description="Recommendations for improving feasibility")
    detailed_analysis: str = Field(..., description="Detailed feasibility analysis")


class DAGAlignmentAssessmentSchema(BaseModel):
    """Schema for organizational alignment assessment of the DAG."""

    alignment_score: float = Field(..., description="Overall organizational alignment score (0.0-1.0)")
    goal_coverage: float = Field(..., description="Coverage of organizational goals (0.0-1.0)")
    priority_alignment: float = Field(..., description="Alignment with organizational priorities (0.0-1.0)")
    uncovered_goals: List[str] = Field(..., description="Organizational goals not addressed by the DAG")
    misaligned_elements: List[str] = Field(..., description="DAG elements that conflict with organizational direction")
    recommendations: List[str] = Field(..., description="Recommendations for improving alignment")
    detailed_analysis: str = Field(..., description="Detailed alignment analysis")


class MutationProposalsSchema(BaseModel):
    """Schema for LLM-generated mutation proposals with reasoning."""

    proposals: List[MutationProposal] = Field(..., description="List of mutation proposals")
    overall_reasoning: str = Field(..., description="Overall reasoning for the proposed mutations")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence in the proposals")




class NodeOperation(BaseModel):
    """Single node operation in a mutation diff.

    Node operations inherently include edge information since every non-root node
    must have a parent, and edges are created/updated as part of node operations.
    """

    operation: Literal["add", "modify", "delete"] = Field(
        ...,
        description="Type of operation to perform"
    )

    node_id: str = Field(
        ...,
        description="Node ID - for modify/delete ops, or new ID for add ops"
    )

    # For add/modify operations - node properties
    node_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Node properties (layer, type, title, description, etc.) for add/modify"
    )

    # For add operations - parent connection
    parent_id: Optional[str] = Field(
        default=None,
        description="Parent node ID when adding a new node"
    )

    edge_to_parent: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Edge properties for connection to parent (conditions, likelihood, cost, etc.)"
    )

    # For modify operations - updating edges to children
    child_edge_updates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Updates to edges leading to child nodes (for modify operations)"
    )

    @field_validator("parent_id")
    @classmethod
    def validate_parent_for_add(cls, v, info):
        """Ensure parent_id is correctly set for add operations based on node type."""
        operation = info.data.get("operation")
        node_data = info.data.get("node_data", {})
        layer = node_data.get("layer")

        if operation == "add":
            if layer == 0 and v is not None:
                raise ValueError("Root nodes (layer 0) should not have a parent_id")
            if layer > 0 and not v:
                raise ValueError("Non-root nodes (layer > 0) must have a parent_id")
        return v

    @field_validator("node_data")
    @classmethod
    def validate_node_data(cls, v, info):
        """Ensure node_data is provided for add/modify operations."""
        operation = info.data.get("operation")
        if operation in ["add", "modify"] and not v:
            raise ValueError(f"node_data is required for {operation} operations")
        return v


class MutationDiffSchema(BaseModel):
    """Diff-based mutation representation using node-centric operations.

    This schema represents mutations as a series of node operations, where each
    operation can add, modify, or delete a node along with its associated edges.
    """

    operations: List[NodeOperation] = Field(
        ...,
        description="List of node operations to apply in order"
    )

    summary: str = Field(
        ...,
        description="Brief summary of what this mutation accomplishes"
    )

    reasoning: str = Field(
        ...,
        description="Detailed reasoning for why these changes improve the path"
    )

    expected_improvements: List[str] = Field(
        ...,
        description="List of specific improvements expected from this mutation"
    )

    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in the mutation's effectiveness (0.0-1.0)"
    )
