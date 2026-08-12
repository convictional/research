from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4
import uuid

from pydantic import BaseModel, Field, field_validator


class NodeType(str, Enum):
    DECISION = "decision"
    OPTION = "option"


class DecisionType(str, Enum):
    """Types of decisions from the experimental PR."""

    IMPLEMENTATION = "implementation"
    RESOURCE = "resource"
    TIMING = "timing"
    RISK = "risk"
    MARKET = "market"
    PRODUCT = "product"
    STRATEGIC = "strategic"


class EdgeType(str, Enum):
    """Edge types for the alternating decision-option pattern."""

    DECISION_TO_OPTION = "decision_to_option"
    OPTION_TO_DECISION = "option_to_decision"


class DecisionReasoningType(str, Enum):
    """Types of decision reasoning for option-to-decision edges."""

    REACTIVE = "reactive"  # Responding to external events
    PROACTIVE = "proactive"  # Planned internal decision
    LOGICAL = "logical"  # Based on data and analysis
    STRATEGIC = "strategic"  # Based on long-term goals
    INTUITIVE = "intuitive"  # Based on experience/gut feeling
    PRACTICAL = "practical"  # Based on feasibility/resources


class WorkflowStatus(str, Enum):
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"
    EVOLVING = "evolving"


class EvolutionStrategy(str, Enum):
    FITNESS_PROPORTIONAL = "fitness_proportional"
    TOURNAMENT = "tournament"
    TOP_K = "top_k"


class DecisionNode(BaseModel):
    """Individual node in the decision DAG representing either a decision or an option."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    layer: int = Field(..., description="Layer depth in the DAG")
    type: NodeType = Field(..., description="Whether this is a decision or option node")
    title: str = Field(..., description="Short descriptive title")
    description: str = Field(..., description="Detailed description")

    # Enhanced fields from PR insights
    decision_type: Optional[DecisionType] = Field(default=None, description="Type of decision (null for option nodes)")
    reasoning_type: Optional[DecisionReasoningType] = Field(default=None, description="How this decision is reached (for decision nodes only)")
    goal_impacts: Dict[str, str] = Field(default_factory=dict, description="Impact on organizational goals")
    people_impacted: List[Dict[str, str]] = Field(default_factory=list, description="People affected and how")
    resource_requirements: Dict[str, str] = Field(default_factory=dict, description="Required resources")

    # Original fields
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Text embedding vector")
    confidence_score: Optional[float] = Field(default=None, description="Generation confidence")

    @field_validator("decision_type")
    @classmethod
    def validate_decision_type(cls, v, info):
        """Ensure decision_type is only set for decision nodes."""
        node_type = info.data.get("type")
        if node_type == NodeType.DECISION and v is None:
            raise ValueError("Decision nodes must have a decision_type")
        if node_type == NodeType.OPTION and v is not None:
            raise ValueError("Option nodes cannot have a decision_type")
        return v

    @field_validator("layer")
    @classmethod
    def validate_alternating_pattern(cls, v, info):
        """Validate alternating decision-option pattern."""
        node_type = info.data.get("type")
        if node_type == NodeType.DECISION and v % 2 != 0:
            raise ValueError(f"Decision nodes must be on even layers, got layer {v}")
        if node_type == NodeType.OPTION and v % 2 == 0:
            raise ValueError(f"Option nodes must be on odd layers, got layer {v}")
        return v

    def is_decision(self) -> bool:
        """Check if this node represents a decision point."""
        return self.type == NodeType.DECISION

    def is_option(self) -> bool:
        """Check if this node represents an option/choice."""
        return self.type == NodeType.OPTION

    def copy(self) -> "DecisionNode":
        """Create a deep copy of this node."""
        return DecisionNode.model_validate(self.model_dump())


class DecisionEdge(BaseModel):
    """Edge connecting two nodes in the decision DAG."""

    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")

    # Enhanced fields from PR insights
    edge_type: EdgeType = Field(..., description="Type of edge for alternating pattern")
    condition: str = Field(..., description="Condition triggering this path")
    decision_reasoning_type: Optional[DecisionReasoningType] = Field(
        default=None, description="Reasoning type for option_to_decision edges"
    )
    likelihood: str = Field(default="medium", description="Probability: low, medium, high, certain")
    label: str = Field(default="", description="Short edge label")

    # Cost and risk fields
    cost_estimate: Optional[str] = Field(default=None, description="Estimated cost category")
    timeline_estimate: Optional[str] = Field(default=None, description="Estimated timeline")
    estimated_cost_dollars: Optional[float] = Field(default=None, description="Numeric cost estimate")
    implementation_risks: Optional[List[str]] = Field(default=None, description="Implementation risks")

    # Legacy field for compatibility
    relationship: str = Field(default="leads_to", description="Type of relationship")
    conditions: List[str] = Field(default_factory=list, description="Conditions for this path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("decision_reasoning_type")
    @classmethod
    def validate_reasoning_type(cls, v, info):
        """Ensure reasoning type is only set for option_to_decision edges."""
        edge_type = info.data.get("edge_type")
        if edge_type == EdgeType.OPTION_TO_DECISION and v is None:
            # Default to LOGICAL for option_to_decision edges if not specified
            return DecisionReasoningType.LOGICAL
        if edge_type == EdgeType.DECISION_TO_OPTION and v is not None:
            # Clear decision_reasoning_type for decision_to_option edges
            return None
        return v

    def copy(self) -> "DecisionEdge":
        """Create a deep copy of this edge."""
        return DecisionEdge.model_validate(self.model_dump())


class DecisionDAG(BaseModel):
    """Main DAG structure representing the complete strategic decision graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    root_nodes: List[DecisionNode] = Field(default_factory=list, description="Starting nodes")
    all_nodes: Dict[str, DecisionNode] = Field(default_factory=dict, description="All nodes by ID")
    edges: List[DecisionEdge] = Field(default_factory=list, description="All edges")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="DAG metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    original_extracted_path_id: Optional[str] = Field(default=None, description="UUID of the original extracted path (persists through evolution)")

    def add_node(self, node: DecisionNode) -> None:
        """Add a node to the DAG."""
        self.all_nodes[node.id] = node
        if node.layer == 0:
            self.root_nodes.append(node)

    def add_edge(self, edge: DecisionEdge) -> None:
        """Add an edge to the DAG."""
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[DecisionNode]:
        """Get a node by ID."""
        return self.all_nodes.get(node_id)

    def get_edge(self, source_id: str, target_id: str) -> Optional[DecisionEdge]:
        """Get an edge between two nodes."""
        for edge in self.edges:
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge
        return None

    def get_children(self, node: DecisionNode) -> List[DecisionNode]:
        """Get all child nodes of a given node."""
        children = []
        for edge in self.edges:
            if edge.source_id == node.id:
                child = self.get_node(edge.target_id)
                if child:
                    children.append(child)
        return children

    def get_parents(self, node: DecisionNode) -> List[DecisionNode]:
        """Get all parent nodes of a given node."""
        parents = []
        for edge in self.edges:
            if edge.target_id == node.id:
                parent = self.get_node(edge.source_id)
                if parent:
                    parents.append(parent)
        return parents

    def get_nodes_at_layer(self, layer: int) -> List[DecisionNode]:
        """Get all nodes at a specific layer."""
        return [node for node in self.all_nodes.values() if node.layer == layer]

    def get_max_layer(self) -> int:
        """Get the maximum layer depth in the DAG."""
        if not self.all_nodes:
            return 0
        return max(node.layer for node in self.all_nodes.values())

    def get_paths(self) -> List[List[DecisionNode]]:
        """Extract all root-to-leaf paths from the DAG."""
        paths = []
        for root in self.root_nodes:
            self._extract_paths_from_node(root, [root], paths)
        return paths

    def _extract_paths_from_node(
        self, node: DecisionNode, current_path: List[DecisionNode], all_paths: List[List[DecisionNode]]
    ) -> None:
        """Recursive helper for path extraction."""
        children = self.get_children(node)
        if not children:
            all_paths.append(current_path.copy())
        else:
            for child in children:
                current_path.append(child)
                self._extract_paths_from_node(child, current_path, all_paths)
                current_path.pop()

    def update_edge(self, source_id: str, target_id: str, updated_edge: DecisionEdge) -> None:
        """Update an existing edge."""
        for i, edge in enumerate(self.edges):
            if edge.source_id == source_id and edge.target_id == target_id:
                self.edges[i] = updated_edge
                break


class StrategicPath(BaseModel):
    """Representation of a strategic path through the decision space."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Path title")
    description: str = Field(..., description="Path description")
    key_milestones: List[str] = Field(default_factory=list, description="Key milestones")
    expected_outcomes: List[str] = Field(default_factory=list, description="Expected outcomes")
    fitness_score: Optional[float] = Field(default=None, description="Fitness evaluation score")
    nodes: List[DecisionNode] = Field(default_factory=list, description="Nodes in this path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DAGBuilderConfig(BaseModel):
    """Configuration for DAG building process."""

    # Parallel processing
    max_concurrent_agents: int = Field(default=10, description="Maximum concurrent agents")
    agent_timeout: float = Field(default=30.0, description="Agent timeout in seconds")

    # DAG structure
    max_layers: int = Field(default=6, description="Maximum DAG depth")
    max_children_per_node: int = Field(default=5, description="Maximum children per node")
    min_children_per_node: int = Field(default=2, description="Minimum children per node")

    # Deduplication
    similarity_threshold: float = Field(default=0.8, description="Similarity threshold for deduplication")
    weak_similarity_threshold: float = Field(default=0.6, description="Weak similarity threshold")

    # LLM settings
    generation_temperature: float = Field(default=0.7, description="Temperature for generation")
    assessment_temperature: float = Field(default=0.3, description="Temperature for assessment")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Quality control
    enable_validation: bool = Field(default=True, description="Enable validation")
    enable_self_correction: bool = Field(default=True, description="Enable self-correction")


class PathEvolutionConfig(BaseModel):
    """Configuration for path evolution process."""

    # Evolution parameters
    max_concurrent_evolutions: int = Field(default=4, description="Maximum concurrent evolutions")
    max_iterations_per_path: int = Field(default=3, description="Maximum iterations per path")
    min_improvement_threshold: float = Field(default=0.1, description="Minimum improvement threshold")

    # Path selection
    top_k_paths: int = Field(default=10, description="Top K paths to select")
    selection_strategy: EvolutionStrategy = Field(
        default=EvolutionStrategy.FITNESS_PROPORTIONAL, description="Selection strategy"
    )

    # Fitness weights
    coherence_weight: float = Field(default=0.3, description="Coherence weight")
    feasibility_weight: float = Field(default=0.25, description="Feasibility weight")
    innovation_weight: float = Field(default=0.2, description="Innovation weight")
    completeness_weight: float = Field(default=0.15, description="Completeness weight")
    uniqueness_weight: float = Field(default=0.1, description="Uniqueness weight")

    # Evolution strategy
    mutation_rate: float = Field(default=0.3, description="Mutation rate")
    crossover_rate: float = Field(default=0.1, description="Crossover rate")
    elite_preservation: float = Field(default=0.2, description="Elite preservation rate")


class FitnessWeights(BaseModel):
    """Weights for fitness evaluation components."""

    w_goals: float = Field(default=0.4, description="Goal alignment weight")
    w_cost: float = Field(default=0.2, description="Cost efficiency weight")
    w_timeline: float = Field(default=0.2, description="Timeline efficiency weight")
    w_risk: float = Field(default=0.1, description="Risk mitigation weight")
    w_complexity: float = Field(default=0.1, description="Complexity weight")


class AlphaEvolutionConfig(BaseModel):
    """Configuration for AlphaEvolve-style path evolution."""

    # Core genetic algorithm parameters
    max_generations: int = Field(default=10, description="Maximum generations")
    population_size: int = Field(default=8, description="Population size")
    mutation_rate: float = Field(default=0.7, description="Mutation rate")
    crossover_rate: float = Field(default=0.3, description="Crossover rate")
    elite_preservation: float = Field(default=0.25, description="Elite preservation rate")

    # AlphaEvolve specific parameters
    proposals_per_generation: int = Field(default=5, description="Proposals per generation")
    min_confidence_threshold: float = Field(default=0.3, description="Minimum confidence threshold")
    max_history_size: int = Field(default=100, description="Maximum history size")

    # Warm-up period configuration
    warmup_generations_ratio: float = Field(default=0.5, description="Ratio of generations for warm-up (0.5 = 50%)")
    track_failed_mutations: bool = Field(default=True, description="Track unsuccessful mutations for learning")
    max_examples_per_category: int = Field(default=3, description="Max few-shot examples per category")

    # Fitness evaluation
    min_improvement_threshold: float = Field(default=0.05, description="Minimum improvement threshold")
    fitness_weights: FitnessWeights = Field(default_factory=FitnessWeights, description="Fitness weights")

    # Objectives for mutation guidance
    objectives: List[str] = Field(
        default_factory=lambda: [
            "Maximize goal alignment with organizational objectives",
            "Minimize implementation costs and resource requirements",
            "Optimize timeline efficiency and reduce delays",
            "Mitigate implementation risks and uncertainty",
            "Maintain strategic coherence and decisiveness",
        ],
        description="Objectives for mutation guidance",
    )

    # Parallelization
    max_concurrent_evolutions: int = Field(default=4, description="Maximum concurrent evolutions")
    enable_self_correction: bool = Field(default=True, description="Enable self-correction")

    # Selection strategy
    selection_strategy: str = Field(
        default="top_k", description="Selection strategy: top_k, fitness_proportional, below_average"
    )
    top_k_paths: int = Field(default=5, description="Number of top paths to select")

    # Plateau detection and exploration
    plateau_detection_window: int = Field(default=3, description="Plateau detection window")
    exploration_burst_enabled: bool = Field(default=True, description="Enable exploration burst")
    exploration_burst_generations: int = Field(default=3, description="Exploration burst generations")


class MutationProposal(BaseModel):
    """A proposed mutation for a path."""

    mutation_type: str = Field(..., description="Type of mutation")
    confidence: float = Field(..., description="Confidence score (0-1)")
    description: str = Field(..., description="Human-readable description")
    operations: Dict[str, Any] = Field(default_factory=dict, description="Specific operations to perform")
    reasoning: str = Field(..., description="Reasoning behind the mutation")
    expected_improvements: List[str] = Field(default_factory=list, description="Expected improvements")


class EvolutionVariant(BaseModel):
    """A variant produced during evolution."""

    variant_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique variant ID")
    generation: int = Field(..., description="Generation number")
    parent_variant_ids: List[str] = Field(default_factory=list, description="Parent variant IDs")
    mutations: List[Dict[str, Any]] = Field(default_factory=list, description="Applied mutations")
    fitness_scores: Dict[str, float] = Field(default_factory=dict, description="Fitness scores")
    evaluation_reasoning: str = Field(default="", description="Evaluation reasoning")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    # Inspiration tracking
    inspiration_sources: List[str] = Field(default_factory=list, description="Inspiration source IDs")
    strong_areas: List[str] = Field(default_factory=list, description="Areas of strength")
    weak_areas: List[str] = Field(default_factory=list, description="Areas of weakness")
    fitness_improvements: Dict[str, float] = Field(
        default_factory=dict, description="Fitness improvements by dimension"
    )

    # Enhanced tracking for dynamic learning
    mutation_success: bool = Field(default=False, description="Whether mutation improved fitness")
    fitness_delta: float = Field(default=0.0, description="Change in fitness from parent")
    mutation_reasoning: str = Field(default="", description="Why this mutation was attempted")
    mutation_type: str = Field(default="", description="Type of mutation applied")
    parent_fitness: float = Field(default=0.0, description="Parent variant's fitness for comparison")


class PathMetrics(BaseModel):
    """Quantitative metrics for a strategic path."""

    total_cost_dollars: float = Field(default=0.0, description="Total estimated cost")
    timeline_weeks: float = Field(default=0.0, description="Total estimated timeline")
    risk_count: int = Field(default=0, description="Number of identified risks")
    complexity_score: float = Field(default=0.0, description="Complexity score (0-1)")
    resource_efficiency: float = Field(default=0.0, description="Resource efficiency score")
    node_count: int = Field(default=0, description="Number of nodes in path")
    branching_factor: float = Field(default=0.0, description="Average branching factor")


class PathExtractionMetrics(BaseModel):
    """Metrics for path extraction process."""

    total_paths: int = Field(default=0, description="Total paths extracted")
    valid_paths: int = Field(default=0, description="Number of valid paths")
    failed_extractions: int = Field(default=0, description="Number of failed extractions")
    duplicate_paths: int = Field(default=0, description="Number of duplicate paths")
    avg_path_length: float = Field(default=0.0, description="Average path length")
    max_path_length: int = Field(default=0, description="Maximum path length")
    min_path_length: int = Field(default=0, description="Minimum path length")
    path_length_distribution: Dict[int, int] = Field(default_factory=dict, description="Distribution of path lengths")
    extraction_errors: List[str] = Field(default_factory=list, description="List of extraction errors")

    @property
    def success_rate(self) -> float:
        """Calculate success rate of path extraction."""
        if self.total_paths == 0:
            return 0.0
        return self.valid_paths / self.total_paths

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate of path extraction."""
        if self.total_paths == 0:
            return 0.0
        return self.failed_extractions / (self.total_paths + self.failed_extractions)

    @property
    def duplicate_rate(self) -> float:
        """Calculate duplicate rate of extracted paths."""
        if self.total_paths == 0:
            return 0.0
        return self.duplicate_paths / self.total_paths


class GoalAlignmentScore(BaseModel):
    """Goal alignment scoring details."""

    score: float = Field(..., ge=0, le=1, description="Alignment score")
    rationale: str = Field(..., description="Reasoning for the score")
    addressed_goals: List[str] = Field(default_factory=list, description="Goals addressed")
    goal_gaps: List[str] = Field(default_factory=list, description="Goals not addressed")


class RiskAssessmentScore(BaseModel):
    """Risk assessment scoring details."""

    score: float = Field(..., ge=0, le=1, description="Risk mitigation score")
    rationale: str = Field(..., description="Reasoning for the score")
    identified_risks: List[str] = Field(default_factory=list, description="Identified risks")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Mitigation strategies")


class CostAssessmentScore(BaseModel):
    """Cost assessment scoring details."""

    score: float = Field(..., ge=0, le=1, description="Cost efficiency score")
    rationale: str = Field(..., description="Reasoning for the score")
    cost_breakdown: Dict[str, float] = Field(default_factory=dict, description="Cost breakdown")
    cost_optimization_opportunities: List[str] = Field(default_factory=list, description="Optimization opportunities")


class TimelineAssessmentScore(BaseModel):
    """Timeline assessment scoring details."""

    score: float = Field(..., ge=0, le=1, description="Timeline efficiency score")
    rationale: str = Field(..., description="Reasoning for the score")
    critical_path_weeks: float = Field(default=0.0, description="Critical path timeline")
    timeline_risks: List[str] = Field(default_factory=list, description="Timeline risks")


class ComprehensiveEvaluation(BaseModel):
    """Comprehensive evaluation from LLM perspective."""

    goal_alignment: GoalAlignmentScore = Field(..., description="Goal alignment assessment")
    risk_assessment: RiskAssessmentScore = Field(..., description="Risk assessment")
    cost_assessment: CostAssessmentScore = Field(..., description="Cost assessment")
    timeline_assessment: TimelineAssessmentScore = Field(..., description="Timeline assessment")


class OrganizationalGoal(BaseModel):
    """Organizational goal from the PR insights."""

    title: str = Field(..., description="Goal title")
    status: str = Field(default="active", description="Goal status")
    target_date: Optional[str] = Field(default=None, description="Target completion date")
    current_metric_value: Optional[str] = Field(default=None, description="Current progress")
    success_conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Success conditions")
    conditions_completed_count: int = Field(default=0, description="Completed conditions")
    conditions_count: int = Field(default=0, description="Total conditions")


class ActivityInsight(BaseModel):
    """Activity-based insights from the PR."""

    activity_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Recent activity patterns")
    engagement_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="User engagement patterns")
    collaboration_indicators: Dict[str, Any] = Field(default_factory=dict, description="Collaboration metrics")
    resource_indicators: Dict[str, Any] = Field(default_factory=dict, description="Resource allocation metrics")


class BuildContext(BaseModel):
    """Context for DAG building operations with PR enhancements."""

    problem_statement: str = Field(..., description="Original problem statement")
    strategic_paths: List[StrategicPath] = Field(default_factory=list, description="Input strategic paths")
    current_layer: int = Field(default=0, description="Current layer being processed")
    parent_nodes: List[DecisionNode] = Field(default_factory=list, description="Parent nodes for context")

    # Enhanced context from PR
    organizational_goals: List[OrganizationalGoal] = Field(default_factory=list, description="Organizational goals")
    activity_insights: Optional[ActivityInsight] = Field(default=None, description="Activity-based insights")
    relevant_data: List[Dict[str, Any]] = Field(default_factory=list, description="Relevant internal data")
    historical_context: List[Dict[str, Any]] = Field(default_factory=list, description="Historical context")
    max_depth: int = Field(default=3, description="Maximum semantic depth")

    # Original fields
    dag_state: Dict[str, Any] = Field(default_factory=dict, description="Current DAG state")
    temperature: float = Field(default=0.7, description="Current generation temperature")


class EvaluationContext(BaseModel):
    """Context for path evaluation operations."""

    organization_id: Optional[str] = Field(default=None, description="Organization ID")
    all_paths: List[DecisionDAG] = Field(default_factory=list, description="All paths for comparison")
    evaluation_criteria: Dict[str, float] = Field(default_factory=dict, description="Custom evaluation criteria")
    domain_context: str = Field(default="", description="Domain-specific context")


class HITLSession(BaseModel):
    """Human-in-the-loop session state."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = Field(..., description="User ID")
    problem_statement: str = Field(..., description="Problem statement")
    current_layer: int = Field(default=0, description="Current layer")
    dag: DecisionDAG = Field(default_factory=DecisionDAG, description="Current DAG state")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Decision history")
    pending_parents: List[DecisionNode] = Field(default_factory=list, description="Pending parent nodes")
    total_time: float = Field(default=0.0, description="Total session time")
    regeneration_count: int = Field(default=0, description="Number of regenerations")
    created_at: datetime = Field(default_factory=datetime.now, description="Session start time")


class NodeSelection(BaseModel):
    """User selection for a node during HITL process."""

    node: DecisionNode = Field(..., description="The node being selected")
    selected: bool = Field(..., description="Whether user selected this node")
    feedback: Optional[str] = Field(default=None, description="User feedback")


class LayerDecision(BaseModel):
    """Record of user decision for a layer during HITL process."""

    layer: int = Field(..., description="Layer number")
    presented: int = Field(..., description="Number of options presented")
    selected: int = Field(..., description="Number of options selected")
    feedback: Optional[str] = Field(default=None, description="User feedback")
    timestamp: datetime = Field(default_factory=datetime.now, description="Decision timestamp")


class LayerResult(BaseModel):
    """Result of processing a layer in HITL mode."""

    next_layer: int = Field(..., description="Next layer to process")
    candidates: List[DecisionNode] = Field(default_factory=list, description="Candidate nodes for next layer")
    is_complete: bool = Field(default=False, description="Whether DAG is complete")
    message: str = Field(default="", description="Status message")


class ChildNodesSchema(BaseModel):
    """Schema for LLM-generated child nodes from PR insights."""

    child_nodes: List[DecisionNode] = Field(..., description="Generated child nodes")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Edges to child nodes")
    reasoning: str = Field(..., description="Reasoning for these children")
    should_continue: bool = Field(default=True, description="Whether to continue building")
    confidence: float = Field(default=0.8, ge=0, le=1, description="Confidence in generation")


class RegenerationGuidance(BaseModel):
    """Guidance for regenerating nodes based on user feedback."""

    direction_feedback: str = Field(default="", description="Direction/focus feedback")
    specificity_feedback: str = Field(default="", description="Specificity feedback")
    missing_options: List[str] = Field(default_factory=list, description="Missing options to include")
    combine_similar: bool = Field(default=False, description="Whether to combine similar options")
    custom_guidance: str = Field(default="", description="Custom text feedback")


class LayerGenerationResult(BaseModel):
    """Result of generating a layer in the DAG."""

    nodes: List[DecisionNode] = Field(default_factory=list, description="Generated nodes")
    edges: List[DecisionEdge] = Field(default_factory=list, description="Generated edges")
    layer_complete: bool = Field(default=False, description="Whether layer is complete")
    reasoning: str = Field(default="", description="Generation reasoning")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")


class ValidationResult(BaseModel):
    """Result of DAG validation."""

    is_valid: bool = Field(..., description="Whether DAG is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    alternating_pattern_valid: bool = Field(default=True, description="Alternating pattern validation")
    connectivity_valid: bool = Field(default=True, description="Connectivity validation")
    mece_violations: List[str] = Field(default_factory=list, description="MECE principle violations")
    node_count: int = Field(default=0, description="Total node count")
    edge_count: int = Field(default=0, description="Total edge count")
    max_layer: int = Field(default=0, description="Maximum layer depth")


class BuildPhase(str, Enum):
    """Phases of DAG building process."""

    INITIALIZATION = "initialization"
    FORWARD_PASS = "forward_pass"
    BACKWARD_PASS = "backward_pass"
    EDGE_ENRICHMENT = "edge_enrichment"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"


class DAGBuildingState(BaseModel):
    """Comprehensive state tracking for DAG building process."""

    current_phase: BuildPhase = Field(..., description="Current build phase")
    current_layer: int = Field(default=0, description="Current layer being processed")
    total_nodes_generated: int = Field(default=0, description="Total nodes generated so far")
    phase_start_time: datetime = Field(..., description="When current phase started")
    layer_results: List[LayerGenerationResult] = Field(default_factory=list, description="Results for each layer")
    validation_results: List[ValidationResult] = Field(default_factory=list, description="Validation results")
    error_count: int = Field(default=0, description="Total number of errors encountered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional state metadata")


class LayerGenerationResult(BaseModel):
    """Result of generating nodes for a single layer."""

    layer: int = Field(..., description="Layer number")
    nodes_generated: int = Field(..., description="Number of nodes generated")
    generation_time_seconds: float = Field(..., description="Time taken to generate layer")
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Success rate for node generation")
    average_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Average confidence score of nodes")
    validation_passed: bool = Field(default=True, description="Whether layer passed validation")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during generation")


# HITL (Human-in-the-Loop) Models
class HITLDecision(str, Enum):
    """Human decisions in the HITL workflow."""

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    SKIP = "skip"
    CONTINUE = "continue"
    STOP = "stop"


class HITLPromptType(str, Enum):
    """Types of HITL prompts."""

    LAYER_APPROVAL = "layer_approval"
    NODE_MODIFICATION = "node_modification"
    PATH_SELECTION = "path_selection"
    QUALITY_REVIEW = "quality_review"
    STRATEGIC_GUIDANCE = "strategic_guidance"


class HITLPrompt(BaseModel):
    """A prompt presented to the human user for decision making."""

    prompt_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique prompt identifier")
    prompt_type: HITLPromptType = Field(..., description="Type of prompt")
    title: str = Field(..., description="Human-readable prompt title")
    description: str = Field(..., description="Detailed prompt description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for the prompt")
    options: List[Dict[str, str]] = Field(default_factory=list, description="Available options for user")
    default_action: Optional[HITLDecision] = Field(default=None, description="Default action if no input")
    timeout_seconds: Optional[int] = Field(default=300, description="Timeout for user response")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When prompt was created")


class HITLResponse(BaseModel):
    """Human response to a HITL prompt."""

    prompt_id: str = Field(..., description="ID of the prompt being responded to")
    decision: HITLDecision = Field(..., description="User's decision")
    feedback: Optional[str] = Field(default=None, description="Optional user feedback")
    modifications: Optional[Dict[str, Any]] = Field(default=None, description="Requested modifications")
    reasoning: Optional[str] = Field(default=None, description="User's reasoning for the decision")
    responded_at: datetime = Field(default_factory=datetime.utcnow, description="When response was given")
    response_time_seconds: float = Field(default=0.0, description="Time taken to respond")


class HITLSessionState(BaseModel):
    """State of a HITL session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session identifier")
    dag_id: Optional[str] = Field(default=None, description="Associated DAG identifier")
    current_prompt: Optional[HITLPrompt] = Field(default=None, description="Currently active prompt")
    prompt_history: List[HITLPrompt] = Field(default_factory=list, description="History of prompts")
    response_history: List[HITLResponse] = Field(default_factory=list, description="History of responses")
    session_start: datetime = Field(default_factory=datetime.utcnow, description="When session started")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    is_active: bool = Field(default=True, description="Whether session is still active")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences for this session")


class HITLWorkflowConfig(BaseModel):
    """Configuration for HITL workflows."""

    enable_layer_approval: bool = Field(default=True, description="Whether to prompt for layer approval")
    enable_node_modification: bool = Field(default=True, description="Whether to allow node modifications")
    enable_path_selection: bool = Field(default=True, description="Whether to prompt for path selection")
    auto_approve_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence threshold for auto-approval"
    )
    max_retries: int = Field(default=3, description="Maximum retries for failed nodes")
    default_timeout: int = Field(default=300, description="Default timeout for prompts in seconds")
    require_approval_layers: List[int] = Field(default_factory=lambda: [0, 1], description="Layers requiring approval")
    skip_approval_on_high_confidence: bool = Field(default=True, description="Skip approval for high confidence nodes")
