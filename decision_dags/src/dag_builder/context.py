"""Context management for DAG building operations."""

from typing import Any, Dict, List, Optional
from datetime import datetime

from ..models import (
    DecisionNode,
    StrategicPath,
    BuildContext,
    DAGBuildingState,
    LayerGenerationResult,
    ValidationResult,
    BuildPhase,
)


class DAGBuildingContext:
    """Enhanced context manager for DAG building operations with comprehensive state tracking."""

    def __init__(
        self,
        problem_statement: str,
        strategic_paths: List[StrategicPath],
        organizational_goals: Optional[List[Dict[str, Any]]] = None,
        activity_insights: Optional[List[Dict[str, Any]]] = None,
        dag_state: Optional[Dict[str, Any]] = None,
        database_context: Optional[Dict[str, Any]] = None,
    ):
        self.problem_statement = problem_statement
        self.strategic_paths = strategic_paths
        self.organizational_goals = organizational_goals or []
        self.activity_insights = activity_insights or []
        self.database_context = database_context or {}

        # Extract database context components for easy access
        if self.database_context:
            self.organizational_goals.extend(self.database_context.get("organizational_goals", []))
            self.past_decisions = self.database_context.get("past_decisions", [])
            self.relevant_content = self.database_context.get("relevant_content", [])
            self.activity_insights.extend([self.database_context.get("activity_insights", {})])
        else:
            self.past_decisions = []
            self.relevant_content = []

        # Initialize state tracking
        self.state = DAGBuildingState(
            current_phase=BuildPhase.INITIALIZATION,
            current_layer=0,
            total_nodes_generated=0,
            phase_start_time=datetime.utcnow(),
            layer_results=[],
            validation_results=[],
            error_count=0,
            metadata=dag_state or {},
        )

        self.layer_history: List[List[DecisionNode]] = []
        self.generation_metrics: Dict[str, Any] = {}

    def create_node_context(
        self, parent_node: DecisionNode, sibling_nodes: Optional[List[DecisionNode]] = None
    ) -> BuildContext:
        """Create context for processing a specific node."""
        # Convert organizational goals to proper format if needed
        org_goals = []
        for goal in self.organizational_goals:
            if isinstance(goal, dict):
                from ..models import OrganizationalGoal

                org_goals.append(OrganizationalGoal(**goal))
            else:
                org_goals.append(goal)

        # Convert activity insights to proper format if needed
        activity_insight = None
        if self.activity_insights:
            from ..models import ActivityInsight

            if isinstance(self.activity_insights, list) and len(self.activity_insights) > 0:
                # Convert list of insights to a single ActivityInsight
                activity_insight = ActivityInsight(
                    activity_patterns=self.activity_insights, engagement_patterns=[],
                    collaboration_indicators={}, resource_indicators={}
                )
            elif isinstance(self.activity_insights, dict):
                activity_insight = ActivityInsight(**self.activity_insights)
            else:
                activity_insight = self.activity_insights

        return BuildContext(
            problem_statement=self.problem_statement,
            strategic_paths=self.strategic_paths,
            organizational_goals=org_goals,
            activity_insights=activity_insight,
            current_layer=self.state.current_layer,
            parent_nodes=[parent_node],
            dag_state=self.state.metadata.copy(),
            temperature=0.7,  # Default temperature
        )

    def advance_layer(self, layer_nodes: List[DecisionNode]) -> None:
        """Advance to the next layer and record history."""
        # Record layer generation result
        layer_result = LayerGenerationResult(
            layer=self.state.current_layer,
            nodes_generated=len(layer_nodes),
            generation_time_seconds=(datetime.utcnow() - self.state.phase_start_time).total_seconds(),
            success_rate=1.0,  # Could be calculated based on failed generations
            average_confidence=sum(node.confidence_score or 0.5 for node in layer_nodes) / len(layer_nodes)
            if layer_nodes
            else 0.0,
            validation_passed=True,  # To be updated after validation
            errors=[],
        )

        self.state.layer_results.append(layer_result)
        self.layer_history.append(layer_nodes)
        self.state.current_layer += 1
        self.state.total_nodes_generated += len(layer_nodes)
        self.state.phase_start_time = datetime.utcnow()  # Reset for next layer

    def update_phase(self, phase: BuildPhase) -> None:
        """Update the current build phase."""
        self.state.current_phase = phase
        self.state.phase_start_time = datetime.utcnow()

    def record_validation_result(self, result: ValidationResult) -> None:
        """Record a validation result."""
        self.state.validation_results.append(result)
        if not result.is_valid:
            self.state.error_count += len(result.errors)

    def record_error(self, error_message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record an error during DAG building."""
        self.state.error_count += 1
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.state.current_phase.value,
            "layer": self.state.current_layer,
            "message": error_message,
            "context": context or {},
        }

        if "errors" not in self.state.metadata:
            self.state.metadata["errors"] = []
        self.state.metadata["errors"].append(error_entry)

    def get_layer_summary(self, layer: int) -> Dict[str, Any]:
        """Get summary information about a specific layer."""
        if layer >= len(self.layer_history):
            return {}

        layer_nodes = self.layer_history[layer]
        return {
            "layer": layer,
            "node_count": len(layer_nodes),
            "node_types": [node.type.value for node in layer_nodes],
            "titles": [node.title for node in layer_nodes],
        }

    def get_dag_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the entire DAG building process."""
        total_time = sum(lr.generation_time_seconds for lr in self.state.layer_results)

        return {
            "state": {
                "current_phase": self.state.current_phase.value,
                "current_layer": self.state.current_layer,
                "total_nodes_generated": self.state.total_nodes_generated,
                "total_generation_time_seconds": total_time,
                "error_count": self.state.error_count,
                "validation_count": len(self.state.validation_results),
            },
            "layers": {
                "total_layers": len(self.layer_history),
                "layers_summary": [self.get_layer_summary(i) for i in range(len(self.layer_history))],
                "layer_performance": [
                    {
                        "layer": lr.layer,
                        "nodes_generated": lr.nodes_generated,
                        "generation_time": lr.generation_time_seconds,
                        "average_confidence": lr.average_confidence,
                        "success_rate": lr.success_rate,
                    }
                    for lr in self.state.layer_results
                ],
            },
            "input": {
                "problem_statement": self.problem_statement,
                "strategic_paths_count": len(self.strategic_paths),
                "organizational_goals_count": len(self.organizational_goals),
                "activity_insights_count": len(self.activity_insights),
            },
            "validation": {
                "total_validations": len(self.state.validation_results),
                "passed_validations": sum(1 for vr in self.state.validation_results if vr.is_valid),
                "failed_validations": sum(1 for vr in self.state.validation_results if not vr.is_valid),
                "total_errors": sum(len(vr.errors) for vr in self.state.validation_results),
                "total_warnings": sum(len(vr.warnings) for vr in self.state.validation_results),
            },
        }

    def get_current_state(self) -> DAGBuildingState:
        """Get the current DAG building state."""
        return self.state

    def is_healthy(self) -> bool:
        """Check if the DAG building process is healthy."""
        # Basic health checks
        if self.state.error_count > 10:  # Too many errors
            return False

        if len(self.state.layer_results) > 0:
            # Check recent layer success rates
            recent_layers = self.state.layer_results[-3:]  # Last 3 layers
            avg_success_rate = sum(lr.success_rate for lr in recent_layers) / len(recent_layers)
            if avg_success_rate < 0.7:  # Success rate too low
                return False

        # Check for stuck progress (no nodes generated in recent layers)
        if len(self.state.layer_results) >= 2:
            recent_node_counts = [lr.nodes_generated for lr in self.state.layer_results[-2:]]
            if all(count == 0 for count in recent_node_counts):
                return False

        return True
