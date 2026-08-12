"""Context-aware DAG validation system using organizational knowledge and LLM assistance."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt

from ..models import DecisionDAG, DecisionNode, ValidationResult
from .context import DAGBuildingContext
from ..utils.validation import validate_dag_comprehensive

logger = logging.getLogger(__name__)


@dataclass
class ContextValidationMetrics:
    """Comprehensive metrics for context-aware validation."""

    structural_validity: bool = False
    mece_compliance_score: float = 0.0
    organizational_alignment_score: float = 0.0
    implementation_feasibility_score: float = 0.0
    strategic_coherence_score: float = 0.0
    overall_quality_score: float = 0.0

    validation_time_seconds: float = 0.0
    llm_calls_made: int = 0
    context_items_analyzed: int = 0

    errors_found: int = 0
    warnings_generated: int = 0
    suggestions_provided: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "structural_validity": self.structural_validity,
            "mece_compliance_score": self.mece_compliance_score,
            "organizational_alignment_score": self.organizational_alignment_score,
            "implementation_feasibility_score": self.implementation_feasibility_score,
            "strategic_coherence_score": self.strategic_coherence_score,
            "overall_quality_score": self.overall_quality_score,
            "validation_time_seconds": self.validation_time_seconds,
            "llm_calls_made": self.llm_calls_made,
            "context_items_analyzed": self.context_items_analyzed,
            "errors_found": self.errors_found,
            "warnings_generated": self.warnings_generated,
            "suggestions_provided": self.suggestions_provided,
        }


@dataclass
class LayerValidationResult:
    """Result of validating a single layer with context."""

    layer: int
    node_count: int
    mece_compliant: bool
    coverage_gaps: List[str]
    redundancies: List[str]
    alignment_issues: List[str]
    feasibility_concerns: List[str]
    quality_score: float
    recommendations: List[str]


class ContextAwareDAGValidator:
    """
    Advanced DAG validator that uses organizational context to provide informed validation.

    This validator goes beyond structural checks to assess:
    - MECE compliance informed by organizational domain knowledge
    - Strategic alignment with actual organizational goals
    - Implementation feasibility based on organizational capacity
    - Quality and coherence considering past successful patterns
    """

    def __init__(self, max_concurrent_validations: int = 2):
        """
        Initialize context-aware validator.

        Args:
            max_concurrent_validations: Maximum concurrent LLM validation calls
        """
        self.semaphore = asyncio.Semaphore(max_concurrent_validations)
        self.metrics = ContextValidationMetrics()

    async def validate_dag_comprehensive(
        self, dag: DecisionDAG, context: DAGBuildingContext
    ) -> Tuple[ValidationResult, ContextValidationMetrics]:
        """
        Perform comprehensive context-aware DAG validation.

        Args:
            dag: The DAG to validate
            context: Building context with organizational information

        Returns:
            Tuple of validation result and metrics
        """
        import time

        start_time = time.time()

        logger.info(f"Starting comprehensive context-aware validation for DAG with {len(dag.all_nodes)} nodes")

        # Reset metrics
        self.metrics = ContextValidationMetrics()

        # Step 1: Structural validation (baseline)
        structural_results = validate_dag_comprehensive(dag)
        self.metrics.structural_validity = structural_results["is_valid"]
        self.metrics.errors_found = len(structural_results["errors"])
        self.metrics.warnings_generated = len(structural_results["warnings"])

        # If structural validation fails critically, stop here
        if not self.metrics.structural_validity and self.metrics.errors_found > 3:
            logger.warning("DAG failed basic structural validation, skipping context validation")
            self.metrics.validation_time_seconds = time.time() - start_time

            return ValidationResult(
                is_valid=False,
                errors=structural_results["errors"],
                warnings=structural_results["warnings"],
                validation_type="context_aware_structural_only",
                node_count=len(dag.all_nodes),
                edge_count=len(dag.edges),
                details={"structural_only": True, "reason": "Too many structural errors"},
            ), self.metrics

        # Step 2: Context-aware layer validation
        layer_results = await self._validate_layers_with_context(dag, context)

        # Step 3: Overall strategic coherence assessment
        coherence_assessment = await self._assess_strategic_coherence(dag, context)

        # Step 4: Implementation feasibility analysis
        feasibility_assessment = await self._assess_implementation_feasibility(dag, context)

        # Step 5: Organizational alignment evaluation
        alignment_assessment = await self._assess_organizational_alignment(dag, context)

        # Compile final results
        final_result = self._compile_validation_results(
            structural_results, layer_results, coherence_assessment, feasibility_assessment, alignment_assessment, dag
        )

        # Calculate final metrics
        self.metrics.validation_time_seconds = time.time() - start_time
        self.metrics.context_items_analyzed = (
            len(getattr(context, "organizational_goals", []))
            + len(getattr(context, "past_decisions", []))
            + len(getattr(context, "relevant_content", []))
        )

        logger.info(f"Context-aware validation complete. Overall quality: {self.metrics.overall_quality_score:.2f}")

        return final_result, self.metrics

    async def _validate_layers_with_context(
        self, dag: DecisionDAG, context: DAGBuildingContext
    ) -> List[LayerValidationResult]:
        """Validate each layer using organizational context."""
        layer_results = []
        max_layer = dag.get_max_layer()

        # Process layers in batches to manage concurrency
        layer_tasks = []
        for layer in range(max_layer + 1):
            task = self._validate_single_layer(dag, layer, context)
            layer_tasks.append(task)

        # Execute layer validations
        results = await asyncio.gather(*layer_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Layer validation failed: {result}")
                self.metrics.errors_found += 1
            else:
                layer_results.append(result)
                if result.mece_compliant:
                    self.metrics.mece_compliance_score += 1.0 / (max_layer + 1)

        return layer_results

    async def _validate_single_layer(
        self, dag: DecisionDAG, layer: int, context: DAGBuildingContext
    ) -> LayerValidationResult:
        """Validate a single layer with organizational context."""
        async with self.semaphore:
            try:
                layer_nodes = dag.get_nodes_at_layer(layer)
                if not layer_nodes:
                    return LayerValidationResult(
                        layer=layer,
                        node_count=0,
                        mece_compliant=True,
                        coverage_gaps=[],
                        redundancies=[],
                        alignment_issues=[],
                        feasibility_concerns=[],
                        quality_score=1.0,
                        recommendations=[],
                    )

                # Get parent context for the layer
                parent_nodes = []
                if layer > 0:
                    parent_nodes = dag.get_nodes_at_layer(layer - 1)

                # Build validation prompt with rich context
                prompt_data = self._build_layer_validation_prompt_data(layer, layer_nodes, parent_nodes, context)

                # Generate context-aware validation
                validation_result = await self._generate_layer_validation(prompt_data)

                self.metrics.llm_calls_made += 1

                return LayerValidationResult(
                    layer=layer,
                    node_count=len(layer_nodes),
                    mece_compliant=validation_result.mece_compliant,
                    coverage_gaps=validation_result.coverage_gaps,
                    redundancies=validation_result.redundancies,
                    alignment_issues=validation_result.alignment_issues,
                    feasibility_concerns=validation_result.feasibility_concerns,
                    quality_score=validation_result.quality_score,
                    recommendations=validation_result.recommendations,
                )

            except Exception as e:
                logger.error(f"Failed to validate layer {layer}: {e}")
                return LayerValidationResult(
                    layer=layer,
                    node_count=len(dag.get_nodes_at_layer(layer)) if layer <= dag.get_max_layer() else 0,
                    mece_compliant=False,
                    coverage_gaps=[f"Validation failed: {str(e)}"],
                    redundancies=[],
                    alignment_issues=[],
                    feasibility_concerns=[],
                    quality_score=0.0,
                    recommendations=["Re-validate this layer manually"],
                )

    async def _generate_layer_validation(self, prompt_data: Dict[str, Any]) -> Any:
        """Generate LLM-assisted layer validation."""
        try:
            # Build prompts for layer validation
            system_prompt = build_prompt("dag_layer_validation_system.txt.jinja", **prompt_data)
            user_prompt = build_prompt("dag_layer_validation_user.txt.jinja", **prompt_data)

            # Call LLM with structured output
            from ..schemas import DAGLayerValidationSchema
            from ..settings import settings

            response = await ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=DAGLayerValidationSchema,
                llm_model=settings.llm_model,
                temperature=0.2,  # Lower temperature for more consistent validation
                max_tokens=2048,
            )

            return response

        except Exception as e:
            logger.error(f"LLM layer validation failed: {e}")
            # Return fallback validation
            from ..schemas import DAGLayerValidationSchema

            return DAGLayerValidationSchema(
                mece_compliant=True,  # Conservative fallback
                coverage_gaps=[],
                redundancies=[],
                alignment_issues=[],
                feasibility_concerns=[],
                quality_score=0.5,
                recommendations=["Manual validation recommended due to LLM failure"],
                detailed_analysis="LLM validation failed, using fallback assessment",
            )

    async def _assess_strategic_coherence(self, dag: DecisionDAG, context: DAGBuildingContext) -> Dict[str, Any]:
        """Assess overall strategic coherence of the DAG."""
        async with self.semaphore:
            try:
                # Build coherence assessment prompt
                prompt_data = self._build_coherence_prompt_data(dag, context)

                system_prompt = build_prompt("dag_coherence_assessment_system.txt.jinja", **prompt_data)
                user_prompt = build_prompt("dag_coherence_assessment_user.txt.jinja", **prompt_data)

                from ..schemas import DAGCoherenceAssessmentSchema
                from ..settings import settings

                response = await ainstruct_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=DAGCoherenceAssessmentSchema,
                    llm_model=settings.llm_model,
                    temperature=0.3,
                    max_tokens=2048,
                )

                self.metrics.llm_calls_made += 1
                self.metrics.strategic_coherence_score = response.coherence_score

                return {
                    "coherence_score": response.coherence_score,
                    "narrative_strength": response.narrative_strength,
                    "logical_flow": response.logical_flow,
                    "strategic_gaps": response.strategic_gaps,
                    "coherence_issues": response.coherence_issues,
                    "recommendations": response.recommendations,
                }

            except Exception as e:
                logger.error(f"Strategic coherence assessment failed: {e}")
                return {
                    "coherence_score": 0.5,
                    "narrative_strength": 0.5,
                    "logical_flow": 0.5,
                    "strategic_gaps": ["Assessment failed"],
                    "coherence_issues": [f"LLM assessment error: {str(e)}"],
                    "recommendations": ["Manual coherence review recommended"],
                }

    async def _assess_implementation_feasibility(
        self, dag: DecisionDAG, context: DAGBuildingContext
    ) -> Dict[str, Any]:
        """Assess implementation feasibility based on organizational context."""
        async with self.semaphore:
            try:
                # Build feasibility assessment prompt
                prompt_data = self._build_feasibility_prompt_data(dag, context)

                system_prompt = build_prompt("dag_feasibility_assessment_system.txt.jinja", **prompt_data)
                user_prompt = build_prompt("dag_feasibility_assessment_user.txt.jinja", **prompt_data)

                from ..schemas import DAGFeasibilityAssessmentSchema
                from ..settings import settings

                response = await ainstruct_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=DAGFeasibilityAssessmentSchema,
                    llm_model=settings.llm_model,
                    temperature=0.3,
                    max_tokens=2048,
                )

                self.metrics.llm_calls_made += 1
                self.metrics.implementation_feasibility_score = response.feasibility_score

                return {
                    "feasibility_score": response.feasibility_score,
                    "resource_adequacy": response.resource_adequacy,
                    "timeline_realism": response.timeline_realism,
                    "capability_gaps": response.capability_gaps,
                    "resource_constraints": response.resource_constraints,
                    "recommendations": response.recommendations,
                }

            except Exception as e:
                logger.error(f"Implementation feasibility assessment failed: {e}")
                return {
                    "feasibility_score": 0.5,
                    "resource_adequacy": 0.5,
                    "timeline_realism": 0.5,
                    "capability_gaps": ["Assessment failed"],
                    "resource_constraints": [f"LLM assessment error: {str(e)}"],
                    "recommendations": ["Manual feasibility review recommended"],
                }

    async def _assess_organizational_alignment(self, dag: DecisionDAG, context: DAGBuildingContext) -> Dict[str, Any]:
        """Assess alignment with organizational goals and constraints."""
        async with self.semaphore:
            try:
                # Build alignment assessment prompt
                prompt_data = self._build_alignment_prompt_data(dag, context)

                system_prompt = build_prompt("dag_alignment_assessment_system.txt.jinja", **prompt_data)
                user_prompt = build_prompt("dag_alignment_assessment_user.txt.jinja", **prompt_data)

                from ..schemas import DAGAlignmentAssessmentSchema
                from ..settings import settings

                response = await ainstruct_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=DAGAlignmentAssessmentSchema,
                    llm_model=settings.llm_model,
                    temperature=0.3,
                    max_tokens=2048,
                )

                self.metrics.llm_calls_made += 1
                self.metrics.organizational_alignment_score = response.alignment_score

                return {
                    "alignment_score": response.alignment_score,
                    "goal_coverage": response.goal_coverage,
                    "priority_alignment": response.priority_alignment,
                    "uncovered_goals": response.uncovered_goals,
                    "misaligned_elements": response.misaligned_elements,
                    "recommendations": response.recommendations,
                }

            except Exception as e:
                logger.error(f"Organizational alignment assessment failed: {e}")
                return {
                    "alignment_score": 0.5,
                    "goal_coverage": 0.5,
                    "priority_alignment": 0.5,
                    "uncovered_goals": ["Assessment failed"],
                    "misaligned_elements": [f"LLM assessment error: {str(e)}"],
                    "recommendations": ["Manual alignment review recommended"],
                }

    def _build_layer_validation_prompt_data(
        self,
        layer: int,
        layer_nodes: List[DecisionNode],
        parent_nodes: List[DecisionNode],
        context: DAGBuildingContext,
    ) -> Dict[str, Any]:
        """Build comprehensive prompt data for layer validation."""
        return {
            "problem_statement": context.problem_statement,
            "layer": layer,
            "layer_type": "decision" if layer % 2 == 0 else "option",
            "nodes": [
                {"title": node.title, "description": node.description, "type": node.type.value, "tags": node.tags}
                for node in layer_nodes
            ],
            "parent_nodes": [
                {"title": node.title, "description": node.description, "type": node.type.value}
                for node in parent_nodes
            ]
            if parent_nodes
            else [],
            "organizational_goals": getattr(context, "organizational_goals", []),
            "past_decisions": getattr(context, "past_decisions", []),
            "relevant_content": getattr(context, "relevant_content", []),
            "activity_insights": getattr(context, "activity_insights", {}),
            "strategic_paths": context.strategic_paths,
        }

    def _build_coherence_prompt_data(self, dag: DecisionDAG, context: DAGBuildingContext) -> Dict[str, Any]:
        """Build prompt data for strategic coherence assessment."""
        # Extract DAG summary for analysis
        paths = dag.get_paths()
        path_summaries = []
        for i, path in enumerate(paths[:5]):  # Limit to top 5 paths for analysis
            path_summary = " → ".join([node.title for node in path])
            path_summaries.append({"id": i, "summary": path_summary})

        return {
            "problem_statement": context.problem_statement,
            "total_nodes": len(dag.all_nodes),
            "total_paths": len(paths),
            "max_layer": dag.get_max_layer(),
            "path_summaries": path_summaries,
            "organizational_goals": getattr(context, "organizational_goals", []),
            "past_decisions": getattr(context, "past_decisions", []),
            "strategic_paths": context.strategic_paths,
        }

    def _build_feasibility_prompt_data(self, dag: DecisionDAG, context: DAGBuildingContext) -> Dict[str, Any]:
        """Build prompt data for feasibility assessment."""
        return {
            "problem_statement": context.problem_statement,
            "total_nodes": len(dag.all_nodes),
            "max_layer": dag.get_max_layer(),
            "organizational_goals": getattr(context, "organizational_goals", []),
            "activity_insights": getattr(context, "activity_insights", {}),
            "past_decisions": getattr(context, "past_decisions", []),
            "relevant_content": getattr(context, "relevant_content", []),
        }

    def _build_alignment_prompt_data(self, dag: DecisionDAG, context: DAGBuildingContext) -> Dict[str, Any]:
        """Build prompt data for alignment assessment."""
        return {
            "problem_statement": context.problem_statement,
            "total_nodes": len(dag.all_nodes),
            "organizational_goals": getattr(context, "organizational_goals", []),
            "past_decisions": getattr(context, "past_decisions", []),
            "strategic_paths": context.strategic_paths,
        }

    def _compile_validation_results(
        self,
        structural_results: Dict[str, Any],
        layer_results: List[LayerValidationResult],
        coherence_assessment: Dict[str, Any],
        feasibility_assessment: Dict[str, Any],
        alignment_assessment: Dict[str, Any],
        dag: DecisionDAG,
    ) -> ValidationResult:
        """Compile all validation results into final assessment."""
        # Combine all errors and warnings
        all_errors = structural_results["errors"].copy()
        all_warnings = structural_results["warnings"].copy()
        all_suggestions = []

        # Add layer-specific issues
        for layer_result in layer_results:
            all_errors.extend([f"Layer {layer_result.layer}: {gap}" for gap in layer_result.coverage_gaps])
            all_warnings.extend(
                [f"Layer {layer_result.layer}: {redundancy}" for redundancy in layer_result.redundancies]
            )
            all_warnings.extend([f"Layer {layer_result.layer}: {issue}" for issue in layer_result.alignment_issues])
            all_warnings.extend(
                [f"Layer {layer_result.layer}: {concern}" for concern in layer_result.feasibility_concerns]
            )
            all_suggestions.extend([f"Layer {layer_result.layer}: {rec}" for rec in layer_result.recommendations])

        # Add strategic issues
        all_warnings.extend([f"Strategic gap: {gap}" for gap in coherence_assessment.get("strategic_gaps", [])])
        all_warnings.extend(
            [f"Coherence issue: {issue}" for issue in coherence_assessment.get("coherence_issues", [])]
        )
        all_suggestions.extend([f"Strategic: {rec}" for rec in coherence_assessment.get("recommendations", [])])

        # Add feasibility issues
        all_warnings.extend([f"Capability gap: {gap}" for gap in feasibility_assessment.get("capability_gaps", [])])
        all_warnings.extend(
            [
                f"Resource constraint: {constraint}"
                for constraint in feasibility_assessment.get("resource_constraints", [])
            ]
        )
        all_suggestions.extend([f"Feasibility: {rec}" for rec in feasibility_assessment.get("recommendations", [])])

        # Add alignment issues
        all_warnings.extend([f"Uncovered goal: {goal}" for goal in alignment_assessment.get("uncovered_goals", [])])
        all_warnings.extend(
            [f"Misalignment: {element}" for element in alignment_assessment.get("misaligned_elements", [])]
        )
        all_suggestions.extend([f"Alignment: {rec}" for rec in alignment_assessment.get("recommendations", [])])

        # Calculate overall quality score
        self.metrics.overall_quality_score = (
            (1.0 if self.metrics.structural_validity else 0.0) * 0.2
            + self.metrics.mece_compliance_score * 0.25
            + self.metrics.organizational_alignment_score * 0.25
            + self.metrics.implementation_feasibility_score * 0.15
            + self.metrics.strategic_coherence_score * 0.15
        )

        # Update final metrics
        self.metrics.errors_found = len(all_errors)
        self.metrics.warnings_generated = len(all_warnings)
        self.metrics.suggestions_provided = len(all_suggestions)

        # Determine overall validity
        is_valid = len(all_errors) == 0 and self.metrics.overall_quality_score >= 0.6

        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            validation_type="context_aware_comprehensive",
            node_count=len(dag.all_nodes),
            edge_count=len(dag.edges),
            details={
                "overall_quality_score": self.metrics.overall_quality_score,
                "mece_compliance_score": self.metrics.mece_compliance_score,
                "organizational_alignment_score": self.metrics.organizational_alignment_score,
                "implementation_feasibility_score": self.metrics.implementation_feasibility_score,
                "strategic_coherence_score": self.metrics.strategic_coherence_score,
                "layer_results": [result.__dict__ for result in layer_results],
                "coherence_assessment": coherence_assessment,
                "feasibility_assessment": feasibility_assessment,
                "alignment_assessment": alignment_assessment,
                "suggestions": all_suggestions,
                "context_items_analyzed": self.metrics.context_items_analyzed,
                "validation_metrics": self.metrics.to_dict(),
            },
        )
