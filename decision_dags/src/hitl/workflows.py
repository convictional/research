"""Specific HITL workflow implementations."""

import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from ..models import HITLResponse, HITLDecision, HITLWorkflowConfig, DecisionNode
from .interface import HITLInterface

logger = logging.getLogger(__name__)


class BaseHITLWorkflow(ABC):
    """Base class for HITL workflows."""

    def __init__(self, interface: HITLInterface, config: HITLWorkflowConfig):
        self.interface = interface
        self.config = config
        self.retry_count = 0

    @abstractmethod
    async def execute(self, **kwargs) -> HITLResponse:
        """Execute the workflow."""
        pass

    async def _retry_on_failure(self, workflow_func, **kwargs) -> HITLResponse:
        """Retry workflow execution on failure."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await workflow_func(**kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Workflow attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries:
                    logger.info(f"Retrying workflow (attempt {attempt + 2}/{self.config.max_retries + 1})")
                    continue
                else:
                    logger.error(f"Workflow failed after {self.config.max_retries + 1} attempts")
                    break

        # Return default response on final failure
        return HITLResponse(
            prompt_id=f"failed_workflow_{workflow_func.__name__}",
            decision=HITLDecision.APPROVE,
            feedback=f"Workflow failed after retries: {str(last_error)}",
            reasoning="Fallback to approval due to workflow failure",
        )


class LayerApprovalWorkflow(BaseHITLWorkflow):
    """Workflow for layer approval in DAG building."""

    async def execute(self, layer: int, nodes: List[DecisionNode], context: Dict[str, Any]) -> HITLResponse:
        """Execute layer approval workflow."""
        return await self._retry_on_failure(self._execute_layer_approval, layer=layer, nodes=nodes, context=context)

    async def _execute_layer_approval(
        self, layer: int, nodes: List[DecisionNode], context: Dict[str, Any]
    ) -> HITLResponse:
        """Internal layer approval execution."""
        logger.info(f"Executing layer approval for layer {layer}")

        # Pre-validation checks
        validation_issues = self._validate_layer(layer, nodes)
        if validation_issues:
            logger.warning(f"Layer {layer} has validation issues: {validation_issues}")
            context["validation_issues"] = validation_issues

        # Calculate layer quality metrics
        quality_metrics = self._calculate_layer_quality(nodes)
        context["quality_metrics"] = quality_metrics

        # Present prompt to user
        response = await self.interface.prompt_layer_approval(layer, nodes, context)

        # Post-process response
        response = await self._process_layer_approval_response(response, layer, nodes, context)

        logger.info(f"Layer approval completed for layer {layer}: {response.decision.value}")
        return response

    async def _process_layer_approval_response(
        self, response: HITLResponse, layer: int, nodes: List[DecisionNode], context: Dict[str, Any]
    ) -> HITLResponse:
        """Process and validate layer approval response."""
        if response.decision == HITLDecision.MODIFY:
            # Handle node selection for modification
            if response.modifications and "selected_nodes" in response.modifications:
                selected_node_ids = response.modifications["selected_nodes"]
                logger.info(f"User selected {len(selected_node_ids)} nodes for modification")

                # Store selected nodes for follow-up processing
                response.modifications["nodes_to_modify"] = [node for node in nodes if node.id in selected_node_ids]

        elif response.decision == HITLDecision.REJECT:
            # Add regeneration context
            if not response.feedback:
                response.feedback = "Layer rejected - will regenerate with improved prompts"

            # Store rejection reasons for prompt improvement
            if response.modifications:
                context["rejection_reasons"] = response.modifications

        return response

    def _validate_layer(self, layer: int, nodes: List[DecisionNode]) -> List[str]:
        """Validate layer structure and content."""
        issues = []

        if not nodes:
            issues.append("Layer is empty")
            return issues

        # Check node consistency
        expected_type = nodes[0].type
        for node in nodes:
            if node.layer != layer:
                issues.append(f"Node {node.id} has incorrect layer {node.layer}, expected {layer}")
            if node.type != expected_type:
                issues.append(f"Node {node.id} has inconsistent type {node.type.value}")

        # Check for duplicate titles (potential issue)
        titles = [node.title.lower().strip() for node in nodes]
        if len(set(titles)) < len(titles):
            issues.append("Some nodes have very similar titles")

        # Check confidence scores
        low_confidence_count = sum(1 for node in nodes if (node.confidence_score or 0.5) < 0.3)
        if low_confidence_count > len(nodes) * 0.3:  # More than 30% low confidence
            issues.append(f"{low_confidence_count} nodes have low confidence scores")

        return issues

    def _calculate_layer_quality(self, nodes: List[DecisionNode]) -> Dict[str, Any]:
        """Calculate quality metrics for the layer."""
        if not nodes:
            return {"error": "No nodes to analyze"}

        # Confidence metrics
        confidences = [node.confidence_score or 0.5 for node in nodes]

        # Diversity metrics
        unique_titles = len(set(node.title.lower().strip() for node in nodes))
        unique_tags = len(set(tag for node in nodes for tag in node.tags))

        return {
            "node_count": len(nodes),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "title_diversity": unique_titles / len(nodes),
            "total_unique_tags": unique_tags,
            "avg_description_length": sum(len(node.description) for node in nodes) / len(nodes),
        }


class NodeModificationWorkflow(BaseHITLWorkflow):
    """Workflow for node modification."""

    async def execute(
        self,
        node: DecisionNode,
        suggested_changes: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> HITLResponse:
        """Execute node modification workflow."""
        return await self._retry_on_failure(
            self._execute_node_modification, node=node, suggested_changes=suggested_changes, context=context or {}
        )

    async def _execute_node_modification(
        self, node: DecisionNode, suggested_changes: Optional[Dict[str, Any]], context: Dict[str, Any]
    ) -> HITLResponse:
        """Internal node modification execution."""
        logger.info(f"Executing node modification for node: {node.title}")

        # Analyze node for potential improvements
        analysis = self._analyze_node_quality(node)
        context["quality_analysis"] = analysis

        # Merge suggested changes with analysis
        combined_suggestions = self._combine_suggestions(suggested_changes, analysis)

        # Present prompt to user
        response = await self.interface.prompt_node_modification(node, combined_suggestions)

        # Validate and process response
        response = self._process_modification_response(response, node)

        logger.info(f"Node modification completed for {node.id}: {response.decision.value}")
        return response

    def _analyze_node_quality(self, node: DecisionNode) -> Dict[str, Any]:
        """Analyze node quality and suggest improvements."""
        suggestions = []

        # Title analysis
        if len(node.title) < 10:
            suggestions.append("Title could be more descriptive")
        elif len(node.title) > 80:
            suggestions.append("Title might be too long")

        # Description analysis
        if len(node.description) < 50:
            suggestions.append("Description could be more detailed")
        elif len(node.description) > 500:
            suggestions.append("Description might be too verbose")

        # Tags analysis
        if not node.tags:
            suggestions.append("Node could benefit from tags")
        elif len(node.tags) > 8:
            suggestions.append("Too many tags - consider reducing")

        # Confidence analysis
        confidence = node.confidence_score or 0.5
        if confidence < 0.4:
            suggestions.append("Low confidence - consider revision")

        return {
            "overall_score": self._calculate_node_score(node),
            "suggestions": suggestions,
            "title_length": len(node.title),
            "description_length": len(node.description),
            "tag_count": len(node.tags),
            "confidence": confidence,
        }

    def _calculate_node_score(self, node: DecisionNode) -> float:
        """Calculate overall quality score for node."""
        score = 0.0

        # Title score (0-0.25)
        title_len = len(node.title)
        if 20 <= title_len <= 60:
            score += 0.25
        elif 10 <= title_len <= 80:
            score += 0.15

        # Description score (0-0.35)
        desc_len = len(node.description)
        if 100 <= desc_len <= 300:
            score += 0.35
        elif 50 <= desc_len <= 500:
            score += 0.25

        # Tags score (0-0.15)
        tag_count = len(node.tags)
        if 2 <= tag_count <= 5:
            score += 0.15
        elif 1 <= tag_count <= 7:
            score += 0.10

        # Confidence score (0-0.25)
        confidence = node.confidence_score or 0.5
        score += confidence * 0.25

        return min(score, 1.0)

    def _combine_suggestions(
        self, suggested_changes: Optional[Dict[str, Any]], analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine external suggestions with quality analysis."""
        combined = {}

        if suggested_changes:
            combined.update(suggested_changes)

        # Add analysis-based suggestions
        if "suggestions" in analysis and analysis["suggestions"]:
            combined["quality_suggestions"] = analysis["suggestions"]

        combined["quality_score"] = analysis.get("overall_score", 0.5)

        return combined

    def _process_modification_response(self, response: HITLResponse, node: DecisionNode) -> HITLResponse:
        """Process and validate modification response."""
        if response.decision == HITLDecision.MODIFY:
            # Validate modification fields
            if response.modifications:
                valid_fields = {"title", "description", "tags", "metadata"}
                invalid_fields = set(response.modifications.keys()) - valid_fields

                if invalid_fields:
                    logger.warning(f"Invalid modification fields ignored: {invalid_fields}")
                    # Remove invalid fields
                    for field in invalid_fields:
                        response.modifications.pop(field, None)

                # Validate field values
                if "title" in response.modifications:
                    title = response.modifications["title"]
                    if not title or len(title.strip()) < 3:
                        logger.warning("Invalid title modification ignored")
                        response.modifications.pop("title", None)

                if "description" in response.modifications:
                    desc = response.modifications["description"]
                    if not desc or len(desc.strip()) < 10:
                        logger.warning("Invalid description modification ignored")
                        response.modifications.pop("description", None)

        return response


class PathSelectionWorkflow(BaseHITLWorkflow):
    """Workflow for path selection and prioritization."""

    async def execute(self, available_paths: List[List[DecisionNode]], context: Dict[str, Any]) -> HITLResponse:
        """Execute path selection workflow."""
        return await self._retry_on_failure(
            self._execute_path_selection, available_paths=available_paths, context=context
        )

    async def _execute_path_selection(
        self, available_paths: List[List[DecisionNode]], context: Dict[str, Any]
    ) -> HITLResponse:
        """Internal path selection execution."""
        logger.info(f"Executing path selection for {len(available_paths)} paths")

        # Analyze paths for quality and diversity
        path_analysis = self._analyze_paths(available_paths)
        context["path_analysis"] = path_analysis

        # Add path recommendations
        recommendations = self._generate_path_recommendations(available_paths, path_analysis)
        context["recommendations"] = recommendations

        # Present prompt to user
        response = await self.interface.prompt_path_selection(available_paths, context)

        # Process response
        response = self._process_path_selection_response(response, available_paths)

        logger.info(f"Path selection completed: {response.decision.value}")
        return response

    def _analyze_paths(self, paths: List[List[DecisionNode]]) -> Dict[str, Any]:
        """Analyze available paths for quality and characteristics."""
        if not paths:
            return {"error": "No paths to analyze"}

        path_metrics = []
        for i, path in enumerate(paths):
            if not path:
                continue

            # Calculate path metrics
            confidences = [node.confidence_score or 0.5 for node in path]
            unique_tags = set(tag for node in path for tag in node.tags)

            metrics = {
                "path_index": i,
                "length": len(path),
                "avg_confidence": sum(confidences) / len(confidences),
                "min_confidence": min(confidences),
                "total_tags": len(unique_tags),
                "start_node": path[0].title if path else "Empty",
                "end_node": path[-1].title if path else "Empty",
                "node_types": [node.type.value for node in path],
            }
            path_metrics.append(metrics)

        # Overall analysis
        all_confidences = [m["avg_confidence"] for m in path_metrics]
        all_lengths = [m["length"] for m in path_metrics]

        return {
            "total_paths": len(paths),
            "path_metrics": path_metrics,
            "avg_path_confidence": sum(all_confidences) / len(all_confidences) if all_confidences else 0,
            "avg_path_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
            "confidence_range": [min(all_confidences), max(all_confidences)] if all_confidences else [0, 0],
            "length_range": [min(all_lengths), max(all_lengths)] if all_lengths else [0, 0],
        }

    def _generate_path_recommendations(self, paths: List[List[DecisionNode]], analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for path selection."""
        recommendations = []

        if "path_metrics" not in analysis:
            return ["Unable to analyze paths"]

        path_metrics = analysis["path_metrics"]

        # Recommend highest confidence paths
        if path_metrics:
            sorted_by_confidence = sorted(path_metrics, key=lambda x: x["avg_confidence"], reverse=True)
            top_confidence = sorted_by_confidence[0]

            if top_confidence["avg_confidence"] > 0.7:
                recommendations.append(
                    f"Path {top_confidence['path_index'] + 1} has highest confidence ({top_confidence['avg_confidence']:.2f})"
                )

        # Recommend diverse path selection
        if len(paths) > 3:
            recommendations.append("Consider selecting 2-3 most promising paths to manage complexity")

        # Warn about low confidence paths
        low_confidence_paths = [m for m in path_metrics if m["avg_confidence"] < 0.4]
        if low_confidence_paths:
            recommendations.append(f"{len(low_confidence_paths)} paths have low confidence - consider regeneration")

        return recommendations

    def _process_path_selection_response(
        self, response: HITLResponse, available_paths: List[List[DecisionNode]]
    ) -> HITLResponse:
        """Process path selection response."""
        if response.decision == HITLDecision.MODIFY:
            # Handle specific path selection
            if response.modifications and "selected_paths" in response.modifications:
                selected_indices = response.modifications["selected_paths"]

                # Validate path indices
                valid_indices = []
                for idx in selected_indices:
                    if isinstance(idx, int) and 0 <= idx < len(available_paths):
                        valid_indices.append(idx)
                    else:
                        logger.warning(f"Invalid path index ignored: {idx}")

                response.modifications["selected_paths"] = valid_indices
                response.modifications["selected_path_count"] = len(valid_indices)

                logger.info(f"User selected {len(valid_indices)} specific paths")

        return response
