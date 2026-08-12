"""HITL interface for user interaction during DAG building."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models import HITLPrompt, HITLResponse, HITLDecision, HITLPromptType, DecisionNode, DecisionDAG

logger = logging.getLogger(__name__)


class HITLInterface:
    """Interface for human interaction during DAG building."""

    def __init__(self, auto_approve_threshold: float = 0.8):
        self.auto_approve_threshold = auto_approve_threshold
        self.pending_prompts: Dict[str, HITLPrompt] = {}
        self.response_futures: Dict[str, asyncio.Future] = {}

    async def prompt_layer_approval(
        self, layer: int, nodes: List[DecisionNode], context: Dict[str, Any]
    ) -> HITLResponse:
        """Prompt user for layer approval."""
        # Check if we can auto-approve based on confidence
        if self._can_auto_approve_layer(nodes):
            logger.info(f"Auto-approving layer {layer} due to high confidence")
            return self._create_auto_approval_response("layer_approval", layer)

        prompt = HITLPrompt(
            prompt_type=HITLPromptType.LAYER_APPROVAL,
            title=f"Review Layer {layer} ({len(nodes)} nodes)",
            description=self._format_layer_description(layer, nodes),
            context={
                "layer": layer,
                "node_count": len(nodes),
                "nodes": [self._node_summary(node) for node in nodes],
                "dag_context": context,
            },
            options=[
                {"key": "approve", "label": "Approve all nodes", "description": "Continue with all generated nodes"},
                {"key": "reject", "label": "Reject layer", "description": "Regenerate this entire layer"},
                {"key": "modify", "label": "Modify nodes", "description": "Select which nodes to keep/modify"},
                {"key": "stop", "label": "Stop building", "description": "Stop DAG construction here"},
            ],
            default_action=HITLDecision.APPROVE,
            timeout_seconds=300,
        )

        return await self._present_prompt(prompt)

    async def prompt_node_modification(
        self, node: DecisionNode, suggested_changes: Optional[Dict[str, Any]] = None
    ) -> HITLResponse:
        """Prompt user for node modification."""
        prompt = HITLPrompt(
            prompt_type=HITLPromptType.NODE_MODIFICATION,
            title=f"Modify Node: {node.title}",
            description=self._format_node_modification_description(node, suggested_changes),
            context={
                "node_id": node.id,
                "current_title": node.title,
                "current_description": node.description,
                "current_tags": node.tags,
                "suggested_changes": suggested_changes or {},
                "layer": node.layer,
                "type": node.type.value,
            },
            options=[
                {"key": "approve", "label": "Keep as-is", "description": "No changes needed"},
                {"key": "modify", "label": "Apply changes", "description": "Modify title, description, or tags"},
                {"key": "reject", "label": "Remove node", "description": "Remove this node entirely"},
            ],
            default_action=HITLDecision.APPROVE,
            timeout_seconds=180,
        )

        return await self._present_prompt(prompt)

    async def prompt_path_selection(
        self, available_paths: List[List[DecisionNode]], context: Dict[str, Any]
    ) -> HITLResponse:
        """Prompt user for path selection/prioritization."""
        prompt = HITLPrompt(
            prompt_type=HITLPromptType.PATH_SELECTION,
            title=f"Select Paths to Continue ({len(available_paths)} available)",
            description=self._format_path_selection_description(available_paths),
            context={
                "path_count": len(available_paths),
                "paths": [self._path_summary(path) for path in available_paths],
                "dag_context": context,
            },
            options=[
                {"key": "continue", "label": "Continue all paths", "description": "Develop all available paths"},
                {"key": "modify", "label": "Select specific paths", "description": "Choose which paths to continue"},
                {"key": "stop", "label": "Stop here", "description": "Stop expansion and finalize DAG"},
            ],
            default_action=HITLDecision.CONTINUE,
            timeout_seconds=240,
        )

        return await self._present_prompt(prompt)

    async def prompt_quality_review(self, dag: DecisionDAG, quality_metrics: Dict[str, Any]) -> HITLResponse:
        """Prompt user for overall DAG quality review."""
        prompt = HITLPrompt(
            prompt_type=HITLPromptType.QUALITY_REVIEW,
            title="Review DAG Quality",
            description=self._format_quality_review_description(dag, quality_metrics),
            context={
                "node_count": len(dag.all_nodes),
                "edge_count": len(dag.edges),
                "layer_count": dag.get_max_layer() + 1,
                "quality_metrics": quality_metrics,
            },
            options=[
                {"key": "approve", "label": "Accept DAG", "description": "DAG quality is acceptable"},
                {"key": "modify", "label": "Request improvements", "description": "Suggest specific improvements"},
                {"key": "reject", "label": "Rebuild DAG", "description": "Start over with different approach"},
            ],
            default_action=HITLDecision.APPROVE,
            timeout_seconds=420,
        )

        return await self._present_prompt(prompt)

    async def _present_prompt(self, prompt: HITLPrompt) -> HITLResponse:
        """Present a prompt to the user and wait for response."""
        logger.info(f"Presenting HITL prompt: {prompt.title}")

        # Store prompt and create future for response
        self.pending_prompts[prompt.prompt_id] = prompt
        response_future = asyncio.Future()
        self.response_futures[prompt.prompt_id] = response_future

        # Print prompt to console (in a real implementation, this would be a proper UI)
        self._display_prompt(prompt)

        try:
            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=prompt.timeout_seconds)
            return response

        except asyncio.TimeoutError:
            logger.warning(f"Prompt {prompt.prompt_id} timed out, using default action")
            return self._create_timeout_response(prompt)

        finally:
            # Clean up
            self.pending_prompts.pop(prompt.prompt_id, None)
            self.response_futures.pop(prompt.prompt_id, None)

    def submit_response(
        self,
        prompt_id: str,
        decision: HITLDecision,
        feedback: Optional[str] = None,
        modifications: Optional[Dict[str, Any]] = None,
        reasoning: Optional[str] = None,
    ) -> bool:
        """Submit a response to a pending prompt."""
        if prompt_id not in self.response_futures:
            logger.error(f"No pending prompt found for ID: {prompt_id}")
            return False

        prompt = self.pending_prompts.get(prompt_id)
        if not prompt:
            logger.error(f"Prompt data not found for ID: {prompt_id}")
            return False

        response_time = (datetime.utcnow() - prompt.created_at).total_seconds()

        response = HITLResponse(
            prompt_id=prompt_id,
            decision=decision,
            feedback=feedback,
            modifications=modifications,
            reasoning=reasoning,
            response_time_seconds=response_time,
        )

        future = self.response_futures[prompt_id]
        if not future.done():
            future.set_result(response)
            logger.info(f"Response submitted for prompt {prompt_id}: {decision.value}")
            return True

        return False

    def _can_auto_approve_layer(self, nodes: List[DecisionNode]) -> bool:
        """Check if layer can be auto-approved based on confidence."""
        if not nodes:
            return False

        # Calculate average confidence
        confidences = [node.confidence_score or 0.5 for node in nodes]
        avg_confidence = sum(confidences) / len(confidences)

        return avg_confidence >= self.auto_approve_threshold

    def _create_auto_approval_response(self, prompt_type: str, context_id: Any) -> HITLResponse:
        """Create an automatic approval response."""
        return HITLResponse(
            prompt_id=f"auto_{prompt_type}_{context_id}",
            decision=HITLDecision.APPROVE,
            feedback="Auto-approved due to high confidence",
            reasoning="Confidence scores exceeded auto-approval threshold",
            response_time_seconds=0.0,
        )

    def _create_timeout_response(self, prompt: HITLPrompt) -> HITLResponse:
        """Create a timeout response with default action."""
        default_decision = prompt.default_action or HITLDecision.APPROVE

        return HITLResponse(
            prompt_id=prompt.prompt_id,
            decision=default_decision,
            feedback="Response timed out, using default action",
            reasoning=f"No response received within {prompt.timeout_seconds} seconds",
            response_time_seconds=prompt.timeout_seconds,
        )

    def _display_prompt(self, prompt: HITLPrompt) -> None:
        """Display prompt to user (console implementation)."""
        print(f"\n{'=' * 60}")
        print(f"HITL PROMPT: {prompt.title}")
        print(f"{'=' * 60}")
        print(f"Type: {prompt.prompt_type.value}")
        print(f"Description: {prompt.description}")
        print("\nOptions:")

        for i, option in enumerate(prompt.options, 1):
            print(f"  {i}. {option['label']} - {option['description']}")

        if prompt.default_action:
            print(f"\nDefault (timeout): {prompt.default_action.value}")

        print(f"Timeout: {prompt.timeout_seconds} seconds")
        print(f"Prompt ID: {prompt.prompt_id}")
        print(f"\nTo respond, call: submit_response('{prompt.prompt_id}', HITLDecision.<DECISION>)")
        print(f"{'=' * 60}\n")

    def _format_layer_description(self, layer: int, nodes: List[DecisionNode]) -> str:
        """Format layer description for user review."""
        node_summaries = []
        for i, node in enumerate(nodes[:5], 1):  # Show first 5 nodes
            node_summaries.append(f"{i}. {node.title} ({node.type.value})")

        if len(nodes) > 5:
            node_summaries.append(f"... and {len(nodes) - 5} more nodes")

        avg_confidence = sum(node.confidence_score or 0.5 for node in nodes) / len(nodes)

        return f"""Layer {layer} contains {len(nodes)} {nodes[0].type.value} nodes:

{chr(10).join(node_summaries)}

Average confidence: {avg_confidence:.2f}

Please review and decide how to proceed."""

    def _format_node_modification_description(
        self, node: DecisionNode, suggested_changes: Optional[Dict[str, Any]]
    ) -> str:
        """Format node modification description."""
        desc = f"""Current Node Details:
Title: {node.title}
Description: {node.description}
Type: {node.type.value}
Layer: {node.layer}
Tags: {", ".join(node.tags) if node.tags else "None"}"""

        if suggested_changes:
            desc += "\n\nSuggested Changes:\n"
            for key, value in suggested_changes.items():
                desc += f"- {key}: {value}\n"

        return desc

    def _format_path_selection_description(self, paths: List[List[DecisionNode]]) -> str:
        """Format path selection description."""
        path_summaries = []
        for i, path in enumerate(paths[:3], 1):  # Show first 3 paths
            path_desc = " → ".join(
                [node.title[:30] + "..." if len(node.title) > 30 else node.title for node in path[-3:]]
            )
            path_summaries.append(f"{i}. {path_desc} ({len(path)} nodes)")

        if len(paths) > 3:
            path_summaries.append(f"... and {len(paths) - 3} more paths")

        return f"""Available paths for continued development:

{chr(10).join(path_summaries)}

Select which paths to continue developing or choose to continue all paths."""

    def _format_quality_review_description(self, dag: DecisionDAG, quality_metrics: Dict[str, Any]) -> str:
        """Format quality review description."""
        return f"""DAG Construction Complete!

Statistics:
- Nodes: {len(dag.all_nodes)}
- Edges: {len(dag.edges)}
- Layers: {dag.get_max_layer() + 1}
- Root nodes: {len(dag.root_nodes)}

Quality Metrics:
{chr(10).join([f"- {k}: {v}" for k, v in quality_metrics.items()])}

Please review the overall DAG quality and decide whether to accept, modify, or rebuild."""

    def _node_summary(self, node: DecisionNode) -> Dict[str, Any]:
        """Create a summary of a node for context."""
        return {
            "id": node.id,
            "title": node.title,
            "description": node.description[:100] + "..." if len(node.description) > 100 else node.description,
            "type": node.type.value,
            "layer": node.layer,
            "confidence": node.confidence_score or 0.5,
            "tags": node.tags[:3],  # First 3 tags
        }

    def _path_summary(self, path: List[DecisionNode]) -> Dict[str, Any]:
        """Create a summary of a path for context."""
        return {
            "length": len(path),
            "start_node": path[0].title if path else "Empty",
            "end_node": path[-1].title if path else "Empty",
            "avg_confidence": sum(node.confidence_score or 0.5 for node in path) / len(path) if path else 0.0,
            "node_types": [node.type.value for node in path],
        }
