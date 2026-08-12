"""HITL manager for coordinating human interaction workflows."""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from ..models import (
    HITLSessionState,
    HITLWorkflowConfig,
    HITLResponse,
    HITLDecision,
    DecisionNode,
    DecisionDAG,
    BuildPhase,
)
from .interface import HITLInterface
from .workflows import LayerApprovalWorkflow, NodeModificationWorkflow, PathSelectionWorkflow

logger = logging.getLogger(__name__)


class HITLManager:
    """Manager for coordinating HITL workflows during DAG building."""

    def __init__(self, config: HITLWorkflowConfig, interface: Optional[HITLInterface] = None):
        self.config = config
        self.interface = interface or HITLInterface(config.auto_approve_threshold)
        self.session = HITLSessionState()

        # Workflow instances
        self.layer_approval = LayerApprovalWorkflow(self.interface, config)
        self.node_modification = NodeModificationWorkflow(self.interface, config)
        self.path_selection = PathSelectionWorkflow(self.interface, config)

        # State tracking
        self.active_workflows: Dict[str, Any] = {}
        self.workflow_history: List[Dict[str, Any]] = []

        # Hook for custom workflow callbacks
        self.workflow_callbacks: Dict[str, Callable] = {}

    async def request_layer_approval(
        self, layer: int, nodes: List[DecisionNode], dag_context: Dict[str, Any], build_phase: BuildPhase
    ) -> HITLResponse:
        """Request approval for a generated layer."""
        if not self._should_request_approval(layer, nodes, build_phase):
            logger.info(f"Skipping approval for layer {layer} based on configuration")
            return self._create_auto_approval("layer_approval", layer)

        logger.info(f"Requesting layer approval for layer {layer} with {len(nodes)} nodes")

        workflow_id = f"layer_approval_{layer}_{datetime.utcnow().isoformat()}"
        self.active_workflows[workflow_id] = {
            "type": "layer_approval",
            "layer": layer,
            "node_count": len(nodes),
            "started_at": datetime.utcnow(),
        }

        try:
            response = await self.layer_approval.execute(layer=layer, nodes=nodes, context=dag_context)

            # Record workflow completion
            self._record_workflow_completion(workflow_id, response)
            self._update_session_state(response)

            return response

        except Exception as e:
            logger.error(f"Layer approval workflow failed: {e}")
            self._record_workflow_error(workflow_id, str(e))

            # Return default approval on error
            return self._create_auto_approval("layer_approval_error", layer)

        finally:
            self.active_workflows.pop(workflow_id, None)

    async def request_node_modification(
        self,
        node: DecisionNode,
        suggested_changes: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> HITLResponse:
        """Request node modification from user."""
        if not self.config.enable_node_modification:
            logger.info(f"Node modification disabled, auto-approving node {node.id}")
            return self._create_auto_approval("node_modification", node.id)

        logger.info(f"Requesting node modification for node: {node.title}")

        workflow_id = f"node_modification_{node.id}_{datetime.utcnow().isoformat()}"
        self.active_workflows[workflow_id] = {
            "type": "node_modification",
            "node_id": node.id,
            "node_title": node.title,
            "started_at": datetime.utcnow(),
        }

        try:
            response = await self.node_modification.execute(
                node=node, suggested_changes=suggested_changes, context=context or {}
            )

            self._record_workflow_completion(workflow_id, response)
            self._update_session_state(response)

            return response

        except Exception as e:
            logger.error(f"Node modification workflow failed: {e}")
            self._record_workflow_error(workflow_id, str(e))

            return self._create_auto_approval("node_modification_error", node.id)

        finally:
            self.active_workflows.pop(workflow_id, None)

    async def request_path_selection(
        self, available_paths: List[List[DecisionNode]], context: Dict[str, Any]
    ) -> HITLResponse:
        """Request path selection from user."""
        if not self.config.enable_path_selection:
            logger.info("Path selection disabled, continuing all paths")
            return self._create_auto_approval("path_selection", len(available_paths))

        logger.info(f"Requesting path selection for {len(available_paths)} paths")

        workflow_id = f"path_selection_{len(available_paths)}_{datetime.utcnow().isoformat()}"
        self.active_workflows[workflow_id] = {
            "type": "path_selection",
            "path_count": len(available_paths),
            "started_at": datetime.utcnow(),
        }

        try:
            response = await self.path_selection.execute(available_paths=available_paths, context=context)

            self._record_workflow_completion(workflow_id, response)
            self._update_session_state(response)

            return response

        except Exception as e:
            logger.error(f"Path selection workflow failed: {e}")
            self._record_workflow_error(workflow_id, str(e))

            return self._create_auto_approval("path_selection_error", len(available_paths))

        finally:
            self.active_workflows.pop(workflow_id, None)

    async def request_quality_review(self, dag: DecisionDAG, quality_metrics: Dict[str, Any]) -> HITLResponse:
        """Request final quality review of completed DAG."""
        logger.info(f"Requesting quality review for DAG with {len(dag.all_nodes)} nodes")

        workflow_id = f"quality_review_{dag.id}_{datetime.utcnow().isoformat()}"
        self.active_workflows[workflow_id] = {
            "type": "quality_review",
            "dag_id": dag.id,
            "node_count": len(dag.all_nodes),
            "started_at": datetime.utcnow(),
        }

        try:
            response = await self.interface.prompt_quality_review(dag, quality_metrics)

            self._record_workflow_completion(workflow_id, response)
            self._update_session_state(response)

            return response

        except Exception as e:
            logger.error(f"Quality review workflow failed: {e}")
            self._record_workflow_error(workflow_id, str(e))

            return self._create_auto_approval("quality_review_error", dag.id)

        finally:
            self.active_workflows.pop(workflow_id, None)

    def apply_node_modifications(self, node: DecisionNode, modifications: Dict[str, Any]) -> DecisionNode:
        """Apply user-requested modifications to a node."""
        logger.info(f"Applying modifications to node {node.id}: {list(modifications.keys())}")

        # Create a copy of the node with modifications
        node_dict = node.dict()

        # Apply modifications
        for key, value in modifications.items():
            if key in node_dict and value is not None:
                if key == "tags" and isinstance(value, str):
                    # Handle comma-separated tags
                    node_dict[key] = [tag.strip() for tag in value.split(",") if tag.strip()]
                else:
                    node_dict[key] = value
                logger.debug(f"Modified {key}: {node_dict[key]}")

        # Add modification metadata
        if "metadata" not in node_dict:
            node_dict["metadata"] = {}

        node_dict["metadata"]["human_modified"] = True
        node_dict["metadata"]["modification_timestamp"] = datetime.utcnow().isoformat()
        node_dict["metadata"]["original_values"] = {
            key: getattr(node, key) for key in modifications.keys() if hasattr(node, key)
        }

        return DecisionNode(**node_dict)

    def register_workflow_callback(self, workflow_type: str, callback: Callable) -> None:
        """Register a callback for workflow events."""
        self.workflow_callbacks[workflow_type] = callback
        logger.info(f"Registered callback for workflow type: {workflow_type}")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the current HITL session."""
        return {
            "session_id": self.session.session_id,
            "dag_id": self.session.dag_id,
            "session_duration_seconds": (datetime.utcnow() - self.session.session_start).total_seconds(),
            "total_prompts": len(self.session.prompt_history),
            "total_responses": len(self.session.response_history),
            "active_workflows": len(self.active_workflows),
            "workflow_history_count": len(self.workflow_history),
            "current_prompt": self.session.current_prompt.title if self.session.current_prompt else None,
            "last_activity": self.session.last_activity.isoformat(),
            "is_active": self.session.is_active,
        }

    def _should_request_approval(self, layer: int, nodes: List[DecisionNode], build_phase: BuildPhase) -> bool:
        """Determine if approval should be requested for this layer."""
        # Check if layer approval is enabled
        if not self.config.enable_layer_approval:
            return False

        # Check if this layer requires approval
        if layer not in self.config.require_approval_layers:
            return False

        # Check if we can skip due to high confidence
        if self.config.skip_approval_on_high_confidence:
            avg_confidence = sum(node.confidence_score or 0.5 for node in nodes) / len(nodes) if nodes else 0.0
            if avg_confidence >= self.config.auto_approve_threshold:
                return False

        return True

    def _create_auto_approval(self, context_type: str, context_id: Any) -> HITLResponse:
        """Create an automatic approval response."""
        return HITLResponse(
            prompt_id=f"auto_{context_type}_{context_id}",
            decision=HITLDecision.APPROVE,
            feedback=f"Auto-approved: {context_type}",
            reasoning="Automatic approval based on configuration or error fallback",
            response_time_seconds=0.0,
        )

    def _record_workflow_completion(self, workflow_id: str, response: HITLResponse) -> None:
        """Record completion of a workflow."""
        workflow_info = self.active_workflows.get(workflow_id, {})
        completion_record = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_info.get("type", "unknown"),
            "started_at": workflow_info.get("started_at"),
            "completed_at": datetime.utcnow(),
            "duration_seconds": (
                datetime.utcnow() - workflow_info.get("started_at", datetime.utcnow())
            ).total_seconds(),
            "decision": response.decision.value,
            "has_feedback": bool(response.feedback),
            "has_modifications": bool(response.modifications),
            "response_time_seconds": response.response_time_seconds,
            "success": True,
        }

        self.workflow_history.append(completion_record)
        logger.debug(f"Recorded workflow completion: {workflow_id}")

    def _record_workflow_error(self, workflow_id: str, error_message: str) -> None:
        """Record workflow error."""
        workflow_info = self.active_workflows.get(workflow_id, {})
        error_record = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_info.get("type", "unknown"),
            "started_at": workflow_info.get("started_at"),
            "error_at": datetime.utcnow(),
            "duration_seconds": (
                datetime.utcnow() - workflow_info.get("started_at", datetime.utcnow())
            ).total_seconds(),
            "error_message": error_message,
            "success": False,
        }

        self.workflow_history.append(error_record)
        logger.warning(f"Recorded workflow error: {workflow_id} - {error_message}")

    def _update_session_state(self, response: HITLResponse) -> None:
        """Update session state with new response."""
        self.session.response_history.append(response)
        self.session.last_activity = datetime.utcnow()

        # Update user preferences based on response patterns
        self._update_user_preferences(response)

    def _update_user_preferences(self, response: HITLResponse) -> None:
        """Update user preferences based on response patterns."""
        # Track decision patterns
        if "decision_patterns" not in self.session.user_preferences:
            self.session.user_preferences["decision_patterns"] = {}

        decision_key = response.decision.value
        patterns = self.session.user_preferences["decision_patterns"]
        patterns[decision_key] = patterns.get(decision_key, 0) + 1

        # Track response times to adjust timeouts
        if "avg_response_time" not in self.session.user_preferences:
            self.session.user_preferences["avg_response_time"] = []

        self.session.user_preferences["avg_response_time"].append(response.response_time_seconds)

        # Keep only last 10 response times
        if len(self.session.user_preferences["avg_response_time"]) > 10:
            self.session.user_preferences["avg_response_time"] = self.session.user_preferences["avg_response_time"][
                -10:
            ]
