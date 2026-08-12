"""JSON-based mutation engine for LLM-guided path evolution."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt

from ..models import (
    DecisionDAG,
    DecisionNode,
    DecisionEdge,
    MutationProposal,
    EdgeType,
    DecisionReasoningType,
    DecisionType,
    NodeType,
)
from ..schemas import MutationProposalsSchema, MutationDiffSchema
from ..settings import settings

logger = logging.getLogger(__name__)


class JSONMutationEngine:
    """Handles LLM-guided mutations on path DAGs using JSON representation."""

    def __init__(self):
        self.llm_model = settings.llm_model
        self.mutation_counter = 0

    async def dag_to_json(self, dag: DecisionDAG) -> Dict[str, Any]:
        """Convert a DecisionDAG to JSON representation for LLM processing."""
        try:
            nodes_json = []
            for node in dag.all_nodes.values():
                node_json = {
                    "id": node.id,
                    "layer": node.layer,
                    "type": node.type.value,
                    "title": node.title,
                    "description": node.description,
                    "decision_type": node.decision_type.value if node.decision_type else None,
                    "goal_impacts": node.goal_impacts,
                    "people_impacted": node.people_impacted,
                    "resource_requirements": node.resource_requirements,
                    "tags": node.tags,
                    "metadata": node.metadata,
                }
                nodes_json.append(node_json)

            edges_json = []
            for i, edge in enumerate(dag.edges):
                edge_json = {
                    "id": f"edge_{i}",
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "type": edge.edge_type.value if edge.edge_type else "decision_to_option",
                    "relationship": edge.relationship,
                    "conditions": edge.conditions,
                    "cost_estimate": edge.cost_estimate,
                    "timeline_estimate": edge.timeline_estimate,
                    "implementation_risks": edge.implementation_risks or [],
                    "success_factors": getattr(edge, "success_factors", []),
                    "metadata": edge.metadata,
                }
                edges_json.append(edge_json)

            dag_json = {
                "id": dag.id,
                "nodes": nodes_json,
                "edges": edges_json,
                "root_node_ids": [node.id for node in dag.root_nodes],
                "metadata": dag.metadata,
            }

            return dag_json

        except Exception as e:
            logger.error(f"Error converting DAG to JSON: {e}")
            raise

    async def json_to_dag(self, dag_json: Dict[str, Any], dag_id: str) -> DecisionDAG:
        """Convert JSON representation back to DecisionDAG."""
        try:
            # Create new DAG
            dag = DecisionDAG(id=dag_id, metadata=dag_json.get("metadata", {}))

            # Create nodes
            for node_json in dag_json["nodes"]:
                # Handle decision_type properly for decision nodes
                node_type = NodeType(node_json["type"])
                decision_type_str = node_json.get("decision_type")

                if node_type == NodeType.DECISION:
                    if decision_type_str:
                        # Try to convert to valid enum, with fallback
                        try:
                            decision_type = DecisionType(decision_type_str)
                        except ValueError:
                            logger.warning(f"Invalid decision_type '{decision_type_str}', defaulting to STRATEGIC")
                            decision_type = DecisionType.STRATEGIC
                    else:
                        decision_type = DecisionType.STRATEGIC  # Default for decision nodes
                elif node_type == NodeType.OPTION:
                    decision_type = None  # Options cannot have decision_type

                node = DecisionNode(
                    id=node_json["id"],
                    layer=node_json["layer"],
                    type=node_type,
                    title=node_json["title"],
                    description=node_json["description"],
                    decision_type=decision_type,
                    goal_impacts=node_json.get("goal_impacts", {}),
                    people_impacted=node_json.get("people_impacted", []),
                    resource_requirements=node_json.get("resource_requirements", {}),
                    tags=node_json.get("tags", []),
                    metadata=node_json.get("metadata", {}),
                )
                dag.add_node(node)

            # Create edges
            for edge_json in dag_json["edges"]:
                # Convert type string to EdgeType enum
                edge_type_str = edge_json.get("type", "decision_to_option")
                edge_type = (
                    EdgeType.DECISION_TO_OPTION
                    if edge_type_str == "decision_to_option"
                    else EdgeType.OPTION_TO_DECISION
                )

                # Set decision_reasoning_type for option-to-decision edges
                decision_reasoning = None
                if edge_type == EdgeType.OPTION_TO_DECISION:
                    # Check if it's in the JSON, otherwise default to LOGICAL
                    reasoning_str = edge_json.get("decision_reasoning_type", "logical")
                    try:
                        decision_reasoning = DecisionReasoningType(reasoning_str)
                    except ValueError:
                        logger.warning(f"Invalid decision_reasoning_type '{reasoning_str}', defaulting to LOGICAL")
                        decision_reasoning = DecisionReasoningType.LOGICAL

                edge = DecisionEdge(
                    source_id=edge_json["source_id"],
                    target_id=edge_json["target_id"],
                    edge_type=edge_type,
                    condition=edge_json.get("conditions", [""])[0] if edge_json.get("conditions") else "",
                    decision_reasoning_type=decision_reasoning,
                    relationship=edge_json.get("relationship", ""),
                    conditions=edge_json.get("conditions", []),
                    cost_estimate=edge_json.get("cost_estimate"),
                    timeline_estimate=edge_json.get("timeline_estimate"),
                    implementation_risks=edge_json.get("implementation_risks", []),
                    metadata=edge_json.get("metadata", {}),
                )
                dag.add_edge(edge)

            return dag

        except Exception as e:
            logger.error(f"Error converting JSON to DAG: {e}")
            raise

    async def propose_mutations(
        self,
        dag_json: Dict[str, Any],
        objectives: List[str],
        current_scores: Dict[str, float],
        generation: int,
        context: str,
        fitness_scorecard: Any = None,
        inspiration_context: Optional[str] = None,
        dynamic_learning_context: Optional[str] = None,
        num_proposals: int = 3,
    ) -> List[MutationProposal]:
        """
        Use LLM to propose strategic mutations based on path weaknesses.

        Args:
            dag_json: JSON representation of the path DAG
            objectives: List of strategic objectives
            current_scores: Current fitness scores by dimension
            generation: Current generation number
            context: Domain/problem context
            organization_id: Organization ID for context
            fitness_scorecard: Detailed fitness scorecard
            inspiration_context: Formatted inspiration from successful variants
            dynamic_learning_context: Dynamic few-shot examples from evolution history
            num_proposals: Number of proposals to generate

        Returns:
            List of mutation proposals with confidence scores
        """
        try:
            # Build system prompt for mutation proposals
            system_prompt = build_prompt(
                "mutation_proposal_system.txt.jinja",
                objectives=objectives,
                generation=generation,
                exploration_mode=generation <= 2,  # Early generations are more exploratory
                has_inspiration=inspiration_context is not None,
            )

            # Build user prompt with path details and context
            user_prompt = build_prompt(
                "mutation_proposal_user.txt.jinja",
                dag_json=dag_json,
                current_scores=current_scores,
                fitness_scorecard=fitness_scorecard,
                context=context,
                inspiration_context=inspiration_context,
                dynamic_learning_context=dynamic_learning_context,
                num_proposals=num_proposals,
            )

            # Get mutation proposals from LLM using structured schema
            response = await ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm_model=settings.llm_model,  # Use configured model
                temperature=0.8,  # Higher temperature for creativity
                max_tokens=6000,
                response_model=MutationProposalsSchema,
            )

            # Extract proposals from structured response
            proposals = response.proposals

            # Filter by confidence threshold
            confident_proposals = [p for p in proposals if p.confidence >= 0.4]

            logger.info(f"Generated {len(confident_proposals)} confident mutation proposals")
            return confident_proposals

        except Exception as e:
            logger.error(f"Error proposing mutations: {e}")
            return []

    async def execute_mutation_plan(
        self,
        dag_json: Dict[str, Any],
        proposal: MutationProposal,
        context: str,
        organization_id: str,
        enable_self_correction: bool = True,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Execute a mutation plan using diff-based approach for efficiency.

        Args:
            dag_json: Original DAG JSON
            proposal: Mutation proposal to execute
            context: Domain context
            organization_id: Organization ID
            enable_self_correction: Whether to enable self-correction

        Returns:
            Tuple of (mutated_dag_json, applied_mutations)
        """
        try:
            # Build system prompt for mutation execution with diff approach
            system_prompt = build_prompt(
                "mutation_execution_diff_system.txt.jinja",
                mutation_type=proposal.mutation_type,
                enable_self_correction=enable_self_correction,
            )

            # Build user prompt with specific mutation instructions
            user_prompt = build_prompt(
                "mutation_execution_diff_user.txt.jinja", dag_json=dag_json, proposal=proposal, context=context
            )

            # Get mutation diff from LLM
            response = await ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm_model=settings.llm_model,
                temperature=0.3,  # Lower temperature for precision
                max_tokens=6000,  # Much smaller token requirement for diffs
                response_model=MutationDiffSchema,
            )

            # Apply the diff to create mutated DAG
            mutated_dag_json = await self.apply_mutation_diff(dag_json, response)

            # Validate the mutated DAG
            is_valid, validation_errors = await self.validate_dag_json(mutated_dag_json)

            if not is_valid and enable_self_correction:
                logger.info("Mutation produced invalid DAG, attempting self-correction")
                mutated_dag_json = await self._self_correct_mutation_diff(
                    dag_json, response, validation_errors, proposal, context
                )

            # Create mutation record
            mutation_record = {
                "mutation_id": f"mut_{self.mutation_counter}",
                "mutation_type": proposal.mutation_type,
                "description": proposal.description,
                "operations": proposal.operations,
                "confidence": proposal.confidence,
                "applied_at": datetime.now().isoformat(),
                "diff_operations": [op.dict() for op in response.operations],
            }

            self.mutation_counter += 1

            return mutated_dag_json, [mutation_record]

        except Exception as e:
            logger.error(f"Error executing mutation plan: {e}")
            # Return original DAG on failure
            return dag_json, []

    async def apply_mutation_diff(
        self, original_dag_json: Dict[str, Any], mutation_diff: MutationDiffSchema
    ) -> Dict[str, Any]:
        """
        Apply a mutation diff to a DAG JSON to produce the mutated version.

        Args:
            original_dag_json: Original DAG JSON
            mutation_diff: Diff operations to apply

        Returns:
            Mutated DAG JSON
        """
        # Deep copy the original to avoid mutations
        import copy

        dag_json = copy.deepcopy(original_dag_json)

        # Create lookup maps for efficient operations
        nodes_by_id = {node["id"]: node for node in dag_json["nodes"]}
        edges_list = dag_json["edges"]

        # Apply each operation in order
        for op in mutation_diff.operations:
            if op.operation == "add":
                # Add new node
                new_node = op.node_data.copy()
                new_node["id"] = op.node_id

                # Ensure required fields
                if "metadata" not in new_node:
                    new_node["metadata"] = {}
                if "tags" not in new_node:
                    new_node["tags"] = []

                # Add to nodes
                dag_json["nodes"].append(new_node)
                nodes_by_id[new_node["id"]] = new_node

                # Create edge to parent
                if op.parent_id and op.edge_to_parent:
                    edge_data = op.edge_to_parent.copy()
                    edge_data["id"] = f"edge_{len(edges_list)}"
                    edge_data["source_id"] = op.parent_id
                    edge_data["target_id"] = op.node_id

                    # Ensure edge type based on node types
                    source_node = nodes_by_id.get(op.parent_id)
                    if source_node:
                        if source_node["type"] == "decision":
                            edge_data["type"] = "decision_to_option"
                        else:
                            edge_data["type"] = "option_to_decision"

                    edges_list.append(edge_data)

            elif op.operation == "modify":
                # Modify existing node
                if op.node_id in nodes_by_id:
                    node = nodes_by_id[op.node_id]

                    # Update node properties
                    if op.node_data:
                        for key, value in op.node_data.items():
                            node[key] = value

                    # Update edges to children if specified
                    if op.child_edge_updates:
                        for edge_update in op.child_edge_updates:
                            # Find the edge to update
                            for edge in edges_list:
                                if edge["source_id"] == op.node_id and edge["target_id"] == edge_update.get(
                                    "target_id"
                                ):
                                    # Update edge properties
                                    edge.update(edge_update)
                                    break

            elif op.operation == "delete":
                # Delete node and associated edges
                if op.node_id in nodes_by_id:
                    # Remove from nodes list
                    dag_json["nodes"] = [node for node in dag_json["nodes"] if node["id"] != op.node_id]

                    # Remove all edges connected to this node
                    dag_json["edges"] = [
                        edge
                        for edge in edges_list
                        if edge["source_id"] != op.node_id and edge["target_id"] != op.node_id
                    ]

                    # Update the edges list reference
                    edges_list = dag_json["edges"]

                    # Remove from lookup
                    del nodes_by_id[op.node_id]

        return dag_json

    async def validate_dag_json(self, dag_json: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a DAG JSON structure."""
        errors = []

        try:
            # Check required fields
            if "nodes" not in dag_json:
                errors.append("Missing 'nodes' field")
            if "edges" not in dag_json:
                errors.append("Missing 'edges' field")

            # Validate nodes
            if "nodes" in dag_json:
                node_ids = set()
                for i, node in enumerate(dag_json["nodes"]):
                    if "id" not in node:
                        errors.append(f"Node {i} missing 'id' field")
                    elif node["id"] in node_ids:
                        errors.append(f"Duplicate node ID: {node['id']}")
                    else:
                        node_ids.add(node["id"])

                    # Check required node fields
                    for field in ["layer", "type", "title", "description"]:
                        if field not in node:
                            errors.append(f"Node {node.get('id', i)} missing '{field}' field")

                    # Validate alternating pattern
                    if "layer" in node and "type" in node:
                        layer = node["layer"]
                        node_type = node["type"]
                        if node_type == "decision" and layer % 2 != 0:
                            errors.append(f"Decision node {node['id']} on odd layer {layer}")
                        elif node_type == "option" and layer % 2 == 0:
                            errors.append(f"Option node {node['id']} on even layer {layer}")

            # Validate edges
            if "edges" in dag_json and "nodes" in dag_json:
                node_ids = {node["id"] for node in dag_json["nodes"]}
                edge_ids = set()

                for i, edge in enumerate(dag_json["edges"]):
                    if "id" not in edge:
                        errors.append(f"Edge {i} missing 'id' field")
                    elif edge["id"] in edge_ids:
                        errors.append(f"Duplicate edge ID: {edge['id']}")
                    else:
                        edge_ids.add(edge["id"])

                    # Check edge connectivity
                    if "source_id" not in edge:
                        errors.append(f"Edge {edge.get('id', i)} missing 'source_id' field")
                    elif edge["source_id"] not in node_ids:
                        errors.append(f"Edge {edge['id']} references non-existent source node: {edge['source_id']}")

                    if "target_id" not in edge:
                        errors.append(f"Edge {edge.get('id', i)} missing 'target_id' field")
                    elif edge["target_id"] not in node_ids:
                        errors.append(f"Edge {edge['id']} references non-existent target node: {edge['target_id']}")

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    async def _self_correct_mutation_diff(
        self,
        original_dag_json: Dict[str, Any],
        failed_diff: MutationDiffSchema,
        validation_errors: List[str],
        original_proposal: MutationProposal,
        context: str,
    ) -> Dict[str, Any]:
        """Attempt to self-correct a failed diff-based mutation."""
        try:
            system_prompt = build_prompt(
                "mutation_self_correction_diff_system.txt.jinja", validation_errors=validation_errors
            )

            user_prompt = build_prompt(
                "mutation_self_correction_diff_user.txt.jinja",
                original_dag_json=original_dag_json,
                failed_diff=failed_diff,
                validation_errors=validation_errors,
                original_proposal=original_proposal,
                context=context,
            )

            # Get corrected diff
            response = await ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm_model=settings.llm_model,
                temperature=0.2,  # Very low temperature for correction
                max_tokens=6000,  # Smaller for diffs
                response_model=MutationDiffSchema,
            )

            # Apply the corrected diff
            corrected_dag_json = await self.apply_mutation_diff(original_dag_json, response)

            # Validate the corrected DAG
            is_valid, _ = await self.validate_dag_json(corrected_dag_json)

            if is_valid:
                logger.info("Self-correction of diff successful")
                return corrected_dag_json
            else:
                logger.warning("Self-correction of diff failed, returning original DAG")
                return original_dag_json

        except Exception as e:
            logger.error(f"Error in diff self-correction: {e}")
            return original_dag_json
