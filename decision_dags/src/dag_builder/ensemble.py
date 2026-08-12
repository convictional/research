"""Main DAG builder ensemble orchestrating the construction process."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from common.instruct_llm import set_async_instructor_client
from common.prompt_template_engine import initialize_and_register_prompt_templates

from ..models import (
    DecisionDAG, DecisionNode, DecisionEdge, StrategicPath, DAGBuilderConfig,
    NodeType, EdgeType, DecisionType, DecisionReasoningType, BuildPhase, ValidationResult,
    HITLWorkflowConfig, HITLDecision, HITLResponse
)
from ..settings import settings
from .context import DAGBuildingContext
from .parallel_agent import ParallelNodeAgent
from .deduplicator import NodeDeduplicator
from .edge_enricher import EdgeEnricher
from .node_enricher import NodeEnricher
from .context_validator import ContextAwareDAGValidator
from ..hitl import HITLManager, HITLInterface
from ..context.database_context import DatabaseContextProvider

logger = logging.getLogger(__name__)


class DAGBuilderEnsemble:
    """Main orchestrator for DAG construction using parallel agents."""

    def __init__(
        self,
        config: DAGBuilderConfig | None = None,
        hitl_config: HITLWorkflowConfig | None = None,
        enable_hitl: bool = True
    ):
        self.config = config or self._create_default_config()
        self.hitl_config = hitl_config or HITLWorkflowConfig()
        self.enable_hitl = enable_hitl

        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)
        self.deduplicator = NodeDeduplicator(self.config.similarity_threshold)
        self.edge_enricher = EdgeEnricher(
            batch_size=15,
            max_concurrent=min(10, self.config.max_concurrent_agents)
        )
        self.node_enricher = NodeEnricher(
            batch_size=10,
            max_concurrent=min(5, self.config.max_concurrent_agents)
        )
        self.context_validator = ContextAwareDAGValidator(
            max_concurrent_validations=min(2, self.config.max_concurrent_agents)
        )
        self.database_context_provider = DatabaseContextProvider()

        # Initialize HITL components
        if self.enable_hitl:
            self.hitl_interface = HITLInterface(self.hitl_config.auto_approve_threshold)
            self.hitl_manager = HITLManager(self.hitl_config, self.hitl_interface)
        else:
            self.hitl_interface = None
            self.hitl_manager = None

        self._initialized = False

    def _create_default_config(self) -> DAGBuilderConfig:
        """Create default configuration from settings."""
        return DAGBuilderConfig(
            max_concurrent_agents=settings.max_concurrent_agents,
            agent_timeout=settings.agent_timeout,
            max_layers=settings.max_layers,
            max_children_per_node=settings.max_children_per_node,
            min_children_per_node=settings.min_children_per_node,
            similarity_threshold=settings.similarity_threshold,
            weak_similarity_threshold=settings.weak_similarity_threshold,
            generation_temperature=settings.generation_temperature,
            assessment_temperature=settings.assessment_temperature,
            max_retries=settings.max_retries
        )

    def _initialize_llm_client(self) -> None:
        """Initialize the LLM client and prompt templates."""
        if not self._initialized:
            # Initialize prompt templates
            prompts_path = settings.root / "src" / "prompts"
            initialize_and_register_prompt_templates(prompts_path)

            # Determine which API key to use based on the model
            api_key = (
                settings.anthropic_api_key
                if settings.llm_model.startswith("claude")
                else settings.openai_api_key
            )

            set_async_instructor_client(
                llm_model=settings.llm_model,
                api_key=api_key
            )
            self._initialized = True

    async def build_dag(
        self,
        problem_statement: str,
        strategic_paths: List[StrategicPath],
        organizational_goals: List[Dict[str, any]] | None = None,
        activity_insights: List[Dict[str, any]] | None = None,
        organization_id: Optional[str] = None
    ) -> DecisionDAG:
        """
        Build a complete DAG from problem statement and strategic paths.

        Args:
            problem_statement: The problem to solve
            strategic_paths: Initial strategic paths to build from
            organizational_goals: Optional organizational goals for context
            activity_insights: Optional activity insights for context
            organization_id: Optional organization ID for database context

        Returns:
            Complete DecisionDAG with comprehensive state tracking
        """
        # Initialize LLM client
        self._initialize_llm_client()

        # Fetch database context if organization_id is provided
        database_context = None
        if organization_id:
            try:
                logger.info(f"Fetching database context for organization: {organization_id}")
                database_context = await self.database_context_provider.get_organizational_context(
                    problem_statement=problem_statement,
                    organization_id=organization_id,
                    top_k=self.config.max_children_per_node * 2  # Get more context for better quality
                )
                logger.info(f"Retrieved database context: {len(database_context.get('organizational_goals', []))} goals, "
                           f"{len(database_context.get('past_decisions', []))} decisions, "
                           f"{len(database_context.get('relevant_content', []))} content items")
            except Exception as e:
                logger.error(f"Failed to fetch database context: {e}")
                database_context = None

        # Initialize DAG and enhanced context
        dag = DecisionDAG()
        context = DAGBuildingContext(
            problem_statement,
            strategic_paths,
            organizational_goals=organizational_goals,
            activity_insights=activity_insights,
            database_context=database_context
        )

        try:
            # Phase 1: Initialization
            context.update_phase(BuildPhase.INITIALIZATION)
            self._initialize_dag_with_paths(dag, strategic_paths)

            # Validate initialization
            init_validation = self._validate_initialization(dag, context)
            context.record_validation_result(init_validation)

            if not init_validation.is_valid:
                logger.error("DAG initialization failed validation")
                raise ValueError(f"Initialization validation failed: {init_validation.errors}")

            # Phase 2: Forward pass (multi-layer generation)
            context.update_phase(BuildPhase.FORWARD_PASS)
            dag = await self._forward_pass(dag, context)

            # Phase 3: Backward pass (deduplication)
            context.update_phase(BuildPhase.BACKWARD_PASS)
            dag = await self._backward_pass(dag, context)

            # Phase 4: Edge enrichment
            context.update_phase(BuildPhase.EDGE_ENRICHMENT)
            dag = await self._edge_enrichment_pass(dag, context)

            # Phase 5: Comprehensive context-aware validation
            context.update_phase(BuildPhase.VALIDATION)
            final_validation, validation_metrics = await self.context_validator.validate_dag_comprehensive(dag, context)
            context.record_validation_result(final_validation)

            # Record validation metrics in context
            context.state.metadata["context_validation_metrics"] = validation_metrics.to_dict()

            # Phase 6: Final HITL quality review
            if self.enable_hitl and self.hitl_manager:
                try:
                    quality_metrics = self._calculate_dag_quality_metrics(dag, context, validation_metrics)
                    quality_response = await self.hitl_manager.request_quality_review(dag, quality_metrics)

                    if quality_response.decision == HITLDecision.REJECT:
                        logger.warning("User rejected DAG quality - would need rebuild logic")
                        # In a full implementation, this would trigger rebuild
                    elif quality_response.decision == HITLDecision.MODIFY:
                        logger.info("User requested DAG modifications")
                        # Store feedback for future improvements
                        if quality_response.feedback:
                            dag.metadata["user_feedback"] = quality_response.feedback

                except Exception as e:
                    logger.error(f"Final quality review failed: {e}")

            # Phase 7: Completion
            context.update_phase(BuildPhase.COMPLETED)

            # Add build metadata to DAG
            dag.metadata.update({
                "build_summary": context.get_dag_summary(),
                "build_state": context.get_current_state().dict(),
                "is_healthy": context.is_healthy(),
                "hitl_enabled": self.enable_hitl,
                "hitl_session_summary": self.hitl_manager.get_session_summary() if self.hitl_manager else None
            })

            logger.info(f"DAG construction complete: {len(dag.all_nodes)} nodes, {len(dag.edges)} edges")
            return dag

        except Exception as e:
            context.record_error(f"DAG building failed: {str(e)}")
            context.update_phase(BuildPhase.FAILED)
            logger.error(f"DAG building failed: {e}")
            raise
        finally:
            # Clean up database connections
            await self.database_context_provider.close()

    def _initialize_dag_with_paths(self, dag: DecisionDAG, strategic_paths: List[StrategicPath]) -> None:
        """Initialize DAG with root nodes derived from strategic paths."""
        if not strategic_paths:
            # Create a default root node if no strategic paths provided
            root_node = DecisionNode(
                layer=0,
                type=NodeType.DECISION,
                title="Strategic Planning Decision",
                description="Determine the strategic approach to address the problem statement",
                decision_type=DecisionType.STRATEGIC,
                tags=["root", "strategic", "planning"]
            )
            dag.add_node(root_node)
            return

        # Create root nodes from strategic paths
        for i, path in enumerate(strategic_paths):
            root_node = DecisionNode(
                layer=0,
                type=NodeType.DECISION,
                title=f"Strategic Path: {path.title}",
                description=path.description,
                decision_type=DecisionType.STRATEGIC,
                tags=["root", "strategic_path"] + path.key_milestones[:3],  # Limit tags
                metadata={
                    "strategic_path_id": path.id,
                    "expected_outcomes": path.expected_outcomes,
                    "key_milestones": path.key_milestones
                }
            )
            dag.add_node(root_node)

    async def _forward_pass(self, dag: DecisionDAG, context: DAGBuildingContext) -> DecisionDAG:
        """Execute forward pass with parallel agent processing."""
        current_layer = 0

        while current_layer < self.config.max_layers:
            # Get nodes at current layer
            layer_nodes = dag.get_nodes_at_layer(current_layer)
            if not layer_nodes:
                break

            logger.info(f"Processing layer {current_layer} with {len(layer_nodes)} nodes")

            # Update context for current layer
            context.current_layer = current_layer
            context.dag_state = {
                "total_nodes": len(dag.all_nodes),
                "current_layer": current_layer,
                "layer_node_count": len(layer_nodes)
            }

            # Process layer in parallel
            next_layer_nodes = await self._process_layer_parallel(layer_nodes, context)

            # Add nodes and edges to DAG
            for parent_id, children in next_layer_nodes.items():
                parent_node = dag.get_node(parent_id)
                for child in children:
                    dag.add_node(child)

                    # Determine edge type based on alternating pattern
                    edge_type = (
                        EdgeType.DECISION_TO_OPTION if parent_node.type == NodeType.DECISION
                        else EdgeType.OPTION_TO_DECISION
                    )

                    # Set decision_reasoning_type for option-to-decision edges
                    decision_reasoning = None
                    if edge_type == EdgeType.OPTION_TO_DECISION:
                        # Use the reasoning type from the child (decision) node
                        decision_reasoning = child.reasoning_type
                        if not decision_reasoning:
                            # Default to LOGICAL if not specified
                            logger.warning(f"Decision node {child.id} missing reasoning_type, defaulting to LOGICAL")
                            decision_reasoning = DecisionReasoningType.LOGICAL

                    edge = DecisionEdge(
                        source_id=parent_id,
                        target_id=child.id,
                        edge_type=edge_type,
                        condition=f"Proceed with {child.title}",
                        decision_reasoning_type=decision_reasoning,
                        relationship="leads_to",
                        metadata={"layer_transition": f"{current_layer} -> {current_layer + 1}"}
                    )
                    dag.add_edge(edge)

            # Collect all new nodes
            all_new_nodes = []
            for children in next_layer_nodes.values():
                all_new_nodes.extend(children)

            if not all_new_nodes:
                logger.info(f"No new nodes generated at layer {current_layer}, stopping")
                break

            # HITL: Request layer approval before advancing
            if self.enable_hitl and self.hitl_manager:
                try:
                    hitl_response = await self.hitl_manager.request_layer_approval(
                        layer=current_layer + 1,
                        nodes=all_new_nodes,
                        dag_context=context.dag_state,
                        build_phase=BuildPhase.FORWARD_PASS
                    )

                    # Process HITL response
                    all_new_nodes = await self._process_layer_hitl_response(
                        hitl_response, all_new_nodes, dag, context
                    )

                except Exception as e:
                    logger.error(f"HITL layer approval failed: {e}")
                    # Continue without HITL on error

            context.advance_layer(all_new_nodes)
            current_layer += 1

        logger.info(f"Forward pass complete: {current_layer} layers, {len(dag.all_nodes)} nodes")
        return dag

    async def _process_layer_parallel(
        self,
        nodes: List[DecisionNode],
        context: DAGBuildingContext
    ) -> Dict[str, List[DecisionNode]]:
        """Process all nodes in a layer concurrently with rate limiting."""
        tasks = []

        for node in nodes:
            # Create context for this node
            node_context = context.create_node_context(node)

            # Create agent and task
            agent = ParallelNodeAgent(node, node_context)
            task = self._create_limited_task(agent.process())
            tasks.append((node, task))

        # Execute all tasks concurrently
        results = {}
        for node, task in tasks:
            try:
                children = await task
                results[node.id] = children
                logger.debug(f"Generated {len(children)} children for node: {node.title}")
            except Exception as e:
                logger.error(f"Agent failed for node {node.id}: {e}")
                results[node.id] = []  # Continue with empty children

        return results

    async def _create_limited_task(self, coro):
        """Create a task with semaphore-based rate limiting."""
        async with self.semaphore:
            try:
                return await asyncio.wait_for(coro, timeout=self.config.agent_timeout)
            except asyncio.TimeoutError:
                logger.error(f"Agent task timed out after {self.config.agent_timeout}s")
                raise

    async def _backward_pass(self, dag: DecisionDAG, context: DAGBuildingContext) -> DecisionDAG:
        """Execute backward pass for deduplication."""
        logger.info("Starting backward pass for deduplication")

        max_layer = dag.get_max_layer()

        # Process each layer from top to bottom for deduplication
        for layer in range(max_layer + 1):
            layer_nodes = dag.get_nodes_at_layer(layer)
            if len(layer_nodes) <= 1:
                continue

            logger.info(f"Deduplicating layer {layer}: {len(layer_nodes)} nodes")

            # Deduplicate nodes in this layer
            deduplicated_nodes = await self.deduplicator.deduplicate_layer(layer_nodes)

            if len(deduplicated_nodes) < len(layer_nodes):
                # Update DAG with deduplicated nodes
                self._update_dag_with_deduplicated_nodes(dag, layer, layer_nodes, deduplicated_nodes)
                logger.info(f"Layer {layer}: reduced from {len(layer_nodes)} to {len(deduplicated_nodes)} nodes")

        logger.info("Backward pass complete")
        return dag

    def _update_dag_with_deduplicated_nodes(
        self,
        dag: DecisionDAG,
        layer: int,
        original_nodes: List[DecisionNode],
        deduplicated_nodes: List[DecisionNode]
    ) -> None:
        """Update DAG with deduplicated nodes, handling edge remapping."""
        # Create mapping from original nodes to deduplicated nodes
        node_mapping = {}
        for dedup_node in deduplicated_nodes:
            if "merged_from" in dedup_node.metadata:
                # This node was merged from multiple nodes
                for original_id in dedup_node.metadata["merged_from"]:
                    node_mapping[original_id] = dedup_node.id
            else:
                # This node was kept as-is
                node_mapping[dedup_node.id] = dedup_node.id

        # Remove original nodes from DAG
        for node in original_nodes:
            if node.id in dag.all_nodes:
                del dag.all_nodes[node.id]

        # Add deduplicated nodes to DAG
        for node in deduplicated_nodes:
            dag.add_node(node)

        # Update edges to point to deduplicated nodes
        updated_edges = []
        for edge in dag.edges:
            new_source_id = node_mapping.get(edge.source_id, edge.source_id)
            new_target_id = node_mapping.get(edge.target_id, edge.target_id)

            # Skip edges that would create self-loops or duplicate edges
            if new_source_id != new_target_id:
                # Handle decision_reasoning_type based on edge_type
                decision_reasoning = None
                if hasattr(edge, 'decision_reasoning_type') and edge.decision_reasoning_type:
                    decision_reasoning = edge.decision_reasoning_type
                elif edge.edge_type == EdgeType.OPTION_TO_DECISION:
                    # Try to get from the target (decision) node
                    target_node = dag.get_node(new_target_id)
                    if target_node and hasattr(target_node, 'reasoning_type') and target_node.reasoning_type:
                        decision_reasoning = target_node.reasoning_type
                    else:
                        # Default to LOGICAL if not specified
                        logger.warning(f"Edge to decision node {new_target_id} missing reasoning_type, defaulting to LOGICAL")
                        decision_reasoning = DecisionReasoningType.LOGICAL

                updated_edge = DecisionEdge(
                    source_id=new_source_id,
                    target_id=new_target_id,
                    edge_type=edge.edge_type,
                    condition=edge.condition,
                    relationship=edge.relationship,
                    conditions=edge.conditions,
                    cost_estimate=edge.cost_estimate,
                    timeline_estimate=edge.timeline_estimate,
                    metadata=edge.metadata,
                    likelihood=edge.likelihood if hasattr(edge, 'likelihood') else None,
                    label=edge.label if hasattr(edge, 'label') else None,
                    decision_reasoning_type=decision_reasoning,
                    estimated_cost_dollars=edge.estimated_cost_dollars if hasattr(edge, 'estimated_cost_dollars') else None,
                    implementation_risks=edge.implementation_risks if hasattr(edge, 'implementation_risks') else None
                )
                updated_edges.append(updated_edge)

        # Remove duplicate edges
        seen_edges = set()
        final_edges = []
        for edge in updated_edges:
            edge_key = (edge.source_id, edge.target_id)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                final_edges.append(edge)

        dag.edges = final_edges

    async def _edge_enrichment_pass(self, dag: DecisionDAG, context: DAGBuildingContext) -> DecisionDAG:
        """Enrich edges and nodes with detailed implementation information using LLM assistance."""
        logger.info(f"Starting comprehensive enrichment pass for {len(dag.edges)} edges and {len(dag.all_nodes)} nodes")

        try:
            # Use the sophisticated edge enricher
            enriched_dag, enrichment_metrics = await self.edge_enricher.enrich_edges(dag, context)

            # Record enrichment metrics in context
            context.state.metadata["edge_enrichment_metrics"] = enrichment_metrics.to_dict()

            logger.info(f"Edge enrichment complete. Success rate: {enrichment_metrics.success_rate:.2%}, "
                       f"Cost estimates: {enrichment_metrics.cost_estimates_added}, "
                       f"Timeline estimates: {enrichment_metrics.timeline_estimates_added}, "
                       f"Risk assessments: {enrichment_metrics.risk_assessments_added}")

            # Now enrich nodes with people_impacted and resource_requirements
            enriched_dag = await self._enrich_nodes(enriched_dag, context)

            return enriched_dag

        except Exception as e:
            logger.error(f"Edge enrichment failed, falling back to basic enrichment: {e}")
            context.record_error("Edge enrichment failed", {"error": str(e), "edges_count": len(dag.edges)})

            # Fallback to basic enrichment
            return await self._basic_edge_enrichment_fallback(dag, context)

    async def _basic_edge_enrichment_fallback(self, dag: DecisionDAG, context: DAGBuildingContext) -> DecisionDAG:
        """Fallback basic edge enrichment when LLM-assisted enrichment fails."""
        logger.info("Applying basic edge enrichment fallback")

        for edge in dag.edges:
            source_node = dag.get_node(edge.source_id)
            target_node = dag.get_node(edge.target_id)

            if source_node and target_node:
                # Add basic relationship metadata
                edge.metadata.update({
                    "source_title": source_node.title,
                    "target_title": target_node.title,
                    "relationship_type": f"{source_node.type.value}_to_{target_node.type.value}",
                    "enriched": True,
                    "enrichment_method": "basic_fallback"
                })

                # Add basic estimates if missing
                if not edge.cost_estimate:
                    edge.cost_estimate = "medium"
                if not edge.timeline_estimate:
                    edge.timeline_estimate = "2-4 weeks"
                if not edge.implementation_risks:
                    edge.implementation_risks = ["Resource availability", "Stakeholder alignment"]

        logger.info("Basic edge enrichment fallback complete")
        return dag

    async def _enrich_nodes(self, dag: DecisionDAG, context: DAGBuildingContext) -> DecisionDAG:
        """Enrich nodes with people_impacted and resource_requirements using LLM assistance."""
        logger.info("Starting node enrichment for people_impacted and resource_requirements")

        try:
            # Use the node enricher
            enriched_dag, enrichment_metrics = await self.node_enricher.enrich_nodes(dag, context)

            # Record enrichment metrics in context
            context.state.metadata["node_enrichment_metrics"] = enrichment_metrics.to_dict()

            logger.info(f"Node enrichment complete. Success rate: {enrichment_metrics.success_rate:.2%}, "
                       f"People impacted added: {enrichment_metrics.people_impacted_added}, "
                       f"Resource requirements added: {enrichment_metrics.resource_requirements_added}, "
                       f"Embeddings added: {enrichment_metrics.embeddings_added}")

            return enriched_dag

        except Exception as e:
            logger.error(f"Node enrichment failed: {e}")
            context.record_error("Node enrichment failed", {"error": str(e), "nodes_count": len(dag.all_nodes)})
            # Return the DAG as-is if enrichment fails
            return dag

    def _validate_initialization(self, dag: DecisionDAG, context: DAGBuildingContext) -> ValidationResult:
        """Validate DAG initialization."""
        errors = []
        warnings = []

        # Check that we have root nodes
        if not dag.root_nodes:
            errors.append("No root nodes created during initialization")

        # Check root nodes are at layer 0
        for root in dag.root_nodes:
            if root.layer != 0:
                errors.append(f"Root node {root.id} not at layer 0: layer={root.layer}")

        # Check that we have the expected number of root nodes
        expected_roots = max(1, len(context.strategic_paths))
        if len(dag.root_nodes) != expected_roots:
            warnings.append(f"Expected {expected_roots} root nodes, got {len(dag.root_nodes)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validation_type="initialization",
            node_count=len(dag.all_nodes),
            edge_count=len(dag.edges)
        )

    async def _validate_final_dag(self, dag: DecisionDAG, context: DAGBuildingContext) -> ValidationResult:
        """Perform comprehensive final validation of the DAG."""
        from ..utils.validation import validate_dag_comprehensive

        # Use comprehensive validation
        validation_results = validate_dag_comprehensive(dag)

        # Convert to ValidationResult format
        return ValidationResult(
            is_valid=validation_results["is_valid"],
            errors=validation_results["errors"],
            warnings=validation_results["warnings"],
            validation_type="final_comprehensive",
            node_count=len(dag.all_nodes),
            edge_count=len(dag.edges),
            details=validation_results["validation_details"]
        )

    async def _process_layer_hitl_response(
        self,
        response: HITLResponse,
        nodes: List[DecisionNode],
        dag: DecisionDAG,
        context: DAGBuildingContext
    ) -> List[DecisionNode]:
        """Process HITL response for layer approval."""
        if response.decision == HITLDecision.APPROVE:
            logger.info("Layer approved by user")
            return nodes

        elif response.decision == HITLDecision.REJECT:
            logger.info("Layer rejected by user - regenerating")
            # In a full implementation, this would trigger regeneration
            # For now, return empty list to stop this layer
            context.record_error("Layer rejected by user", {"rejected_nodes": len(nodes)})
            return []

        elif response.decision == HITLDecision.MODIFY:
            logger.info("User requested layer modifications")

            if response.modifications and "nodes_to_modify" in response.modifications:
                # Process specific node modifications
                nodes_to_modify = response.modifications["nodes_to_modify"]
                modified_nodes = []

                for node in nodes:
                    if any(mod_node.id == node.id for mod_node in nodes_to_modify):
                        # Request individual node modification
                        try:
                            node_response = await self.hitl_manager.request_node_modification(node)
                            if node_response.decision == HITLDecision.APPROVE:
                                modified_nodes.append(node)
                            elif node_response.decision == HITLDecision.MODIFY and node_response.modifications:
                                modified_node = self.hitl_manager.apply_node_modifications(node, node_response.modifications)
                                modified_nodes.append(modified_node)
                            # Skip rejected nodes
                        except Exception as e:
                            logger.error(f"Node modification failed for {node.id}: {e}")
                            modified_nodes.append(node)  # Keep original on error
                    else:
                        modified_nodes.append(node)

                return modified_nodes
            else:
                # No specific modifications, return original nodes
                return nodes

        elif response.decision == HITLDecision.STOP:
            logger.info("User requested to stop DAG building")
            context.record_error("User requested stop", {"stopped_at_layer": context.state.current_layer})
            return []

        else:
            logger.warning(f"Unknown HITL decision: {response.decision}")
            return nodes

    def _calculate_dag_quality_metrics(
        self,
        dag: DecisionDAG,
        context: DAGBuildingContext,
        validation_metrics = None
    ) -> Dict[str, Any]:
        """Calculate quality metrics for final DAG review."""
        # Basic structure metrics
        metrics = {
            "total_nodes": len(dag.all_nodes),
            "total_edges": len(dag.edges),
            "max_layer": dag.get_max_layer(),
            "root_nodes": len(dag.root_nodes),
            "leaf_nodes": len([node for node in dag.all_nodes.values() if not dag.get_children(node.id)])
        }

        # Confidence metrics
        all_confidences = [node.confidence_score or 0.5 for node in dag.all_nodes.values()]
        if all_confidences:
            metrics.update({
                "avg_confidence": sum(all_confidences) / len(all_confidences),
                "min_confidence": min(all_confidences),
                "max_confidence": max(all_confidences),
                "low_confidence_nodes": sum(1 for c in all_confidences if c < 0.4)
            })

        # Layer distribution
        layer_distribution = {}
        for node in dag.all_nodes.values():
            layer_distribution[node.layer] = layer_distribution.get(node.layer, 0) + 1
        metrics["layer_distribution"] = layer_distribution

        # Alternating pattern validation
        pattern_violations = 0
        for node in dag.all_nodes.values():
            expected_decision = (node.layer % 2 == 0)
            is_decision = (node.type == NodeType.DECISION)
            if expected_decision != is_decision:
                pattern_violations += 1
        metrics["pattern_violations"] = pattern_violations

        # Build process metrics from context
        build_summary = context.get_dag_summary()
        if "state" in build_summary:
            metrics.update({
                "build_time_seconds": build_summary["state"].get("total_generation_time_seconds", 0),
                "build_errors": build_summary["state"].get("error_count", 0),
                "validation_success_rate": (
                    build_summary["validation"]["passed_validations"] /
                    max(1, build_summary["validation"]["total_validations"])
                ) if "validation" in build_summary else 1.0
            })

        # Add context-aware validation metrics if available
        if validation_metrics:
            metrics.update({
                "overall_quality_score": validation_metrics.overall_quality_score,
                "mece_compliance_score": validation_metrics.mece_compliance_score,
                "organizational_alignment_score": validation_metrics.organizational_alignment_score,
                "implementation_feasibility_score": validation_metrics.implementation_feasibility_score,
                "strategic_coherence_score": validation_metrics.strategic_coherence_score,
                "context_validation_errors": validation_metrics.errors_found,
                "context_validation_warnings": validation_metrics.warnings_generated,
                "validation_llm_calls": validation_metrics.llm_calls_made,
                "context_items_analyzed": validation_metrics.context_items_analyzed,
                "validation_time_seconds": validation_metrics.validation_time_seconds
            })

        return metrics

    def get_hitl_interface(self) -> Optional[HITLInterface]:
        """Get the HITL interface for external interaction."""
        return self.hitl_interface

    def submit_hitl_response(
        self,
        prompt_id: str,
        decision: HITLDecision,
        feedback: Optional[str] = None,
        modifications: Optional[Dict[str, Any]] = None,
        reasoning: Optional[str] = None
    ) -> bool:
        """Submit a response to a pending HITL prompt."""
        if not self.hitl_interface:
            logger.error("HITL interface not available")
            return False

        return self.hitl_interface.submit_response(
            prompt_id=prompt_id,
            decision=decision,
            feedback=feedback,
            modifications=modifications,
            reasoning=reasoning
        )
