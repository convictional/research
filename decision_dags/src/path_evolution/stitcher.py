"""Path stitching engine for merging evolved paths back into a unified DAG.

This module implements path merging and deduplication by creating a composite DAG
from multiple evolved path DAGs and using the existing DAG deduplication infrastructure.

The approach:
1. Create a temporary composite DAG containing all best evolved paths as disconnected components
2. Use the existing DAGBuilderEnsemble._backward_pass() method for deduplication
3. Return the deduplicated composite DAG with merged path components

This leverages the robust deduplication system that already handles:
- Embedding-based similarity clustering
- LLM-guided duplicate assessment
- Pattern preservation (decision-option alternating)
- Relationship consolidation
"""

import logging
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..models import DecisionDAG, DecisionNode, DecisionEdge

logger = logging.getLogger(__name__)


@dataclass
class PathStitchingResult:
    """Result of path stitching operation."""

    stitched_dag: DecisionDAG
    original_path_count: int
    nodes_deduplicated: int
    edges_consolidated: int
    stitching_time_seconds: float
    deduplication_metadata: Dict[str, Any]


@dataclass
class PathStitchingConfig:
    """Configuration for path stitching operations."""

    # DAG creation settings
    composite_dag_name: str = "Evolved Paths Composite"
    composite_dag_description: str = "Merged composite of best evolved paths"

    # Deduplication settings (leverages existing DAGBuilderEnsemble settings)
    strong_similarity_threshold: float = 0.8
    weak_similarity_threshold: float = 0.6
    preserve_path_boundaries: bool = False  # If True, less aggressive merging

    # Performance settings
    enable_parallel_processing: bool = True
    max_concurrent_operations: int = 5


class PathStitchingEngine:
    """Engine for stitching evolved paths together using existing DAG deduplication infrastructure."""

    def __init__(self, config: Optional[PathStitchingConfig] = None):
        self.config = config or PathStitchingConfig()
        self.dag_builder = None

    async def stitch_paths(
        self,
        evolved_paths: List[DecisionDAG],
        original_dag: Optional[DecisionDAG] = None,
        config: Optional[PathStitchingConfig] = None,
    ) -> PathStitchingResult:
        """
        Main path stitching workflow:
        1. Create composite DAG with all evolved paths as disconnected components
        2. Apply existing deduplication logic to merge similar nodes/edges
        3. Return unified, deduplicated DAG

        Args:
            evolved_paths: List of evolved path DAGs
            original_dag: Original DAG (for metadata)
            config: Optional configuration override

        Returns:
            PathStitchingResult with stitched DAG and metadata
        """
        config = config or self.config
        stitching_start = datetime.now()

        logger.info(f"🔗 Starting path stitching for {len(evolved_paths)} evolved paths")

        if not evolved_paths:
            logger.warning("No evolved paths provided for stitching")
            if original_dag:
                return PathStitchingResult(
                    stitched_dag=original_dag,
                    original_path_count=0,
                    nodes_deduplicated=0,
                    edges_consolidated=0,
                    stitching_time_seconds=0.0,
                    deduplication_metadata={},
                )
            else:
                raise ValueError("No paths to stitch and no original DAG provided")

        if len(evolved_paths) == 1:
            logger.info("Only one evolved path - returning it directly")
            return PathStitchingResult(
                stitched_dag=evolved_paths[0],
                original_path_count=1,
                nodes_deduplicated=0,
                edges_consolidated=0,
                stitching_time_seconds=(datetime.now() - stitching_start).total_seconds(),
                deduplication_metadata={},
            )

        # Step 1: Create composite DAG with all paths as disconnected components
        logger.info("📦 Step 1: Creating composite DAG from evolved paths")
        composite_dag, composition_metadata = await self._create_composite_dag(evolved_paths, original_dag, config)

        # Count original nodes/edges before deduplication
        original_node_count = len(composite_dag.all_nodes)
        original_edge_count = len(composite_dag.edges)

        logger.info(f"  Composite DAG created: {original_node_count} nodes, {original_edge_count} edges")

        # Step 2: Apply deduplication using existing infrastructure
        logger.info("🔍 Step 2: Applying deduplication to composite DAG")
        deduplicated_dag, deduplication_metadata = await self._apply_deduplication(composite_dag)

        # Count final nodes/edges after deduplication
        final_node_count = len(deduplicated_dag.all_nodes)
        final_edge_count = len(deduplicated_dag.edges)

        nodes_deduplicated = original_node_count - final_node_count
        edges_consolidated = original_edge_count - final_edge_count

        logger.info(
            f"  Deduplication complete: {final_node_count} nodes ({nodes_deduplicated} deduplicated), "
            f"{final_edge_count} edges ({edges_consolidated} consolidated)"
        )

        # Step 3: Finalize and return result
        stitching_time = (datetime.now() - stitching_start).total_seconds()

        logger.info(f"🎉 Path stitching complete in {stitching_time:.1f}s")
        logger.info(f"   Final result: {final_node_count} nodes, {final_edge_count} edges")

        return PathStitchingResult(
            stitched_dag=deduplicated_dag,
            original_path_count=len(evolved_paths),
            nodes_deduplicated=nodes_deduplicated,
            edges_consolidated=edges_consolidated,
            stitching_time_seconds=stitching_time,
            deduplication_metadata={**composition_metadata, **deduplication_metadata},
        )

    async def _create_composite_dag(
        self, evolved_paths: List[DecisionDAG], original_dag: Optional[DecisionDAG], config: PathStitchingConfig
    ) -> tuple[DecisionDAG, Dict[str, Any]]:
        """
        Create a composite DAG containing all evolved paths as disconnected components.
        """
        # Create new composite DAG
        composite_dag = DecisionDAG(
            id="composite_dag",
            metadata={
                "description": config.composite_dag_description,
                "stitching_config": {
                    "strong_similarity_threshold": config.strong_similarity_threshold,
                    "weak_similarity_threshold": config.weak_similarity_threshold,
                    "preserve_path_boundaries": config.preserve_path_boundaries,
                },
                "evolved_path_count": len(evolved_paths),
                "original_dag_id": original_dag.id if original_dag else None,
                "merge_method": "path_stitching",
                "status": "stitched",
                "generation_method": "path_stitching",
            },
        )

        # Track composition metadata
        path_metadata = []
        node_mapping = {}  # old_node_id -> new_node_id
        edge_mapping = {}  # old_edge_id -> new_edge_id

        # Process each evolved path
        for path_index, path_dag in enumerate(evolved_paths):
            logger.debug(f"  Processing path {path_index + 1}/{len(evolved_paths)}: {path_dag.id}")

            # Copy all nodes from this path
            path_node_count = len(path_dag.all_nodes)

            for node_id, node in path_dag.all_nodes.items():
                # Create new node in composite DAG with unique ID
                new_node_id = f"path{path_index}_{node_id}"

                # Create new node with enriched metadata
                new_node = DecisionNode(
                    id=new_node_id,
                    layer=node.layer,
                    type=node.type,
                    title=node.title,
                    description=node.description,
                    decision_type=node.decision_type,
                    goal_impacts=node.goal_impacts.copy(),
                    people_impacted=node.people_impacted.copy(),
                    resource_requirements=node.resource_requirements.copy(),
                    tags=node.tags.copy(),
                    metadata={
                        **node.metadata,
                        # Mark which path this came from
                        "source_path_index": path_index,
                        "source_path_dag_id": path_dag.id,
                        "original_node_id": node_id,  # Preserve original for reference
                        "stitching_metadata": {
                            "source_path_name": path_dag.metadata.get("name", f"path_{path_index}"),
                            "evolved_from": path_dag.metadata.get("evolved_from"),
                            "generation": path_dag.metadata.get("generation", 0),
                        },
                    },
                )

                composite_dag.add_node(new_node)
                node_mapping[f"{path_index}_{node_id}"] = new_node_id

            # Copy all edges from this path
            path_edge_count = len(path_dag.edges)

            for edge in path_dag.edges:
                # Get mapped node IDs
                source_key = f"{path_index}_{edge.source_id}"
                target_key = f"{path_index}_{edge.target_id}"

                new_source_id = node_mapping.get(source_key)
                new_target_id = node_mapping.get(target_key)

                if new_source_id and new_target_id:
                    # Create new edge
                    new_edge = DecisionEdge(
                        source_id=new_source_id,
                        target_id=new_target_id,
                        edge_type=edge.edge_type,
                        condition=edge.condition,
                        decision_reasoning_type=edge.decision_reasoning_type,
                        relationship=edge.relationship,
                        conditions=edge.conditions.copy(),
                        cost_estimate=edge.cost_estimate,
                        timeline_estimate=edge.timeline_estimate,
                        estimated_cost_dollars=edge.estimated_cost_dollars,
                        implementation_risks=edge.implementation_risks.copy() if edge.implementation_risks else [],
                        metadata={
                            **edge.metadata,
                            "source_path_index": path_index,
                            "source_path_dag_id": path_dag.id,
                            "original_edge_source": edge.source_id,
                            "original_edge_target": edge.target_id,
                        },
                    )

                    composite_dag.add_edge(new_edge)
                    edge_mapping[f"{path_index}_{edge.source_id}_{edge.target_id}"] = (
                        f"{new_source_id}_{new_target_id}"
                    )

            path_metadata.append(
                {
                    "path_index": path_index,
                    "path_dag_id": path_dag.id,
                    "path_name": path_dag.metadata.get("name", f"path_{path_index}"),
                    "nodes_copied": path_node_count,
                    "edges_copied": path_edge_count,
                }
            )

        logger.debug(f"  Composite DAG creation complete: {len(node_mapping)} nodes, {len(edge_mapping)} edges")

        composition_metadata = {
            "composite_creation": {
                "paths_processed": len(evolved_paths),
                "total_nodes_copied": len(node_mapping),
                "total_edges_copied": len(edge_mapping),
                "path_details": path_metadata,
            }
        }

        return composite_dag, composition_metadata

    async def _apply_deduplication(self, composite_dag: DecisionDAG) -> tuple[DecisionDAG, Dict[str, Any]]:
        """
        Apply existing deduplication logic to the composite DAG.
        """
        logger.debug("  Initializing DAGBuilderEnsemble for deduplication")

        # Count nodes before deduplication for metadata
        nodes_before = len(composite_dag.all_nodes)
        edges_before = len(composite_dag.edges)

        logger.debug(f"  Starting deduplication: {nodes_before} nodes, {edges_before} edges")

        try:
            # For now, use the existing deduplicator directly instead of DAGBuilderEnsemble
            # This is a simpler approach that should work with our current setup
            from ..dag_builder.deduplicator import NodeDeduplicator

            deduplicator = NodeDeduplicator()

            # Process each layer for deduplication
            max_layer = max(node.layer for node in composite_dag.all_nodes.values()) if composite_dag.all_nodes else 0

            for layer in range(max_layer + 1):
                layer_nodes = composite_dag.get_nodes_at_layer(layer)
                if len(layer_nodes) > 1:
                    # Apply deduplication to this layer
                    deduplicated_layer_nodes = await deduplicator.deduplicate_layer(layer_nodes)

                    # Update the composite DAG with deduplicated nodes
                    # Remove original nodes from this layer
                    for node in layer_nodes:
                        if node.id in composite_dag.all_nodes:
                            del composite_dag.all_nodes[node.id]

                    # Add deduplicated nodes
                    for node in deduplicated_layer_nodes:
                        composite_dag.all_nodes[node.id] = node
                        if node.layer == 0:
                            composite_dag.root_nodes = [n for n in composite_dag.root_nodes if n.id != node.id]
                            composite_dag.root_nodes.append(node)

            # Get the deduplicated DAG
            deduplicated_dag = composite_dag

            # Count nodes after deduplication
            nodes_after = len(deduplicated_dag.all_nodes)
            edges_after = len(deduplicated_dag.edges)

            nodes_deduplicated = nodes_before - nodes_after
            edges_consolidated = edges_before - edges_after

            logger.debug(
                f"  Deduplication complete: {nodes_after} nodes ({nodes_deduplicated} removed), "
                f"{edges_after} edges ({edges_consolidated} consolidated)"
            )

            deduplication_metadata = {
                "deduplication_applied": {
                    "nodes_before": nodes_before,
                    "nodes_after": nodes_after,
                    "nodes_deduplicated": nodes_deduplicated,
                    "edges_before": edges_before,
                    "edges_after": edges_after,
                    "edges_consolidated": edges_consolidated,
                    "deduplication_success": True,
                }
            }

            return deduplicated_dag, deduplication_metadata

        except Exception as e:
            logger.error(f"Failed to apply deduplication: {e}")
            # Return original composite DAG if deduplication fails
            deduplication_metadata = {
                "deduplication_applied": {
                    "deduplication_success": False,
                    "error": str(e),
                    "fallback": "returned_original_composite",
                }
            }
            return composite_dag, deduplication_metadata

    def _get_path_max_layer(self, path: DecisionDAG) -> int:
        """Get maximum layer in a path DAG."""
        if not path.all_nodes:
            return 0
        return max(node.layer for node in path.all_nodes.values())

    async def _process_layer(self, merged_dag: DecisionDAG, source_paths: List[DecisionDAG], layer: int) -> None:
        """Process a single layer, collecting and deduplicating nodes."""
        # Collect all nodes at this layer from all paths
        layer_nodes = []
        for path in source_paths:
            path_layer_nodes = self._get_nodes_at_layer(path, layer)
            layer_nodes.extend(path_layer_nodes)

        if not layer_nodes:
            return

        logger.debug(f"Processing layer {layer}: {len(layer_nodes)} nodes before deduplication")

        # Deduplicate nodes at this layer
        deduplicated_nodes = await self.deduplicator.deduplicate_layer(layer_nodes)

        # Add deduplicated nodes to merged DAG
        for node in deduplicated_nodes:
            merged_dag.add_node(node)

        logger.debug(f"Layer {layer}: {len(deduplicated_nodes)} nodes after deduplication")

    def _get_nodes_at_layer(self, path: DecisionDAG, layer: int) -> List[DecisionNode]:
        """Get all nodes at a specific layer in a path."""
        return [node for node in path.all_nodes.values() if node.layer == layer]

    async def _reconstruct_edges(self, merged_dag: DecisionDAG, source_paths: List[DecisionDAG]) -> None:
        """Reconstruct edges in merged DAG from source paths."""
        # Build node mapping from original to merged nodes
        node_mapping = self._build_node_mapping(merged_dag, source_paths)

        # Track added edges to avoid duplicates
        added_edges: Set[tuple] = set()

        # Add edges based on source paths
        for path in source_paths:
            for edge in path.edges:
                # Find corresponding nodes in merged DAG
                merged_source_id = node_mapping.get((path.id, edge.source_id))
                merged_target_id = node_mapping.get((path.id, edge.target_id))

                if merged_source_id and merged_target_id:
                    edge_key = (merged_source_id, merged_target_id)

                    if edge_key not in added_edges:
                        # Create new edge
                        new_edge = DecisionEdge(
                            source_id=merged_source_id,
                            target_id=merged_target_id,
                            edge_type=edge.edge_type,
                            condition=edge.condition,
                            decision_reasoning_type=edge.decision_reasoning_type,
                            relationship=edge.relationship,
                            conditions=edge.conditions,
                            cost_estimate=edge.cost_estimate,
                            timeline_estimate=edge.timeline_estimate,
                            estimated_cost_dollars=edge.estimated_cost_dollars,
                            implementation_risks=edge.implementation_risks,
                            metadata=edge.metadata.copy(),
                        )

                        # Add source path information to edge metadata
                        new_edge.metadata["source_paths"] = new_edge.metadata.get("source_paths", [])
                        if path.id not in new_edge.metadata["source_paths"]:
                            new_edge.metadata["source_paths"].append(path.id)

                        merged_dag.add_edge(new_edge)
                        added_edges.add(edge_key)
                    else:
                        # Merge edge information for existing edge
                        existing_edge = merged_dag.get_edge(merged_source_id, merged_target_id)
                        if existing_edge:
                            self._merge_edge_information(existing_edge, edge, path.id)

    def _build_node_mapping(self, merged_dag: DecisionDAG, source_paths: List[DecisionDAG]) -> Dict[tuple, str]:
        """
        Build mapping from (path_id, original_node_id) to merged_node_id.

        This handles cases where nodes were merged during deduplication.
        """
        node_mapping = {}

        for merged_node in merged_dag.all_nodes.values():
            if "merged_from" in merged_node.metadata:
                # This node was created by merging multiple nodes
                merged_from_ids = merged_node.metadata["merged_from"]

                # Find which paths these original nodes came from
                for path in source_paths:
                    for original_id in merged_from_ids:
                        if original_id in path.all_nodes:
                            node_mapping[(path.id, original_id)] = merged_node.id
            else:
                # This node was kept as-is, find which path it came from
                for path in source_paths:
                    if merged_node.id in path.all_nodes:
                        node_mapping[(path.id, merged_node.id)] = merged_node.id
                        break

        return node_mapping

    def _merge_edge_information(
        self, existing_edge: DecisionEdge, new_edge: DecisionEdge, source_path_id: str
    ) -> None:
        """Merge information from a new edge into an existing edge."""
        # Merge conditions
        if new_edge.conditions:
            existing_conditions = set(existing_edge.conditions)
            for condition in new_edge.conditions:
                if condition not in existing_conditions:
                    existing_edge.conditions.append(condition)

        # Merge primary condition if different
        if new_edge.condition and new_edge.condition != existing_edge.condition:
            # Add to conditions list if not already there
            if new_edge.condition not in existing_edge.conditions:
                existing_edge.conditions.append(new_edge.condition)

        # Merge cost estimates (take average or more conservative estimate)
        if new_edge.estimated_cost_dollars and existing_edge.estimated_cost_dollars:
            # Take average of cost estimates
            existing_edge.estimated_cost_dollars = (
                existing_edge.estimated_cost_dollars + new_edge.estimated_cost_dollars
            ) / 2
        elif new_edge.estimated_cost_dollars and not existing_edge.estimated_cost_dollars:
            existing_edge.estimated_cost_dollars = new_edge.estimated_cost_dollars

        # Merge timeline estimates (take more conservative estimate)
        if new_edge.timeline_estimate:
            if not existing_edge.timeline_estimate:
                existing_edge.timeline_estimate = new_edge.timeline_estimate
            # Could add logic to compare and take longer timeline

        # Merge implementation risks
        if new_edge.implementation_risks:
            existing_risks = set(existing_edge.implementation_risks or [])
            for risk in new_edge.implementation_risks:
                if risk not in existing_risks:
                    if existing_edge.implementation_risks is None:
                        existing_edge.implementation_risks = []
                    existing_edge.implementation_risks.append(risk)

        # Update source paths in metadata
        existing_paths = existing_edge.metadata.get("source_paths", [])
        if source_path_id not in existing_paths:
            existing_paths.append(source_path_id)
        existing_edge.metadata["source_paths"] = existing_paths

    def validate_stitched_dag(self, dag: DecisionDAG) -> Dict[str, any]:
        """
        Validate the stitched DAG for correctness.

        Returns:
            Validation results dictionary
        """
        from ..utils.validation import validate_dag_comprehensive

        validation_results = validate_dag_comprehensive(dag)

        # Add stitching-specific validations
        stitching_results = {
            "source_path_count": len(dag.metadata.get("source_paths", [])),
            "has_multiple_roots": len(dag.root_nodes) > 1,
            "path_convergence": self._analyze_path_convergence(dag),
            "edge_multiplicity": self._analyze_edge_multiplicity(dag),
        }

        validation_results["stitching_analysis"] = stitching_results

        return validation_results

    def _analyze_path_convergence(self, dag: DecisionDAG) -> Dict[str, any]:
        """Analyze how paths converge in the stitched DAG."""
        convergence_points = []

        for node in dag.all_nodes.values():
            parents = dag.get_parents(node)
            if len(parents) > 1:
                # This is a convergence point
                convergence_points.append(
                    {"node_id": node.id, "node_title": node.title, "layer": node.layer, "parent_count": len(parents)}
                )

        return {"convergence_point_count": len(convergence_points), "convergence_points": convergence_points}

    def _analyze_edge_multiplicity(self, dag: DecisionDAG) -> Dict[str, any]:
        """Analyze edges that represent multiple source paths."""
        multi_path_edges = []

        for edge in dag.edges:
            source_paths = edge.metadata.get("source_paths", [])
            if len(source_paths) > 1:
                multi_path_edges.append(
                    {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "path_count": len(source_paths),
                        "source_paths": source_paths,
                    }
                )

        return {"multi_path_edge_count": len(multi_path_edges), "multi_path_edges": multi_path_edges}


# Convenience functions for common path stitching scenarios


async def stitch_paths_balanced(
    evolved_paths: List[DecisionDAG], original_dag: Optional[DecisionDAG] = None
) -> PathStitchingResult:
    """Stitch paths with balanced deduplication settings."""
    config = PathStitchingConfig(
        strong_similarity_threshold=0.8, weak_similarity_threshold=0.6, preserve_path_boundaries=False
    )
    engine = PathStitchingEngine(config)
    return await engine.stitch_paths(evolved_paths, original_dag, config)


async def stitch_paths_conservative(
    evolved_paths: List[DecisionDAG], original_dag: Optional[DecisionDAG] = None
) -> PathStitchingResult:
    """Stitch paths with conservative deduplication (preserves more diversity)."""
    config = PathStitchingConfig(
        strong_similarity_threshold=0.9,  # Higher threshold = less aggressive merging
        weak_similarity_threshold=0.8,
        preserve_path_boundaries=True,
        composite_dag_name="Conservatively Merged Paths",
    )
    engine = PathStitchingEngine(config)
    return await engine.stitch_paths(evolved_paths, original_dag, config)


async def stitch_paths_aggressive(
    evolved_paths: List[DecisionDAG], original_dag: Optional[DecisionDAG] = None
) -> PathStitchingResult:
    """Stitch paths with aggressive deduplication (maximizes consolidation)."""
    config = PathStitchingConfig(
        strong_similarity_threshold=0.7,  # Lower threshold = more aggressive merging
        weak_similarity_threshold=0.5,
        preserve_path_boundaries=False,
        composite_dag_name="Aggressively Merged Paths",
    )
    engine = PathStitchingEngine(config)
    return await engine.stitch_paths(evolved_paths, original_dag, config)
