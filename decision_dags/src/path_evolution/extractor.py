"""Path extraction engine for converting DAG paths to individual sub-DAGs."""

import logging
from typing import List, Tuple

from ..models import DecisionDAG, DecisionNode, DecisionEdge, PathExtractionMetrics

logger = logging.getLogger(__name__)


class PathExtractionEngine:
    """Extract all paths from a DAG as individual sub-DAGs for evolution."""

    def __init__(self):
        self.path_id_counter = 0

    def extract_paths(self, dag: DecisionDAG) -> Tuple[List[DecisionDAG], PathExtractionMetrics]:
        """
        Extract all paths from DAG as individual sub-DAGs with comprehensive metrics.

        Args:
            dag: Source DAG to extract paths from

        Returns:
            Tuple of (List of path DAGs, extraction metrics)
        """
        if not dag.root_nodes:
            logger.warning("DAG has no root nodes, cannot extract paths")
            return [], PathExtractionMetrics(
                total_paths=0,
                valid_paths=0,
                failed_extractions=0,
                duplicate_paths=0,
                avg_path_length=0.0,
                max_path_length=0,
                min_path_length=0,
                path_length_distribution={},
                extraction_errors=["No root nodes in DAG"],
            )

        # Find all root-to-leaf paths using DFS
        all_paths = []
        for root in dag.root_nodes:
            root_paths = self._extract_paths_from_node(dag, root, [root])
            all_paths.extend(root_paths)

        # Convert node paths to sub-DAGs
        path_dags = []
        failed_extractions = 0
        extraction_errors = []

        for path_nodes in all_paths:
            try:
                path_dag = self._create_path_dag(path_nodes, dag)
                path_dags.append(path_dag)
            except Exception as e:
                logger.error(f"Failed to create path DAG: {e}")
                failed_extractions += 1
                extraction_errors.append(str(e))
                continue

        # Calculate metrics
        metrics = self._calculate_extraction_metrics(path_dags, failed_extractions, extraction_errors)

        logger.info(f"Extracted {len(path_dags)} paths from DAG with {len(dag.all_nodes)} nodes")
        return path_dags, metrics

    def _extract_paths_from_node(
        self, dag: DecisionDAG, current_node: DecisionNode, current_path: List[DecisionNode]
    ) -> List[List[DecisionNode]]:
        """
        Recursive DFS to find all paths from a node.

        Args:
            dag: Source DAG
            current_node: Current node in traversal
            current_path: Current path being built

        Returns:
            List of complete paths (each path is a list of nodes)
        """
        children = dag.get_children(current_node)

        if not children:
            # Leaf node - return the complete path
            return [current_path.copy()]

        # Continue traversal to children
        all_paths = []
        for child in children:
            child_path = current_path + [child]
            child_paths = self._extract_paths_from_node(dag, child, child_path)
            all_paths.extend(child_paths)

        return all_paths

    def _create_path_dag(self, path_nodes: List[DecisionNode], original_dag: DecisionDAG) -> DecisionDAG:
        """
        Create a sub-DAG from a path.

        Args:
            path_nodes: Nodes in the path
            original_dag: Original DAG for edge information

        Returns:
            Sub-DAG representing the path
        """
        path_dag = DecisionDAG(
            id=f"path_{self.path_id_counter}",
            metadata={"source_dag_id": original_dag.id, "path_length": len(path_nodes), "is_path_dag": True},
        )
        self.path_id_counter += 1

        # Add nodes to path DAG
        for node in path_nodes:
            # Create a copy of the node to avoid modifying the original
            node_copy = node.copy()
            path_dag.add_node(node_copy)

        # Add edges between consecutive nodes in the path
        for i in range(len(path_nodes) - 1):
            source = path_nodes[i]
            target = path_nodes[i + 1]

            # Find the original edge between these nodes
            original_edge = original_dag.get_edge(source.id, target.id)
            if original_edge:
                # Create a copy of the edge
                edge_copy = original_edge.copy()
                path_dag.add_edge(edge_copy)
            else:
                # Create a basic edge if original not found
                logger.warning(f"Original edge not found between {source.id} and {target.id}")

                # Determine edge type based on source and target node types
                from ..models import EdgeType, DecisionReasoningType, NodeType
                edge_type = (
                    EdgeType.DECISION_TO_OPTION if source.type == NodeType.DECISION
                    else EdgeType.OPTION_TO_DECISION
                )

                # Set decision_reasoning_type for option-to-decision edges
                decision_reasoning = None
                if edge_type == EdgeType.OPTION_TO_DECISION:
                    # Try to get from the target (decision) node
                    if hasattr(target, 'reasoning_type') and target.reasoning_type:
                        decision_reasoning = target.reasoning_type
                    else:
                        decision_reasoning = DecisionReasoningType.LOGICAL

                basic_edge = DecisionEdge(
                    source_id=source.id,
                    target_id=target.id,
                    edge_type=edge_type,
                    condition=f"Path from {source.title} to {target.title}",
                    decision_reasoning_type=decision_reasoning,
                    relationship="path_sequence",
                    metadata={"synthetic": True},
                )
                path_dag.add_edge(basic_edge)

        return path_dag

    def get_path_summary(self, path_dag: DecisionDAG) -> dict:
        """
        Get a summary of a path DAG.

        Args:
            path_dag: Path DAG to summarize

        Returns:
            Dictionary with path summary information
        """
        nodes = list(path_dag.all_nodes.values())

        # Sort nodes by layer for proper path order
        nodes.sort(key=lambda n: n.layer)

        summary = {
            "path_id": path_dag.id,
            "length": len(nodes),
            "start_node": nodes[0].title if nodes else "Empty",
            "end_node": nodes[-1].title if nodes else "Empty",
            "decision_count": sum(1 for n in nodes if n.is_decision()),
            "option_count": sum(1 for n in nodes if n.is_option()),
            "node_titles": [n.title for n in nodes],
            "metadata": path_dag.metadata,
        }

        return summary

    def _calculate_extraction_metrics(
        self, path_dags: List[DecisionDAG], failed_extractions: int, extraction_errors: List[str]
    ) -> PathExtractionMetrics:
        """Calculate comprehensive metrics for path extraction."""
        if not path_dags:
            return PathExtractionMetrics(
                total_paths=0,
                valid_paths=0,
                failed_extractions=failed_extractions,
                duplicate_paths=0,
                avg_path_length=0.0,
                max_path_length=0,
                min_path_length=0,
                path_length_distribution={},
                extraction_errors=extraction_errors,
            )

        # Calculate path lengths
        path_lengths = [len(path.all_nodes) for path in path_dags]

        # Calculate length distribution
        length_distribution = {}
        for length in path_lengths:
            length_distribution[length] = length_distribution.get(length, 0) + 1

        # Check for duplicates
        duplicate_count = self._count_duplicate_paths(path_dags)

        # Validate paths
        valid_paths = sum(1 for path in path_dags if self._validate_single_path(path))

        return PathExtractionMetrics(
            total_paths=len(path_dags),
            valid_paths=valid_paths,
            failed_extractions=failed_extractions,
            duplicate_paths=duplicate_count,
            avg_path_length=sum(path_lengths) / len(path_lengths) if path_lengths else 0.0,
            max_path_length=max(path_lengths) if path_lengths else 0,
            min_path_length=min(path_lengths) if path_lengths else 0,
            path_length_distribution=length_distribution,
            extraction_errors=extraction_errors,
        )

    def _count_duplicate_paths(self, path_dags: List[DecisionDAG]) -> int:
        """Count duplicate paths based on signatures."""
        signatures = set()
        duplicates = 0

        for path in path_dags:
            signature = self._get_path_signature(path)
            if signature in signatures:
                duplicates += 1
            else:
                signatures.add(signature)

        return duplicates

    def extract_paths_with_criteria(
        self, dag: DecisionDAG, min_length: int = 2, max_length: int = 20, include_incomplete: bool = False
    ) -> Tuple[List[DecisionDAG], PathExtractionMetrics]:
        """
        Extract paths with specific criteria and filtering.

        Args:
            dag: Source DAG to extract paths from
            min_length: Minimum path length to include
            max_length: Maximum path length to include
            include_incomplete: Whether to include incomplete paths

        Returns:
            Tuple of (filtered path DAGs, metrics)
        """
        # Extract all paths first
        all_paths, base_metrics = self.extract_paths(dag)

        # Filter paths based on criteria
        filtered_paths = []
        filtered_out_count = 0

        for path in all_paths:
            path_length = len(path.all_nodes)

            # Check length criteria
            if path_length < min_length or path_length > max_length:
                filtered_out_count += 1
                continue

            # Check completeness criteria
            if not include_incomplete and not self._is_complete_path(path):
                filtered_out_count += 1
                continue

            filtered_paths.append(path)

        # Update metrics
        filtered_metrics = self._update_metrics_for_filtering(base_metrics, filtered_paths, filtered_out_count)

        logger.info(f"Filtered paths: {len(filtered_paths)} remaining after criteria filtering")
        return filtered_paths, filtered_metrics

    def _is_complete_path(self, path: DecisionDAG) -> bool:
        """Check if a path is complete (has proper decision-option alternation)."""
        nodes = list(path.all_nodes.values())
        nodes.sort(key=lambda n: n.layer)

        # Check alternating pattern
        for i, node in enumerate(nodes):
            expected_type = "decision" if i % 2 == 0 else "option"
            if node.type.value != expected_type:
                return False

        return True

    def _update_metrics_for_filtering(
        self, base_metrics: PathExtractionMetrics, filtered_paths: List[DecisionDAG], filtered_out_count: int
    ) -> PathExtractionMetrics:
        """Update metrics after filtering."""
        if not filtered_paths:
            return PathExtractionMetrics(
                total_paths=0,
                valid_paths=0,
                failed_extractions=base_metrics.failed_extractions,
                duplicate_paths=0,
                avg_path_length=0.0,
                max_path_length=0,
                min_path_length=0,
                path_length_distribution={},
                extraction_errors=base_metrics.extraction_errors + [f"Filtered out {filtered_out_count} paths"],
            )

        # Recalculate metrics for filtered paths
        path_lengths = [len(path.all_nodes) for path in filtered_paths]

        length_distribution = {}
        for length in path_lengths:
            length_distribution[length] = length_distribution.get(length, 0) + 1

        duplicate_count = self._count_duplicate_paths(filtered_paths)
        valid_paths = sum(1 for path in filtered_paths if self._validate_single_path(path))

        return PathExtractionMetrics(
            total_paths=len(filtered_paths),
            valid_paths=valid_paths,
            failed_extractions=base_metrics.failed_extractions,
            duplicate_paths=duplicate_count,
            avg_path_length=sum(path_lengths) / len(path_lengths),
            max_path_length=max(path_lengths),
            min_path_length=min(path_lengths),
            path_length_distribution=length_distribution,
            extraction_errors=base_metrics.extraction_errors + [f"Filtered out {filtered_out_count} paths"],
        )

    def validate_extracted_paths(self, path_dags: List[DecisionDAG]) -> dict:
        """
        Validate the extracted paths for correctness.

        Args:
            path_dags: List of extracted path DAGs

        Returns:
            Validation results dictionary
        """
        results = {
            "total_paths": len(path_dags),
            "valid_paths": 0,
            "invalid_paths": 0,
            "validation_errors": [],
            "path_lengths": [],
            "duplicate_paths": 0,
        }

        seen_path_signatures = set()

        for path_dag in path_dags:
            try:
                # Validate path structure
                is_valid = self._validate_single_path(path_dag)

                if is_valid:
                    results["valid_paths"] += 1
                else:
                    results["invalid_paths"] += 1
                    results["validation_errors"].append(f"Invalid path structure: {path_dag.id}")

                # Track path length
                results["path_lengths"].append(len(path_dag.all_nodes))

                # Check for duplicates
                path_signature = self._get_path_signature(path_dag)
                if path_signature in seen_path_signatures:
                    results["duplicate_paths"] += 1
                else:
                    seen_path_signatures.add(path_signature)

            except Exception as e:
                results["invalid_paths"] += 1
                results["validation_errors"].append(f"Error validating {path_dag.id}: {e}")

        return results

    def _validate_single_path(self, path_dag: DecisionDAG) -> bool:
        """Validate that a path DAG represents a valid linear path."""
        nodes = list(path_dag.all_nodes.values())

        if not nodes:
            return False

        # Should have exactly one root node
        root_nodes = [n for n in nodes if n.layer == 0]
        if len(root_nodes) != 1:
            return False

        # Should have exactly one leaf node
        leaf_nodes = []
        for node in nodes:
            children = path_dag.get_children(node)
            if not children:
                leaf_nodes.append(node)

        if len(leaf_nodes) != 1:
            return False

        # Each node (except leaf) should have exactly one child
        for node in nodes:
            children = path_dag.get_children(node)
            if len(children) > 1:
                return False

            # Each node (except root) should have exactly one parent
            parents = path_dag.get_parents(node)
            if node.layer > 0 and len(parents) != 1:
                return False

        # Layers should be consecutive
        layers = sorted(set(n.layer for n in nodes))
        if layers != list(range(len(layers))):
            return False

        return True

    def _get_path_signature(self, path_dag: DecisionDAG) -> str:
        """Get a signature for a path to detect duplicates."""
        nodes = list(path_dag.all_nodes.values())
        nodes.sort(key=lambda n: n.layer)

        # Create signature from node titles
        titles = [n.title.lower().strip() for n in nodes]
        return "|".join(titles)
