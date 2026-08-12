import logging
from typing import List, Tuple

from ..models import DecisionDAG, DecisionNode, NodeType

logger = logging.getLogger(__name__)


class DAGValidator:
    """Validator for decision DAG structure and constraints."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_dag(self, dag: DecisionDAG) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete DAG structure.

        Args:
            dag: DAG to validate

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Basic structure validation
        self._validate_basic_structure(dag)

        # Node validation
        self._validate_nodes(dag)

        # Edge validation
        self._validate_edges(dag)

        # Layer validation
        self._validate_layers(dag)

        # Connectivity validation
        self._validate_connectivity(dag)

        # Alternating pattern validation
        self._validate_alternating_pattern(dag)

        is_valid = len(self.errors) == 0
        return is_valid, self.errors.copy(), self.warnings.copy()

    def _validate_basic_structure(self, dag: DecisionDAG) -> None:
        """Validate basic DAG structure."""
        if not dag.all_nodes:
            self.errors.append("DAG has no nodes")
            return

        if not dag.root_nodes:
            self.errors.append("DAG has no root nodes")

        # Check for missing nodes referenced in edges
        node_ids = set(dag.all_nodes.keys())
        for edge in dag.edges:
            if edge.source_id not in node_ids:
                self.errors.append(f"Edge references missing source node: {edge.source_id}")
            if edge.target_id not in node_ids:
                self.errors.append(f"Edge references missing target node: {edge.target_id}")

    def _validate_nodes(self, dag: DecisionDAG) -> None:
        """Validate individual nodes."""
        for node_id, node in dag.all_nodes.items():
            if node.id != node_id:
                self.errors.append(f"Node ID mismatch: key={node_id}, node.id={node.id}")

            if not node.title.strip():
                self.errors.append(f"Node {node_id} has empty title")

            if not node.description.strip():
                self.warnings.append(f"Node {node_id} has empty description")

            if node.layer < 0:
                self.errors.append(f"Node {node_id} has negative layer: {node.layer}")

    def _validate_edges(self, dag: DecisionDAG) -> None:
        """Validate edges."""
        edge_pairs = set()

        for edge in dag.edges:
            # Check for duplicate edges
            edge_pair = (edge.source_id, edge.target_id)
            if edge_pair in edge_pairs:
                self.errors.append(f"Duplicate edge: {edge.source_id} -> {edge.target_id}")
            edge_pairs.add(edge_pair)

            # Check for self-loops
            if edge.source_id == edge.target_id:
                self.errors.append(f"Self-loop detected: {edge.source_id}")

    def _validate_layers(self, dag: DecisionDAG) -> None:
        """Validate layer structure."""
        # Check root nodes are at layer 0
        for root in dag.root_nodes:
            if root.layer != 0:
                self.errors.append(f"Root node {root.id} not at layer 0: layer={root.layer}")

        # Check layer progression in edges
        for edge in dag.edges:
            source = dag.get_node(edge.source_id)
            target = dag.get_node(edge.target_id)

            if source and target:
                if target.layer != source.layer + 1:
                    self.errors.append(
                        f"Invalid layer progression: {source.id} (layer {source.layer}) -> "
                        f"{target.id} (layer {target.layer})"
                    )

    def _validate_connectivity(self, dag: DecisionDAG) -> None:
        """Validate DAG connectivity."""
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            rec_stack.add(node_id)

            # Check all children
            for edge in dag.edges:
                if edge.source_id == node_id:
                    if has_cycle(edge.target_id):
                        return True

            rec_stack.remove(node_id)
            return False

        # Check for cycles from each root
        for root in dag.root_nodes:
            visited = set()
            rec_stack = set()
            if has_cycle(root.id, visited, rec_stack):
                self.errors.append(f"Cycle detected starting from root {root.id}")
                break

        # Check for unreachable nodes
        reachable = set()

        def mark_reachable(node_id: str):
            if node_id in reachable:
                return
            reachable.add(node_id)
            for edge in dag.edges:
                if edge.source_id == node_id:
                    mark_reachable(edge.target_id)

        for root in dag.root_nodes:
            mark_reachable(root.id)

        unreachable = set(dag.all_nodes.keys()) - reachable
        for node_id in unreachable:
            self.warnings.append(f"Unreachable node: {node_id}")

    def _validate_alternating_pattern(self, dag: DecisionDAG) -> None:
        """Validate the alternating decision-option pattern."""
        for node in dag.all_nodes.values():
            expected_type = NodeType.DECISION if node.layer % 2 == 0 else NodeType.OPTION
            if node.type != expected_type:
                self.errors.append(
                    f"Node {node.id} at layer {node.layer} should be {expected_type.value}, but is {node.type.value}"
                )


class NodeValidator:
    """Validator for individual nodes."""

    @staticmethod
    def validate_node(node: DecisionNode) -> Tuple[bool, List[str]]:
        """
        Validate a single node.

        Args:
            node: Node to validate

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        if not node.id:
            errors.append("Node ID is empty")

        if not node.title.strip():
            errors.append("Node title is empty")

        if not node.description.strip():
            errors.append("Node description is empty")

        if node.layer < 0:
            errors.append(f"Invalid layer: {node.layer}")

        # Validate type based on layer
        expected_type = NodeType.DECISION if node.layer % 2 == 0 else NodeType.OPTION
        if node.type != expected_type:
            errors.append(
                f"Node type {node.type.value} doesn't match expected type {expected_type.value} for layer {node.layer}"
            )

        if node.confidence_score is not None:
            if not 0 <= node.confidence_score <= 1:
                errors.append(f"Invalid confidence score: {node.confidence_score}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_children_for_parent(
        parent: DecisionNode, children: List[DecisionNode], min_children: int = 2, max_children: int = 5
    ) -> Tuple[bool, List[str]]:
        """
        Validate children nodes for a given parent.

        Args:
            parent: Parent node
            children: List of child nodes
            min_children: Minimum number of children required
            max_children: Maximum number of children allowed

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        if len(children) < min_children:
            errors.append(f"Too few children: {len(children)} < {min_children}")

        if len(children) > max_children:
            errors.append(f"Too many children: {len(children)} > {max_children}")

        expected_child_layer = parent.layer + 1
        expected_child_type = NodeType.OPTION if parent.type == NodeType.DECISION else NodeType.DECISION

        for child in children:
            if child.layer != expected_child_layer:
                errors.append(f"Child {child.id} has wrong layer: {child.layer} != {expected_child_layer}")

            if child.type != expected_child_type:
                errors.append(f"Child {child.id} has wrong type: {child.type.value} != {expected_child_type.value}")

        # Check for duplicate titles (potential duplicates)
        titles = [child.title.lower().strip() for child in children]
        if len(set(titles)) < len(titles):
            errors.append("Duplicate child titles detected")

        return len(errors) == 0, errors


class ConstraintValidator:
    """Validator for business logic constraints."""

    @staticmethod
    def validate_strategic_constraints(dag: DecisionDAG) -> Tuple[bool, List[str], List[str]]:
        """
        Validate strategic planning constraints.

        Args:
            dag: DAG to validate

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Check for minimum path lengths
        paths = dag.get_paths()
        for i, path in enumerate(paths):
            if len(path) < 3:
                warnings.append(f"Path {i} is very short: {len(path)} nodes")

        # Check for balanced branching
        for layer in range(dag.get_max_layer()):
            layer_nodes = dag.get_nodes_at_layer(layer)
            if not layer_nodes:
                continue

            children_counts = []
            for node in layer_nodes:
                children = dag.get_children(node)
                children_counts.append(len(children))

            if children_counts:
                avg_children = sum(children_counts) / len(children_counts)
                if avg_children < 2:
                    warnings.append(f"Layer {layer} has low branching factor: {avg_children:.1f}")
                elif avg_children > 6:
                    warnings.append(f"Layer {layer} has high branching factor: {avg_children:.1f}")

        # Check for leaf nodes (paths should end somewhere)
        leaf_count = 0
        for node in dag.all_nodes.values():
            children = dag.get_children(node)
            if not children:
                leaf_count += 1

        if leaf_count == 0:
            errors.append("DAG has no leaf nodes (infinite paths)")
        elif leaf_count < len(paths) * 0.5:
            warnings.append(f"Few leaf nodes relative to paths: {leaf_count} leaves, {len(paths)} paths")

        return len(errors) == 0, errors, warnings


def validate_dag_comprehensive(dag: DecisionDAG) -> dict:
    """
    Perform comprehensive validation of a DAG.

    Args:
        dag: DAG to validate

    Returns:
        Dictionary with validation results
    """
    results = {"is_valid": True, "errors": [], "warnings": [], "validation_details": {}}

    # Structure validation
    dag_validator = DAGValidator()
    is_structure_valid, structure_errors, structure_warnings = dag_validator.validate_dag(dag)

    results["validation_details"]["structure"] = {
        "is_valid": is_structure_valid,
        "errors": structure_errors,
        "warnings": structure_warnings,
    }

    # Strategic constraints validation
    is_strategy_valid, strategy_errors, strategy_warnings = ConstraintValidator.validate_strategic_constraints(dag)

    results["validation_details"]["strategy"] = {
        "is_valid": is_strategy_valid,
        "errors": strategy_errors,
        "warnings": strategy_warnings,
    }

    # Combine results
    all_errors = structure_errors + strategy_errors
    all_warnings = structure_warnings + strategy_warnings

    results["is_valid"] = len(all_errors) == 0
    results["errors"] = all_errors
    results["warnings"] = all_warnings

    return results
