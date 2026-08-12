import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..settings import settings

logger = logging.getLogger(__name__)


class SimpleCSVLogger:
    """Simple CSV logger that doesn't interfere with async operations."""

    def __init__(self, log_name: str, headers: List[str]):
        self.log_name = log_name
        self.headers = headers
        self.file_path = self._get_csv_file_path()
        self._initialized = False

    def _get_csv_file_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.log_name}_{timestamp}.csv"
        return settings.output_path / filename

    def _initialize_csv(self):
        """Initialize CSV file with headers if not already done."""
        if not self._initialized:
            try:
                with open(self.file_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                    writer.writeheader()
                self._initialized = True
                logger.info(f"Initialized CSV log: {self.file_path}")
            except Exception as e:
                logger.error(f"Failed to initialize CSV file: {e}")

    def log_row(self, data: Dict[str, Any]):
        """Log a single row to the CSV file."""
        try:
            self._initialize_csv()

            # Add timestamp if not already present
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().isoformat()

            # Filter data to only include headers and fill missing ones
            filtered_data = {}
            for header in self.headers:
                if header in data:
                    filtered_data[header] = str(data[header]) if data[header] is not None else ""
                else:
                    filtered_data[header] = ""

            with open(self.file_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                writer.writerow(filtered_data)

        except Exception as e:
            logger.error(f"Failed to log CSV row: {e}")

    def log_dag_structure(self, dag, phase: str = "final"):
        """Log the complete DAG structure to CSV."""
        try:
            # Skip if DAG is None or empty
            if dag is None:
                logger.warning(f"Cannot log DAG structure for phase {phase}: DAG is None")
                return

            # Log nodes
            nodes_logger = SimpleCSVLogger(
                f"dag_nodes_{phase}",
                [
                    "timestamp", "node_id", "node_type", "layer", "title",
                    "description", "parent_ids", "child_ids", "goal_impact_score",
                    "resource_requirements", "feasibility_score", "risk_level"
                ]
            )

            # Safely iterate over nodes
            nodes = getattr(dag, 'nodes', []) or []
            for node in nodes:
                try:
                    nodes_logger.log_row({
                        "node_id": getattr(node, 'id', 'unknown'),
                        "node_type": getattr(node, 'node_type', 'unknown'),
                        "layer": getattr(node, 'layer', 0),
                        "title": getattr(node, 'title', ''),
                        "description": getattr(node, 'description', ''),
                        "parent_ids": ",".join(getattr(node, 'parent_ids', []) or []),
                        "child_ids": ",".join(getattr(node, 'child_ids', []) or []),
                        "goal_impact_score": getattr(node, 'goal_impact_score', 0.0),
                        "resource_requirements": str(getattr(node, 'resource_requirements', '') or ''),
                        "feasibility_score": getattr(node, 'feasibility_score', 0.0),
                        "risk_level": getattr(node, 'risk_level', 'unknown')
                    })
                except Exception as node_error:
                    logger.warning(f"Failed to log node {getattr(node, 'id', 'unknown')}: {node_error}")

            # Log edges
            edges_logger = SimpleCSVLogger(
                f"dag_edges_{phase}",
                [
                    "timestamp", "edge_id", "source_id", "target_id", "edge_type",
                    "weight", "conditional_logic", "confidence_score"
                ]
            )

            # Safely iterate over edges
            edges = getattr(dag, 'edges', []) or []
            for edge in edges:
                try:
                    edges_logger.log_row({
                        "edge_id": getattr(edge, 'id', 'unknown'),
                        "source_id": getattr(edge, 'source_id', 'unknown'),
                        "target_id": getattr(edge, 'target_id', 'unknown'),
                        "edge_type": getattr(edge, 'edge_type', 'unknown'),
                        "weight": getattr(edge, 'weight', 0.0),
                        "conditional_logic": getattr(edge, 'conditional_logic', ''),
                        "confidence_score": getattr(edge, 'confidence_score', 0.0)
                    })
                except Exception as edge_error:
                    logger.warning(f"Failed to log edge {getattr(edge, 'id', 'unknown')}: {edge_error}")

            logger.info(f"Logged DAG structure for phase: {phase} - {len(nodes)} nodes, {len(edges)} edges")

        except Exception as e:
            logger.error(f"Failed to log DAG structure: {e}")


# Global CSV loggers - created once and reused
_orchestration_logger = None
_dag_building_logger = None
_path_extraction_logger = None
_evolution_logger = None
_stitching_logger = None
_node_processing_logger = None


def get_orchestration_logger() -> SimpleCSVLogger:
    """Get or create the orchestration CSV logger."""
    global _orchestration_logger
    if _orchestration_logger is None:
        _orchestration_logger = SimpleCSVLogger(
            "orchestration_metrics",
            [
                "timestamp", "session_id", "phase", "duration_seconds", "status",
                "error_message", "nodes_processed", "edges_created", "total_nodes",
                "total_edges", "max_layer_reached", "success_rate"
            ]
        )
    return _orchestration_logger


def get_dag_building_logger() -> SimpleCSVLogger:
    """Get or create the DAG building CSV logger."""
    global _dag_building_logger
    if _dag_building_logger is None:
        _dag_building_logger = SimpleCSVLogger(
            "dag_building_metrics",
            [
                "timestamp", "session_id", "layer", "nodes_in_layer", "edges_created",
                "processing_time_seconds", "failed_nodes", "timeout_count", "retry_count"
            ]
        )
    return _dag_building_logger


def get_path_extraction_logger() -> SimpleCSVLogger:
    """Get or create the path extraction CSV logger."""
    global _path_extraction_logger
    if _path_extraction_logger is None:
        _path_extraction_logger = SimpleCSVLogger(
            "path_extraction_metrics",
            [
                "timestamp", "session_id", "total_paths_extracted", "avg_path_length",
                "max_path_length", "min_path_length", "extraction_time_seconds"
            ]
        )
    return _path_extraction_logger


def get_evolution_logger() -> SimpleCSVLogger:
    """Get or create the evolution CSV logger."""
    global _evolution_logger
    if _evolution_logger is None:
        _evolution_logger = SimpleCSVLogger(
            "evolution_metrics",
            [
                "timestamp", "session_id", "generation", "population_size",
                "fitness_improvement", "best_fitness", "avg_fitness", "evolution_time_seconds"
            ]
        )
    return _evolution_logger


def get_stitching_logger() -> SimpleCSVLogger:
    """Get or create the stitching CSV logger."""
    global _stitching_logger
    if _stitching_logger is None:
        _stitching_logger = SimpleCSVLogger(
            "stitching_metrics",
            [
                "timestamp", "session_id", "paths_stitched", "conflicts_resolved",
                "new_edges_added", "nodes_merged", "stitching_time_seconds"
            ]
        )
    return _stitching_logger


def get_node_processing_logger() -> SimpleCSVLogger:
    """Get or create the node processing CSV logger."""
    global _node_processing_logger
    if _node_processing_logger is None:
        _node_processing_logger = SimpleCSVLogger(
            "node_processing_metrics",
            [
                "timestamp", "session_id", "node_id", "node_type", "layer",
                "processing_time_seconds", "status", "retry_count", "error_message"
            ]
        )
    return _node_processing_logger
