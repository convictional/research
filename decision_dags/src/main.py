"""Main orchestration logic for the Decision DAG system."""

import logging
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from .models import (
    StrategicPath,
    DAGBuilderConfig,
    HITLWorkflowConfig,
    DecisionDAG,
    AlphaEvolutionConfig,
    EvaluationContext,
)
from .dag_builder.ensemble import DAGBuilderEnsemble
from .path_evolution.extractor import PathExtractionEngine
from .path_evolution.evolver import PathEvolutionEngine
from .path_evolution.stitcher import PathStitchingResult
from .utils.csv_logger import (
    get_orchestration_logger,
    get_dag_building_logger,
    get_path_extraction_logger,
    get_evolution_logger,
    get_stitching_logger,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Complete result of the decision DAG orchestration process."""

    # Core outputs
    original_dag: DecisionDAG
    evolved_dag: Optional[DecisionDAG] = None

    # Metrics and tracking
    orchestration_time_seconds: float = 0.0
    extraction_metrics: Dict[str, Any] = None
    evolution_metrics: Dict[str, Any] = None
    stitching_result: Optional[PathStitchingResult] = None

    # Process tracking
    phases_completed: List[str] = None
    total_phases: int = 4

    # Error tracking
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.phases_completed is None:
            self.phases_completed = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class OrchestrationConfig:
    """Configuration for the complete orchestration process."""

    # DAG building
    dag_config: Optional[DAGBuilderConfig] = None
    hitl_config: Optional[HITLWorkflowConfig] = None
    enable_hitl: bool = False

    # Path evolution
    evolution_config: Optional[AlphaEvolutionConfig] = None
    enable_evolution: bool = True

    # Path extraction
    min_path_length: int = 2
    max_path_length: int = 20
    include_incomplete_paths: bool = False

    # Path stitching
    stitching_strategy: str = "balanced"  # balanced, conservative, aggressive

    # Performance
    max_concurrent_operations: int = 4
    enable_detailed_logging: bool = True


async def orchestrate_decision_dag(
    problem_statement: str,
    strategic_paths: List[StrategicPath],
    organization_id: Optional[str] = None,
    config: Optional[OrchestrationConfig] = None,
) -> OrchestrationResult:
    """
    Main orchestration function for the complete decision DAG workflow.

    Phases:
    1. DAG Building: Create initial DAG from strategic paths
    2. Path Extraction: Extract strategic paths from DAG
    3. Path Evolution: Evolve paths using AlphaEvolve genetic algorithm
    4. Path Stitching: Merge evolved paths back into unified DAG

    Args:
        problem_statement: The problem to solve with the DAG
        strategic_paths: Initial strategic paths to build from
        organization_id: Optional organization ID for database context
        config: Optional orchestration configuration

    Returns:
        OrchestrationResult with complete process tracking and outputs
    """
    config = config or OrchestrationConfig()
    orchestration_start = datetime.now()
    session_id = str(uuid.uuid4())

    logger.info(f"🚀 Starting Decision DAG orchestration for: {problem_statement}")
    logger.info(f"📊 Session ID: {session_id}")

    # Get CSV loggers
    orchestration_logger = get_orchestration_logger()

    # Log orchestration start
    orchestration_logger.log_row(
        {
            "session_id": session_id,
            "phase": "orchestration_start",
            "duration_seconds": 0.0,
            "status": "started",
            "nodes_processed": 0,
            "edges_created": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "max_layer_reached": 0,
            "success_rate": 0.0,
        }
    )

    # Initialize result tracking
    result = OrchestrationResult(
        original_dag=None,  # Will be set after DAG building
        orchestration_time_seconds=0.0,
    )

    try:
        # ===== PHASE 1: DAG BUILDING =====
        logger.info("📊 Phase 1: Building Initial DAG")
        phase_start = datetime.now()

        original_dag = await _build_dag_phase(
            problem_statement=problem_statement,
            strategic_paths=strategic_paths,
            organization_id=organization_id,
            config=config,
            result=result,
            session_id=session_id,
        )
        result.original_dag = original_dag
        result.phases_completed.append("dag_building")

        # Log DAG building completion
        phase_duration = (datetime.now() - phase_start).total_seconds()
        orchestration_logger.log_row(
            {
                "session_id": session_id,
                "phase": "dag_building",
                "duration_seconds": phase_duration,
                "status": "completed",
                "nodes_processed": len(original_dag.all_nodes),
                "edges_created": len(original_dag.edges),
                "total_nodes": len(original_dag.all_nodes),
                "total_edges": len(original_dag.edges),
                "max_layer_reached": original_dag.get_max_layer(),
                "success_rate": 1.0,
            }
        )

        # Log DAG structure (only if DAG has nodes)
        if original_dag and len(original_dag.all_nodes) > 0:
            orchestration_logger.log_dag_structure(original_dag, "dag_building")

        if not config.enable_evolution:
            logger.info("⚡ Evolution disabled - returning original DAG")
            result.evolved_dag = original_dag
            result.orchestration_time_seconds = (datetime.now() - orchestration_start).total_seconds()
            return result

        # ===== PHASE 2: PATH EXTRACTION =====
        logger.info("🔍 Phase 2: Extracting Strategic Paths")
        phase_start = datetime.now()

        extracted_paths, extraction_metrics = await _extract_paths_phase(
            dag=original_dag, organization_id=organization_id, config=config, result=result, session_id=session_id
        )
        result.extraction_metrics = extraction_metrics
        result.phases_completed.append("path_extraction")

        # Log path extraction completion
        phase_duration = (datetime.now() - phase_start).total_seconds()
        orchestration_logger.log_row(
            {
                "session_id": session_id,
                "phase": "path_extraction",
                "duration_seconds": phase_duration,
                "status": "completed",
                "nodes_processed": len(extracted_paths) if extracted_paths else 0,
                "edges_created": 0,
                "total_nodes": len(original_dag.all_nodes),
                "total_edges": len(original_dag.edges),
                "max_layer_reached": original_dag.get_max_layer(),
                "success_rate": 1.0 if extracted_paths else 0.0,
            }
        )

        # ===== PHASE 3: PATH EVOLUTION =====
        logger.info("🧬 Phase 3: Evolving Strategic Paths")
        phase_start = datetime.now()

        evolved_paths, evolution_metrics = await _evolve_paths_phase(
            paths=extracted_paths,
            original_dag=original_dag,
            organization_id=organization_id,
            config=config,
            result=result,
            session_id=session_id,
        )
        result.evolution_metrics = evolution_metrics
        result.phases_completed.append("path_evolution")

        # Log path evolution completion
        phase_duration = (datetime.now() - phase_start).total_seconds()
        orchestration_logger.log_row(
            {
                "session_id": session_id,
                "phase": "path_evolution",
                "duration_seconds": phase_duration,
                "status": "completed",
                "nodes_processed": len(evolved_paths) if evolved_paths else 0,
                "edges_created": 0,
                "total_nodes": len(original_dag.all_nodes),
                "total_edges": len(original_dag.edges),
                "max_layer_reached": original_dag.get_max_layer(),
                "success_rate": 1.0 if evolved_paths else 0.0,
            }
        )

        # ===== PHASE 4: PATH STITCHING =====
        logger.info("🔗 Phase 4: Stitching Evolved Paths")
        phase_start = datetime.now()

        stitching_result = await _stitch_paths_phase(
            evolved_paths=evolved_paths, original_dag=original_dag, config=config, result=result, session_id=session_id
        )
        result.stitching_result = stitching_result
        result.evolved_dag = stitching_result.stitched_dag
        result.phases_completed.append("path_stitching")

        # Log path stitching completion
        phase_duration = (datetime.now() - phase_start).total_seconds()
        orchestration_logger.log_row(
            {
                "session_id": session_id,
                "phase": "path_stitching",
                "duration_seconds": phase_duration,
                "status": "completed",
                "nodes_processed": len(stitching_result.stitched_dag.all_nodes),
                "edges_created": len(stitching_result.stitched_dag.edges),
                "total_nodes": len(stitching_result.stitched_dag.all_nodes),
                "total_edges": len(stitching_result.stitched_dag.edges),
                "max_layer_reached": stitching_result.stitched_dag.get_max_layer(),
                "success_rate": 1.0,
            }
        )

        # Log final DAG structure (only if DAG has nodes)
        if stitching_result.stitched_dag and len(stitching_result.stitched_dag.all_nodes) > 0:
            orchestration_logger.log_dag_structure(stitching_result.stitched_dag, "final")

        # ===== COMPLETION =====
        result.orchestration_time_seconds = (datetime.now() - orchestration_start).total_seconds()

        logger.info(f"🎉 Orchestration complete in {result.orchestration_time_seconds:.1f}s")
        logger.info(
            f"📈 Final result: {len(result.evolved_dag.all_nodes)} nodes, {len(result.evolved_dag.edges)} edges"
        )

        # Log orchestration completion
        orchestration_logger.log_row(
            {
                "session_id": session_id,
                "phase": "orchestration_complete",
                "duration_seconds": result.orchestration_time_seconds,
                "status": "completed",
                "nodes_processed": len(result.evolved_dag.all_nodes),
                "edges_created": len(result.evolved_dag.edges),
                "total_nodes": len(result.evolved_dag.all_nodes),
                "total_edges": len(result.evolved_dag.edges),
                "max_layer_reached": result.evolved_dag.get_max_layer(),
                "success_rate": len(result.phases_completed) / result.total_phases,
            }
        )

        return result

    except Exception as e:
        error_msg = f"Orchestration failed in phase {len(result.phases_completed) + 1}: {e}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.orchestration_time_seconds = (datetime.now() - orchestration_start).total_seconds()

        # Log orchestration failure
        orchestration_logger.log_row(
            {
                "session_id": session_id,
                "phase": f"orchestration_failed_phase_{len(result.phases_completed) + 1}",
                "duration_seconds": result.orchestration_time_seconds,
                "status": "failed",
                "error_message": str(e),
                "nodes_processed": len(result.original_dag.all_nodes) if result.original_dag else 0,
                "edges_created": len(result.original_dag.edges) if result.original_dag else 0,
                "total_nodes": len(result.original_dag.all_nodes) if result.original_dag else 0,
                "total_edges": len(result.original_dag.edges) if result.original_dag else 0,
                "max_layer_reached": result.original_dag.get_max_layer() if result.original_dag else 0,
                "success_rate": len(result.phases_completed) / result.total_phases,
            }
        )

        raise


async def build_decision_dag(
    problem_statement: str,
    strategic_paths: List[StrategicPath],
    organization_id: Optional[str] = None,
    enable_hitl: bool = False,
    dag_config: Optional[DAGBuilderConfig] = None,
    hitl_config: Optional[HITLWorkflowConfig] = None,
) -> DecisionDAG:
    """
    Legacy function for building decision DAGs (backward compatibility).

    For new code, use orchestrate_decision_dag() for full workflow.
    """
    start_time = datetime.now()
    session_id = str(uuid.uuid4())

    logger.info(f"Starting DAG building for problem: {problem_statement}")
    logger.info(f"📊 Session ID: {session_id}")

    # Get CSV loggers
    orchestration_logger = get_orchestration_logger()
    dag_building_logger = get_dag_building_logger()

    # Log DAG building start
    orchestration_logger.log_row(
        {
            "session_id": session_id,
            "phase": "legacy_dag_building_start",
            "duration_seconds": 0.0,
            "status": "started",
            "nodes_processed": 0,
            "edges_created": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "max_layer_reached": 0,
            "success_rate": 0.0,
        }
    )

    # Initialize ensemble builder
    ensemble = DAGBuilderEnsemble(config=dag_config, hitl_config=hitl_config, enable_hitl=enable_hitl)

    try:
        # Build the DAG
        dag = await ensemble.build_dag(
            problem_statement=problem_statement, strategic_paths=strategic_paths, organization_id=organization_id
        )

        build_duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"DAG building complete: {len(dag.all_nodes)} nodes, {len(dag.edges)} edges")

        # Log DAG building completion
        orchestration_logger.log_row(
            {
                "session_id": session_id,
                "phase": "legacy_dag_building_complete",
                "duration_seconds": build_duration,
                "status": "completed",
                "nodes_processed": len(dag.all_nodes),
                "edges_created": len(dag.edges),
                "total_nodes": len(dag.all_nodes),
                "total_edges": len(dag.edges),
                "max_layer_reached": dag.get_max_layer(),
                "success_rate": 1.0,
            }
        )

        # Log DAG building details
        dag_building_logger.log_row(
            {
                "session_id": session_id,
                "layer": dag.get_max_layer(),
                "nodes_in_layer": len(dag.all_nodes),
                "edges_created": len(dag.edges),
                "processing_time_seconds": build_duration,
                "failed_nodes": 0,
                "timeout_count": 0,
                "retry_count": 0,
            }
        )

        # Log complete DAG structure (only if DAG has nodes)
        if dag and len(dag.all_nodes) > 0:
            orchestration_logger.log_dag_structure(dag, "legacy_build")

        return dag

    except Exception as e:
        build_duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"DAG building failed: {e}")

        # Log DAG building failure
        orchestration_logger.log_row(
            {
                "session_id": session_id,
                "phase": "legacy_dag_building_failed",
                "duration_seconds": build_duration,
                "status": "failed",
                "error_message": str(e),
                "nodes_processed": 0,
                "edges_created": 0,
                "total_nodes": 0,
                "total_edges": 0,
                "max_layer_reached": 0,
                "success_rate": 0.0,
            }
        )

        raise


# ===== PHASE IMPLEMENTATION FUNCTIONS =====


async def _build_dag_phase(
    problem_statement: str,
    strategic_paths: List[StrategicPath],
    organization_id: Optional[str],
    config: OrchestrationConfig,
    result: OrchestrationResult,
    session_id: str,
) -> DecisionDAG:
    """Phase 1: Build the initial DAG from strategic paths."""
    try:
        # Initialize ensemble builder
        ensemble = DAGBuilderEnsemble(
            config=config.dag_config, hitl_config=config.hitl_config, enable_hitl=config.enable_hitl
        )

        # Build the DAG
        dag = await ensemble.build_dag(
            problem_statement=problem_statement, strategic_paths=strategic_paths, organization_id=organization_id
        )

        logger.info(f"✅ DAG building complete: {len(dag.all_nodes)} nodes, {len(dag.edges)} edges")

        # Log DAG building details
        dag_building_logger = get_dag_building_logger()
        dag_building_logger.log_row(
            {
                "session_id": session_id,
                "layer": dag.get_max_layer(),
                "nodes_in_layer": len(dag.all_nodes),
                "edges_created": len(dag.edges),
                "processing_time_seconds": 0.0,
                "failed_nodes": 0,
                "timeout_count": 0,
                "retry_count": 0,
            }
        )

        return dag

    except Exception as e:
        error_msg = f"DAG building failed: {e}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        raise


async def _extract_paths_phase(
    dag: DecisionDAG,
    organization_id: Optional[str],
    config: OrchestrationConfig,
    result: OrchestrationResult,
    session_id: str,
) -> tuple[List[DecisionDAG], Dict[str, Any]]:
    """Phase 2: Extract strategic paths from the DAG."""
    try:
        # Initialize path extractor
        extractor = PathExtractionEngine()

        # Extract paths with criteria
        extracted_paths, extraction_metrics = extractor.extract_paths_with_criteria(
            dag=dag,
            min_length=config.min_path_length,
            max_length=config.max_path_length,
            include_incomplete=config.include_incomplete_paths,
        )

        logger.info(f"✅ Path extraction complete: {len(extracted_paths)} paths extracted")
        logger.info(f"📊 Extraction metrics: {extraction_metrics}")

        if not extracted_paths:
            warning_msg = "No paths extracted from DAG - evolution will be skipped"
            logger.warning(warning_msg)
            result.warnings.append(warning_msg)

        # Log path extraction details
        path_extraction_logger = get_path_extraction_logger()
        path_extraction_logger.log_row(
            {
                "session_id": session_id,
                "total_paths_extracted": len(extracted_paths),
                "avg_path_length": extraction_metrics.avg_path_length
                if hasattr(extraction_metrics, "avg_path_length")
                else 0.0,
                "max_path_length": extraction_metrics.max_path_length
                if hasattr(extraction_metrics, "max_path_length")
                else 0,
                "min_path_length": extraction_metrics.min_path_length
                if hasattr(extraction_metrics, "min_path_length")
                else 0,
                "extraction_time_seconds": 0.0,
            }
        )

        return extracted_paths, extraction_metrics._asdict()

    except Exception as e:
        error_msg = f"Path extraction failed: {e}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        raise


async def _evolve_paths_phase(
    paths: List[DecisionDAG],
    original_dag: DecisionDAG,
    organization_id: Optional[str],
    config: OrchestrationConfig,
    result: OrchestrationResult,
    session_id: str,
) -> tuple[List[DecisionDAG], Dict[str, Any]]:
    """Phase 3: Evolve the extracted paths using AlphaEvolve."""
    try:
        if not paths:
            logger.info("⚡ No paths to evolve - returning empty list")
            return [], {"evolved_paths": 0, "improved_paths": 0}

        # Initialize evolution engine
        evolution_config = config.evolution_config or AlphaEvolutionConfig()
        evolver = PathEvolutionEngine(evolution_config)

        # Create evaluation context
        evaluation_context = EvaluationContext(
            organization_id=organization_id or "default",
            domain_context="strategic_planning",
            user_preferences={"risk_tolerance": "medium"},
            temporal_context={"current_date": datetime.now().isoformat()},
            resource_constraints={"max_budget": 1000000, "max_timeline_weeks": 52},
        )

        # Evolve paths
        evolved_paths = await evolver.evolve_paths(
            paths=paths,
            context=evaluation_context,
            organizational_goals=[
                {"type": "growth", "weight": 0.4, "target": 0.8},
                {"type": "efficiency", "weight": 0.3, "target": 0.7},
                {"type": "risk_mitigation", "weight": 0.2, "target": 0.6},
                {"type": "innovation", "weight": 0.1, "target": 0.5},
            ],
        )

        # Get evolution summary
        evolution_summary = evolver.get_evolution_summary()

        logger.info(f"✅ Path evolution complete: {len(evolved_paths)} evolved paths")
        logger.info(f"📊 Evolution summary: {evolution_summary}")

        evolution_metrics = {
            "original_paths": len(paths),
            "evolved_paths": len(evolved_paths),
            "improved_paths": len([p for p in evolved_paths if "evolved_from" in p.metadata]),
            "evolution_summary": evolution_summary,
        }

        # Log evolution details
        evolution_logger = get_evolution_logger()
        evolution_logger.log_row(
            {
                "session_id": session_id,
                "generation": evolution_summary.get("generations", 0) if evolution_summary else 0,
                "population_size": len(paths),
                "fitness_improvement": evolution_summary.get("fitness_improvement", 0.0) if evolution_summary else 0.0,
                "best_fitness": evolution_summary.get("best_fitness", 0.0) if evolution_summary else 0.0,
                "avg_fitness": evolution_summary.get("avg_fitness", 0.0) if evolution_summary else 0.0,
                "evolution_time_seconds": 0.0,
            }
        )

        return evolved_paths, evolution_metrics

    except Exception as e:
        error_msg = f"Path evolution failed: {e}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        raise


async def _stitch_paths_phase(
    evolved_paths: List[DecisionDAG],
    original_dag: DecisionDAG,
    config: OrchestrationConfig,
    result: OrchestrationResult,
    session_id: str,
) -> PathStitchingResult:
    """Phase 4: Stitch evolved paths back into a unified DAG."""
    try:
        if not evolved_paths:
            logger.info("⚡ No evolved paths to stitch - returning original DAG")
            return PathStitchingResult(
                stitched_dag=original_dag,
                original_path_count=0,
                nodes_deduplicated=0,
                edges_consolidated=0,
                stitching_time_seconds=0.0,
                deduplication_metadata={},
            )

        # Choose stitching strategy
        if config.stitching_strategy == "conservative":
            from .path_evolution.stitcher import stitch_paths_conservative

            stitching_result = await stitch_paths_conservative(evolved_paths, original_dag)
        elif config.stitching_strategy == "aggressive":
            from .path_evolution.stitcher import stitch_paths_aggressive

            stitching_result = await stitch_paths_aggressive(evolved_paths, original_dag)
        else:  # balanced (default)
            from .path_evolution.stitcher import stitch_paths_balanced

            stitching_result = await stitch_paths_balanced(evolved_paths, original_dag)

        logger.info(
            f"✅ Path stitching complete: {len(stitching_result.stitched_dag.all_nodes)} nodes, {len(stitching_result.stitched_dag.edges)} edges"
        )
        logger.info(
            f"📊 Stitching metrics: {stitching_result.nodes_deduplicated} nodes deduplicated, {stitching_result.edges_consolidated} edges consolidated"
        )

        # Log stitching details
        stitching_logger = get_stitching_logger()
        stitching_logger.log_row(
            {
                "session_id": session_id,
                "paths_stitched": len(evolved_paths),
                "conflicts_resolved": 0,
                "new_edges_added": len(stitching_result.stitched_dag.edges) - len(original_dag.edges),
                "nodes_merged": stitching_result.nodes_deduplicated,
                "stitching_time_seconds": stitching_result.stitching_time_seconds,
            }
        )

        return stitching_result

    except Exception as e:
        error_msg = f"Path stitching failed: {e}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        raise


async def run_evolution_workflow(
    built_dag: DecisionDAG,
    problem_statement: str,
    organization_id: Optional[str] = None,
    config: Optional[OrchestrationConfig] = None,
) -> OrchestrationResult:
    """
    Run just the evolution workflow (extract, evolve, stitch) on an already built DAG.

    Args:
        built_dag: The already constructed DAG to evolve
        problem_statement: The original problem statement
        organization_id: Optional organization ID for database context
        config: Optional orchestration configuration

    Returns:
        OrchestrationResult with evolution results
    """
    config = config or OrchestrationConfig()
    orchestration_start = datetime.now()
    session_id = str(uuid.uuid4())

    logger.info(f"🧬 Starting evolution workflow for: {problem_statement}")
    logger.info(f"📊 Session ID: {session_id}")

    # Initialize result tracking
    result = OrchestrationResult(
        original_dag=built_dag,
        phases_completed=[],
        total_phases=3,  # Extract, Evolve, Stitch
    )

    try:
        # Phase 1: Extract paths from the built DAG
        logger.info("📊 Phase 1: Extracting paths from built DAG")
        extracted_paths, extraction_metrics = await _extract_paths_phase(
            dag=built_dag, organization_id=organization_id, config=config, result=result, session_id=session_id
        )
        result.extraction_metrics = extraction_metrics.to_dict() if hasattr(extraction_metrics, "to_dict") else {}
        result.phases_completed.append("extract_paths")

        # Phase 2: Evolve paths (if any were extracted)
        if extracted_paths and config.enable_evolution:
            logger.info("📊 Phase 2: Evolving extracted paths")
            evolved_paths, evolution_metrics = await _evolve_paths_phase(
                paths=extracted_paths,
                original_dag=built_dag,
                organization_id=organization_id,
                config=config,
                result=result,
                session_id=session_id,
            )
            result.evolution_metrics = evolution_metrics
            result.phases_completed.append("evolve_paths")
        else:
            logger.warning("⚠️ Skipping evolution phase - no paths to evolve or evolution disabled")
            evolved_paths = extracted_paths

        # Phase 3: Stitch evolved paths back together
        if evolved_paths and len(evolved_paths) > 0:
            logger.info("📊 Phase 3: Stitching evolved paths")
            stitching_result = await _stitch_paths_phase(
                evolved_paths=evolved_paths,
                original_dag=built_dag,
                config=config,
                result=result,
                session_id=session_id,
            )
            result.stitching_result = stitching_result
            result.evolved_dag = stitching_result.stitched_dag
            result.phases_completed.append("stitch_paths")
        else:
            logger.warning("⚠️ Skipping stitching phase - no evolved paths to stitch")
            result.evolved_dag = built_dag  # Use original DAG as fallback

        # Calculate total time
        result.orchestration_time_seconds = (datetime.now() - orchestration_start).total_seconds()

        logger.info(f"🎉 Evolution workflow complete in {result.orchestration_time_seconds:.1f}s")
        logger.info(f"📊 Phases completed: {', '.join(result.phases_completed)}")

        return result

    except Exception as e:
        logger.error(f"Evolution workflow failed: {e}")
        result.orchestration_time_seconds = (datetime.now() - orchestration_start).total_seconds()
        raise


def create_sample_strategic_paths() -> List[StrategicPath]:
    """Create sample strategic paths for testing."""
    return [
        StrategicPath(
            title="Aggressive Growth Strategy",
            description="Rapid expansion through acquisition and market penetration",
            key_milestones=["Market analysis", "Target identification", "Acquisition", "Integration"],
            expected_outcomes=["50% revenue growth", "Market leadership", "Operational synergies"],
        ),
        StrategicPath(
            title="Organic Growth Strategy",
            description="Steady growth through product development and customer acquisition",
            key_milestones=["Product roadmap", "Team expansion", "Marketing campaign", "Customer onboarding"],
            expected_outcomes=["25% revenue growth", "Product portfolio expansion", "Customer satisfaction"],
        ),
    ]
