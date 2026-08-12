"""Enhanced CLI for Decision DAGs with database persistence."""

import argparse
import asyncio
import logging
import sys
from uuid import UUID
from pathlib import Path
import json
import csv
from datetime import datetime

from .persistence import init_db, close_db, create_tables, dag_repository
from .models import EvaluationContext, AlphaEvolutionConfig
from .main import create_sample_strategic_paths, OrchestrationConfig
from .path_evolution.extractor import PathExtractionEngine
from .path_evolution.evolver import PathEvolutionEngine
from .path_evolution.stitcher import stitch_paths_balanced, stitch_paths_conservative, stitch_paths_aggressive
from .settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def print_table(headers, rows):
    """Print a simple table."""
    if not rows:
        return

    col_widths = []
    for i in range(len(headers)):
        width = len(headers[i])
        for row in rows:
            if i < len(row):
                width = max(width, len(str(row[i])))
        col_widths.append(width)

    # Print headers
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Decision DAGs - Strategic Planning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new DAG
  python -m decision_dags create --problem "How to expand internationally?"

  # List all DAGs
  python -m decision_dags list

  # Show DAG details
  python -m decision_dags show --dag-id <uuid>

  # Extract paths from a DAG
  python -m decision_dags extract-paths --dag-id <uuid>

  # Evolve paths
  python -m decision_dags evolve --dag-id <uuid>
        """,
    )

    # Global arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for troubleshooting")

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create command
    create_cmd = subparsers.add_parser("create", help="Create a new Decision DAG")
    create_cmd.add_argument("-p", "--problem", type=str, required=True, help="Problem statement to solve")
    create_cmd.add_argument(
        "--max-layers",
        type=int,
        default=settings.max_layers,
        help=f"Maximum DAG depth (default: {settings.max_layers})",
    )
    create_cmd.add_argument(
        "--timeout",
        type=float,
        default=settings.agent_timeout,
        help=f"Agent timeout in seconds (default: {settings.agent_timeout})",
    )
    create_cmd.add_argument("--enable-hitl", action="store_true", help="Enable Human-in-the-Loop workflows")
    create_cmd.add_argument(
        "--enable-evolution", action="store_true", default=True, help="Enable path evolution (default: True)"
    )
    create_cmd.add_argument(
        "--no-evolution", dest="enable_evolution", action="store_false", help="Disable path evolution"
    )
    create_cmd.add_argument(
        "--stitching-strategy",
        choices=["balanced", "conservative", "aggressive"],
        default="balanced",
        help="Path stitching strategy (default: balanced)",
    )

    # List command
    list_cmd = subparsers.add_parser("list", help="List saved DAGs")
    list_cmd.add_argument("--filter-by", choices=["build", "extracted", "evolved"], help="Filter by generation method")
    list_cmd.add_argument("--limit", type=int, default=10, help="Maximum results to show (default: 10)")
    list_cmd.add_argument("--offset", type=int, default=0, help="Number of results to skip")
    list_cmd.add_argument(
        "--sort-by",
        choices=["created_at", "updated_at", "node_count", "max_layers"],
        default="created_at",
        help="Sort field (default: created_at)",
    )
    list_cmd.add_argument("--ascending", action="store_true", help="Sort in ascending order")

    # Show command
    show_cmd = subparsers.add_parser("show", help="Show details of a specific DAG")
    show_cmd.add_argument("--dag-id", type=str, required=True, help="UUID of the DAG to show")
    show_cmd.add_argument("--include-nodes", action="store_true", help="Include node details")
    show_cmd.add_argument("--include-edges", action="store_true", help="Include edge details")
    show_cmd.add_argument("--include-metrics", action="store_true", help="Include detailed metrics")

    # Extract paths command
    extract_cmd = subparsers.add_parser("extract-paths", help="Extract strategic paths from a DAG")
    extract_cmd.add_argument("--dag-id", type=str, required=True, help="UUID of the DAG to extract paths from")
    extract_cmd.add_argument("--min-length", type=int, default=3, help="Minimum path length (default: 3)")
    extract_cmd.add_argument("--max-length", type=int, default=10, help="Maximum path length (default: 10)")
    extract_cmd.add_argument("--include-incomplete", action="store_true", help="Include incomplete paths")

    # Evolve command
    evolve_cmd = subparsers.add_parser("evolve", help="Evolve paths from a DAG")
    evolve_cmd.add_argument("--dag-id", type=str, required=True, help="UUID of the DAG with paths to evolve")
    evolve_cmd.add_argument(
        "--generations", type=int, default=10, help="Number of evolution generations (default: 10)"
    )
    evolve_cmd.add_argument("--population-size", type=int, default=8, help="Population size (default: 8)")
    evolve_cmd.add_argument("--mutation-rate", type=float, default=0.7, help="Mutation rate (default: 0.7)")
    evolve_cmd.add_argument(
        "--top-k-paths",
        type=int,
        default=5,
        help="Number of top paths to select for evolution based on initial fitness (default: 5, use -1 for all)",
    )

    # Stitch command
    stitch_cmd = subparsers.add_parser("stitch", help="Stitch evolved paths back into a DAG")
    stitch_cmd.add_argument("--dag-id", type=str, required=True, help="UUID of the evolved DAG to stitch")
    stitch_cmd.add_argument(
        "--strategy",
        choices=["balanced", "conservative", "aggressive"],
        default="balanced",
        help="Stitching strategy (default: balanced)",
    )

    # Export command
    export_cmd = subparsers.add_parser("export", help="Export a DAG to various formats")
    export_cmd.add_argument("--dag-id", type=str, required=True, help="UUID of the DAG to export")
    export_cmd.add_argument(
        "--format", choices=["csv", "json", "graphviz"], default="json", help="Export format (default: json)"
    )
    export_cmd.add_argument(
        "--output-dir", type=Path, default=Path("./exports"), help="Output directory (default: ./exports)"
    )

    # Delete command
    delete_cmd = subparsers.add_parser("delete", help="Delete a DAG")
    delete_cmd.add_argument("--dag-id", type=str, required=True, help="UUID of the DAG to delete")
    delete_cmd.add_argument("--cascade", action="store_true", help="Delete child DAGs as well")

    # Database commands
    db_cmd = subparsers.add_parser("db", help="Database management commands")
    db_subparsers = db_cmd.add_subparsers(dest="db_command", help="Database commands")

    db_subparsers.add_parser("init", help="Initialize database tables")
    db_subparsers.add_parser("drop", help="Drop all database tables (WARNING: deletes all data)")

    # Visualize command
    visualize_cmd = subparsers.add_parser("visualize", help="Launch interactive DAG visualization server")
    visualize_cmd.add_argument("--dag-id", type=str, help="UUID of the DAG to visualize initially")
    visualize_cmd.add_argument(
        "--port", type=int, default=5006, help="Port to run the visualization server on (default: 5006)"
    )
    visualize_cmd.add_argument(
        "--host", type=str, default="localhost", help="Host to run the visualization server on (default: localhost)"
    )
    visualize_cmd.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    return parser


async def handle_create(args):
    """Handle the create command."""
    print("\n=== Creating Decision DAG ===")
    print(f"Problem: {args.problem}")
    print(f"Max Layers: {args.max_layers}")
    print(f"Timeout: {args.timeout}s")
    print(f"HITL: {'Enabled' if args.enable_hitl else 'Disabled'}")
    print(f"Evolution: {'Enabled' if args.enable_evolution else 'Disabled'}")
    print(f"Stitching Strategy: {args.stitching_strategy}")

    # Update settings
    settings.max_layers = args.max_layers
    settings.agent_timeout = args.timeout

    # Create config
    config = OrchestrationConfig(
        enable_hitl=args.enable_hitl,
        enable_evolution=args.enable_evolution,
        stitching_strategy=args.stitching_strategy,
        enable_detailed_logging=True,
    )

    # Create strategic paths
    strategic_paths = create_sample_strategic_paths()

    print("\nBuilding DAG...")
    import time

    build_start_time = time.time()

    try:
        # Phase 1: Build the initial DAG
        from .main import build_decision_dag

        original_dag = await build_decision_dag(
            problem_statement=args.problem,
            strategic_paths=strategic_paths,
            organization_id=settings.organization_id,
            enable_hitl=config.enable_hitl,
            dag_config=config.dag_config,
            hitl_config=config.hitl_config,
        )

        build_time = time.time() - build_start_time
        print(f"DAG building completed in {build_time:.1f}s")
        print("Saving built DAG to database...")

        # Save the built DAG first
        try:
            built_dag_id = await dag_repository.save_dag(
                dag=original_dag,
                problem_statement=args.problem,
                generation_method="build",
                metadata={
                    "build_time_seconds": build_time,
                    "max_layers": args.max_layers,
                    "timeout": args.timeout,
                    "hitl_enabled": args.enable_hitl,
                },
            )
            print(f"✓ Built DAG saved with ID: {built_dag_id}")
            print(
                f"Built DAG - Nodes: {len(original_dag.all_nodes)}, Edges: {len(original_dag.edges)}, Layers: {original_dag.get_max_layer()}"
            )
        except Exception as db_error:
            print(f"Database save error for built DAG: {db_error}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            raise

        final_dag_id = built_dag_id
        total_time = build_time

        # Phase 2: Run evolution if enabled
        if args.enable_evolution:
            print("\nStarting evolution phases...")
            evolution_start_time = time.time()

            try:
                # Run just the evolution workflow on the already built DAG
                from .main import run_evolution_workflow

                result = await run_evolution_workflow(
                    built_dag=original_dag,
                    problem_statement=args.problem,
                    organization_id=settings.organization_id,
                    config=config,
                )

                evolution_time = time.time() - evolution_start_time

                if result.evolved_dag:
                    print(f"Evolution completed in {evolution_time:.1f}s")
                    print("Saving evolved DAG to database...")

                    try:
                        evolved_dag_id = await dag_repository.save_dag(
                            dag=result.evolved_dag,
                            problem_statement=args.problem,
                            generation_method="evolved",
                            parent_dag_id=built_dag_id,
                            metadata={
                                "phases_completed": result.phases_completed,
                                "build_time_seconds": build_time,
                                "evolution_time_seconds": evolution_time,
                                "total_orchestration_time": result.orchestration_time_seconds,
                                "extraction_metrics": result.extraction_metrics,
                                "evolution_metrics": result.evolution_metrics,
                                "stitching_result": result.stitching_result.__dict__
                                if result.stitching_result
                                else None,
                                "parent_dag_id": str(built_dag_id),
                            },
                        )
                        print(f"✓ Evolved DAG saved with ID: {evolved_dag_id}")
                        print(
                            f"Evolved DAG - Nodes: {len(result.evolved_dag.all_nodes)}, Edges: {len(result.evolved_dag.edges)}, Layers: {result.evolved_dag.get_max_layer()}"
                        )
                        final_dag_id = evolved_dag_id
                        total_time = build_time + evolution_time
                    except Exception as db_error:
                        print(f"Database save error for evolved DAG: {db_error}")
                        if args.verbose:
                            import traceback

                            traceback.print_exc()
                        print("Built DAG is still available in database")
                        final_dag_id = built_dag_id
                        total_time = build_time
                else:
                    print("Evolution produced no results, using built DAG")
                    final_dag_id = built_dag_id
                    total_time = build_time + evolution_time

            except Exception as evolution_error:
                print(f"Evolution failed: {evolution_error}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()
                print("Built DAG is still available in database")
                final_dag_id = built_dag_id
                total_time = build_time
        else:
            print("Evolution disabled, using built DAG as final result")

        print("\n✓ DAG creation process complete!")
        print(f"Final DAG ID: {final_dag_id}")
        print(f"Built DAG ID: {built_dag_id}")
        print(f"Total Time: {total_time:.1f}s")

    except Exception as e:
        print(f"\nError creating DAG: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        raise


async def handle_list(args):
    """Handle the list command."""
    dags = await dag_repository.list_dags(
        filter_by=args.filter_by, limit=args.limit, offset=args.offset, sort_by=args.sort_by, ascending=args.ascending
    )

    if not dags:
        print("No DAGs found")
        return

    print("\n=== Decision DAGs ===")
    if args.filter_by:
        print(f"Filter: {args.filter_by}")
    print()

    # Prepare table data
    headers = ["ID", "Problem Statement", "Method", "Nodes", "Edges", "Layers", "Created"]
    rows = []

    for dag in dags:
        rows.append(
            [
                str(dag.id)[:8] + "...",
                dag.problem_statement[:50] + "..." if len(dag.problem_statement) > 50 else dag.problem_statement,
                dag.generation_method,
                str(dag.node_count),
                str(dag.edge_count),
                str(dag.max_layers),
                dag.created_at.strftime("%Y-%m-%d %H:%M"),
            ]
        )

    print_table(headers, rows)

    if args.offset > 0 or len(dags) == args.limit:
        print(f"\nShowing results {args.offset + 1}-{args.offset + len(dags)}")


async def handle_show(args):
    """Handle the show command."""
    try:
        dag_id = UUID(args.dag_id)
    except ValueError:
        print(f"Error: Invalid UUID: {args.dag_id}")
        return

    try:
        info = await dag_repository.get_dag_info(dag_id)

        # Basic info
        print("\n=== DAG Information ===")
        print(f"ID: {info['id']}")
        print(f"Problem: {info['problem_statement']}")
        print(f"Generation Method: {info['generation_method']}")
        print(f"Created: {info['created_at']}")
        print(f"Updated: {info['updated_at']}")

        # Structure info
        print("\n=== Structure ===")
        print(f"Nodes: {info['node_count']}")
        print(f"Edges: {info['edge_count']}")
        print(f"Max Layers: {info['max_layers']}")

        # Lineage
        if info["parent"]:
            print("\n=== Parent DAG ===")
            print(f"ID: {info['parent']['id'][:8]}...")
            print(f"Problem: {info['parent']['problem_statement']}")
            print(f"Method: {info['parent']['generation_method']}")

        if info["children"]:
            print(f"\n=== Child DAGs ({len(info['children'])}) ===")
            for child in info["children"][:3]:
                print(f"- {child['id'][:8]}... ({child['generation_method']}) - {child['created_at']}")
            if len(info["children"]) > 3:
                print(f"... and {len(info['children']) - 3} more")

        # Metadata
        if args.include_metrics and info["metadata"]:
            print("\n=== Metadata ===")
            print(json.dumps(info["metadata"], indent=2))

        # Load full DAG if needed
        if args.include_nodes or args.include_edges:
            dag = await dag_repository.load_dag(dag_id)

            if args.include_nodes:
                print("\n=== Nodes (first 10) ===")
                for i, (node_id, node) in enumerate(list(dag.all_nodes.items())[:10]):
                    print(f"Layer {node.layer} - {node.type.value}: {node.title}")
                if len(dag.all_nodes) > 10:
                    print(f"... and {len(dag.all_nodes) - 10} more nodes")

            if args.include_edges:
                print("\n=== Edges (first 10) ===")
                for i, edge in enumerate(dag.edges[:10]):
                    source = dag.get_node(edge.source_id)
                    target = dag.get_node(edge.target_id)
                    print(
                        f"{source.title if source else edge.source_id} -> {target.title if target else edge.target_id}"
                    )
                if len(dag.edges) > 10:
                    print(f"... and {len(dag.edges) - 10} more edges")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


async def handle_extract_paths(args):
    """Handle the extract-paths command."""
    try:
        dag_id = UUID(args.dag_id)
    except ValueError:
        print(f"Error: Invalid UUID: {args.dag_id}")
        return

    print("\n=== Extracting Paths ===")
    print(f"DAG ID: {args.dag_id}")
    print(f"Min Length: {args.min_length}")
    print(f"Max Length: {args.max_length}")
    print(f"Include Incomplete: {args.include_incomplete}")

    try:
        print("\nLoading DAG...")
        # Load the DAG
        dag = await dag_repository.load_dag(dag_id)
        dag_info = await dag_repository.get_dag_info(dag_id)

        print("Extracting paths...")

        # Extract paths
        extractor = PathExtractionEngine()
        extracted_paths, metrics = extractor.extract_paths_with_criteria(
            dag=dag, min_length=args.min_length, max_length=args.max_length, include_incomplete=args.include_incomplete
        )

        print(f"Found {len(extracted_paths)} paths")
        print("Saving extracted paths...")

        # Save each extracted path as a new DAG
        saved_ids = []
        for i, path in enumerate(extracted_paths):
            path_id = await dag_repository.save_dag(
                dag=path,
                problem_statement=f"Path {i + 1} from: {dag_info['problem_statement']}",
                generation_method="extracted",
                parent_dag_id=dag_id,
                metadata={
                    "path_index": i,
                    "extraction_metrics": metrics._asdict() if hasattr(metrics, "_asdict") else {},
                },
            )
            saved_ids.append(path_id)

        print("\n✓ Path extraction complete!")
        print(f"Extracted Paths: {len(extracted_paths)}")
        print("\nSaved Path IDs:")
        for i, path_id in enumerate(saved_ids[:5]):
            print(f"  {i + 1}. {path_id}")
        if len(saved_ids) > 5:
            print(f"  ... and {len(saved_ids) - 5} more")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


async def handle_evolve(args):
    """Handle the evolve command."""
    try:
        dag_id = UUID(args.dag_id)
    except ValueError:
        print(f"Error: Invalid UUID: {args.dag_id}")
        return

    # Initialize prompt templates and LLM client (required for evolution)
    from . import prompts  # This triggers prompt template registration
    from common.instruct_llm import set_async_instructor_client

    # Set up async instructor client for LLM calls
    # Determine which API key to use based on the model (following ensemble.py pattern)
    api_key = settings.anthropic_api_key if settings.llm_model.startswith("claude") else settings.openai_api_key

    set_async_instructor_client(llm_model=settings.llm_model, api_key=api_key)

    print("\n=== Evolving Paths ===")
    print(f"DAG ID: {args.dag_id}")
    print(f"Generations: {args.generations}")
    print(f"Population Size: {args.population_size}")
    print(f"Mutation Rate: {args.mutation_rate}")

    try:
        print("\nLoading DAG...")
        # Load the DAG
        dag = await dag_repository.load_dag(dag_id)
        dag_info = await dag_repository.get_dag_info(dag_id)

        # Initialize extraction_metrics
        extraction_metrics = None

        # Check if this is a path DAG
        if dag_info["generation_method"] == "extracted":
            # This is a single path, load all sibling paths
            parent_id = UUID(dag_info["parent"]["id"]) if dag_info["parent"] else None
            if parent_id:
                # Get all extracted paths from the same parent
                sibling_dags = await dag_repository.list_dags(filter_by="extracted", limit=100)
                paths = []
                for sibling in sibling_dags:
                    if sibling.parent_dag_id == parent_id:
                        path_dag = await dag_repository.load_dag(sibling.id)
                        paths.append(path_dag)
                print(f"Loaded {len(paths)} paths from parent DAG")
            else:
                paths = [dag]
        else:
            # Extract paths from this DAG
            print("Extracting paths from DAG...")
            extractor = PathExtractionEngine()
            paths, extraction_metrics = extractor.extract_paths_with_criteria(
                dag=dag, min_length=3, max_length=20, include_incomplete=False
            )
            print(f"Extracted {len(paths)} paths")

            # We'll save extracted paths after selecting which ones to evolve

        if not paths:
            print("No paths to evolve")
            return

        # Configure evolution
        evolution_config = AlphaEvolutionConfig(
            max_generations=args.generations,
            population_size=args.population_size,
            mutation_rate=args.mutation_rate,
            top_k_paths=args.top_k_paths if args.top_k_paths > 0 else len(paths),  # -1 means all paths
        )

        # Show how many paths will actually be evolved
        paths_to_evolve = min(evolution_config.top_k_paths, len(paths))
        print(f"\nEvolving top {paths_to_evolve} of {len(paths)} paths based on initial fitness...")

        # Create evolver
        evolver = PathEvolutionEngine(evolution_config)

        # Create evaluation context
        context = EvaluationContext(
            organization_id="default",
            domain_context="strategic_planning",
            user_preferences={"risk_tolerance": "medium"},
            temporal_context={"current_date": datetime.now().isoformat()},
            resource_constraints={"max_budget": 1000000, "max_timeline_weeks": 52},
        )

        # Evaluate all paths to determine which ones to evolve
        from .path_evolution.evaluator import PathFitnessEvaluator

        evaluator = PathFitnessEvaluator()
        organizational_goals = [
            {"type": "growth", "weight": 0.3, "target": 0.8},
            {"type": "efficiency", "weight": 0.2, "target": 0.7},
            {"type": "risk_mitigation", "weight": 0.2, "target": 0.6},
            {"type": "innovation", "weight": 0.1, "target": 0.5},
            {"type": "pmf_finding", "weight": 0.2, "target": 0.8},
        ]

        print("Evaluating paths to select top-k for evolution...")
        baseline_fitness = await evaluator.evaluate_path_batch(
            paths, context, organizational_goals, evolution_config.max_concurrent_evolutions
        )

        # Select top-k paths based on fitness
        sorted_paths = sorted(paths, key=lambda p: baseline_fitness.get(p.id, 0.0), reverse=True)
        selected_paths = sorted_paths[:paths_to_evolve]

        # Now save only the selected extracted paths to database
        print(f"\nSaving {len(selected_paths)} selected extracted paths to database...")
        saved_path_ids = []
        path_id_mapping = {}

        # Save selected paths and build mapping
        path_index = 0
        for path in selected_paths:
            try:
                path_id = await dag_repository.save_dag(
                    dag=path,
                    problem_statement=f"Extracted Path {path_index + 1} from: {dag_info['problem_statement'][:50]}...",
                    generation_method="extracted",
                    parent_dag_id=dag_id,
                    metadata={
                        "path_index": path_index,
                        "fitness_score": baseline_fitness.get(path.id, 0.0),
                        "extraction_metrics": extraction_metrics._asdict()
                        if extraction_metrics and hasattr(extraction_metrics, "_asdict")
                        else {},
                    },
                )
                saved_path_ids.append(path_id)
                # Map the path's internal ID to its database ID
                path_id_mapping[path.id] = path_id
                # Set the original_extracted_path_id on the path object
                path.original_extracted_path_id = str(path_id)
                if args.verbose:
                    print(f"  Saved path with internal ID {path.id} as database ID {path_id}")
                path_index += 1
            except Exception as e:
                print(f"Failed to save path {path_index}: {e}")

        print(f"Saved {len(saved_path_ids)} extracted paths to database")

        if dag_info["generation_method"] == "extracted":
            # We're evolving an already-extracted path or set of paths
            # Map each path's internal ID to its database ID
            for path in paths:
                if path.id not in path_id_mapping:
                    # For extracted paths loaded from DB, their internal ID matches their DB ID
                    path_id_mapping[path.id] = path.id
                # Ensure the path has its original_extracted_path_id set
                if not path.original_extracted_path_id:
                    path.original_extracted_path_id = str(dag_id)

        # Evolve paths - we'll do this manually to capture ALL evolved paths
        # First evaluate baseline fitness (already done above)

        # Select paths for evolution based on fitness (already done above)

        # Create a mapping of original path internal IDs to selected paths
        # This is needed because evolution uses internal IDs
        original_to_selected = {}
        for selected_path in selected_paths:
            # Find the original path this selected path corresponds to
            for original_path in paths:
                if original_path.id == selected_path.id:
                    original_to_selected[original_path.id] = selected_path
                    # Ensure the selected path has the original_extracted_path_id
                    selected_path.original_extracted_path_id = original_path.original_extracted_path_id
                    break

        # Evolve selected paths without filtering by improvement
        print("\nEvolving selected paths...")
        all_evolved_paths = await evolver._evolve_paths_parallel(selected_paths, context, organizational_goals)

        # Extract just the evolved DAGs (not the fitness scores)
        evolved_paths = [dag for dag, _ in all_evolved_paths]

        # Get evolution summary
        evolution_summary = evolver.get_evolution_summary()

        print("\n✓ Evolution complete!")
        print(f"Selected Paths: {len(selected_paths)}")
        print(f"Evolved Paths Generated: {len(evolved_paths)}")
        if evolution_summary:
            print(
                f"Generations Run: {evolution_summary.get('generations_run', 0)} (warm-up: {evolution_summary.get('warmup_generations', 0)})"
            )
            print(f"Best Fitness: {evolution_summary.get('best_fitness', 0):.3f}")
            print(f"Average Fitness: {evolution_summary.get('avg_fitness', 0):.3f}")
            print(f"Mutation Success Rate: {evolution_summary.get('mutation_success_rate', 0):.1%}")
            print(f"Successful Mutations: {evolution_summary.get('successful_mutations', 0)}")
            print(f"Failed Mutations: {evolution_summary.get('unsuccessful_mutations', 0)}")

        # Save evolved paths
        print("\nSaving evolved paths...")
        saved_ids = []
        evolved_count = 0
        for i, evolved_path in enumerate(evolved_paths):
            # Only save paths that were actually evolved (have evolved_from metadata)
            if not hasattr(evolved_path, "metadata") or "evolved_from" not in evolved_path.metadata:
                # This is an original path that wasn't selected for evolution
                continue

            evolved_count += 1

            # Find the parent path ID from the original_extracted_path_id
            parent_path_id = evolved_path.original_extracted_path_id

            # Debug logging
            if args.verbose:
                print(f"  Debug: evolved_path.original_extracted_path_id={parent_path_id}")
                print(f"  Debug: evolved_from={evolved_path.metadata.get('evolved_from')}")

            # If we couldn't find it in the path object, it might be because
            # the user ran evolve on an already-extracted path
            if parent_path_id is None and dag_info["generation_method"] == "extracted":
                # The current DAG is an extracted path, use its ID
                parent_path_id = str(dag_id)
                if args.verbose:
                    print(f"  Debug: Using current DAG ID as parent: {dag_id}")

            # If we still don't have a parent_path_id, log a warning with more details
            if parent_path_id is None:
                print(f"Warning: Could not determine parent path ID for evolved path {evolved_count}")
                print(f"  - evolved_from: {evolved_path.metadata.get('evolved_from')}")
                print(f"  - original_extracted_path_id: {evolved_path.original_extracted_path_id}")
                print(f"  - Current DAG generation method: {dag_info['generation_method']}")
                # Skip saving this evolved path to avoid incorrect relationships
                continue

            evolved_id = await dag_repository.save_dag(
                dag=evolved_path,
                problem_statement=f"Evolved path {evolved_count} from: {dag_info['problem_statement']}",
                generation_method="evolved",
                parent_dag_id=parent_path_id,
                metadata={
                    "evolution_config": evolution_config.model_dump(),
                    "evolution_summary": evolution_summary,
                    "path_index": evolved_count - 1,
                    "original_dag_id": str(dag_id),  # Keep reference to original DAG
                    "parent_path_internal_id": evolved_path.metadata.get("evolved_from", "unknown"),
                },
            )
            saved_ids.append(evolved_id)

        print(f"\nSaved {len(saved_ids)} evolved paths")
        if len(saved_ids) == 0 and len(evolved_paths) > 0:
            print(f"Note: {len(evolved_paths)} evolved paths were generated but none had proper parent mappings")
            print("This might indicate an issue with the evolution process")

        for i, path_id in enumerate(saved_ids[:5]):
            print(f"  {i + 1}. {path_id}")
        if len(saved_ids) > 5:
            print(f"  ... and {len(saved_ids) - 5} more")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


async def handle_stitch(args):
    """Handle the stitch command."""
    try:
        dag_id = UUID(args.dag_id)
    except ValueError:
        print(f"Error: Invalid UUID: {args.dag_id}")
        return

    print("\n=== Stitching Paths ===")
    print(f"DAG ID: {args.dag_id}")
    print(f"Strategy: {args.strategy}")

    try:
        print("\nLoading evolved paths...")
        # Get DAG info
        dag_info = await dag_repository.get_dag_info(dag_id)

        # Find all evolved paths with this parent
        evolved_dags = await dag_repository.list_dags(filter_by="evolved", limit=100)

        evolved_paths = []
        for evolved in evolved_dags:
            if evolved.parent_dag_id == dag_id or (
                dag_info["parent"] and str(evolved.parent_dag_id) == dag_info["parent"]["id"]
            ):
                path_dag = await dag_repository.load_dag(evolved.id)
                evolved_paths.append(path_dag)

        if not evolved_paths:
            print("No evolved paths found for this DAG")
            return

        print(f"Found {len(evolved_paths)} evolved paths")

        # Load original DAG for context
        original_dag = None
        if dag_info["parent"]:
            original_dag = await dag_repository.load_dag(UUID(dag_info["parent"]["id"]))
        else:
            original_dag = await dag_repository.load_dag(dag_id)

        print(f"Stitching with {args.strategy} strategy...")

        # Choose stitching function
        if args.strategy == "conservative":
            stitching_result = await stitch_paths_conservative(evolved_paths, original_dag)
        elif args.strategy == "aggressive":
            stitching_result = await stitch_paths_aggressive(evolved_paths, original_dag)
        else:  # balanced
            stitching_result = await stitch_paths_balanced(evolved_paths, original_dag)

        print("Saving stitched DAG...")

        # Save stitched DAG
        stitched_id = await dag_repository.save_dag(
            dag=stitching_result.stitched_dag,
            problem_statement=f"Stitched: {dag_info['problem_statement']}",
            generation_method="evolved",
            parent_dag_id=dag_id,
            metadata={
                "stitching_strategy": args.strategy,
                "original_path_count": stitching_result.original_path_count,
                "nodes_deduplicated": stitching_result.nodes_deduplicated,
                "edges_consolidated": stitching_result.edges_consolidated,
                "stitching_time_seconds": stitching_result.stitching_time_seconds,
            },
        )

        print("\n✓ Stitching complete!")
        print(f"Stitched DAG ID: {stitched_id}")
        print(f"Nodes: {len(stitching_result.stitched_dag.all_nodes)}")
        print(f"Edges: {len(stitching_result.stitched_dag.edges)}")
        print(f"Nodes Deduplicated: {stitching_result.nodes_deduplicated}")
        print(f"Edges Consolidated: {stitching_result.edges_consolidated}")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


async def handle_export(args):
    """Handle the export command."""
    try:
        dag_id = UUID(args.dag_id)
    except ValueError:
        print(f"Error: Invalid UUID: {args.dag_id}")
        return

    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Load DAG
        print(f"Loading DAG {args.dag_id}...")
        dag = await dag_repository.load_dag(dag_id)
        dag_info = await dag_repository.get_dag_info(dag_id)

        if args.format == "json":
            # Export as JSON
            output_file = args.output_dir / f"dag_{dag_id}.json"
            export_data = {
                "info": dag_info,
                "nodes": [node.model_dump() for node in dag.all_nodes.values()],
                "edges": [edge.model_dump() for edge in dag.edges],
            }
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"✓ Exported to {output_file}")

        elif args.format == "csv":
            # Export nodes and edges as separate CSV files
            nodes_file = args.output_dir / f"dag_{dag_id}_nodes.csv"
            edges_file = args.output_dir / f"dag_{dag_id}_edges.csv"

            # Export nodes
            with open(nodes_file, "w", newline="") as f:
                if dag.all_nodes:
                    fieldnames = ["id", "layer", "type", "title", "description", "decision_type", "confidence_score"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for node in dag.all_nodes.values():
                        writer.writerow(
                            {
                                "id": node.id,
                                "layer": node.layer,
                                "type": node.type.value,
                                "title": node.title,
                                "description": node.description,
                                "decision_type": node.decision_type.value if node.decision_type else "",
                                "confidence_score": node.confidence_score or "",
                            }
                        )

            # Export edges
            with open(edges_file, "w", newline="") as f:
                if dag.edges:
                    fieldnames = [
                        "source_id",
                        "target_id",
                        "edge_type",
                        "condition",
                        "likelihood",
                        "cost_estimate",
                        "timeline_estimate",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for edge in dag.edges:
                        writer.writerow(
                            {
                                "source_id": edge.source_id,
                                "target_id": edge.target_id,
                                "edge_type": edge.edge_type.value,
                                "condition": edge.condition,
                                "likelihood": edge.likelihood,
                                "cost_estimate": edge.cost_estimate or "",
                                "timeline_estimate": edge.timeline_estimate or "",
                            }
                        )

            print(f"✓ Exported to {nodes_file} and {edges_file}")

        elif args.format == "graphviz":
            print("Graphviz export not yet implemented")
            return

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


async def handle_delete(args):
    """Handle the delete command."""
    try:
        dag_id = UUID(args.dag_id)
    except ValueError:
        print(f"Error: Invalid UUID: {args.dag_id}")
        return

    try:
        # Get DAG info first
        info = await dag_repository.get_dag_info(dag_id)

        # Confirm deletion
        print("\nWarning! You are about to delete:")
        print(f"Problem: {info['problem_statement']}")
        print(f"Nodes: {info['node_count']}")
        print(f"Edges: {info['edge_count']}")
        if info["children"]:
            print(f"\nThis DAG has {len(info['children'])} child DAGs")

        confirm = input("\nType 'yes' to confirm deletion: ")
        if confirm.lower() != "yes":
            print("Deletion cancelled")
            return

        # Delete the DAG
        await dag_repository.delete_dag(dag_id, cascade=args.cascade)
        print(f"✓ DAG {dag_id} deleted successfully")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


async def handle_db_init(args):
    """Handle database initialization."""
    print("Initializing database...")
    try:
        await create_tables()
        print("✓ Database tables created successfully")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


async def handle_db_drop(args):
    """Handle database drop."""
    print("Warning! This will delete all data!")
    confirm = input("Type 'yes' to confirm: ")
    if confirm.lower() != "yes":
        print("Operation cancelled")
        return

    try:
        from .persistence.database import drop_tables

        await drop_tables()
        print("✓ Database tables dropped")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


async def handle_visualize(args):
    """Handle the visualize command."""
    print("\n=== Starting DAG Visualization Server ===")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")

    if args.dag_id:
        print(f"Initial DAG: {args.dag_id}")

    print("\nPress Ctrl+C to stop the server")

    try:
        from .visualization import run_server

        # Run the visualization server
        await run_server(port=args.port, address=args.host, show=not args.no_browser, title="Decision DAG Visualizer")
    except ImportError as e:
        print(f"Error: Missing visualization dependencies - {e}")
        print("Please ensure panel and plotly are installed")
    except Exception as e:
        print(f"Error starting visualization server: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Also set debug level for path evolution modules
        logging.getLogger("decision_dags.src.path_evolution").setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Initialize database connection
    await init_db()

    try:
        # Handle commands
        if args.command == "create":
            await handle_create(args)
        elif args.command == "list":
            await handle_list(args)
        elif args.command == "show":
            await handle_show(args)
        elif args.command == "extract-paths":
            await handle_extract_paths(args)
        elif args.command == "evolve":
            await handle_evolve(args)
        elif args.command == "stitch":
            await handle_stitch(args)
        elif args.command == "export":
            await handle_export(args)
        elif args.command == "delete":
            await handle_delete(args)
        elif args.command == "db":
            if args.db_command == "init":
                await handle_db_init(args)
            elif args.db_command == "drop":
                await handle_db_drop(args)
            else:
                parser.print_help()
        elif args.command == "visualize":
            await handle_visualize(args)
        else:
            parser.print_help()

    finally:
        # Close database connection
        await close_db()


def run():
    """Run the CLI."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()
