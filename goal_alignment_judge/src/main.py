import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from .pipelines.dspy_pointwise.dspy_optimizer import run_gepa, run_miprov2
from .pipelines.dspy_pointwise.dspy_scorer import score_from_saved, score_with_dspy
from .pipelines.pointwise.extract_ratings import extract_ratings
from .pipelines.pointwise.filter_ratings import filter_to_common_pairs
from .pipelines.pointwise.pointwise_data_loader import load_pointwise_data, load_pointwise_split
from .pipelines.pointwise.pointwise_evaluator import (
    evaluate_pointwise,
    find_optimal_thresholds,
    save_pointwise_results,
)
from .scripts.ablation_scorer_downgrade import run_study as run_scorer_downgrade_study
from .scripts.ablation_train_size import run_study as run_train_size_study
from .scripts.compare_raters import compare_raters
from .scripts.create_seed_program import create_seed_program
from .settings import DATABASE, logger, settings


async def cmd_extract_ratings(args: argparse.Namespace) -> None:
    output = settings.root / args.output
    database = args.database or DATABASE
    await extract_ratings(output_path=output, database=database)


async def cmd_filter_to_pairs(args: argparse.Namespace) -> None:
    def _resolve(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else settings.root / p

    csv_a = _resolve(args.csv_a)
    csv_b = _resolve(args.csv_b) if args.csv_b else settings.pointwise_input_csv
    output_a = _resolve(args.output_a) if args.output_a else settings.root / "input" / f"{csv_a.stem}_filtered.csv"
    output_b = _resolve(args.output_b) if args.output_b else settings.root / "input" / f"{csv_b.stem}_filtered.csv"
    filter_to_common_pairs(csv_a, csv_b, output_a, output_b)


async def cmd_create_seed_program(args: argparse.Namespace) -> None:
    instructions_path = Path(args.instructions)
    if not instructions_path.is_absolute():
        instructions_path = settings.root / instructions_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = settings.root / args.output
    create_seed_program(instructions_path, output_dir)


async def cmd_ablation_train_size(args: argparse.Namespace) -> None:
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None
    await run_train_size_study(num_repetitions=args.repetitions, sizes=sizes)


async def cmd_ablation_scorer_downgrade(args: argparse.Namespace) -> None:
    run_scorer_downgrade_study()


async def cmd_clear_dspy_cache(args: argparse.Namespace) -> None:
    """Clear DSPy LLM response cache to force fresh optimization runs.

    DSPy 3.x caches all LLM responses in ~/.dspy_cache/ (diskcache FanoutCache,
    16 shards). Clearing this cache forces fresh LLM calls on the next run.
    The cache location can be overridden via DSPY_CACHEDIR env var.
    """
    cache_dir = Path(os.environ.get("DSPY_CACHEDIR", Path.home() / ".dspy_cache"))

    if cache_dir.exists():
        size_mb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        shutil.rmtree(cache_dir)
        print(f"  Cleared DSPy cache: {cache_dir} ({size_mb:.1f} MB)")
    else:
        print(f"  No cache found at {cache_dir}")


async def cmd_compare_raters(args: argparse.Namespace) -> None:
    def _resolve(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else settings.root / p

    csv_a = _resolve(args.csv_a)
    csv_b = _resolve(args.csv_b) if args.csv_b else settings.pointwise_input_csv
    output = _resolve(args.output) if args.output else settings.eda_path / "rater_comparison.html"
    compare_raters(csv_a, csv_b, output)


def _resolve_input_csv(raw: str | None) -> Path | None:
    """Resolve --input-csv relative to experiment root if not absolute."""
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else settings.root / p


async def cmd_load_pointwise_data(args: argparse.Namespace) -> None:
    input_csv = _resolve_input_csv(args.input_csv)
    await load_pointwise_data(input_csv=input_csv)


async def cmd_dspy_optimize(args: argparse.Namespace) -> None:
    subdir = Path(args.input_csv).stem if args.input_csv else None
    train = load_pointwise_split("train", subdir=subdir)
    dev = load_pointwise_split("dev", subdir=subdir)

    scorer_model = args.scorer_model or settings.scorer_model
    optimizer_model = args.optimizer_model or settings.rubric_model

    if args.method == "miprov2":
        await run_miprov2(
            train, dev,
            scorer_model=scorer_model,
            optimizer_model=optimizer_model,
            auto=args.auto,
            num_threads=args.num_threads,
        )
    elif args.method == "gepa":
        await run_gepa(
            train, dev,
            scorer_model=scorer_model,
            optimizer_model=optimizer_model,
            auto=args.auto,
            num_threads=args.num_threads,
        )


async def cmd_evaluate_dspy(args: argparse.Namespace) -> None:
    subdir = Path(args.input_csv).stem if args.input_csv else None
    examples = load_pointwise_split(args.split, subdir=subdir)
    module_path = Path(args.module_path)
    scorer_model = args.scorer_model or settings.scorer_model

    scored = score_from_saved(examples, module_path, scorer_model=scorer_model)
    result = evaluate_pointwise(scored, split=args.split)
    save_pointwise_results(result, f"dspy_{module_path.name}", prefix=subdir)


async def cmd_dspy_pipeline(args: argparse.Namespace) -> None:
    method = args.method
    scorer_model = args.scorer_model or settings.scorer_model
    optimizer_model = args.optimizer_model or settings.rubric_model
    auto = args.auto if hasattr(args, "auto") else "medium"
    num_threads = args.num_threads if hasattr(args, "num_threads") else 4
    comments = args.comments if hasattr(args, "comments") else ""

    print("\n" + "=" * 60)
    print("  PHASE 5: DSPy Pointwise Optimization")
    print("=" * 60)

    subdir = Path(args.input_csv).stem if args.input_csv else None
    train = load_pointwise_split("train", subdir=subdir)
    dev = load_pointwise_split("dev", subdir=subdir)
    test = load_pointwise_split("test", subdir=subdir)

    print(f"\n  Method: {method}")
    print(f"  Scorer model: {scorer_model}")
    print(f"  Optimizer model: {optimizer_model}")
    if subdir:
        print(f"  Dataset: {subdir}")
    print(f"  Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")

    print("\n" + "-" * 40)
    print("  Step 1: Optimizing...")
    print("-" * 40)

    seed_module = Path(args.seed_module) if hasattr(args, "seed_module") and args.seed_module else None
    if seed_module and not seed_module.is_absolute():
        seed_module = settings.root / seed_module

    if method == "miprov2":
        optimized, save_path = await run_miprov2(
            train, dev,
            scorer_model=scorer_model,
            optimizer_model=optimizer_model,
            auto=auto,
            num_threads=num_threads,
        )
    else:
        optimized, save_path = await run_gepa(
            train, dev,
            scorer_model=scorer_model,
            optimizer_model=optimizer_model,
            auto=auto,
            num_threads=num_threads,
            seed_module=seed_module,
        )

    # Create timestamped results folder — use current time (not program dir)
    # so re-evaluations of cached programs get their own folder
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_results_dir = settings.results_path / f"{method}_{timestamp}"
    run_results_dir.mkdir(parents=True, exist_ok=True)

    # Copy optimized program into results folder
    shutil.copytree(save_path, run_results_dir / "program")
    logger.info(f"Copied program from {save_path} to {run_results_dir / 'program'}")

    print("\n" + "-" * 40)
    print("  Step 2: Scoring dev and test...")
    print("-" * 40)

    dev_scored = score_with_dspy(dev, optimized, scorer_model=scorer_model)
    test_scored = score_with_dspy(test, optimized, scorer_model=scorer_model)

    # Evaluate with default thresholds
    print("\n  --- Default thresholds (pinned≥0.4, deleted<0.15) ---")
    dev_result = evaluate_pointwise(dev_scored, split="dev")
    test_result = evaluate_pointwise(test_scored, split="test")

    # Find optimal thresholds on dev, apply to test
    print("\n" + "-" * 40)
    print("  Step 3: Post-hoc threshold optimization on dev...")
    print("-" * 40)

    opt_pt, opt_dt, opt_dev_f1 = find_optimal_thresholds(dev_scored)
    print(f"\n  Optimal thresholds: pinned≥{opt_pt}, deleted<{opt_dt}")
    print(f"  Dev macro F1 with optimal thresholds: {opt_dev_f1:.4f}")

    dev_result_opt = evaluate_pointwise(dev_scored, split="dev", pinned_threshold=opt_pt, deleted_threshold=opt_dt)
    test_result_opt = evaluate_pointwise(test_scored, split="test", pinned_threshold=opt_pt, deleted_threshold=opt_dt)

    # Save all results
    (run_results_dir / "eval_dev.json").write_text(dev_result.model_dump_json(indent=2))
    (run_results_dir / "eval_test.json").write_text(test_result.model_dump_json(indent=2))
    (run_results_dir / "eval_dev_optimized_thresholds.json").write_text(dev_result_opt.model_dump_json(indent=2))
    (run_results_dir / "eval_test_optimized_thresholds.json").write_text(test_result_opt.model_dump_json(indent=2))

    default_gap = dev_result.accuracy.macro_f1 - test_result.accuracy.macro_f1
    posthoc_gap_trigger = 0.10
    use_optimized = default_gap > posthoc_gap_trigger

    metadata = {
        "scorer_model": scorer_model,
        "optimizer_model": optimizer_model,
        "algorithm": method,
        "dataset": subdir or "default",
        "train_ratio": settings.train_ratio,
        "dev_ratio": settings.dev_ratio,
        "train_size": len(train),
        "dev_size": len(dev),
        "test_size": len(test),
        "default_thresholds": {"pinned": 0.4, "deleted": 0.15},
        "dev_macro_f1": round(dev_result.accuracy.macro_f1, 4),
        "test_macro_f1": round(test_result.accuracy.macro_f1, 4),
        "dev_test_gap": round(default_gap, 4),
        "optimized_thresholds": {"pinned": opt_pt, "deleted": opt_dt},
        "dev_macro_f1_optimized": round(dev_result_opt.accuracy.macro_f1, 4),
        "test_macro_f1_optimized": round(test_result_opt.accuracy.macro_f1, 4),
        "dev_test_gap_optimized": round(dev_result_opt.accuracy.macro_f1 - test_result_opt.accuracy.macro_f1, 4),
        "gepa_params": {"auto": auto, "num_threads": num_threads},
        "program_source": str(save_path),
        "seed_module": str(seed_module) if seed_module else None,
        "comments": comments,
        "posthoc_gap_trigger": posthoc_gap_trigger,
        "posthoc_triggered": use_optimized,
        "recommended_thresholds": "optimized" if use_optimized else "default",
        "recommended_test_macro_f1": round(
            test_result_opt.accuracy.macro_f1 if use_optimized else test_result.accuracy.macro_f1, 4
        ),
    }
    (run_results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\n" + "=" * 60)
    print(f"  DSPy {method.upper()} Results")
    print("=" * 60)
    marker_default = " <-- recommended" if not use_optimized else ""
    marker_opt = " <-- recommended" if use_optimized else ""
    print(f"  Default thresholds (pinned≥0.4, deleted<0.15):{marker_default}")
    print(f"    Dev macro F1:  {dev_result.accuracy.macro_f1:.4f}")
    print(f"    Test macro F1: {test_result.accuracy.macro_f1:.4f}")
    print(f"    Gap:           {default_gap:.4f}")
    print(f"  Optimized thresholds (pinned≥{opt_pt}, deleted<{opt_dt}):{marker_opt}")
    print(f"    Dev macro F1:  {dev_result_opt.accuracy.macro_f1:.4f}")
    print(f"    Test macro F1: {test_result_opt.accuracy.macro_f1:.4f}")
    print(f"    Gap:           {dev_result_opt.accuracy.macro_f1 - test_result_opt.accuracy.macro_f1:.4f}")
    print(f"  Post-hoc threshold trigger: gap > {posthoc_gap_trigger} → {'TRIGGERED' if use_optimized else 'not triggered'}")
    print(f"  Saved module:  {save_path}")
    print(f"  Results:       {run_results_dir}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Goal Alignment Judge Experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data extraction and filtering
    extract_parser = subparsers.add_parser(
        "extract_ratings", help="Extract goal alignment ratings from local dev DB as CSV"
    )
    extract_parser.add_argument(
        "--output", type=str, default="input/goal_alignments_extracted.csv",
        help="Output CSV path relative to experiment root (default: input/goal_alignments_extracted.csv)",
    )
    extract_parser.add_argument(
        "--database", type=str, default=None,
        help="PostgreSQL database name (default: experiment DATABASE constant)",
    )

    filter_parser = subparsers.add_parser(
        "filter_to_pairs", help="Intersect two ratings CSVs to their common (goal, content) pairs"
    )
    filter_parser.add_argument(
        "--csv-a", required=True, help="First ratings CSV (e.g., input/goal_alignments_adam.csv)"
    )
    filter_parser.add_argument(
        "--csv-b", type=str, default=None,
        help="Second ratings CSV (default: input/goal_alignments_rated.csv)",
    )
    filter_parser.add_argument(
        "--output-a", type=str, default=None,
        help="Filtered output for csv-a (default: input/{csv_a_stem}_filtered.csv)",
    )
    filter_parser.add_argument(
        "--output-b", type=str, default=None,
        help="Filtered output for csv-b (default: input/{csv_b_stem}_filtered.csv)",
    )

    compare_parser = subparsers.add_parser(
        "compare_raters", help="Compare two ratings CSVs: overlap, agreement, distributions → HTML report"
    )
    compare_parser.add_argument(
        "--csv-a", required=True, help="First ratings CSV (e.g., input/goal_alignments_adam.csv)"
    )
    compare_parser.add_argument(
        "--csv-b", type=str, default=None,
        help="Second ratings CSV (default: goal_alignments_rated.csv)",
    )
    compare_parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML path (default: output/eda/rater_comparison.html)",
    )

    # Data loading
    pw_load_parser = subparsers.add_parser(
        "load_pointwise_data", help="Load pin/dismiss CSV, enrich from DB, create train/dev/test splits"
    )
    pw_load_parser.add_argument(
        "--input-csv", type=str, default=None,
        help="Input ratings CSV (default: goal_alignments_rated.csv)",
    )

    # DSPy optimization commands
    dspy_opt_parser = subparsers.add_parser(
        "dspy_optimize", help="Optimize pointwise scorer with DSPy (MIPROv2 or GEPA)"
    )
    dspy_opt_parser.add_argument(
        "--method", required=True, choices=["miprov2", "gepa"], help="Optimization method"
    )
    dspy_opt_parser.add_argument("--scorer-model", type=str, default=None, help="Scorer model (default: Sonnet)")
    dspy_opt_parser.add_argument("--optimizer-model", type=str, default=None, help="Optimizer model (default: Opus)")
    dspy_opt_parser.add_argument(
        "--auto", type=str, default="medium", choices=["light", "medium", "heavy"],
        help="GEPA/MIPROv2 auto preset (default: medium)",
    )
    dspy_opt_parser.add_argument("--num-threads", type=int, default=4, help="Parallel threads (default: 4)")
    dspy_opt_parser.add_argument(
        "--input-csv", type=str, default=None,
        help="Input ratings CSV (loads splits from matching subdir)",
    )

    dspy_eval_parser = subparsers.add_parser(
        "evaluate_dspy", help="Evaluate a saved DSPy-optimized module"
    )
    dspy_eval_parser.add_argument("--module-path", required=True, help="Path to saved DSPy program directory")
    dspy_eval_parser.add_argument("--split", default="dev", choices=["dev", "test"])
    dspy_eval_parser.add_argument("--scorer-model", type=str, default=None, help="Scorer model (default: Sonnet)")
    dspy_eval_parser.add_argument(
        "--input-csv", type=str, default=None,
        help="Input ratings CSV (loads splits from matching subdir)",
    )

    dspy_pipe_parser = subparsers.add_parser(
        "dspy_pipeline", help="End-to-end DSPy optimize + evaluate on dev and test"
    )
    dspy_pipe_parser.add_argument(
        "--method", required=True, choices=["miprov2", "gepa"], help="Optimization method"
    )
    dspy_pipe_parser.add_argument("--scorer-model", type=str, default=None, help="Scorer model (default: Sonnet)")
    dspy_pipe_parser.add_argument("--optimizer-model", type=str, default=None, help="Optimizer model (default: Opus)")
    dspy_pipe_parser.add_argument(
        "--input-csv", type=str, default=None,
        help="Input ratings CSV (loads splits from matching subdir)",
    )
    dspy_pipe_parser.add_argument(
        "--auto", type=str, default="medium", choices=["light", "medium", "heavy"],
        help="GEPA/MIPROv2 auto preset (default: medium)",
    )
    dspy_pipe_parser.add_argument("--num-threads", type=int, default=4, help="Parallel threads (default: 4)")
    dspy_pipe_parser.add_argument(
        "--seed-module", type=str, default=None,
        help="Path to a saved program directory to warm-start from",
    )
    dspy_pipe_parser.add_argument(
        "--comments", type=str, default="",
        help="Free-text comments to store in run metadata",
    )

    # Ablation studies and tooling
    ablation_ts_parser = subparsers.add_parser(
        "ablation_train_size", help="Study 5: Learning curve — test GEPA with subsampled training data"
    )
    ablation_ts_parser.add_argument(
        "--repetitions", type=int, default=1,
        help="Number of independent runs per config (default: 1)",
    )
    ablation_ts_parser.add_argument(
        "--sizes", type=str, default=None,
        help="Comma-separated subsample sizes to run, e.g. '20,35'. Omit for all sizes.",
    )

    subparsers.add_parser(
        "ablation_scorer_downgrade", help="Study 1: Evaluate best programs with Haiku vs Sonnet scorer"
    )

    seed_parser = subparsers.add_parser(
        "create_seed_program", help="Create a DSPy program pickle from a text file with prompt instructions"
    )
    seed_parser.add_argument(
        "--instructions", required=True, help="Path to text/markdown file with prompt instructions"
    )
    seed_parser.add_argument(
        "--output", required=True,
        help="Output directory for the seed program (e.g., output/dspy/seed_generic)",
    )

    subparsers.add_parser(
        "clear_dspy_cache", help="Clear DSPy/LiteLLM response cache for fresh optimization runs"
    )

    args = parser.parse_args(sys.argv[1:])

    command_map = {
        "extract_ratings": cmd_extract_ratings,
        "filter_to_pairs": cmd_filter_to_pairs,
        "compare_raters": cmd_compare_raters,
        "load_pointwise_data": cmd_load_pointwise_data,
        "dspy_optimize": cmd_dspy_optimize,
        "evaluate_dspy": cmd_evaluate_dspy,
        "dspy_pipeline": cmd_dspy_pipeline,
        "ablation_train_size": cmd_ablation_train_size,
        "ablation_scorer_downgrade": cmd_ablation_scorer_downgrade,
        "create_seed_program": cmd_create_seed_program,
        "clear_dspy_cache": cmd_clear_dspy_cache,
    }

    await command_map[args.command](args)
