"""Study 5: Minimum Viable Training Set Size + Variance Measurement

Subsamples each rater's training data independently (stratified by action)
and runs GEPA optimization for each size. Dev/test stay fixed.

Clears both DSPy disk and in-memory caches between each run to ensure
fresh optimization trajectories that match cmd_dspy_pipeline behavior.
No rollout_id or custom gepa_seed — uses exact same defaults as the CLI.

Usage:
    make run_experiment ARGS="goal_alignment_judge ablation_train_size --repetitions 3"
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import dspy

from ..pipelines.dspy_pointwise.dspy_optimizer import run_gepa
from ..pipelines.dspy_pointwise.dspy_scorer import score_with_dspy
from ..pipelines.pointwise.pointwise_data_loader import load_pointwise_split, subsample_train
from ..pipelines.pointwise.pointwise_evaluator import evaluate_pointwise, find_optimal_thresholds
from ..settings import CLAUDE_OPUS, CLAUDE_SONNET, logger, settings

# --- Configuration ---
RATERS = [
    ("adam", "goal_alignments_adam_filtered"),
    ("matt", "goal_alignments_matt_032526_filtered"),
]
SUBSAMPLE_SIZES = [10, 20, 35, 50]  # full is added automatically
SCORER_MODEL = CLAUDE_SONNET
OPTIMIZER_MODEL = CLAUDE_OPUS
AUTO = "medium"
NUM_THREADS = 4
SEED = 42


def _clear_all_dspy_caches() -> None:
    """Clear both disk and in-memory DSPy caches for a fresh GEPA run.

    DSPy caches LLM responses at two levels:
    - Disk: ~/.dspy_cache/ (diskcache FanoutCache, persists across processes)
    - Memory: cachetools LRUCache (persists within a process)

    Both must be cleared to ensure a fully independent optimization trajectory.
    Uses dspy.configure_cache() to reinitialize cleanly rather than deleting
    the directory (which leaves the FanoutCache object with dangling references).
    """
    if hasattr(dspy, "cache") and dspy.cache is not None:
        # Clear disk cache contents (keeps directory intact)
        if hasattr(dspy.cache, "disk_cache") and hasattr(dspy.cache.disk_cache, "clear"):
            dspy.cache.disk_cache.clear()
        # Clear in-memory cache
        dspy.cache.memory_cache.clear()

    # Reinitialize the cache to ensure a clean state
    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
    )
    logger.info("Cleared and reinitialized DSPy disk + memory caches")


async def _run_single(
    rater_name: str,
    subdir: str,
    train_size: int | None,
    output_dir: Path,
    repetition: int = 0,
) -> dict:
    """Run a single GEPA optimization + evaluation and return results."""
    train = load_pointwise_split("train", subdir=subdir)
    dev = load_pointwise_split("dev", subdir=subdir)
    test = load_pointwise_split("test", subdir=subdir)

    original_train_size = len(train)
    if train_size is not None and train_size < len(train):
        train = subsample_train(train, train_size, seed=SEED)

    label = f"n{len(train)}" if train_size else "full"
    print(f"\n{'=' * 60}")
    print(f"  {rater_name} / train={len(train)} (from {original_train_size}) / dev={len(dev)} / test={len(test)} / rep={repetition}")
    print(f"{'=' * 60}")

    # Clear all caches before each run for a fully independent trajectory.
    # No rollout_id, no custom gepa_seed — matches cmd_dspy_pipeline exactly.
    _clear_all_dspy_caches()

    optimized, save_path = await run_gepa(
        train, dev,
        scorer_model=SCORER_MODEL,
        optimizer_model=OPTIMIZER_MODEL,
        auto=AUTO,
        num_threads=NUM_THREADS,
    )

    # Score dev and test
    dev_scored = score_with_dspy(dev, optimized, scorer_model=SCORER_MODEL)
    test_scored = score_with_dspy(test, optimized, scorer_model=SCORER_MODEL)

    # Evaluate with default thresholds
    dev_result = evaluate_pointwise(dev_scored, split="dev")
    test_result = evaluate_pointwise(test_scored, split="test")

    # Post-hoc threshold optimization
    opt_pt, opt_dt, _ = find_optimal_thresholds(dev_scored)
    dev_result_opt = evaluate_pointwise(dev_scored, split="dev", pinned_threshold=opt_pt, deleted_threshold=opt_dt)
    test_result_opt = evaluate_pointwise(test_scored, split="test", pinned_threshold=opt_pt, deleted_threshold=opt_dt)

    # Determine recommended thresholds
    default_gap = dev_result.accuracy.macro_f1 - test_result.accuracy.macro_f1
    use_opt = default_gap > 0.10

    # Save results to per-run folder
    run_ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{rater_name}_{label}_rep{repetition}_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "eval_dev.json").write_text(dev_result.model_dump_json(indent=2))
    (run_dir / "eval_test.json").write_text(test_result.model_dump_json(indent=2))
    (run_dir / "eval_dev_optimized_thresholds.json").write_text(dev_result_opt.model_dump_json(indent=2))
    (run_dir / "eval_test_optimized_thresholds.json").write_text(test_result_opt.model_dump_json(indent=2))

    # Copy program
    shutil.copytree(save_path, run_dir / "program", dirs_exist_ok=True)

    metadata = {
        "rater": rater_name,
        "dataset": subdir,
        "repetition": repetition,
        "train_size_original": original_train_size,
        "train_size_subsampled": len(train),
        "dev_size": len(dev),
        "test_size": len(test),
        "scorer_model": SCORER_MODEL,
        "optimizer_model": OPTIMIZER_MODEL,
        "auto": AUTO,
        "subsample_seed": SEED,
        "default_thresholds": {"pinned": 0.4, "deleted": 0.15},
        "dev_macro_f1": round(dev_result.accuracy.macro_f1, 4),
        "test_macro_f1": round(test_result.accuracy.macro_f1, 4),
        "dev_test_gap": round(default_gap, 4),
        "optimized_thresholds": {"pinned": opt_pt, "deleted": opt_dt},
        "dev_macro_f1_optimized": round(dev_result_opt.accuracy.macro_f1, 4),
        "test_macro_f1_optimized": round(test_result_opt.accuracy.macro_f1, 4),
        "dev_test_gap_optimized": round(dev_result_opt.accuracy.macro_f1 - test_result_opt.accuracy.macro_f1, 4),
        "recommended": "optimized" if use_opt else "default",
        "recommended_test_f1": round(
            test_result_opt.accuracy.macro_f1 if use_opt else test_result.accuracy.macro_f1, 4
        ),
        "program_source": str(save_path),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return metadata


async def run_study(num_repetitions: int = 1, sizes: list[int] | None = None) -> None:
    """Run the full training size ablation study.

    Args:
        num_repetitions: Number of independent runs per (rater, size) configuration.
            Each run clears all DSPy caches for a fully fresh optimization trajectory.
        sizes: List of training subsample sizes to run. None means full train only.
            If empty or not provided, runs all sizes including full: [full, 10, 20, 35, 50].
    """
    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = settings.results_path / "ablation_train_size"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_file = output_dir / f"summary_{run_timestamp}.json"

    # Build size list: None = full train, integers = subsamples
    if sizes is not None:
        run_sizes = sizes  # Use exactly what was requested; no implicit full train
    else:
        run_sizes = [None] + SUBSAMPLE_SIZES  # Default: full + all configured sizes

    for rep in range(num_repetitions):
        for rater_name, subdir in RATERS:
            for size in run_sizes:
                result = await _run_single(
                    rater_name, subdir, size, output_dir,
                    repetition=rep,
                )
                all_results.append(result)

                # Save running summary after each run (in case of interruption)
                summary_file.write_text(json.dumps(all_results, indent=2))

    # Print final summary
    print(f"\n\n{'=' * 90}")
    print("  STUDY 5 SUMMARY: Minimum Viable Training Set Size")
    print(f"{'=' * 90}")
    print(f"{'Rater':<6} {'Train':<7} {'Rep':<4} {'Dev F1':<8} {'Test F1':<9} {'Test Opt':<9} {'Gap':<7} {'Rec':<10} {'Rec F1':<8}")
    print("-" * 90)
    for r in all_results:
        size_label = str(r["train_size_subsampled"])
        if r["train_size_subsampled"] == r["train_size_original"]:
            size_label += "*"
        print(
            f"{r['rater']:<6} {size_label:<7} {r['repetition']:<4} {r['dev_macro_f1']:<8} {r['test_macro_f1']:<9} "
            f"{r['test_macro_f1_optimized']:<9} {r['dev_test_gap']:<7} {r['recommended']:<10} "
            f"{r['recommended_test_f1']:<8}"
        )

    print(f"\n  Results saved to {output_dir}")
    print(f"  Summary: {summary_file}")
    print(f"  * = full train set (no subsampling)")
    if num_repetitions > 1:
        print(f"  Repetitions: {num_repetitions} (cache cleared between each run)")
