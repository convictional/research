"""Study 1: Scorer Downgrade at Inference

Can we optimize with Sonnet but deploy with Haiku?

Re-evaluates existing best GEPA programs with different scorer models.
No optimization needed — inference only, ~5 min per run.

Usage:
    make run_experiment ARGS="goal_alignment_judge ablation_scorer_downgrade"
"""

import json
import sys
from pathlib import Path

from ..pipelines.dspy_pointwise.dspy_optimizer import load_optimized_module
from ..pipelines.dspy_pointwise.dspy_scorer import score_with_dspy
from ..pipelines.pointwise.pointwise_data_loader import load_pointwise_split
from ..pipelines.pointwise.pointwise_evaluator import evaluate_pointwise, find_optimal_thresholds
from ..settings import settings

# Configure runs: (label, program_dir, data_subdir, scorer_model, scorer_label)
RUNS = [
    ("Adam", "gepa_20260326_025538", "goal_alignments_adam_filtered", "claude-sonnet-4-6", "Sonnet"),
    ("Adam", "gepa_20260326_025538", "goal_alignments_adam_filtered", "claude-haiku-4-5-20251001", "Haiku"),
    ("Matt", "gepa_20260326_220254", "goal_alignments_matt_032526_filtered", "claude-sonnet-4-6", "Sonnet"),
    ("Matt", "gepa_20260326_220254", "goal_alignments_matt_032526_filtered", "claude-haiku-4-5-20251001", "Haiku"),
]


def run_study() -> None:
    results = []

    for rater, prog_dir, subdir, scorer_model, scorer_label in RUNS:
        prog_path = settings.dspy_path / prog_dir
        if not (prog_path / "program.pkl").exists():
            print(f"  SKIP {rater}/{scorer_label}: program not found at {prog_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {rater} / {scorer_label} scorer")
        print(f"  Program: {prog_dir}")
        print(f"{'=' * 60}")

        module = load_optimized_module(prog_path)
        dev = load_pointwise_split("dev", subdir=subdir)
        test = load_pointwise_split("test", subdir=subdir)

        print(f"\n  Scoring dev ({len(dev)} items)...")
        dev_scored = score_with_dspy(dev, module, scorer_model=scorer_model)
        dev_result = evaluate_pointwise(dev_scored, split="dev")

        print(f"\n  Scoring test ({len(test)} items)...")
        test_scored = score_with_dspy(test, module, scorer_model=scorer_model)
        test_result = evaluate_pointwise(test_scored, split="test")

        # Post-hoc threshold optimization
        opt_pt, opt_dt, _ = find_optimal_thresholds(dev_scored)
        test_result_opt = evaluate_pointwise(
            test_scored, split="test", pinned_threshold=opt_pt, deleted_threshold=opt_dt,
        )

        default_gap = dev_result.accuracy.macro_f1 - test_result.accuracy.macro_f1
        use_opt = default_gap > 0.10

        row = {
            "rater": rater,
            "scorer": scorer_label,
            "program": prog_dir,
            "dev_f1": round(dev_result.accuracy.macro_f1, 4),
            "test_f1_default": round(test_result.accuracy.macro_f1, 4),
            "test_f1_optimized": round(test_result_opt.accuracy.macro_f1, 4),
            "gap_default": round(default_gap, 4),
            "opt_thresholds": {"pinned": opt_pt, "deleted": opt_dt},
            "recommended": "optimized" if use_opt else "default",
            "recommended_test_f1": round(
                test_result_opt.accuracy.macro_f1 if use_opt else test_result.accuracy.macro_f1, 4
            ),
            "spearman": round(test_result.score_correlation, 4),
            "critical_errors": test_result.accuracy.critical_errors,
        }
        results.append(row)

    # Summary table
    print(f"\n\n{'=' * 90}")
    print("  STUDY 1 SUMMARY: Scorer Downgrade at Inference")
    print(f"{'=' * 90}")
    header = f"{'Rater':<6} {'Scorer':<8} {'Dev F1':<8} {'Test F1':<9} {'Test Opt':<9} {'Gap':<7} {'Rec':<10} {'Rec F1':<8} {'Spearman':<9} {'Crit':<5}"
    print(header)
    print("-" * 90)
    for r in results:
        print(
            f"{r['rater']:<6} {r['scorer']:<8} {r['dev_f1']:<8} {r['test_f1_default']:<9} "
            f"{r['test_f1_optimized']:<9} {r['gap_default']:<7} {r['recommended']:<10} "
            f"{r['recommended_test_f1']:<8} {r['spearman']:<9} {r['critical_errors']:<5}"
        )

    # Save results
    output_path = settings.results_path / "ablation_study1_scorer_downgrade.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {output_path}")
