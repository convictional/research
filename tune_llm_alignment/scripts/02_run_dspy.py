#!/usr/bin/env python3
"""Run DSPy optimization for priority prediction.

This script optimizes the PriorityPredictor module using MIPROv2,
comparable to the OPRO baseline in 01_run_opro.py.

Usage:
    uv run python scripts/02_run_dspy.py [--team] [--recall-metric]

Options:
    --team          Use team-level data (all team members' priorities)
    --recall-metric Use simpler recall-based metric instead of full judge
"""

import json
import os
import sys
from datetime import datetime, UTC
from pathlib import Path

import dspy
from dspy.teleprompt import MIPROv2

from src.config import config
from src.dspy_modules import PriorityPredictor, alignment_metric, load_examples
from src.dspy_modules.metrics import team_recall_metric


def main():
    # Parse args
    use_team_data = "--team" in sys.argv
    use_recall_metric = "--recall-metric" in sys.argv

    # Config loads .env.secrets automatically
    api_key = config.anthropic_api_key
    os.environ["ANTHROPIC_API_KEY"] = api_key  # DSPy reads from env

    print("=" * 60)
    print("DSPy OPTIMIZATION")
    print("=" * 60)
    print(f"  Data: {'team (all members)' if use_team_data else 'individual (Adam only)'}")
    print(f"  Metric: {'recall' if use_recall_metric else 'full judge'}")

    # Configure LLMs
    print("\nConfiguring models...")

    generator_lm = dspy.LM(
        "anthropic/claude-haiku-4-5-20251001",
        temperature=0.9,
        max_tokens=4096,
    )
    judge_lm = dspy.LM(
        "anthropic/claude-sonnet-4-5-20250929",
        temperature=0.3,
        max_tokens=4096,
    )
    optimizer_lm = dspy.LM(
        "anthropic/claude-opus-4-5-20251101",
        temperature=1.0,
        max_tokens=4096,
    )

    # Set default LM for generation
    dspy.configure(lm=generator_lm)

    print(f"  Generator: claude-haiku-4-5 (temp=0.9)")
    print(f"  Judge: claude-sonnet-4-5 (temp=0.3)")
    print(f"  Optimizer: claude-opus-4-5 (temp=1.0)")

    # Load data
    print("\nLoading data...")
    data_dir = Path(__file__).parent.parent / ("data_team" if use_team_data else "data")

    train_set = load_examples(data_dir / "train.jsonl")
    dev_set = load_examples(data_dir / "dev.jsonl")
    test_set = load_examples(data_dir / "test.jsonl")

    print(f"  Train: {len(train_set)} examples")
    print(f"  Dev: {len(dev_set)} examples")
    print(f"  Test: {len(test_set)} examples")

    # Create metric that uses judge LM
    base_metric = team_recall_metric if use_recall_metric else alignment_metric

    def metric_with_judge(example, prediction, trace=None):
        """Wrap metric to use judge LM."""
        with dspy.context(lm=judge_lm):
            return base_metric(example, prediction, trace)

    # Create base module
    print("\nInitializing PriorityPredictor...")
    predictor = PriorityPredictor()

    # Configure optimizer
    print("\nConfiguring MIPROv2 optimizer...")
    optimizer = MIPROv2(
        metric=metric_with_judge,
        auto="medium",  # Automatic hyperparameter selection
        num_threads=4,
        prompt_model=optimizer_lm,  # Use Opus for instruction generation
    )

    # Run optimization
    print("\n" + "=" * 60)
    print("Starting optimization...")
    print("=" * 60)

    start_time = datetime.now(UTC)

    optimized_predictor = optimizer.compile(
        predictor,
        trainset=train_set,
        valset=dev_set,
    )

    end_time = datetime.now(UTC)
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")

    # Save optimized module
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = results_dir / f"dspy_optimized_{timestamp}.json"

    optimized_predictor.save(str(model_path))
    print(f"\nSaved optimized module to: {model_path}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    test_scores = []
    for i, example in enumerate(test_set):
        with dspy.context(lm=generator_lm):
            prediction = optimized_predictor(
                context=example.context,
                target_date=example.target_date,
            )
        score = metric_with_judge(example, prediction)
        test_scores.append(score)
        print(f"  Test {i+1}/{len(test_set)}: {score:.3f}")

    avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0
    print(f"\nAverage test score: {avg_test_score:.3f} ({avg_test_score * 100:.1f}/100)")

    # Save results summary
    results = {
        "timestamp": timestamp,
        "duration_seconds": duration,
        "config": {
            "data": "team" if use_team_data else "individual",
            "metric": "recall" if use_recall_metric else "full_judge",
        },
        "train_size": len(train_set),
        "dev_size": len(dev_set),
        "test_size": len(test_set),
        "test_scores": test_scores,
        "avg_test_score": avg_test_score,
        "models": {
            "generator": "claude-haiku-4-5-20251001",
            "judge": "claude-sonnet-4-5-20250929",
            "optimizer": "claude-opus-4-5-20251101",
        },
    }

    summary_path = results_dir / f"dspy_results_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results summary to: {summary_path}")


if __name__ == "__main__":
    main()
