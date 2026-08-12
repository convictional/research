"""
Main experiment script - Run OPRO optimization.

Usage:
    uv run python scripts/01_run_opro.py
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from src.config import config
from src.standup_parser import parse_all_standups
from src.context_retriever import ContextRetriever
from src.data_splitter import DataSplitter
from src.generator import PriorityGenerator
from src.judge import AlignmentJudge
from src.optimizer import OPROOptimizer
from src.opro_loop import OPROLoop


async def main():
    """Run the full OPRO optimization experiment."""

    print("=" * 80)
    print("LLM DECISION ALIGNMENT VIA OPRO")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load API keys
    gemini_api_key = config.gemini_api_key
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env.secrets")

    # Parse standup entries
    print("📄 Parsing standup documents...")
    db_cutoff = datetime.strptime(
        config.get("data.db_content_cutoff", "2025-05-19"), "%Y-%m-%d"
    )

    standup_entries = parse_all_standups(
        username="Adam",
        h1_file=config.get("data.standup_h1_file", "Product Standups 2025H1.md"),
        h2_file=config.get("data.standup_h2_file", "Product Standups 2025H2.md"),
        before_date=db_cutoff,
    )

    print(f"✓ Found {len(standup_entries)} standup entries")
    print(f"  Date range: {standup_entries[0].date.date()} to {standup_entries[-1].date.date()}\n")

    # Initialize context retriever
    print("🔍 Initializing context retriever...")
    context_retriever = ContextRetriever(openai_api_key)
    await context_retriever.connect()
    print("✓ Connected to database\n")

    # Prepare dataset
    print("📊 Preparing dataset with context...")
    data_splitter = DataSplitter(context_retriever)

    data_dir = Path("data")
    if (data_dir / "train.jsonl").exists():
        print("  Loading cached splits...")
        train_examples = DataSplitter.load_examples(data_dir / "train.jsonl")
        dev_examples = DataSplitter.load_examples(data_dir / "dev.jsonl")
        test_examples = DataSplitter.load_examples(data_dir / "test.jsonl")
        print(f"  ✓ Loaded {len(train_examples)} train, {len(dev_examples)} dev, {len(test_examples)} test examples\n")
    else:
        print("  Preparing dataset from scratch (this may take a few minutes)...")
        train_examples, dev_examples, test_examples = await data_splitter.prepare_dataset(
            standup_entries, output_dir=data_dir
        )
        print()

    # Initialize LLMs
    print("🤖 Initializing Gemini models...")
    generator = PriorityGenerator(gemini_api_key)
    judge = AlignmentJudge(gemini_api_key)
    optimizer = OPROOptimizer(gemini_api_key)
    print("✓ Models initialized\n")

    # Run OPRO optimization
    print("🚀 Starting OPRO optimization...\n")
    opro = OPROLoop(generator, judge, optimizer)

    result = await opro.optimize(
        train_examples=train_examples,
        dev_examples=dev_examples,
        test_examples=test_examples,  # Pass test set for periodic logging
    )

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"opro_result_{timestamp}.json"

    opro.save_result(result, result_file)

    # Save detailed predictions log for debugging
    predictions_log_file = results_dir / f"predictions_log_{timestamp}.jsonl"
    opro.save_predictions_log(predictions_log_file)

    print(f"\n💾 Results saved to: {result_file}")
    print(f"💾 Predictions log saved to: {predictions_log_file}")

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION")
    print("=" * 80)

    test_score = await opro._evaluate_instruction(
        result.best_instruction,
        test_examples,
        "final_test",
    )

    print(f"Test set score: {test_score:.1f}")
    print(f"Best instruction: {result.best_instruction}")

    # Cleanup
    await context_retriever.close()

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total duration: {result.duration_seconds:.1f}s")
    print(f"Final improvement: {result.improvement:.1f} points")
    print(f"Test set performance: {test_score:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
