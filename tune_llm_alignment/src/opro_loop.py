"""
Main OPRO optimization loop.

Orchestrates Generator, Judge, and Optimizer to find better instructions.
"""

import asyncio
import random
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import json

from tqdm import tqdm

from .models import (
    ExampleData,
    TrajectoryPoint,
    OptimizationResult,
)
from .generator import PriorityGenerator
from .judge import AlignmentJudge
from .optimizer import OPROOptimizer
from .config import config


class OPROLoop:
    """
    Main optimization loop implementing OPRO algorithm.

    Iteratively:
    1. Optimizer generates candidate instructions
    2. Generator predicts priorities using each candidate
    3. Judge evaluates predictions
    4. Best candidates added to trajectory
    5. Repeat until convergence
    """

    def __init__(
        self,
        generator: PriorityGenerator,
        judge: AlignmentJudge,
        optimizer: OPROOptimizer,
    ):
        self.generator = generator
        self.judge = judge
        self.optimizer = optimizer
        self.predictions_log = []  # Log all predictions for debugging

    async def optimize(
        self,
        train_examples: List[ExampleData],
        dev_examples: List[ExampleData],
        test_examples: Optional[List[ExampleData]] = None,
        max_iterations: Optional[int] = None,
        candidates_per_iteration: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Run OPRO optimization loop.

        Args:
            train_examples: Training data
            dev_examples: Development data for validation
            max_iterations: Maximum optimization iterations
            candidates_per_iteration: Number of candidates to generate per iteration

        Returns:
            OptimizationResult with best instruction and trajectory
        """
        # Get config
        max_iterations = max_iterations or config.get("optimization.max_iterations", 30)
        candidates_per_iteration = candidates_per_iteration or config.get(
            "optimization.candidates_per_iteration", 8
        )
        mini_batch_size = config.get("optimization.mini_batch_size", 10)
        plateau_threshold = config.get("optimization.plateau_threshold", 5)

        # Initialize trajectory with baseline instruction(s)
        baseline_instruction = config.get(
            "baseline.instruction", "Analyze the context and predict today's top priorities."
        )

        print("=" * 60)
        print("OPRO OPTIMIZATION")
        print("=" * 60)
        print(f"Train examples: {len(train_examples)}")
        print(f"Dev examples: {len(dev_examples)}")
        print(f"Max iterations: {max_iterations}")
        print(f"Candidates per iteration: {candidates_per_iteration}")
        print(f"Mini-batch size: {mini_batch_size}")
        print(f"\nBaseline instruction: {baseline_instruction}")
        print("=" * 60)

        # Evaluate baseline
        print("\n[Iteration 0] Evaluating baseline...")
        baseline_score = await self._evaluate_instruction(
            baseline_instruction,
            train_examples[:mini_batch_size],
            "baseline",
        )

        trajectory = [
            TrajectoryPoint(
                iteration=0,
                instruction=baseline_instruction,
                score=baseline_score,
                candidate_index=0,
            )
        ]

        best_score = baseline_score
        no_improvement_count = 0
        start_time = datetime.utcnow()

        # Sample exemplars for meta-prompt (stay consistent)
        exemplars = self._prepare_exemplars(train_examples[:3])

        # Main optimization loop
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'=' * 60}")
            print(f"[Iteration {iteration}/{max_iterations}]")
            print(f"{'=' * 60}")

            # Generate candidate instructions
            print("Generating candidate instructions...")
            trajectory_pairs = [(t.instruction, t.score) for t in trajectory]
            candidates = await self.optimizer.generate_candidates(
                trajectory=trajectory_pairs,
                exemplars=exemplars,
                n=candidates_per_iteration,
            )

            print(f"Generated {len(candidates)} candidates")

            # Evaluate each candidate
            print("Evaluating candidates...")
            for i, candidate in enumerate(tqdm(candidates, desc="Candidates")):
                # Use mini-batch for efficiency
                batch = random.sample(
                    train_examples, min(mini_batch_size, len(train_examples))
                )

                score = await self._evaluate_instruction(
                    candidate,
                    batch,
                    f"iter{iteration}_cand{i}",
                )

                # Add to trajectory
                trajectory.append(
                    TrajectoryPoint(
                        iteration=iteration,
                        instruction=candidate,
                        score=score,
                        candidate_index=i,
                    )
                )

                print(f"  Candidate {i+1}: {score:.1f}")

                # Update best
                if score > best_score:
                    best_score = score
                    no_improvement_count = 0
                    print(f"  ✓ New best score: {best_score:.1f}")

            # Check for improvement this iteration
            iteration_best = max(t.score for t in trajectory if t.iteration == iteration)
            if iteration_best <= best_score:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            print(f"\nBest score so far: {best_score:.1f}")
            print(f"No improvement for {no_improvement_count} iterations")

            # Validate on dev and test sets every 5 iterations
            if iteration % 5 == 0:
                best_instruction = max(trajectory, key=lambda t: t.score).instruction

                dev_score = await self._evaluate_instruction(
                    best_instruction,
                    dev_examples,
                    f"dev_iter{iteration}",
                )
                print(f"\n📊 Dev set score: {dev_score:.1f}")

                # Also evaluate on test set (for logging only, no reward signal)
                if test_examples:
                    test_score = await self._evaluate_instruction(
                        best_instruction,
                        test_examples,
                        f"test_iter{iteration}",
                    )
                    print(f"📊 Test set score: {test_score:.1f} (logging only, no optimization signal)")

            # Check stopping criteria
            if no_improvement_count >= plateau_threshold:
                print(f"\n🛑 Stopping: No improvement for {plateau_threshold} iterations")
                break

        # Final results
        end_time = datetime.utcnow()
        best_point = max(trajectory, key=lambda t: t.score)

        result = OptimizationResult(
            best_instruction=best_point.instruction,
            best_score=best_point.score,
            trajectory=trajectory,
            total_iterations=iteration,
            stopping_reason=(
                f"plateau_{no_improvement_count}"
                if no_improvement_count >= plateau_threshold
                else "max_iterations"
            ),
            start_time=start_time,
            end_time=end_time,
        )

        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Best score: {result.best_score:.1f}")
        print(f"Improvement: {result.improvement:.1f}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"\nBest instruction:")
        print(f"  {result.best_instruction}")
        print("=" * 60)

        return result

    async def _evaluate_instruction(
        self,
        instruction: str,
        examples: List[ExampleData],
        label: str,
    ) -> float:
        """Evaluate an instruction on a set of examples."""
        scores = []

        for example in examples:
            # Generate prediction
            prediction = await self.generator.generate(
                instruction=instruction,
                context=example.context,
            )

            # Judge prediction
            judge_score = await self.judge.evaluate(
                prediction=prediction,
                ground_truth=example.standup_entry,
                context=example.context,
            )

            scores.append(judge_score.overall_score)

            # Log prediction details for debugging
            self.predictions_log.append(
                {
                    "label": label,
                    "date": example.standup_entry.date.isoformat(),
                    "instruction": instruction,
                    "predicted": [p.description for p in prediction.priorities],
                    "ground_truth": [p.description for p in example.standup_entry.priorities],
                    "reasoning": prediction.reasoning,
                    "judge_score": judge_score.overall_score,
                    "judge_reasoning": judge_score.reasoning,
                    "judge_issues": judge_score.specific_issues,
                    "judge_suggestions": judge_score.suggestions,
                }
            )

        return sum(scores) / len(scores) if scores else 0.0

    def _prepare_exemplars(self, examples: List[ExampleData]) -> List[dict]:
        """Prepare exemplars for the optimizer meta-prompt."""
        exemplars = []
        for ex in examples:
            exemplars.append(
                {
                    "date": ex.standup_entry.date.strftime("%Y-%m-%d"),
                    "context_summary": f"{ex.context.total_items} items (emails, meetings, tasks, discussions)",
                    "ground_truth_preview": ex.standup_entry.priorities[0].description[
                        :100
                    ],
                }
            )
        return exemplars

    def save_result(self, result: OptimizationResult, filepath: Path):
        """Save optimization result to JSON."""
        with open(filepath, "w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

    def save_predictions_log(self, filepath: Path):
        """Save detailed predictions log for debugging."""
        with open(filepath, "w") as f:
            for entry in self.predictions_log:
                f.write(json.dumps(entry, default=str) + "\n")
