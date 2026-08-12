import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import dspy
from dspy.teleprompt import MIPROv2

from ..pointwise.pointwise_models import PointwiseExample
from ...settings import CLAUDE_OPUS, CLAUDE_SONNET, logger, settings
from .dspy_data import pointwise_to_dspy
from .dspy_metric import gepa_metric, macro_f1_metric
from .dspy_module import GoalAlignmentScorer


def configure_dspy_lms(
    scorer_model: str = CLAUDE_SONNET,
    optimizer_model: str = CLAUDE_OPUS,
    optimizer_rollout_id: int | None = None,
) -> tuple[dspy.LM, dspy.LM]:
    """Configure DSPy language models and set ANTHROPIC_API_KEY in env.

    Args:
        optimizer_rollout_id: If set, adds a rollout_id to the optimizer LM's
            cache key, forcing fresh API calls for the reflection/proposal LM
            while keeping caching enabled for within-run consistency. The scorer
            LM always uses the shared cache (deterministic at temp=0.0).

            Use this for ablation studies: each run gets a different rollout_id,
            producing independent optimization trajectories while maintaining
            deterministic scoring within each run.
    """
    os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key.get_secret_value()

    # Scorer: always cached, no rollout_id. temp=0.0 gives near-deterministic scoring.
    # Cached results are shared across all runs for consistent evaluation.
    generator_lm = dspy.LM(
        f"anthropic/{scorer_model}",
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    # Optimizer: temp=1.0 for diverse proposals. Use rollout_id to force
    # fresh proposals across independent runs.
    optimizer_kwargs = {
        "temperature": 1.0,
        "max_tokens": settings.max_tokens,
    }
    if optimizer_rollout_id is not None:
        optimizer_kwargs["rollout_id"] = optimizer_rollout_id
    optimizer_lm = dspy.LM(
        f"anthropic/{optimizer_model}",
        **optimizer_kwargs,
    )

    dspy.configure(lm=generator_lm)
    return generator_lm, optimizer_lm


def _save_program(
    optimized: GoalAlignmentScorer,
    method: str,
    scorer_model: str,
    optimizer_model: str,
    train_size: int,
    dev_size: int,
    duration: float,
    extra_meta: dict | None = None,
) -> Path:
    """Save optimized program and metadata to output/dspy/.

    Uses save_program=True (cloudpickle) to preserve the full module architecture,
    since optimizers like MIPROv2 modify the module structure (adding few-shot demos,
    rewriting instructions) in ways that don't round-trip through JSON state saving.
    """
    output_dir = settings.dspy_path
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    save_dir = output_dir / f"{method}_{timestamp}"
    optimized.save(str(save_dir), save_program=True)

    meta = {
        "method": method,
        "scorer_model": scorer_model,
        "optimizer_model": optimizer_model,
        "train_size": train_size,
        "dev_size": dev_size,
        "duration_seconds": round(duration, 1),
        "timestamp": timestamp,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = save_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n  Saved optimized program to: {save_dir}")
    return save_dir


async def run_miprov2(
    train: list[PointwiseExample],
    dev: list[PointwiseExample],
    scorer_model: str = CLAUDE_SONNET,
    optimizer_model: str = CLAUDE_OPUS,
    auto: str = "medium",
    num_threads: int = 4,
) -> tuple[GoalAlignmentScorer, Path]:
    """Run MIPROv2 optimization on the pointwise scorer."""
    _, optimizer_lm = configure_dspy_lms(scorer_model, optimizer_model)

    train_dspy = [pointwise_to_dspy(ex) for ex in train]
    dev_dspy = [pointwise_to_dspy(ex) for ex in dev]

    module = GoalAlignmentScorer()

    optimizer = MIPROv2(
        metric=macro_f1_metric,
        auto=auto,
        num_threads=num_threads,
        prompt_model=optimizer_lm,
    )

    print(f"\n  Starting MIPROv2 optimization (auto={auto}, threads={num_threads})...")
    print(f"  Scorer: {scorer_model}, Optimizer: {optimizer_model}")
    print(f"  Train: {len(train_dspy)}, Dev: {len(dev_dspy)}")

    start_time = datetime.now(UTC)

    optimized = await asyncio.to_thread(
        optimizer.compile,
        module,
        trainset=train_dspy,
        valset=dev_dspy,
    )

    duration = (datetime.now(UTC) - start_time).total_seconds()
    logger.info(f"MIPROv2 optimization complete in {duration:.1f}s")

    save_path = _save_program(
        optimized,
        "miprov2",
        scorer_model,
        optimizer_model,
        len(train),
        len(dev),
        duration,
        extra_meta={"auto": auto, "num_threads": num_threads},
    )
    return optimized, save_path


async def run_gepa(
    train: list[PointwiseExample],
    dev: list[PointwiseExample],
    scorer_model: str = CLAUDE_SONNET,
    optimizer_model: str = CLAUDE_OPUS,
    auto: str = "medium",
    num_threads: int = 4,
    seed_module: Path | None = None,
    optimizer_rollout_id: int | None = None,
    gepa_seed: int = 0,
) -> tuple[GoalAlignmentScorer, Path]:
    """Run GEPA optimization on the pointwise scorer.

    Args:
        seed_module: Optional path to a previously optimized program directory.
            If provided, GEPA seeds from its instructions instead of the base
            signature. The prompt's general structure carries over; GEPA adapts
            calibration for the new data.
        optimizer_rollout_id: If set, forces fresh optimizer LM calls by adding
            a unique rollout_id to the cache key. Scorer LM stays cached for
            deterministic evaluation. Use for independent ablation runs.
        gepa_seed: RNG seed for GEPA's internal sampling and candidate selection.
            Vary alongside optimizer_rollout_id for fully independent runs.
    """
    _, optimizer_lm = configure_dspy_lms(scorer_model, optimizer_model, optimizer_rollout_id=optimizer_rollout_id)

    train_dspy = [pointwise_to_dspy(ex) for ex in train]
    dev_dspy = [pointwise_to_dspy(ex) for ex in dev]

    if seed_module:
        module = load_optimized_module(seed_module)
        logger.info(f"Warm-starting from seed module: {seed_module}")
    else:
        module = GoalAlignmentScorer()

    optimizer = dspy.GEPA(
        metric=gepa_metric,
        auto=auto,
        reflection_lm=optimizer_lm,
        num_threads=num_threads,
        seed=gepa_seed,
    )

    rollout_label = f", rollout_id={optimizer_rollout_id}" if optimizer_rollout_id is not None else ""
    seed_mod_label = f", seed_module={seed_module.name}" if seed_module else ""
    print(f"\n  Starting GEPA optimization (auto={auto}, threads={num_threads}, seed={gepa_seed}{rollout_label}{seed_mod_label})...")
    print(f"  Scorer: {scorer_model}, Optimizer: {optimizer_model}")
    print(f"  Train: {len(train_dspy)}, Dev: {len(dev_dspy)}")

    start_time = datetime.now(UTC)

    optimized = await asyncio.to_thread(
        optimizer.compile,
        student=module,
        trainset=train_dspy,
        valset=dev_dspy,
    )

    duration = (datetime.now(UTC) - start_time).total_seconds()
    logger.info(f"GEPA optimization complete in {duration:.1f}s")

    save_path = _save_program(
        optimized,
        "gepa",
        scorer_model,
        optimizer_model,
        len(train),
        len(dev),
        duration,
        extra_meta={
            "auto": auto,
            "num_threads": num_threads,
            "gepa_seed": gepa_seed,
            "seed_module": str(seed_module) if seed_module else None,
            "optimizer_rollout_id": optimizer_rollout_id,
        },
    )
    return optimized, save_path


def load_optimized_module(path: Path) -> GoalAlignmentScorer:
    """Load a previously saved optimized DSPy program directory."""
    import cloudpickle

    pkl_path = path / "program.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"No program.pkl found in {path}")

    with open(pkl_path, "rb") as f:
        return cloudpickle.load(f)


def find_latest_module(method: str) -> Path | None:
    """Find the latest saved program directory by method name."""
    output_dir = settings.dspy_path
    if not output_dir.exists():
        return None

    candidates = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith(f"{method}_")],
        reverse=True,
    )
    return candidates[0] if candidates else None
