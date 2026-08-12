from pathlib import Path

import dspy

from ..pointwise.pointwise_models import PointwiseExample, ScoredPointwiseExample
from ...settings import CLAUDE_SONNET, logger
from .dspy_data import pointwise_to_dspy, prediction_to_scored
from .dspy_optimizer import configure_dspy_lms, load_optimized_module


def score_with_dspy(
    examples: list[PointwiseExample],
    module: dspy.Module,
    scorer_model: str = CLAUDE_SONNET,
) -> list[ScoredPointwiseExample]:
    """Score examples using a DSPy module (in-memory or loaded).

    Returns ScoredPointwiseExample list compatible with existing evaluate_pointwise().
    """
    configure_dspy_lms(scorer_model=scorer_model)

    logger.info(f"Scoring {len(examples)} examples with DSPy module")

    scored = []
    for i, example in enumerate(examples):
        dspy_ex = pointwise_to_dspy(example)
        prediction = module(
            goal_title=dspy_ex.goal_title,
            goal_description=dspy_ex.goal_description,
            content_type=dspy_ex.content_type,
            content_title=dspy_ex.content_title,
            content_body=dspy_ex.content_body,
        )
        scored.append(prediction_to_scored(example, prediction))

        if (i + 1) % 10 == 0:
            logger.info(f"  DSPy scored {i + 1}/{len(examples)}")

    logger.info(f"DSPy scoring complete: {len(scored)} examples")
    return scored


def score_from_saved(
    examples: list[PointwiseExample],
    module_path: Path,
    scorer_model: str = CLAUDE_SONNET,
) -> list[ScoredPointwiseExample]:
    """Load a saved DSPy program and score examples with it."""
    module = load_optimized_module(module_path)
    return score_with_dspy(examples, module, scorer_model=scorer_model)
