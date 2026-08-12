"""Create a DSPy program pickle from a text file containing prompt instructions.

Used to create warm-start seed programs for GEPA from manually curated templates.

Usage:
    make run_experiment ARGS="goal_alignment_judge create_seed_program \
        --instructions path/to/template.txt --output output/dspy/seed_generic"
"""

import cloudpickle
from pathlib import Path

from ..pipelines.dspy_pointwise.dspy_module import GoalAlignmentScorer
from ..settings import logger


def create_seed_program(instructions_path: Path, output_dir: Path) -> Path:
    """Create a DSPy program with custom instructions and save as pickle."""
    instructions = instructions_path.read_text()

    # Strip frontmatter if present (erdos note format)
    if instructions.startswith("---"):
        parts = instructions.split("---", 2)
        if len(parts) >= 3:
            instructions = parts[2]

    # Strip everything before the first actual prompt line
    # (skip metadata lines like "Program:", "Config:", "Result:", "Derived from")
    lines = instructions.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Assess ") or stripped.startswith("You will receive"):
            start = i
            break
        # Also catch if the template starts with a heading
        if stripped.startswith("## ") and i > 3:
            start = i
            break
    instructions = "\n".join(lines[start:]).strip()

    if not instructions:
        raise ValueError(f"No instructions found in {instructions_path}")

    # Create module and inject instructions
    module = GoalAlignmentScorer()
    for _, pred in module.named_predictors():
        pred.signature.instructions = instructions

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "program.pkl"
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(module, f)

    logger.info(f"Created seed program at {output_dir}")
    logger.info(f"  Instructions: {len(instructions)} chars, ~{len(instructions.split())} words")

    return output_dir
