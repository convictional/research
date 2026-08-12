#!/usr/bin/env python3
"""Run full baseline evaluation on all available internal data.

This script runs comprehensive baseline evaluation using all available author pairs.
Results are saved to baselines/results/ for analysis.

Usage:
    cd experiments/writing_identification
    poetry run python scripts/run_baseline_evaluation.py
    poetry run python scripts/run_baseline_evaluation.py --models custom --checkpoint-path models/checkpoints/best_model.pt
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.evaluate import main


if __name__ == "__main__":
    asyncio.run(main())
