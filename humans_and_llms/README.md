# Human-LLM Scaling: Mathematical Framework and Analysis

An exploration of how trust in LLMs impacts workforce composition and organizational structure.

## Overview

This project explores the mathematical relationship between human trust in LLM outputs and workforce composition. As trust in LLMs increases, we model how organizations might balance human employees and LLM agents to optimize costs while meeting operational demands.

## Running the Analysis

To run the analysis and generate all visualizations:

```bash
# From the root directory
make run_experiment ARGS="humans_and_llms"

# Or navigate to the experiments directory and run
python -m humans_and_llms
```
## Interactive Dashboard

The analysis generates an interactive HTML dashboard that presents all findings in an organized format:

- **Location**: `output/humans_and_llms.html`
- **Contents**:
  - Introduction and model explanation
  - Key visualizations with explanations
  - Methodology and limitations
