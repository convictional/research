# OPRO Implementation: Technical Documentation

This document describes our custom OPRO (Optimization by PROmpting) implementation for aligning LLM priority predictions with actual work behavior.

## Overview

**Goal**: Automatically optimize instruction prompts so that Gemini Flash better predicts what a user will work on each day, given historical context (emails, meetings, tasks, discussions).

**Why This Matters**: Manual prompt engineering requires reading content data and iterating based on intuition. Under privacy constraints where we can't see user content, we need automated optimization that works on metrics alone.

## Architecture

### Three-LLM System

```
┌─────────────────────────────────────────────────────────────────┐
│                        OPRO Loop                                │
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   Optimizer  │───▶│   Generator  │───▶│    Judge     │     │
│   │ (Gemini Pro) │    │(Gemini Flash)│    │ (Gemini Pro) │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                                        │              │
│         │              trajectory                │              │
│         └────────────────────────────────────────┘              │
│                    (instruction, score) pairs                   │
└─────────────────────────────────────────────────────────────────┘
```

**LLM 1 - Generator** (`src/generator.py`)
- Model: `gemini-flash-latest`
- Temperature: 0.9 (diverse predictions)
- Role: Takes an instruction + context, outputs 5 ranked priorities
- This is the model being optimized

**LLM 2 - Judge** (`src/judge.py`)
- Model: `gemini-pro-latest`
- Temperature: 0.3 (consistent evaluation)
- Role: Scores predictions against ground truth
- Provides the reward signal for optimization

**LLM 3 - Optimizer** (`src/optimizer.py`)
- Model: `gemini-pro-latest`
- Temperature: 1.3 (creative instruction proposals)
- Role: Analyzes trajectory, proposes better instructions
- Implements the OPRO meta-prompt pattern

### Data Flow

1. **Context Retrieval** (`src/context_retriever.py`)
   - Hybrid search: 70% vector similarity + 30% full-text (PostgreSQL tsvector)
   - Temporal filtering: only content created before target date
   - Returns: emails, meetings, tasks, discussions

2. **Ground Truth** (`src/standup_parser.py`)
   - Parses markdown standup files
   - Extracts what was actually worked on each day

3. **Optimization Loop** (`src/opro_loop.py`)
   - Generates N candidate instructions per iteration
   - Evaluates each on a mini-batch of training examples
   - Adds (instruction, score) pairs to trajectory
   - Feeds trajectory back to Optimizer for next iteration

## Scoring Rubric

The Judge evaluates on four dimensions (each 0-10):

| Metric | Focus |
|--------|-------|
| **Correctness** | Are ground truth items in the predictions? (recall) |
| **Completeness** | What % of ground truth items are captured? |
| **Ordering** | Are ground truth items ranked highly (top 1-3)? |
| **Context Usage** | Did the model derive predictions from context? |

**Overall Score** = (sum of 4 metrics) × 2.5 = 0-100

## Configuration

Key hyperparameters from `config.toml`:

```toml
[optimization]
max_iterations = 30
candidates_per_iteration = 8
mini_batch_size = 10
plateau_threshold = 5  # stop after N iterations without improvement

[generation]
temperature_generator = 0.9
temperature_judge = 0.3
temperature_optimizer = 1.3

[context]
top_k = 20  # items retrieved from hybrid search
```

## Results

### Run: 2024-11-24

- **Duration**: ~7.5 hours
- **Iterations**: 12 (stopped due to plateau)
- **Candidates evaluated**: 97 total

| Metric | Value |
|--------|-------|
| Baseline score | 18.75 |
| Best score | 50.0 |
| Improvement | +31.25 |

### Score Trajectory

```
Iter 0 (baseline): 18.75
Iter 1: best=39.0, avg=23.0
Iter 2: best=43.25, avg=33.8
Iter 3: best=39.5, avg=29.8
Iter 4: best=44.25, avg=29.8
Iter 5: best=32.25, avg=23.6
Iter 6: best=36.25, avg=24.1
Iter 7: best=41.0, avg=26.1
Iter 8: best=50.0 ← NEW BEST
Iter 9-12: no improvement (plateau)
```

### Best Instruction Found

```
From the provided context, identify the specific tasks the user actually
worked on for the target date. Rank these tasks from most to least
significant, ensuring the day's primary accomplishment or main focus is
listed first. Your analysis must prioritize tasks with direct evidence
of execution over planned items or general discussions.
```

### Key Patterns in High-Scoring Instructions

1. **Focus on execution evidence**: "tasks with direct evidence of execution"
2. **Distinguish from plans**: "over planned items or general discussions"
3. **Recall emphasis**: "identify the specific tasks the user actually worked on"
4. **Ranking matters**: "primary accomplishment or main focus is listed first"
5. **Action verbs**: "executed", "demonstrably worked on", "tangible progress"

### Low-Scoring Patterns (to avoid)

- Vague instructions: "predict today's priorities"
- Future-focused: "what they will work on"
- No filtering guidance: treating all context equally

## Learnings

### What Worked

1. **High Optimizer temperature (1.3)**: Generated diverse instruction candidates
2. **Mini-batch evaluation**: Faster iteration without losing signal
3. **Trajectory-based meta-prompt**: OPRO pattern effectively builds on past attempts
4. **Structured Judge output**: Consistent scoring across dimensions

### Limitations

1. **Score ceiling at 50**: Plateau suggests either:
   - Task is inherently difficult (human decisions are unpredictable)
   - Judge scoring has a ceiling
   - Need more/better training data

2. **High variance**: Same-quality instructions can score 20-40 points apart on different mini-batches

3. **Slow iteration**: ~7.5 hours for 12 iterations due to:
   - Sequential candidate evaluation
   - Large context windows
   - API latency

4. **No hyperparameter optimization**: Fixed top_k=20, temperatures, etc.

## File Structure

```
tune_llm_alignment/
├── src/
│   ├── config.py           # Config loader
│   ├── models.py           # Pydantic data models
│   ├── generator.py        # LLM 1: priority prediction
│   ├── judge.py            # LLM 2: scoring
│   ├── optimizer.py        # LLM 3: instruction generation
│   ├── opro_loop.py        # Main optimization orchestrator
│   ├── context_retriever.py # Hybrid search
│   ├── standup_parser.py   # Ground truth extraction
│   └── data_splitter.py    # Train/dev/test splits
├── scripts/
│   └── 01_run_opro.py      # Entry point
├── data/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── results/
│   └── opro_result_*.json  # Optimization results
└── config.toml             # Configuration
```

## Running the Experiment

```bash
cd experiments/tune_llm_alignment
make run_experiment ARGS="tune_llm_alignment scripts/01_run_opro.py"
```

Required environment variables:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY` (for embeddings)
- Database connection to `local_research_db`

## Next Steps

See `02_dspy_migration.md` for the proposed DSPy refactor.
