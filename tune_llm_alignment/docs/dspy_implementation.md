# DSPy Implementation: Technical Documentation

This document describes our DSPy implementation for optimizing LLM priority predictions, building on the OPRO baseline (see `01_opro_implementation.md`).

## Overview

**Goal**: Replace manual OPRO orchestration with DSPy's declarative framework, enabling systematic prompt optimization with less code and more optimizer choices.

**Key Insight**: DSPy's MIPROv2 optimizer not only rewrites instructions but also **automatically selects few-shot examples** - something our OPRO implementation didn't do.

## Architecture

### Three-LLM System (Same Pattern, Different Models)

```
┌─────────────────────────────────────────────────────────────────┐
│                      DSPy MIPROv2                               │
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   Optimizer  │───▶│   Generator  │───▶│    Judge     │     │
│   │(Claude Opus) │    │(Claude Haiku)│    │(Claude Sonnet)│    │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                                        │              │
│         │       instruction + few-shot demos     │              │
│         └────────────────────────────────────────┘              │
│                    (handled internally by DSPy)                 │
└─────────────────────────────────────────────────────────────────┘
```

**LLM 1 - Generator** (`claude-haiku-4-5-20251001`)
- Role: Takes instruction + context, outputs 5 ranked priorities
- Temperature: 0.9
- This is the model being optimized

**LLM 2 - Judge** (`claude-sonnet-4-5-20250929`)
- Role: Scores predictions against ground truth (same 4-criteria rubric as OPRO)
- Temperature: 0.3

**LLM 3 - Optimizer** (`claude-opus-4-5-20251101`)
- Role: MIPROv2's "prompt model" - generates instruction candidates
- Temperature: 1.0

### OPRO vs DSPy Mapping

| OPRO Component | DSPy Equivalent |
|----------------|-----------------|
| `generator.py` | `dspy.ChainOfThought(PrioritySignature)` |
| `judge.py` | Metric function using `dspy.Predict(JudgeSignature)` |
| `optimizer.py` | MIPROv2's internal prompt model |
| Manual trajectory | MIPROv2 has `verbose`, `track_stats`, `log_dir` options |
| No few-shot | **Automatic few-shot demo selection** |

## Core Components

### Signatures (`src/dspy_modules/signatures.py`)

DSPy signatures define input/output contracts:

```python
class PrioritySignature(dspy.Signature):
    """Predict a user's top work priorities from historical context."""

    context: str = dspy.InputField(desc="Historical content...")
    target_date: str = dspy.InputField(desc="The date to predict for")

    reasoning: str = dspy.OutputField(desc="Analysis of context...")
    priority_1: str = dspy.OutputField(desc="Highest priority")
    priority_2: str = dspy.OutputField(desc="Second priority")
    # ... through priority_5
```

### Module (`src/dspy_modules/predictor.py`)

```python
class PriorityPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(PrioritySignature)

    def forward(self, context: str, target_date: str) -> dspy.Prediction:
        return self.predict(context=context, target_date=target_date)
```

### Metrics (`src/dspy_modules/metrics.py`)

Two metric functions available:

1. **`alignment_metric`** - Full LLM judge (same 4-criteria scoring as OPRO)
2. **`team_recall_metric`** - Simpler recall-based metric for faster iteration

Both return 0-1 scores normalized from the 0-100 scale.

## Results

### Run 1: Individual Data (Nov 26, 2025)

| Metric | Value |
|--------|-------|
| Duration | ~35 minutes |
| Data | Individual (Adam only) |
| Train/Dev/Test | 36 / 12 / 12 |
| **Avg Test Score** | **9.4/100** |

### Run 2: Team Data (Nov 27, 2025)

| Metric | Value |
|--------|-------|
| Duration | ~43 minutes |
| Data | Team (all members pooled) |
| Train/Dev/Test | 50 / 16 / 18 |
| **Avg Test Score** | **26.8/100** |

### What MIPROv2 Optimized

The optimizer made two key changes:

**1. Rewrote Instructions**

Original (baseline):
```
Predict a user's top work priorities from historical context...
```

Optimized (excerpt):
```
Analyze the provided GitHub activity (issues, comments, and discussions)
to predict a software developer's top 5 work priorities for the target date.

**Focus your analysis on:**
1. Deployment follow-ups: Features recently deployed that need documentation...
2. Active feature development: Issues created in the 24-48 hours before...
3. Explicit requests: Direct asks from team members with time-sensitive language...
...
```

**2. Selected 4 Few-Shot Demos**

MIPROv2 automatically selected 4 examples from the training set to include as demonstrations, showing the model what good outputs look like.

## Comparison: OPRO vs DSPy

| Aspect | OPRO | DSPy |
|--------|------|------|
| Runtime | ~5.5 hours | ~43 minutes |
| Best train/val score | 59.5/100 | N/A (internal) |
| **Test score** | **12.8/100** | **26.8/100** |
| Code complexity | ~500 lines | ~100 lines |
| Few-shot | No | Yes (automatic) |
| Trajectory visible | Yes (custom logging) | Configurable (`verbose`, `log_dir`) |

### Key Finding: Poor Generalization

Both approaches show the same pattern: optimization improves train/val performance but **generalizes poorly to the held-out test set**.

**Hypothesis**: Daily priority prediction is inherently unpredictable - humans make last-minute decisions based on factors not captured in historical context.

## Running the Experiment

```bash
# Individual data (Adam only)
make run_experiment ARGS="tune_llm_alignment scripts/02_run_dspy.py"

# Team data (all members pooled)
make run_experiment ARGS="tune_llm_alignment scripts/02_run_dspy.py --team"

# With simpler recall metric
make run_experiment ARGS="tune_llm_alignment scripts/02_run_dspy.py --team --recall-metric"
```

### Prerequisites

1. **Database**: Local `decide_development` database with content seed
   ```bash
   # From the main decide/ directory
   make db_seed  # Includes content data needed for hybrid search
   ```

2. **Environment variables**:
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY` (for embeddings in context retrieval)

## File Structure

```
tune_llm_alignment/
├── src/dspy_modules/
│   ├── __init__.py         # Exports
│   ├── signatures.py       # Input/output contracts
│   ├── predictor.py        # DSPy module
│   ├── metrics.py          # Evaluation functions
│   └── data_loader.py      # JSONL → dspy.Example
├── scripts/
│   └── 02_run_dspy.py      # Entry point
├── data/                   # Individual dataset
└── data_team/              # Team dataset
```

## Learnings

### What DSPy Does Well

1. **Automatic few-shot selection**: MIPROv2 picks good examples without manual curation
2. **Instruction rewriting**: Generated specific, structured instructions
3. **Less code**: ~100 lines vs ~500 for OPRO
4. **Faster iteration**: 43 min vs 5.5 hours
5. **Configurable logging**: Has `verbose`, `track_stats`, and `log_dir` options (we haven't fully explored these)

### Limitations

1. **Test generalization**: Still poor (~27%) - the task may be too hard
2. **Cost**: Three Claude models adds up (~$10-30 per run)
3. **Framework churn**: DSPy API evolves rapidly

### When to Use DSPy vs OPRO

**Use DSPy when:**
- You want quick iteration
- Few-shot examples help your task
- You're okay with less custom control

**Use OPRO when:**
- You need full custom control over the optimization loop
- You want to implement novel optimization strategies
- You're debugging why optimization isn't working

## Next Steps

The poor test generalization suggests daily priority prediction may not be the right task for automated optimization. Better candidates:

1. **Classification tasks** (email routing, urgency detection)
2. **Extraction tasks** (action items from meetings)
3. **Transformation tasks** (standardized summaries)

These have clearer ground truth and should generalize better.

See `02_dspy_migration.md` for the original architecture proposal and production application ideas.
