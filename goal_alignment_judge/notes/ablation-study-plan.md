---
id: 7A2B3C4D-5E6F-7890-ABCD-EF1234567890
type: general
title: Ablation Study Plan — Rater Convergence & Pipeline Cost
pinned: true
created: 2026-03-27T00:00:00Z
updated: 2026-03-27T15:45:29Z
---

See full plan at `.claude/plans/resilient-finding-bumblebee.md` and below.

## Priority order

1. **Scorer downgrade at inference** (P0, ~30 min): Can Haiku execute Sonnet-optimized prompts? Inference-only, no optimization.
2. **Optimizer downgrade** (P0, ~5 hrs): Can Sonnet-Sonnet replace Opus-Sonnet? ~5x cost reduction.
3. **GEPA effort** (P1, ~3 hrs): Does `auto="light"` maintain quality at ~half time?
4. **Warm-starting** (P1, ~4 hrs): Seed from existing program to converge faster. Needs small code change.
5. **Minimum viable training set** (P2, ~8 hrs): Learning curve — how few items does a user need?

## Target deliverable

Production recommendation: optimizer model, scorer model (opt + inference), GEPA effort, warm-start strategy, minimum items per user, expected time and F1.


——

# Ablation Study Plan: Rater Convergence & Pipeline Cost

## Context

Single-rater GEPA optimization achieves 0.70-0.79 test macro F1 (Adam: 0.764 with post-hoc thresholds, Matt: 0.791 default). The current configuration (Opus optimizer + Sonnet scorer, ~84 min/user, ~134 rated items) works but is expensive for production. We need to determine: (1) how few items a user needs, and (2) how to make the pipeline cheaper/faster.

## Study 1: Scorer Downgrade at Inference (P0 — run first)

**Question:** Can we optimize with Sonnet but deploy with Haiku?

Re-evaluate existing best programs with Haiku as scorer — no optimization needed, just inference. ~5 min per run.

| Run | Program | Scorer | Dataset |
|-----|---------|--------|---------|
| 1a | `gepa_20260326_025538` | Sonnet | Adam (baseline: 0.764) |
| 1b | `gepa_20260326_025538` | Haiku | Adam |
| 1c | `gepa_20260326_220254` | Sonnet | Matt (baseline: 0.791) |
| 1d | `gepa_20260326_220254` | Haiku | Matt |

Use `evaluate_dspy` command — already supports `--scorer-model`. Run each 3x to measure LLM variance (inference is cheap).

**Success:** Haiku test F1 within 0.05 of Sonnet. **Failure:** Drop >0.10.

**Code changes:** None.

## Study 2: Optimizer Downgrade — Sonnet-Sonnet (P0)

**Question:** Can Sonnet serve as both optimizer and scorer?

Opus reflection is the biggest cost driver (~$15/$75 per MTok vs Sonnet at ~$3/$15). If Sonnet can self-reflect effectively, optimization cost drops ~5x.

| Run | Optimizer | Scorer | Dataset | Expected time |
|-----|-----------|--------|---------|---------------|
| 2a | Opus | Sonnet | Adam | ~84 min (baseline) |
| 2b | Sonnet | Sonnet | Adam | ~60-80 min |
| 2c | Sonnet | Sonnet | Matt | ~60-80 min |

```bash
make run_experiment ARGS="inter_rater_goal_alignment_scores dspy_pipeline \
  --method gepa --input-csv input/goal_alignments_adam_filtered.csv \
  --scorer-model claude-sonnet-4-6 --optimizer-model claude-sonnet-4-6 \
  --comments 'Ablation 2b: Sonnet-Sonnet, Adam'"
```

**Success:** Test F1 within 0.03 of Opus baseline. **Failure:** Drop >0.05.

**Code changes:** None — `--optimizer-model` already exists.

## Study 3: GEPA Effort — Light vs Medium (P1)

**Question:** Does `auto="light"` (6 candidates vs 12) maintain quality at ~half the time?

Must clear DSPy cache first to avoid reusing cached medium programs.

| Run | Auto | Optimizer | Dataset | Expected time |
|-----|------|-----------|---------|---------------|
| 3a | medium | best from S2 | Adam | baseline |
| 3b | light | best from S2 | Adam | ~35-45 min |
| 3c | light | best from S2 | Matt | ~35-45 min |

**Success:** Test F1 within 0.04 of medium. **Failure:** Drop >0.06.

**Code changes:** None.

## Study 4: Warm-Starting from Existing Program (P1)

**Question:** Can we seed GEPA with an existing optimized prompt to converge faster?

Instead of starting from the base signature, load a previously optimized program as the GEPA student. The prompt's general structure/principles carry over; GEPA adapts calibration for the new user.

| Run | Seed | Auto | Dataset | What it tests |
|-----|------|------|---------|---------------|
| 4a | base signature | medium | Matt | Baseline (have: 0.791) |
| 4b | Adam-optimized | light | Matt | Transfer + light refinement |
| 4c | Adam-optimized | medium | Matt | Transfer + full refinement |
| 4d | base signature | light | Matt | Light from scratch (control) |

**Success:** 4b (warm+light) within 0.03 of 4a (cold+medium) — warm-starting compensates for lighter effort.

**Code changes required:**
- `dspy_optimizer.py`: Add `seed_module: Path | None = None` param to `run_gepa`. If provided, `module = load_optimized_module(seed_module)` instead of `GoalAlignmentScorer()`. ~5 lines.
- `main.py`: Add `--seed-module PATH` to `dspy_pipeline` and `dspy_optimize` subparsers. Thread through to `run_gepa`. ~10 lines.

## Study 5: Minimum Viable Training Set (P2)

**Question:** How many rated items does a user need before GEPA produces a useful scorer?

Subsample training data while keeping dev/test fixed. Uses best optimizer/scorer from Studies 2-3.

| Train size | Approx. total items | What it represents |
|-----------|--------------------|--------------------|
| 67 (full) | 134 | Current setup |
| 50 | ~100 | Moderate adoption |
| 35 | ~70 | Early adopter |
| 20 | ~40 | First week |
| 10 | ~20 | Day one |

**Code changes required:**
- Add `--train-subsample N` flag to `dspy_pipeline`. Before calling `run_gepa`, randomly subsample `train` to N items (stratified by action to preserve class balance). ~15 lines in `main.py`.

**Success:** Identify the "knee" where test F1 stabilizes within 0.05 of the full-train result.

Run endpoints first (full, 10), then midpoint (35), then fill in.

## Execution Order & Dependencies

```
Day 1 (~2 hrs):  Study 1 (inference-only, quick)
Day 1-2 (~5 hrs): Study 2 (Sonnet-Sonnet optimizer)
Day 2 (~3 hrs):  Study 3 (light effort, uses S2 winner)
Day 2-3 (~4 hrs): Study 4 (warm-starting, needs small code change)
Day 3-4 (~8 hrs): Study 5 (learning curve, needs small code change)
```

Studies 1 and 2 can run in parallel. Studies 3-5 depend on S2 results.

## Code Changes Summary

| Study | File | Change |
|-------|------|--------|
| 4 | `dspy_optimizer.py` | Add `seed_module` param to `run_gepa` (~5 lines) |
| 4 | `main.py` | Add `--seed-module` CLI flag (~10 lines) |
| 5 | `main.py` | Add `--train-subsample` CLI flag (~15 lines) |

Studies 1-3 need zero code changes.

## Expected Production Recommendation

After all studies, deliverable is:

> **Optimizer:** [Opus / Sonnet], **Scorer (opt):** Sonnet, **Scorer (inference):** [Sonnet / Haiku]
> **Effort:** [light / medium], **Warm-start:** [yes / no]
> **Minimum items:** [N], **Optimization time:** [X] min, **Test F1:** [0.XX] +/- [0.YY]

## Verification

Each study produces a results folder (`output/results/gepa_YYYYMMDD_HHMMSS/`) with metadata.json documenting the configuration. Compare test macro F1, dev-test gap, and Spearman across configurations. Create a summary observation note after each study group.
