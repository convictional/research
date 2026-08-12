# Goal Alignment Scoring — Research Log

What we've tried, what we've learned, and where the open problems are. This is meant to give someone picking up the experiment full context for choosing next directions.

## The Problem

Users of our goal-tracking product can pin (confirm), delete (reject), or leave (neutral) content items that the system surfaces as relevant to their goals. We want an automated scorer that replicates these judgments — and are trying both pointwise (is this content aligned with this goal?) and pairwise (which of two items is more aligned?) approaches, with a theory that using a fitted Bradley-Terry model, we can convert a well performing pairwise ranker into a pointwise scorer.

## Data

- **Source**: One organization, 5 goals with signal (where at least one item was pinned or deleted). 7 additional goals excluded — all-neutral, no discriminatory signal.
- **Pointwise**: 403 rows in the input CSV across 19 goals. Reduced to 101 usable items (16 pinned, 12 deleted, 73 neutral) through three steps: (1) DB enrichment drops ~19 items whose content records aren't in the local seed database, (2) 3 items held out as few-shot calibration examples (all from Conversion goal), (3) `filter_goals_with_signal` keeps only 5 goals that have at least one pin or delete — the other 14 goals are all-neutral with no discriminatory signal. Splits: ~50% train, ~25% dev (28 items), ~25% test (27 items). Scarce classes (pinned/deleted) rebalanced into dev/test.
- **Pairwise**: 120 training pairs derived from pointwise boundary comparisons (e.g. pinned > neutral > deleted) + 20 human-rated A/B pairs (5 raters per pair - prefer goal-owner ratings; fall back on majority vote). Dev/test: 50 pairs each (25A/25B, no ties).
- **Inter-rater agreement**: ICC(2,1) = 0.624 across 8 raters — moderate agreement. Easy pairs (clear similarity gap) ICC = 0.688, hard pairs ICC = 0.540. "Goal alignment" is partly subjective. Note, easy and hard pairs mean little — see below.

## Similarity Distributions

![Embedding similarity by human action](similarity_by_action.png)

The cosine similarity between content and goal embeddings provides essentially zero signal for distinguishing pin/delete/neutral. The three distributions overlap almost completely: pinned mean=0.509, neutral mean=0.475, deleted mean=0.466. A deleted item is just as likely to have high embedding similarity as a pinned one.

This matters for two reasons:
1. **The "easy" vs "hard" pair distinction from Phase 1 is misleading.** Pairs were constructed based on similarity gaps, but similarity doesn't predict human preference — so "easy" pairs aren't necessarily easier for the task.
2. **Embedding similarity is not a useful feature for scoring.** Any scorer that relies on topical proximity to the goal will fail. The pin/delete decision is about the *quality and directness* of the connection, not whether the content is *about* the goal's topic. Most content surfaced by the system is already topically relevant — the human judgment is about whether it's *actionable, specific, and directly advances* the goal.

## What We Tried

### Phase 1: Three-Class Pointwise (pinned/neutral/deleted)

Switched to three-class. Much harder — macro F1 dropped to 0.41–0.49. The pinned class is essentially unlearnable with 2–5 items per split. The scorer consistently struggles with the neutral/pinned and neutral/deleted boundaries.

**Eight runs, chronological:**

I include v1-v6 evn though I had included the 7 goals without ratings which causes extremely noisy signal because there were still some worthwile learnings (e.g. pipes starting with a low-instruction, broad prompt work better than seeding strategies). V7 and V8 ran on the _actually_ rated 5 goals.

| Run | Key Change | Dev F1 | Test F1 | Rounds | Key Finding |
|-----|-----------|--------|---------|--------|-------------|
| v1 | Binary, ensemble=3 | 0.725 | 0.689 | 3 | Binary works, three-class is harder |
| v2 | Three-class, ensemble=1 | 0.412 | 0.491 | 3 | Baseline three-class |
| v2b | Three-class, ensemble=3 | — | 0.393 | 3 | Ensemble amplifies conservative bias, makes it worse |
| v3 | Intent + counterfactual prompts | 0.431 | 0.456 | 3 | Prompt seeding doesn't help — wrong bottleneck |
| v4 | Minimal prompts, 5 rounds | 0.480 | 0.514 | 5 | Best generalizing config. LLM discovered calibration dimensions on its own |
| v5 | Resumed v4 to v9 | 0.749 | 0.441 | 9 | Dev overfit — rubric accumulated case-specific rules |
| v6 | Generalization constraints, 18 rounds | 0.769 | 0.365 | 18 | Constraints didn't prevent overfitting, just slowed it |
| v7 | Clean data, no outer loop | 0.880 | 0.535 | 7 | Corrected labels helped, but same overfit pattern |
| v8 | Clean data + outer loop | 1.000 | 0.644 | 20 | Outer loop improved test ceiling, best test F1 so far |

### Pairwise Comparison

Pairwise ("which is more aligned?") is cognitively easier for LLMs and avoids absolute calibration. With the generalization outer loop:

| Run | Dev Acc | Test Acc | Rounds |
|-----|---------|----------|--------|
| No outer loop | 0.72 | 0.52 | 1 (early-stopped) |
| With outer loop | 0.94 | **0.74** | 10 |

Test 0.74 is the strongest result across all experiments. Pairwise > pointwise for this task - however, to work with our product, we need a ranking and labels. The Bradley-Terry Ranking below is an attempt to derive scores based on overall pairwise rankings.

#### Bradley-Terry Ranking

Used pairwise rubric v5 (best test accuracy = 0.74) to score 234 boundary pairs, then fit per-goal BT models. Ran with both Haiku and Sonnet as the scorer:

| Goal | Haiku | Sonnet | Notes |
|------|-------|--------|-------|
| Activation | Excellent | Excellent | Both pinned #1-#2. Clean BT score separation |
| Operations | Good | Excellent | Haiku: pinned #1,#3. Sonnet: both pinned #1-#2 |
| Conversion | Decent | Decent | Pinned in top-5 for both. Some BT score ties |
| AI Moat | **Poor** | **Good** | Haiku: deleted #1-#2. Sonnet: pinned #1-#2, deleted #6,#8 |
| Discover | N/A | N/A | Only 3 items, 2 pairs — insufficient for BT |

**Model capability matters.** The same rubric v5 produces dramatically different AI Moat rankings depending on the scorer — Sonnet completely fixes it. This validates the distillation path: optimize with Sonnet, test whether improvements transfer to Haiku for production. BT scores provide natural class separation when the judge is accurate — Activation's pinned-neutral gap (41/40 vs 28) is clean enough to threshold into pointwise classes. Not yet formally evaluated on ranking metrics (NDCG, Kendall tau). See `notes/observation--bt-ranking-*.md` for full breakdowns.

**Scaling caveat**: N-choose-2 is combinatorially brutal. 30 items = 435 pairs, 100 items = 4,950. In production, goals accumulate content continuously — Haiku is the only viable scorer at this scale. Even then, we'll need Swiss-style adaptive tournaments (O(N log N) vs O(N²)) and transitive filtering (if A > B and B > C, skip A vs C) to keep costs manageable. See `notes/next_step_ideas.md` for details.

## Key Lessons

### 1. The inner loop always overfits

Every refinement loop that optimizes a rubric against dev disagreements eventually memorizes dev. The pattern is consistent: dev keeps climbing, test plateaus or declines after 3–5 rounds. More rounds = wider gap.

| Rounds | Best Test F1 (pointwise) |
|--------|-------------------------|
| 3 | 0.491–0.514 |
| 5 | 0.514 |
| 7 | 0.535 |
| 9 | 0.441 |
| 18 | 0.365 |
| 20 (with outer loop) | 0.644 |

The outer loop helps by periodically broadening, but doesn't fully solve the problem.

### 2. Minimal prompts generalize better than elaborate ones

Stripping the scorer and rubric prompts to bare essentials (v4) produced the best generalizing pointwise config. The LLM discovered its own calibration dimensions — including a "topical relevance floor" and "frequency test" that directly addressed our bottleneck. Seeded strategies (intent reasoning, counterfactual impact, generalization constraints) either didn't help or actively hurt by constraining the search space.

### 3. Ensemble majority vote hurts three-class scoring

Ensemble=3 with majority vote amplified the scorer's conservative bias, pushing more neutrals into deleted. The error mode is asymmetric (more likely to reject borderline content), so majority vote doubles down on systematic error rather than canceling noise. Score averaging is slightly better but still doesn't solve the calibration problem.

### 4. The pinned class is nearly unlearnable

With 5 pinned items in dev and 5 in test, the signal is too thin. Pinned F1 on test rarely exceeds 0.35 and frequently hits 0.00 (all pinned scored as neutral). The rubric learns to identify dev-specific pin patterns that don't transfer. This drags macro F1 down even when neutral and deleted are reasonable.

### 5. Pairwise comparison is easier, but converting back to pointwise is hard

Pairwise accuracy of 0.74 on test beats any pointwise macro F1. Relative comparison avoids the calibration problem — the LLM doesn't need to decide "how aligned" something is, just "which is more aligned." However, converting BT rankings back to pointwise classes (pinned/neutral/deleted) via learned thresholds doesn't generalize at this data scale. Thresholds learned on dev achieve 0.700 macro F1 on dev but only 0.356 on test — the per-goal test splits are too small and class-imbalanced (e.g., Operations test is all-neutral) for thresholds to transfer. This is a data limitation that may be overcome with more labels.

### 6. Clean labels matter

Correcting mislabeled items improved early generalization (test F1 at round 3: 0.465 clean vs 0.355 dirty). Garbage in, garbage out — even a few bad labels in a dataset this small distort the rubric.

### 7. Rubric size correlates with overfitting

The more iteration rounds, the larger the rubric grows (v5: ~40 signals, v9: ~98, v18: ~135). Larger rubrics memorize more dev-specific patterns. The generalization-constrained run (v6) actually produced the largest rubric (135 signals) because the LLM worked around the constraints with longer, more elaborate descriptions.

### 8. Embedding similarity provides zero signal for pin/delete

Cosine similarity between content and goal embeddings: pinned mean=0.509, neutral mean=0.475, deleted mean=0.466. Completely overlapping distributions. The pin/delete decision is about quality and directness, not topical proximity.

### 9. Rubrics are coupled to the scorer model they were optimized for

Rubric v5 was discovered and refined with Sonnet as the scorer. When used with Sonnet for BT ranking, it works well (AI Moat pinned at #1-#2). When used with Opus — a more capable model — it actually performs worse (rankings degrade). The rubric encodes instructions calibrated for how Sonnet interprets them; a different model interprets the same rubric differently. This means rubric discovery and scoring must use the same model, and if the production scorer is Haiku, the rubric should be discovered with Haiku too.

## Open Problems

### Pointwise: closing the test gap

Best test macro F1 is 0.644 (v8 outer loop), target is 0.70. The bottleneck is pinned recall (0.20 on test). Possible directions:
- **Binary evaluation** (keep vs delete) may be more realistic — the neutral/pinned boundary is less consequential in production
- **Per-goal-owner rubrics** as more data accumulates — pin decisions are subjective and likely owner-specific
- **More labeled data** — 5 pinned test items means any metric is extremely noisy
- **Rubric pruning** — start from the best rubric and systematically remove signals to find the minimal generalizing set

### Pairwise: validating the 0.74

The 0.74 test accuracy is promising but:
- Mild A-preference bias (10 B→A errors vs 3 A→B, at v5)
- Haven't tested whether the BT rankings downstream are useful in production
- The dev-test gap (0.20) suggests there's still some overfitting
- Should evaluate on ranking metrics (NDCG, Kendall tau) rather than just pairwise accuracy

### Cross-pipeline

- **Can pairwise rankings improve pointwise scoring?** Use BT scores as calibration anchors for pointwise signal strength
- **Active learning**: use the scorer's confidence to identify which new labels would be most informative
- **Goal-type stratification**: abstract goals (AI Moat) consistently score worse — may need different rubric approaches than concrete goals (Conversion)

## Configuration That Works Best

For someone picking this up, the strongest configs so far:

**Pointwise**: Minimal prompts, clean labels, generalization outer loop enabled, ~10-20 rounds. `target_pointwise_f1=0.70`, `generalization_gap_threshold=0.15`, `generalization_guard_rail=0.75`, `generalization_min_rounds=4`.

**Pairwise**: Same outer loop, `target_pairwise_accuracy=0.90`, 10 rounds. Sweet spot is around v5 — after that dev continues but test degrades.
