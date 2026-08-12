# Train Research Report Judge (LLM-as-a-Judge)

An experiment to train an automated judge that replicates expert quality scoring (0–3 scale) of research reports. The scorer discovers quality rubrics from expert-labeled data, then iteratively refines them through disagreement analysis.

**Status**: Complete — 11 trials exhausted. The scorer reliably matches the expert's score *distribution* (adjacent match 0.95) but cannot predict which *specific* reports are good or bad (best Spearman 0.326, target ≥0.7). Prompt engineering, ensemble voting, claim decomposition, metadata injection, and RAG-based source verification all failed to improve discrimination. The most promising next direction is pairwise ranking — we have a strong A/B dataset from human eval outputs that could support this.

## Core blocker

The scorer can't reliably discriminate score 1 ("adequate with gaps") from score 2 ("adequate with evidence"). These reports look structurally identical — similar length, citation density, section organization. The distinguishing factor appears to be **factual correctness and contextual judgment**: the expert brings domain knowledge about whether claims accurately represent the underlying situation, whether recommendations are actually relevant, whether the analysis addresses the real question vs a superficial reading of it.

The LLM can evaluate surface features (structure, coherence, Q-A alignment, claim specificity) but cannot replicate the expert's contextual judgment from the report text alone. 11 trials confirmed this as a ceiling for pointwise scoring approaches — whether prompt-only or RAG-augmented. Ensemble voting, claim-level decomposition, contrastive calibration pairs, metadata injection, and source-level fact verification all failed to improve discrimination.

Trial 11 tested whether RAG verification could break through this ceiling. It couldn't: verification verdicts don't correlate with expert scores because all reports draw from the same source material and are roughly equally "factually grounded." The expert discriminates on reasoning quality and analytical depth, not factual accuracy.

## Dataset

399 research reports scored 0–3 by expert, split 60/20/20 stratified by score × variant:

| Score | Meaning | Count | % |
|-------|---------|-------|---|
| 0 | Inadequate | 54 | 14% |
| 1 | Adequate with gaps | 156 | 39% |
| 2 | Adequate with evidence | 169 | 42% |
| 3 | Exceptional | 20 | 5% |

Train=238, Dev=80, Test=81.

## Directory structure

```
experiments/train_research_report_judge/
├── __init__.py
├── __main__.py                          # Entry point
├── src/
│   ├── main.py                          # CLI dispatcher + orchestration
│   ├── settings.py                      # Pydantic settings (model, paths, thresholds)
│   ├── models.py                        # Data models (RatedReport, rubric schemas, LLM response models)
│   ├── data_loader.py                   # CSV loading + stratified train/dev/test splitting
│   ├── rubric_discovery.py              # Rubric creation from training data + iterative refinement
│   ├── pointwise_scorer.py              # Two-pass scoring: critic → scorer, with ensemble/claims
│   ├── evaluator.py                     # Metrics: Spearman, MAE, exact/adjacent match
│   ├── disagreement_analyzer.py         # Finds worst scorer–expert disagreements for refinement
│   ├── content_search.py                # Hybrid search (pgvector + text) against content DB (Trial 11)
│   ├── claim_verifier.py                # Per-claim RAG verification with Haiku (Trial 11)
│   ├── format_scorer.py                 # Surface quality assessment with Haiku (Trial 11)
│   ├── rag_scorer.py                    # RAG scorer orchestration: extract → verify → format → score (Trial 11)
│   └── prompts/
│       ├── pointwise_scorer_system.txt.jinja   # Scorer system (rubric + calibration + distribution)
│       ├── pointwise_scorer_user.txt.jinja     # Scorer user (report + critique + claims)
│       ├── critic_system.txt.jinja             # Harsh critic (no rubric access, finds weaknesses)
│       ├── claim_analysis_system.txt.jinja     # Claim extraction + classification
│       ├── claim_analysis_user.txt.jinja
│       ├── qa_alignment_gate_system.txt.jinja  # Q-A alignment gate (deprecated, Trials 1-3)
│       ├── qa_alignment_gate_user.txt.jinja
│       ├── rubric_discovery_system.txt.jinja   # Batch analysis for rubric discovery
│       ├── rubric_discovery_user.txt.jinja
│       ├── rubric_synthesis_user.txt.jinja     # Multi-batch rubric synthesis
│       ├── rubric_refinement_user.txt.jinja    # Rubric refinement from disagreements
│       ├── disagreement_analysis_system.txt.jinja
│       ├── disagreement_analysis_user.txt.jinja
│       ├── claim_verification_system.txt.jinja # Per-claim evidence verification (Trial 11)
│       ├── claim_verification_user.txt.jinja
│       ├── format_scorer_system.txt.jinja      # Surface quality assessment (Trial 11)
│       ├── format_scorer_user.txt.jinja
│       ├── rag_final_scorer_system.txt.jinja   # Final score from structured signals (Trial 11)
│       └── rag_final_scorer_user.txt.jinja
├── data/
│   └── processed/
│       ├── train.csv
│       ├── dev.csv
│       └── test.csv
└── output/
    ├── rubric/                          # rubric_v1.json through rubric_v4.json
    ├── results/                         # eval_dev_v1.json, eval_test_v2.json, etc.
    └── service/                         # Exported rubric + prompts for production use
```

## How to run

All commands run from the `experiments/` directory using `make run_experiment`. Requires `ANTHROPIC_API_KEY` in `.env` or `.env.secrets`. Raw data goes in `tmp/research_output_evals.csv`.

```bash
cd experiments/

# Full automated pipeline (load → discover rubric → iterate → test)
make run_experiment ARGS="train_research_report_judge full_pipeline"

# Or step by step:
make run_experiment ARGS="train_research_report_judge load_data"
make run_experiment ARGS="train_research_report_judge discover_rubric --version 1"
make run_experiment ARGS="train_research_report_judge evaluate_scorer --split dev --rubric-version 1"
make run_experiment ARGS="train_research_report_judge analyze_disagreements --split dev --rubric-version 1"

# Score a single report
make run_experiment ARGS="train_research_report_judge score --query 'What are best practices for X?' --report-file /path/to/report.md"

# Export scorer config for production
make run_experiment ARGS="train_research_report_judge export_service --rubric-version 4"
```

### Trial 11: RAG scorer

Requires additional env vars: `OPENAI_API_KEY`, `ORGANIZATION_ID`, `POSTGRES_URL` (e.g. `asyncpg://decide:@localhost:5432/decide_yourdb`). The database must have pgvector and a populated `content` table.

```bash
make run_experiment ARGS="train_research_report_judge evaluate_rag_scorer --split dev"
```

### CLI flags for trial variants

```bash
# Ensemble scoring (Trial 8)
... evaluate_scorer --ensemble-n 5

# Disable claim analysis (default: enabled since Trial 9)
... evaluate_scorer --no-claims

# Disable metadata + contrastive pairs (default: enabled since Trial 10)
... evaluate_scorer --no-metadata
```

## Trial history

### Phase 1: Solving calibration (Run 1 + Trials 1–7c)

The initial problem was massive over-scoring — the LLM predicted score 3 for 46–67 of 80 dev reports when the expert only gave 3 to 4.

**Run 1 (rubric refinement, 4 iterations)**: Discovered a 5-dimension rubric from training data, then iteratively refined it via disagreement analysis. Four rounds produced diminishing returns. Score-3 over-prediction barely budged (50→46→46→42). Best Spearman was 0.303 (v2).

**Trial 1 (rich calibration + Q-A gate)**: Added 8 full-text calibration examples and a separate Q-A alignment gate LLM call that caps the final score. Gate gave score 3 to 69/80 reports — too lenient. Full-text examples bloated the prompt. Everything got worse.

**Trial 2 (stricter gate prompt)**: Rewrote Q-A gate with distribution table and "default to 1" guidance. Gate started discriminating (38 score-3s, down from 69). MAE improved to 1.0. Still far from targets.

**Trial 3 (custom gate scale)**: Replaced rubric-based gate with a stricter custom scale. Backfired — removing rubric anchors made the gate more lenient (62/80 score-3). Custom scales without examples don't constrain LLMs.

**Trial 4 (weakness-first scoring, no gate)**: Dropped the gate. Added structured fields forcing the scorer to identify weaknesses *before* assigning a score. Score-1 prediction nearly perfect (29 predicted vs 31 actual). But bimodal: lots of 1s and 3s, almost no 2s.

**Trial 5 (consistency rules)**: Added rules like "if you listed ANY weakness, you CANNOT give score 3." The model game-planned — it generated fewer weaknesses to justify higher scores.

**Trial 7 (decoupled critic + scorer)**: The breakthrough for calibration. Two separate LLM passes: (1) a deliberately harsh critic with no rubric access finds weaknesses, (2) the scorer receives the critique alongside calibration and distribution guidance. Adjacent match jumped to 0.950, MAE dropped to 0.588. But the harsh critic over-suppressed — zero score-3 predictions.

**Trial 7c (softened severity mapping)**: Told the scorer the critic is deliberately harsh, softened severity→score mapping. Best calibration achieved: adjacent match 0.925, MAE 0.600, exact match 0.475, distribution shape matches expert. Spearman stuck at 0.237.

### Phase 2: Improving discrimination (Trials 8–10)

With calibration solved, the remaining problem was rank correlation (Spearman ~0.24 vs target ≥0.7).

**Trial 8 (ensemble N=5)**: Ran 5 scoring passes at temp=0.5, aggregated via median. No improvement — errors are systematic, not stochastic. Voting over a consistently biased estimator reproduces the bias.

**Trial 9 (claim-level decomposition)**: Added a claim analysis pass that extracts individual claims and classifies them for specificity, hedging, citation support, and relevance. Passed claim stats to the scorer. Spearman ticked up marginally (0.246 int, 0.264 continuous) but MAE got slightly worse. Claim quality is a directionally correct but weak proxy for the expert's contextual judgment.

**Trial 10 (ceiling assessment)**: Stacked everything — ensemble + claims + variant/community metadata + contrastive calibration pairs (3 length-matched score-1 vs score-2 reports in system prompt). Worst trial yet for MAE (0.763) and exact match (0.350). The system prompt grew too large; contrastive pairs diluted the core scoring instructions.

### Phase 3: RAG-based verification (Trial 11)

**Trial 11 (claim-level RAG verification)**: A fundamentally different scoring architecture. Instead of evaluating the report as a whole, decompose it into claims and verify each one individually against retrieved source content via hybrid search (pgvector + text). The pipeline:

1. **Claim extraction** (Sonnet) — reuses existing `_analyze_claims`, extracts 10–15 key claims
2. **Per-claim RAG verification** (parallel Haiku) — for each claim, hybrid search the content DB for evidence, then Haiku determines: supported / partially supported / unsupported / no evidence found
3. **Format scoring** (Haiku) — evaluates surface quality: structure, length, tone, Q-A alignment
4. **Final scoring** (Sonnet) — takes claim verification roll-up + format assessment → quality score 0–3. Does NOT see the full report text — forced to use the structured signals.

**Hypothesis**: Even though reports are generated from the same source material, the verification step is different from generation. A claim verifier can catch fabrications, distortions, and unsupported assertions that are invisible to a scorer reading the report alone.

**Cost**: ~$2–3 for the full dev set (80 reports × ~12 claims × embedding + Haiku calls + Sonnet calls).

**Result (standalone)**: Spearman 0.232, MAE 0.613, adjacent 0.938. No improvement over Trial 7c. Diagnostics revealed that verification verdicts don't correlate with expert scores — supported% was 33→21→47→40 across expert scores 0→1→2→3. "Partially supported" acts as a catch-all that collapses signal. The format scorer gives near-perfect marks to everything scored 1+.

**Trial 11b (hybrid — RAG fed into Trial 7c pipeline)**: Instead of replacing the critic+scorer pipeline, feed the RAG verification roll-up as a supplementary signal into the existing pointwise scorer prompt. Spearman 0.258, MAE 0.613, adjacent 0.950. Marginal Spearman bump (+0.02) — noise-level.

**Conclusion**: Source-level fact verification doesn't provide discriminative signal for this task. The expert evaluates reasoning quality and analytical depth, not factual grounding. All reports draw from the same source material, so they're all roughly equally "factually grounded." The verification step, despite being a different task than generation, doesn't capture what the expert cares about.

## Results

### All trials (dev set, n=80)

| Trial | Config | Spearman | MAE | Adj. match | Exact |
|-------|--------|----------|-----|------------|-------|
| Run 1 v1 | Single-pass, rubric v1 | 0.294 | 1.213 | 0.663 | 0.163 |
| Run 1 v2 | Rubric v2 | 0.303 | 1.163 | 0.700 | 0.175 |
| Run 1 v3 | Rubric v3 | 0.251 | 1.175 | 0.663 | 0.188 |
| Run 1 v4 | Rubric v4 | 0.263 | 1.088 | 0.700 | 0.250 |
| Trial 1 | Rich calibration + Q-A gate | 0.248 | 1.325 | 0.600 | — |
| Trial 2 | Stricter gate prompt | 0.276 | 1.000 | 0.750 | — |
| Trial 3 | Custom gate scale | 0.326 | 1.250 | 0.600 | — |
| Trial 4 | Weakness-first, no gate | 0.181 | 1.025 | 0.763 | — |
| Trial 5 | Consistency rules | 0.260 | 1.138 | 0.700 | — |
| Trial 7 | Decoupled critic + scorer | 0.276 | 0.588 | **0.950** | 0.463 |
| Trial 7c | Softened severity | 0.237 | 0.600 | 0.925 | 0.475 |
| Trial 8 | Ensemble N=5, temp=0.5 | 0.230 | 0.675 | 0.925 | 0.413 |
| Trial 9 | + Claim analysis | 0.246 | 0.688 | 0.913 | 0.413 |
| Trial 10 | + Metadata + contrastive pairs | 0.235 | 0.763 | 0.888 | 0.350 |
| Trial 11 | Claim-level RAG verification | 0.232 | 0.613 | 0.938 | 0.438 |
| Trial 11b | RAG verification → Trial 7c hybrid | 0.258 | 0.613 | 0.950 | 0.438 |

### Targets

| Metric | Target | Best achieved | Trial |
|--------|--------|---------------|-------|
| Spearman | ≥ 0.7 | 0.326 | Trial 3 (but MAE was 1.25) |
| MAE | ≤ 0.5 | 0.588 | Trial 7 |
| Adjacent match | ≥ 0.85 | 0.950 | Trial 7 |

No trial simultaneously met all three targets. The calibration–discrimination tradeoff was never resolved.

## Key lessons

**What worked (for calibration):**
- Decoupled critic + scorer (Trial 7) was the single most impactful change — prevents the scorer from gaming its own weakness analysis
- Distribution awareness with explicit percentages constrains the LLM more than vague guidance
- Fewer, truncated calibration examples (5 at 2000 chars) outperform more full-text examples (8 at unlimited)
- Weakness-first structured output correctly identifies low-quality reports

**What didn't work (for discrimination):**
- Q-A alignment gate as a separate LLM call (Trials 1–3) — consistently too lenient regardless of prompt strictness
- Consistency rules linking weaknesses to scores (Trial 5) — LLM game-plans the weakness list
- Ensemble voting (Trial 8) — doesn't help when errors are systematic
- Claim-level decomposition (Trial 9) — weak proxy for what the expert actually evaluates
- Adding more context to the prompt (Trial 10) — degrades signal-to-noise ratio
- Rubric refinement across 4 iterations (Run 1) — diminishing returns, language gets more precise but behavior barely changes
- RAG-based source verification (Trial 11/11b) — verification verdicts don't correlate with expert scores; all reports are roughly equally factually grounded since they draw from the same sources

## Current status

The Spearman blocker (~0.24) means the scorer can't predict which specific reports are good or bad — only that roughly the right *proportion* should fall in each bucket. 11 trials confirmed this as a ceiling for pointwise scoring — the task of assigning an absolute 0–3 score to a report in isolation.

**Why fine-tuning likely won't help**: Fine-tuning gives the model the same information as in-context learning, just encoded differently. The problem isn't that the model can't follow instructions — it matches the distribution perfectly. The problem is that the discriminative signal the expert uses (reasoning quality, analytical depth, contextual judgment) isn't extractable from the report text alone via any pointwise evaluation. Fine-tuning would likely overfit to surface correlations in the training set without improving generalization.

**Pairwise ranking — future direction**: The pointwise task asks "how good is this report on an absolute scale?" — a question that requires the expert's domain context to answer. A pairwise task asks "which of these two reports is better?" — a fundamentally easier judgment that even surface-level features can support. Pairwise preferences can then be aggregated into a ranking via Bradley-Terry or similar, potentially recovering the ordinal signal that pointwise scoring couldn't capture.

We investigated existing A/B human eval data (165 ratings, 17 raters, 6 experiments comparing pipeline variants). The data has the right structure — same question, two report variants, human picks a winner — but only 40 unique (question, answer_A, answer_B) cases, which is too thin for reliable train/dev splits. Additionally, inter-rater disagreement patterns need to be understood before using the labels (median 4 raters per case, 16% tie rate). This is worth revisiting once we've collected more unique pairs and developed a strategy for aggregating multi-rater preferences.
