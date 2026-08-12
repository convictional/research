# Decisions-to-Goals: Phase 3 Results

---

## 1. Experiment Framing

> **this experiment ranks schemas on judge-perceived mapping quality under a no-ground-truth (no-GTX) protocol. It does NOT measure real-world correctness of individual decision→goal links. Treat conclusions as schema-fit signals, not as factual claims about Convictional's organization.**

This experiment evaluates three decision-to-goal mapping schemas under a pure
mixture-of-experts LLM-as-a-judge (MoE LLMaaj) protocol with no external ground truth.

---

## 2. Pipeline Summary

**Phase 1 — Goal Mining** (5 steps per condition):
1. Unstated goal extraction from activity corpus (skipped for `stated` condition)
2. Stated goal validation against activity evidence (no-op for `unstated` condition)
3. Consolidation (embedding-based deduplication)
4. Alignment report (synergies and tensions)
5. Summary and FinalizedGoalSet production

**Three conditions** (goal-set provenance; decision corpus is identical across all three):
- **`unstated`**: LLM-mined unstated goals only (Step 1 only; models fresh-onboarded company with 0 written goals)
- **`stated`**: Human-written stated goals only (Step 2 only; models company with formal goals, no unstated corpus)
- **`mixed`**: Both — LLM-mined unstated + human-written stated, merged (Steps 1+2+3; models future platform state)

**Phase 2 — Mapping** (three schemas, all schema-masked, then compressed to fixed-length summaries):
- **Single-goal**: each decision mapped to at most one goal
- **Scored**: each decision scored against all goals (≥0.20 threshold)
- **Graph**: decisions and goals form a labeled relationship graph (8-relation vocabulary; goal↔goal cross-edges only)

**Obfuscation layer**: each Phase-2 artifact (varying in length 16k–30k words) is compressed by an LLM into a
fixed-length neutral research summary (~450–600 words) before judging, so the judge compares information density
rather than volume. Schema structural differences are deliberately hidden.

**Phase 3 — Judging**: 3 models × 3 roles = 9 judges per cell. See MoE ensemble below.

---

## 3. Calibration Pilot

**Result: ✓ PASSED**

**Check A — Length normalization across schemas:**

| Schema | Summary word count |
|--------|-------------------|
| dm | 598 |
| dsm | 607 |
| gm | 646 |

Length band OK: True · Max/min ratio: 1.08

**Check B — Padding-bias guard (on summary):**

| Metric | Value |
|--------|-------|
| Real summary trimmed mean   | 4.29 |
| Padded summary trimmed mean | 4.00 |
| Delta (padded − real)       | -0.29 (threshold: 0.2) |

Both checks passed: the obfuscation layer normalizes volume across schemas, and the ensemble's `information_density` dimension is functioning as a length-bias guard.

---

## 4. Table A — Within-Condition Schema Rankings (Primary Reading)

> Within-row comparisons are the load-bearing reading. All three conditions use the
> same decision corpus and the same judge ensemble, so schema differences within a row
> are cleanly interpretable.

| Condition | Best schema | 2nd | 3rd | Margin (1st–3rd) |
|-----------|------------|-----|-----|-----------------|
| unstated | gm (4.9) | dm (4.7) | dsm (4.6) | 0.3 |
| stated   | dm (4.9) | gm (4.9) | dsm (4.3) | 0.6 |
| mixed    | dm (4.9) | dsm (4.3) | gm (4.3) | 0.6 |

---

## 5. Table B — 3×3 Headline Matrix

> **NOTE:** Down-column comparisons (across conditions) are confounded by goal-set
> size and composition differences across conditions.
> Within-row schema comparisons (Table A) are the load-bearing reading.

| Condition | Single-goal | Scored | Graph |
|-----------|-------------|--------|-------|
| unstated | 4.7         | 4.6    | 4.9   |
| stated   | 4.9         | 4.3    | 4.9   |
| mixed    | 4.9         | 4.3    | 4.3   |

---

## 6. Per-Cell Variance and Decomposition

Cells with `inter_judge_variance > 1.5` are flagged (high disagreement — judge SD > ~1.2 of 5).

| Cell | Trimmed mean | Variance | Opus | Sonnet | Haiku | Analyst | Ops | Skeptic |
|------|-------------|----------|------|--------|-------|---------|-----|---------|
| unstated__dm | 4.7 | 0.5 | 5.0 | 4.3 | 4.3 | 4.7 | 4.3 | 4.7 |
| unstated__dsm | 4.6 | 0.5 | 4.7 | 4.3 | 4.3 | 4.7 | 4.0 | 4.7 |
| unstated__gm | 4.9 | 0.5 | 5.0 | 5.0 | 4.0 | 5.0 | 4.7 | 4.3 |
| stated__dm | 4.9 | 1.0 | 5.0 | 5.0 | 3.7 | 5.0 | 4.7 | 4.0 |
| stated__dsm | 4.3 | 2.0 ⚠ | 5.0 | 4.7 | 2.3 | 4.3 | 4.3 | 3.3 |
| stated__gm | 4.9 | 0.5 | 5.0 | 4.7 | 4.3 | 5.0 | 4.0 | 5.0 |
| mixed__dm | 4.9 | 0.5 | 5.0 | 4.7 | 4.3 | 4.7 | 4.3 | 5.0 |
| mixed__dsm | 4.3 | 0.9 | 4.7 | 5.0 | 3.0 | 4.3 | 4.3 | 4.0 |
| mixed__gm | 4.3 | 0.7 | 5.0 | 4.3 | 3.3 | 4.0 | 4.0 | 4.7 |

---

## 7. Judging Temperature

All cells were judged at a single temperature: **T=0.0**. Cross-temperature comparison is intentionally not part of this experiment — the temperature is reported for the record, not treated as an experimental variable.

| Judge model | Honors requested temperature |
|-------------|------------------------------|
| claude-haiku-4-5-20251001 | yes |
| claude-opus-4-7 | no (runs at API default) |
| claude-sonnet-4-6 | yes |

> Note: claude-opus-4-7 reject the temperature parameter and run at their API default regardless of the requested value.

---

## 8. Limitations

1. **No external ground truth.** All scores reflect judge-perceived quality, not factual correctness.
2. **Correlated judge errors.** Three roles share an underlying LLM architecture; systematic blind spots may affect all.
3. **Cross-condition confounding.** Goal-set sizes differ across conditions; down-column score comparisons (Table B) are not clean.
4. **No retrieval pre-filter.** Phase 1 deliberately omits the embedding+keyword retrieval pre-filter from `linking_tasks_to_goals/approach_10`. That is a candidate enhancement tested in a future experiment, not a flaw here.
5. **Decision↔decision edges absent.** The GM mapper analyzes one decision per call and never has another decision's UUID available, so decision↔decision edges are structurally impossible. Future option: pass a neighbor set to the mapper to enable these edges.
6. **Summarizer overlap with judges.** The obfuscation-layer summarizer uses Claude Sonnet, which is also one of the three judge models. This known overlap is documented in `research_summary.py` but cannot be fully eliminated without a non-Claude summarizer.

---

## 9. Next Experiments

1. **Add the retrieval pre-filter**: replicate this experiment with the `approach_10` keyword+entity+cosine pre-filter enabled and compare mapping quality (specifically coverage and fidelity).
2. **Human rater validation**: have two goal owners rate a sample of decision-to-goal links from the winning schema and compute agreement with the MoE judge scores.
3. **Longitudinal stability**: re-run Phase 1 after a 90-day gap with the same date-bound cutoff to check whether the canonical goal set is stable over time.
4. **Enable decision↔decision edges**: feed the GM mapper a neighbor set so genuine cross-decision relationships can form.
