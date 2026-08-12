"""Assemble the 3×3 results matrix and write RESULTS.md + CSV/JSON.

Two output tables:
  Table A: Within-condition schema rankings (primary reading).
  Table B: Headline 3×3 matrix (down-column comparisons are confounded).
"""
import csv
import json
from pathlib import Path

from common.io import load_pickle_file

from ..instruct_helper import model_supports_temperature
from ..settings import logger, settings
from .aggregator import aggregate, flag_divergent_runs
from .moe_judge import cache_filename
from .rubric import CellAggregate

CONDITIONS = ["unstated", "stated", "mixed"]
SCHEMAS = ["dm", "dsm", "gm"]
DEFAULT_JUDGE_TEMPERATURE = 0.0

_NO_GTX_FRAMING = (
    "this experiment ranks schemas on judge-perceived mapping quality under a no-ground-truth "
    "(no-GTX) protocol. It does NOT measure real-world correctness of individual decision→goal "
    "links. Treat conclusions as schema-fit signals, not as factual claims about Convictional's "
    "organization."
)


def _load_runs(condition: str, schema: str, temperature: float) -> list | None:
    pkl = settings.condition_output_path(condition) / cache_filename(schema, temperature)
    if not pkl.exists():
        return None
    return load_pickle_file(pkl)


def _load_all_aggregates(temperature: float) -> dict[str, CellAggregate]:
    """Load and aggregate all 9 cells at a given temperature. Returns dict keyed by cell_id."""
    aggs = {}
    for cond in CONDITIONS:
        for schema in SCHEMAS:
            cid = f"{cond}__{schema}"
            runs = _load_runs(cond, schema, temperature)
            if runs is None:
                logger.warning(f"No judge runs found for {cid} at T={temperature}")
                continue
            agg = aggregate(runs, cond, schema)
            aggs[cid] = agg
    return aggs


def _temperature_support(aggs: dict[str, CellAggregate]) -> dict[str, bool]:
    """Map each judge model that actually ran to whether it honors a requested temperature.

    Models that reject the temperature parameter (e.g. claude-opus-4-7) run at their API
    default regardless of the requested value — reported here, not evaluated against.
    """
    judge_models = sorted({r.model_id for agg in aggs.values() for r in agg.judge_runs})
    return {m: model_supports_temperature(m) for m in judge_models}


def build_results_matrix(temperature: float = DEFAULT_JUDGE_TEMPERATURE, load_from_cache: bool = True) -> dict:
    """Build the full results, write RESULTS.md + output/results_matrix.{csv,json}.

    Judging happens at a single temperature (the cross-temperature sweep was removed);
    the temperature used and which judge models honor it are reported, not evaluated.
    """
    print(f"Loading judge aggregates at T={temperature}...")
    aggs = _load_all_aggregates(temperature)

    if not aggs:
        raise RuntimeError(
            f"No judge results found at T={temperature}. Run 'judge_all --temperature {temperature}' first."
        )

    temperature_support = _temperature_support(aggs)

    # Load calibration pilot result
    pilot_pkl = settings.output_path / "calibration_pilot.pkl"
    pilot = load_pickle_file(pilot_pkl) if pilot_pkl.exists() else None

    # ── Build output data structures ──────────────────────────────────────────
    cells_data = []
    for cond in CONDITIONS:
        for schema in SCHEMAS:
            cid = f"{cond}__{schema}"
            if cid not in aggs:
                continue
            agg = aggs[cid]
            flagged = flag_divergent_runs(agg.judge_runs)
            cells_data.append({
                "cell_id": cid,
                "condition": cond,
                "schema": schema,
                "trimmed_mean_overall": agg.trimmed_mean_overall,
                "per_dimension_mean": agg.per_dimension_mean,
                "inter_judge_variance": agg.inter_judge_variance,
                "model_decomposition": agg.model_decomposition,
                "role_decomposition": agg.role_decomposition,
                "flagged_divergent_runs": len(flagged),
            })

    # Write CSV
    csv_path = settings.output_path / "results_matrix.csv"
    _write_csv(cells_data, csv_path)

    # Write JSON
    json_path = settings.output_path / "results_matrix.json"
    json_path.write_text(json.dumps({
        "meta": {
            "no_gtx_framing": _NO_GTX_FRAMING,
            "pilot_passed": pilot.passed if pilot else None,
            "judge_temperature": temperature,
            "temperature_support": temperature_support,
        },
        "cells": cells_data,
    }, indent=2))

    # Write RESULTS.md
    md_path = settings.root / "RESULTS.md"
    _write_results_md(cells_data, aggs, pilot, temperature, temperature_support, md_path)

    print(f"\n  Results matrix: {csv_path}")
    print(f"  Results JSON:   {json_path}")
    print(f"  RESULTS.md:     {md_path}")

    return {"cells": cells_data, "pilot": pilot}


def _write_csv(cells_data: list, path: Path) -> None:
    dims = ["coverage", "fidelity", "synthesis_quality", "interpretability", "information_density"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "condition", "schema", "trimmed_mean_overall"] + dims + ["variance"])
        for c in cells_data:
            writer.writerow([
                c["cell_id"], c["condition"], c["schema"], c["trimmed_mean_overall"],
                *[c["per_dimension_mean"].get(d, "") for d in dims],
                c["inter_judge_variance"],
            ])


def _fmt_score(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "—"


def _write_results_md(cells, aggs, pilot, temperature, temperature_support, path: Path) -> None:
    def cell(cond, schema) -> dict | None:
        cid = f"{cond}__{schema}"
        return next((c for c in cells if c["cell_id"] == cid), None)

    def score(cond, schema) -> str:
        c = cell(cond, schema)
        return _fmt_score(c["trimmed_mean_overall"]) if c else "—"

    def within_row_winner(cond: str) -> str:
        row = [(schema, cell(cond, schema)) for schema in SCHEMAS]
        row = [(s, c) for s, c in row if c is not None]
        if not row:
            return "—"
        best = max(row, key=lambda x: x[1]["trimmed_mean_overall"])
        return best[0]

    lines = [
        "# Decisions-to-Goals: Phase 3 Results",
        "",
        "---",
        "",
        "## 1. Experiment Framing",
        "",
        f"> **{_NO_GTX_FRAMING}**",
        "",
        "This experiment evaluates three decision-to-goal mapping schemas under a pure",
        "mixture-of-experts LLM-as-a-judge (MoE LLMaaj) protocol with no external ground truth.",
        "",
        "---",
        "",
        "## 2. Pipeline Summary",
        "",
        "**Phase 1 — Goal Mining** (5 steps per condition):",
        "1. Unstated goal extraction from activity corpus (skipped for `stated` condition)",
        "2. Stated goal validation against activity evidence (no-op for `unstated` condition)",
        "3. Consolidation (embedding-based deduplication)",
        "4. Alignment report (synergies and tensions)",
        "5. Summary and FinalizedGoalSet production",
        "",
        "**Three conditions** (goal-set provenance; decision corpus is identical across all three):",
        "- **`unstated`**: LLM-mined unstated goals only (Step 1 only; models fresh-onboarded company with 0 written goals)",
        "- **`stated`**: Human-written stated goals only (Step 2 only; models company with formal goals, no unstated corpus)",
        "- **`mixed`**: Both — LLM-mined unstated + human-written stated, merged (Steps 1+2+3; models future platform state)",
        "",
        "**Phase 2 — Mapping** (three schemas, all schema-masked, then compressed to fixed-length summaries):",
        "- **Single-goal**: each decision mapped to at most one goal",
        "- **Scored**: each decision scored against all goals (≥0.20 threshold)",
        "- **Graph**: decisions and goals form a labeled relationship graph (8-relation vocabulary; goal↔goal cross-edges only)",
        "",
        "**Obfuscation layer**: each Phase-2 artifact (varying in length 16k–30k words) is compressed by an LLM into a",
        "fixed-length neutral research summary (~450–600 words) before judging, so the judge compares information density",
        "rather than volume. Schema structural differences are deliberately hidden.",
        "",
        "**Phase 3 — Judging**: 3 models × 3 roles = 9 judges per cell. See MoE ensemble below.",
        "",
        "---",
        "",
        "## 3. Calibration Pilot",
        "",
    ]

    if pilot:
        status = "✓ PASSED" if pilot.passed else "✗ FAILED"
        lines += [
            f"**Result: {status}**",
            "",
            "**Check A — Length normalization across schemas:**",
            "",
        ]
        if pilot.summary_word_counts:
            lines += [
                "| Schema | Summary word count |",
                "|--------|-------------------|",
            ]
            for schema, wc in pilot.summary_word_counts.items():
                lines.append(f"| {schema} | {wc} |")
            lines += [
                "",
                f"Length band OK: {pilot.length_band_ok} · Max/min ratio: {pilot.max_pairwise_word_ratio:.2f}",
                "",
            ]
        lines += [
            "**Check B — Padding-bias guard (on summary):**",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Real summary trimmed mean   | {pilot.real_trimmed_mean:.2f} |",
            f"| Padded summary trimmed mean | {pilot.padded_trimmed_mean:.2f} |",
            f"| Delta (padded − real)       | {pilot.delta:.2f} (threshold: {pilot.threshold}) |",
            "",
        ]
        if not pilot.passed:
            lines.append(f"> ⚠ **{pilot.warning_message}**")
            lines.append("")
        else:
            lines.append(
                "Both checks passed: the obfuscation layer normalizes volume across schemas, and the "
                "ensemble's `information_density` dimension is functioning as a length-bias guard."
            )
    else:
        lines += ["*Calibration pilot not yet run. Run `calibration_pilot` first.*", ""]

    lines += [
        "",
        "---",
        "",
        "## 4. Table A — Within-Condition Schema Rankings (Primary Reading)",
        "",
        "> Within-row comparisons are the load-bearing reading. All three conditions use the",
        "> same decision corpus and the same judge ensemble, so schema differences within a row",
        "> are cleanly interpretable.",
        "",
        "| Condition | Best schema | 2nd | 3rd | Margin (1st–3rd) |",
        "|-----------|------------|-----|-----|-----------------|",
    ]

    for cond in CONDITIONS:
        row = sorted(
            [(schema, cell(cond, schema)) for schema in SCHEMAS if cell(cond, schema)],
            key=lambda x: x[1]["trimmed_mean_overall"],
            reverse=True,
        )
        if len(row) < 3:
            continue
        s1, s2, s3 = row[0][0], row[1][0], row[2][0]
        margin = row[0][1]["trimmed_mean_overall"] - row[2][1]["trimmed_mean_overall"]
        lines.append(
            f"| {cond:8} | {s1} ({score(cond,s1)}) | {s2} ({score(cond,s2)}) "
            f"| {s3} ({score(cond,s3)}) | {margin:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Table B — 3×3 Headline Matrix",
        "",
        "> **NOTE:** Down-column comparisons (across conditions) are confounded by goal-set",
        "> size and composition differences across conditions.",
        "> Within-row schema comparisons (Table A) are the load-bearing reading.",
        "",
        "| Condition | Single-goal | Scored | Graph |",
        "|-----------|-------------|--------|-------|",
    ]

    for cond in CONDITIONS:
        lines.append(f"| {cond:8} | {score(cond,'dm'):11} | {score(cond,'dsm'):6} | {score(cond,'gm'):5} |")

    lines += [
        "",
        "---",
        "",
        "## 6. Per-Cell Variance and Decomposition",
        "",
        "Cells with `inter_judge_variance > 1.5` are flagged (high disagreement — judge SD > ~1.2 of 5).",
        "",
        "| Cell | Trimmed mean | Variance | Opus | Sonnet | Haiku | Analyst | Ops | Skeptic |",
        "|------|-------------|----------|------|--------|-------|---------|-----|---------|",
    ]

    for c in cells:
        md_ = c["model_decomposition"]
        rd_ = c["role_decomposition"]
        flag = " ⚠" if c["inter_judge_variance"] > 1.5 else ""
        lines.append(
            f"| {c['cell_id']} | {c['trimmed_mean_overall']:.1f} | "
            f"{c['inter_judge_variance']:.1f}{flag} | "
            f"{_fmt_score(md_.get('claude-opus-4-7'))} | "
            f"{_fmt_score(md_.get('claude-sonnet-4-6'))} | "
            f"{_fmt_score(md_.get('claude-haiku-4-5-20251001'))} | "
            f"{_fmt_score(rd_.get('strategy_analyst'))} | "
            f"{_fmt_score(rd_.get('ops_reviewer'))} | "
            f"{_fmt_score(rd_.get('skeptic'))} |"
        )

    # ── Section 7: Judging temperature (reported, not evaluated) ───────────────
    not_honored = [m for m, ok in temperature_support.items() if not ok]
    lines += [
        "",
        "---",
        "",
        "## 7. Judging Temperature",
        "",
        f"All cells were judged at a single temperature: **T={temperature}**. "
        "Cross-temperature comparison is intentionally not part of this experiment — the "
        "temperature is reported for the record, not treated as an experimental variable.",
        "",
        "| Judge model | Honors requested temperature |",
        "|-------------|------------------------------|",
    ]
    for model in sorted(temperature_support):
        lines.append(f"| {model} | {'yes' if temperature_support[model] else 'no (runs at API default)'} |")
    if not_honored:
        lines += [
            "",
            f"> Note: {', '.join(not_honored)} reject the temperature parameter and run at their API "
            "default regardless of the requested value.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 8. Limitations",
        "",
        "1. **No external ground truth.** All scores reflect judge-perceived quality, not factual correctness.",
        "2. **Correlated judge errors.** Three roles share an underlying LLM architecture; systematic blind spots may affect all.",
        "3. **Cross-condition confounding.** Goal-set sizes differ across conditions; down-column score comparisons (Table B) are not clean.",
        "4. **No retrieval pre-filter.** Phase 1 deliberately omits the embedding+keyword retrieval pre-filter from `linking_tasks_to_goals/approach_10`. That is a candidate enhancement tested in a future experiment, not a flaw here.",
        "5. **Decision↔decision edges absent.** The GM mapper analyzes one decision per call and never has another decision's UUID available, so decision↔decision edges are structurally impossible. Future option: pass a neighbor set to the mapper to enable these edges.",
        "6. **Summarizer overlap with judges.** The obfuscation-layer summarizer uses Claude Sonnet, which is also one of the three judge models. This known overlap is documented in `research_summary.py` but cannot be fully eliminated without a non-Claude summarizer.",
        "",
        "---",
        "",
        "## 9. Next Experiments",
        "",
        "1. **Add the retrieval pre-filter**: replicate this experiment with the `approach_10` keyword+entity+cosine pre-filter enabled and compare mapping quality (specifically coverage and fidelity).",
        "2. **Human rater validation**: have two goal owners rate a sample of decision-to-goal links from the winning schema and compute agreement with the MoE judge scores.",
        "3. **Longitudinal stability**: re-run Phase 1 after a 90-day gap with the same date-bound cutoff to check whether the canonical goal set is stable over time.",
        "4. **Enable decision↔decision edges**: feed the GM mapper a neighbor set so genuine cross-decision relationships can form.",
        "",
    ]

    path.write_text("\n".join(lines))
    print(f"  RESULTS.md written ({len(lines)} lines)")
