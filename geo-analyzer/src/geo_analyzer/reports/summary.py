"""Assemble summary.md from analytical sections.

Pure function over typed inputs (Run + tasks + scores + catalog + goals + today).
The CLI reads disks and calls this; the renderer never touches I/O.

Layout:
  # Run <id> (status, wall-time)
  ## TL;DR              (one bullet per subject — the headline)
  ## <subject 1>        (per-subject narrative + per-model table)
  ## <subject 2>
  ...
  ## Goal progress
  ## Grounded vs ungrounded gap
  ## Cost & runtime
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from geo_analyzer.reports.funnel import compute_funnel, render_sparkline
from geo_analyzer.reports.goals import GoalStatus, evaluate_goal
from geo_analyzer.reports.grounded_gap import compute_grounded_gaps
from geo_analyzer.runtime import Run, Score, Task, TaskStatus
from geo_analyzer.types import Catalog, Goal, Subject, SubjectKind


@dataclass(frozen=True)
class SummaryInputs:
    run: Run
    tasks: list[Task]
    scores: list[Score]
    catalog: Catalog
    goals: list[Goal]
    today: date


def render_summary(inputs: SummaryInputs) -> str:
    reported = _reported_subjects(inputs)
    sections: list[str] = []
    sections.append(_header(inputs.run))
    sections.append(_tldr(inputs, reported))
    for subject in reported:
        sections.append(_subject_section(subject, inputs))
    sections.append(_goals_section(inputs))
    sections.append(_grounded_gap_section(inputs))
    sections.append(_cost_section(inputs))
    sections.append(_prompts_by_tier_section(inputs))
    return "\n\n".join(s for s in sections if s).rstrip() + "\n"


def _reported_subjects(inputs: SummaryInputs) -> list[Subject]:
    """Subjects worth narrating: appear in some prompt's targets, or have any score row.

    Filters out alias-only catalog subjects (e.g. competitor brands carried only
    so SoV and conflation know about them) — they'd otherwise show up as empty
    'no data' bullets and headerless sections.
    """
    targeted = {t for p in inputs.catalog.prompts for t in p.targets}
    scored = {s.subject_id for s in inputs.scores}
    return [s for s in inputs.catalog.subjects if s.id in targeted or s.id in scored]


# --- header ---------------------------------------------------------------


def _header(run: Run) -> str:
    elapsed = ""
    if run.finished_at and run.started_at:
        delta = (run.finished_at - run.started_at).total_seconds()
        elapsed = f" • {_format_duration(delta)}"
    return f"# Run {run.id} ({run.status.value}{elapsed})"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h{m:02d}m"


# --- TL;DR ----------------------------------------------------------------


def _tldr(inputs: SummaryInputs, subjects: list[Subject]) -> str:
    """Compact table: one row per reported subject, one column per tier.

    Each cell is `k/n` — fired cohorts over total cohorts at that tier — so the
    reader can see the funnel shape (does signal land at L1 or only L4?) at a
    glance, instead of a single rolled-up percentage that hides everything.
    """
    rows: list[str] = []
    for subject in subjects:
        row = _subject_tldr_row(subject, inputs)
        if row:
            rows.append(row)
    if not rows:
        return ""
    header = ["| subject | metric | L1 | L2 | L3 | L4 |", "|---|---|---|---|---|---|"]
    return "## TL;DR\n\n" + "\n".join(header + rows)


def _subject_tldr_row(subject: Subject, inputs: SummaryInputs) -> str:
    """One row of the TL;DR table — per-tier `k/n` for this subject's metric."""
    if subject.kind in (SubjectKind.BRAND, SubjectKind.CATEGORY):
        metric = "mention_presence"
        label = "category awareness" if subject.kind == SubjectKind.CATEGORY else "brand mention"
    else:
        metric = "brand_legacy_conflation"
        label = "conflation (want down)"
    by_tier = _binary_counts_by_tier(inputs, subject_id=subject.id, metric=metric)
    cells: list[str] = []
    for tier in ("L1", "L2", "L3", "L4"):
        bucket = by_tier.get(tier, [])
        if not bucket:
            cells.append("--")
            continue
        k = sum(1 for v in bucket if v)
        cells.append(f"{k}/{len(bucket)}")
    if all(c == "--" for c in cells):
        return ""
    return f"| `{subject.id}` | {label} | " + " | ".join(cells) + " |"


# --- per-subject sections -------------------------------------------------


def _subject_section(subject: Subject, inputs: SummaryInputs) -> str:
    """Per-subject narrative: tier funnel + per-model breakdown + worst prompts."""
    parts: list[str] = [f"## {subject.id}", ""]

    if subject.kind == SubjectKind.ANTI_BRAND:
        parts.extend(_anti_brand_body(subject, inputs))
    else:
        parts.extend(_brand_or_category_body(subject, inputs))

    return "\n".join(parts).rstrip()


def _brand_or_category_body(subject: Subject, inputs: SummaryInputs) -> list[str]:
    parts: list[str] = []

    funnel = compute_funnel(inputs.scores, inputs.catalog, subject_id=subject.id)
    spark = render_sparkline([t.rate for t in funnel])
    tiers_str = "  ".join(f"{t.tier}={_pct(t.rate)}({t.n})" for t in funnel)
    parts.append(f"**Funnel** (mention rate by tier): `{spark}`  {tiers_str}")
    parts.append("")

    table = _per_model_rate_table(inputs, subject_id=subject.id, metric_binary="mention_presence")
    if table:
        parts.append("**Per-model mention rate:**")
        parts.append("")
        parts.append(table)
        parts.append("")

    rank_mean = _mean_metric(inputs.scores, subject.id, "ordinal_rank")
    if rank_mean is not None:
        parts.append(f"**Mean ordinal rank** (when in a list): {rank_mean:.2f}")
        parts.append("")

    sov_note = _share_of_voice_summary(inputs.scores, subject.id, inputs.catalog)
    if sov_note:
        parts.append(sov_note)
        parts.append("")

    prompt_table = _per_prompt_rate_table(inputs, subject_id=subject.id, metric_binary="mention_presence")
    if prompt_table:
        parts.append("**Per-prompt mention rate** (cohorts that mentioned this subject, by tier and mode):")
        parts.append("")
        parts.append(prompt_table)
        parts.append("")

    return parts


def _anti_brand_body(subject: Subject, inputs: SummaryInputs) -> list[str]:
    parts: list[str] = []

    parts.append(_anti_brand_funnel(subject, inputs))
    parts.append("")

    table = _per_model_rate_table(inputs, subject_id=subject.id, metric_binary="brand_legacy_conflation")
    if table:
        parts.append("**Per-model conflation rate:**")
        parts.append("")
        parts.append(table)
        parts.append("")

    prompt_table = _per_prompt_rate_table(inputs, subject_id=subject.id, metric_binary="brand_legacy_conflation")
    if prompt_table:
        parts.append(
            "**Per-prompt conflation rate** "
            "(cohorts where brand and legacy term co-occurred, by tier and mode):"
        )
        parts.append("")
        parts.append(prompt_table)
        parts.append("")

    return parts


def _prompts_by_tier_section(inputs: SummaryInputs) -> str:
    """Reference section: every prompt's full text, grouped by tier."""
    if not inputs.catalog.prompts:
        return ""
    tier_label = {
        "L1": "L1 (broadest)",
        "L2": "L2 (category-adjacent)",
        "L3": "L3 (category-named)",
        "L4": "L4 (brand-named)",
    }
    by_tier: dict[str, list[tuple[str, str]]] = {"L1": [], "L2": [], "L3": [], "L4": []}
    for p in inputs.catalog.prompts:
        by_tier[p.tier].append((p.id, p.text))
    lines = ["## Prompts (by tier)"]
    for tier in ("L1", "L2", "L3", "L4"):
        bucket = by_tier.get(tier, [])
        if not bucket:
            continue
        lines.append("")
        lines.append(f"### {tier_label[tier]}")
        for pid, text in sorted(bucket):
            lines.append(f"- `{pid}` — {text}")
    return "\n".join(lines)


def _anti_brand_funnel(subject: Subject, inputs: SummaryInputs) -> str:
    """Conflation rate by tier."""
    prompt_tier = {p.id: p.tier for p in inputs.catalog.prompts}
    by_tier: dict[str, list[bool]] = defaultdict(list)
    for s in inputs.scores:
        if s.subject_id != subject.id or s.metric != "brand_legacy_conflation":
            continue
        if not isinstance(s.value, bool):
            continue
        tier = prompt_tier.get(s.prompt_id)
        if tier is None:
            continue
        by_tier[tier].append(s.value)
    rates: list[float | None] = []
    tier_strs: list[str] = []
    for tier in ("L1", "L2", "L3", "L4"):
        bucket = by_tier.get(tier, [])
        if bucket:
            rate = sum(1 for v in bucket if v) / len(bucket)
            rates.append(rate)
            tier_strs.append(f"{tier}={_pct(rate)}({len(bucket)})")
        else:
            rates.append(None)
            tier_strs.append(f"{tier}=-(0)")
    spark = render_sparkline(rates)
    return f"**Conflation by tier**: `{spark}`  " + "  ".join(tier_strs)


# --- per-model breakdown table -------------------------------------------


def _per_model_rate_table(
    inputs: SummaryInputs,
    *,
    subject_id: str,
    metric_binary: str,
) -> str:
    """Markdown table: rows = model_name, cols = ungrounded | grounded."""
    buckets: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for s in inputs.scores:
        if s.subject_id != subject_id or s.metric != metric_binary:
            continue
        if not isinstance(s.value, bool):
            continue
        sm = _stem_and_mode(s.model_id)
        if sm is None:
            continue
        buckets[sm].append(s.value)
    if not buckets:
        return ""

    model_names = sorted({stem for stem, _ in buckets})
    lines = ["| model | ungrounded | grounded |", "|---|---|---|"]
    for stem in model_names:
        u = buckets.get((stem, "ungrounded"), [])
        g = buckets.get((stem, "grounded"), [])
        u_str = _pct(sum(u) / len(u)) if u else "—"
        g_str = _pct(sum(g) / len(g)) if g else "—"
        lines.append(f"| `{stem}` | {u_str} | {g_str} |")
    return "\n".join(lines)


def _stem_and_mode(model_id: str) -> tuple[str, str] | None:
    parts = model_id.rsplit(":", 1)
    if len(parts) != 2 or parts[1] not in ("grounded", "ungrounded"):
        return None
    return parts[0], parts[1]


# --- per-prompt breakdown table ------------------------------------------


def _per_prompt_rate_table(
    inputs: SummaryInputs,
    *,
    subject_id: str,
    metric_binary: str,
) -> str:
    """Markdown table: rows = (tier, prompt), cols = ungrounded | grounded.

    Each cell is `k/n` — cohorts where the metric fired over total cohorts of
    that mode for the prompt. This is the per-prompt detail beneath the tier
    funnel: shows which specific prompts pull weight (and which never do).
    """
    buckets: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for s in inputs.scores:
        if s.subject_id != subject_id or s.metric != metric_binary:
            continue
        if not isinstance(s.value, bool):
            continue
        sm = _stem_and_mode(s.model_id)
        if sm is None:
            continue
        _, mode = sm
        buckets[(s.prompt_id, mode)].append(s.value)
    if not buckets:
        return ""

    prompt_tier = {p.id: p.tier for p in inputs.catalog.prompts}
    prompt_text = {p.id: p.text for p in inputs.catalog.prompts}
    seen_prompts = sorted({pid for pid, _ in buckets})
    tier_order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}

    rows: list[tuple[int, str, str, str, str, str]] = []
    for pid in seen_prompts:
        tier = prompt_tier.get(pid, "?")
        u = buckets.get((pid, "ungrounded"), [])
        g = buckets.get((pid, "grounded"), [])
        u_cell = f"{sum(u)}/{len(u)}" if u else "--"
        g_cell = f"{sum(g)}/{len(g)}" if g else "--"
        slug = pid.rsplit(".", 1)[-1]
        text = prompt_text.get(pid, "").strip().replace("|", "\\|").replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        rows.append((tier_order.get(tier, 9), tier, slug, u_cell, g_cell, text))

    rows.sort()
    lines = ["| tier | prompt | ungrounded | grounded | text |", "|---|---|---|---|---|"]
    for _, tier, slug, u_cell, g_cell, text in rows:
        lines.append(f"| {tier} | `{slug}` | {u_cell} | {g_cell} | {text} |")
    return "\n".join(lines)


def _binary_counts_by_tier(
    inputs: SummaryInputs,
    *,
    subject_id: str,
    metric: str,
) -> dict[str, list[bool]]:
    """Group cohort-level binary verdicts by prompt tier."""
    prompt_tier = {p.id: p.tier for p in inputs.catalog.prompts}
    by_tier: dict[str, list[bool]] = defaultdict(list)
    for s in inputs.scores:
        if s.subject_id != subject_id or s.metric != metric:
            continue
        if not isinstance(s.value, bool):
            continue
        tier = prompt_tier.get(s.prompt_id)
        if tier is None:
            continue
        by_tier[tier].append(s.value)
    return by_tier


# --- aggregation helpers --------------------------------------------------


def _mean_metric(scores: list[Score], subject_id: str, metric: str) -> float | None:
    matched = [s for s in scores if s.subject_id == subject_id and s.metric == metric]
    nums = [float(s.value) for s in matched if isinstance(s.value, int | float) and not isinstance(s.value, bool)]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _share_of_voice_summary(scores: list[Score], subject_id: str, catalog: Catalog) -> str:
    """Short SoV note, or '' if uninformative this run.

    Uninformative = every value is exactly 1.0. Two distinct causes:
    (a) the subject declares no competitors in the catalog (denominator collapses
        to its own mentions), or (b) competitors are declared but none appeared
        in any response. Surface (a) explicitly so the reader doesn't think the
        responses are competitor-free.
    """
    matched = [s for s in scores if s.subject_id == subject_id and s.metric == "share_of_voice"]
    nums = [float(s.value) for s in matched if isinstance(s.value, int | float) and not isinstance(s.value, bool)]
    if not nums:
        return ""
    if all(abs(v - 1.0) < 1e-9 for v in nums):
        subject = next((s for s in catalog.subjects if s.id == subject_id), None)
        if subject is not None and not subject.competitors:
            return (
                "_Share of voice: uninformative — this subject declares no "
                "competitors in the catalog, so the denominator is just its own mentions._"
            )
        return (
            "_Share of voice: uninformative this run "
            "(no competitor mentions detected in any response)._"
        )
    mean = sum(nums) / len(nums)
    return f"**Share of voice**: {_pct(mean)} (mean across cohorts that mentioned anyone)"


# --- goals ----------------------------------------------------------------


def _goals_section(inputs: SummaryInputs) -> str:
    if not inputs.goals:
        return ""
    lines = ["## Goal progress", ""]
    for goal in inputs.goals:
        ev = evaluate_goal(goal, scores=inputs.scores, catalog=inputs.catalog, today=inputs.today)
        light = _LIGHTS[ev.status]
        actual = "—" if ev.actual is None else f"{ev.actual:.3f}"
        expected = "—" if ev.expected is None else f"{ev.expected:.3f}"
        lines.append(
            f"- {light} **{goal.id}** ({goal.subject}/{goal.metric}/{goal.tier}, "
            f"target={goal.target} by {goal.target_date}): "
            f"actual={actual} expected={expected}"
        )
    return "\n".join(lines)


# --- grounded vs ungrounded gap ------------------------------------------


def _grounded_gap_section(inputs: SummaryInputs) -> str:
    gaps = compute_grounded_gaps(inputs.scores, top_n=5)
    if not gaps:
        return ""
    lines = ["## Grounded vs ungrounded gap (top 5 by |gap|)", ""]
    for g in gaps:
        lines.append(
            f"- `{g.prompt_id}` x `{g.model_stem}` / {g.subject_id} / {g.metric}: "
            f"grounded={_fmt_value(g.grounded_value)} ungrounded={_fmt_value(g.ungrounded_value)} "
            f"gap={g.gap:+.3f}"
        )
    return "\n".join(lines)


# --- cost -----------------------------------------------------------------


def _cost_section(inputs: SummaryInputs) -> str:
    n_total = len(inputs.tasks)
    n_success = sum(1 for t in inputs.tasks if t.status == TaskStatus.SUCCESS)
    n_failed = n_total - n_success
    total_cost = sum(t.cost_usd_estimate for t in inputs.tasks)
    total_in = sum(t.tokens_in for t in inputs.tasks)
    total_out = sum(t.tokens_out for t in inputs.tasks)
    elapsed = ""
    if inputs.run.finished_at and inputs.run.started_at:
        seconds = (inputs.run.finished_at - inputs.run.started_at).total_seconds()
        elapsed = f"; wall_time={_format_duration(seconds)}"
    return (
        "## Cost & runtime\n\n"
        f"- tasks: total={n_total} success={n_success} failed={n_failed}\n"
        f"- tokens: in={total_in:,} out={total_out:,}\n"
        f"- cost (token counts x pricing table): ${total_cost:.4f}{elapsed}"
    )


_LIGHTS = {
    GoalStatus.GREEN: "[GREEN]",
    GoalStatus.YELLOW: "[YELLOW]",
    GoalStatus.RED: "[RED]",
    GoalStatus.PENDING: "[PENDING]",
}


# --- formatting ----------------------------------------------------------


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _fmt_value(v: float | int | None) -> str:
    if v is None:
        return "-"
    if isinstance(v, int):
        return str(v)
    return f"{v:.3f}"
