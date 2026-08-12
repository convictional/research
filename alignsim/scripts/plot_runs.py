#!/usr/bin/env python3
"""Plot AlignSim run metrics from the database.

Outputs a multi-page PDF:
  Page 1 — Time-series grid (MRR, runway, cash, tech debt, customers, capacity)
  Page 2 — Scatter plots with Pareto frontiers (when ≥2 scored runs)
  Page 3 — Score distributions by group (when any group has ≥2 runs)

Examples:
  plot_runs.py --commit 3e46d9 --condition c2
  plot_runs.py --commit 3e46d9 --seeds 100,104 --model opus
  plot_runs.py --run-ids UUID1 UUID2
  plot_runs.py -n 20 --condition c3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alignsim.src.persistence.database import get_db_url
from alignsim.src.persistence.models import RunModel, TurnSnapshotModel

from tortoise import Tortoise

# ── Constants ──────────────────────────────────────────────────────────────

GROUP_COLORS = [
    "#E07B39", "#5B8DEF", "#8BC34A", "#AB47BC",
    "#EF5350", "#26A69A", "#FFA726", "#78909C",
    "#EC407A", "#7E57C2", "#29B6F6", "#D4E157",
]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
FUNC_ORDER = ["engineering", "sales", "support", "marketing", "ops"]
ALIGNMENT_METRIC_ORDER = [
    "support_timing", "bug_responsiveness", "debt_management",
    "sales_focus", "ops_engagement",
]
TOKEN_COMPONENT_ORDER = [
    ("input_tokens", "Input"),
    ("output_tokens", "Output"),
    ("cache_creation_input_tokens", "Cache write"),
    ("cache_read_input_tokens", "Cache read"),
]
AGGREGATE_THRESHOLD = 8
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ── Helpers ────────────────────────────────────────────────────────────────


def short_model_name(model: str | None) -> str:
    if not model:
        return "unknown"
    replacements = {
        "claude-opus-4-6": "Opus 4.6",
        "claude-opus-4-7": "Opus 4.7",
        "claude-opus-4-8": "Opus 4.8",
        "claude-sonnet-4-6": "Sonnet 4.6",
        "claude-haiku-4-5": "Haiku 4.5",
        "gemma-4": "Gemma 4",
    }
    return replacements.get(model, model)


def short_condition(condition: str) -> str:
    _MAP = {"condition2": "C2", "condition3": "C3", "condition4a": "C4a", "condition4b": "C4b"}
    return _MAP.get(condition, condition[:6].upper())


# The autonomous player_types: llm_agent (C2), multi_agent (C3/C4), and the legacy `llm`. These are the
# baseline and stay unlabeled. Any other value (human_guided, human, or a future type) is a human-in-the-
# loop / non-standard treatment and gets a group suffix so it never pools with the autonomous runs.
_AUTONOMOUS_PLAYER_TYPES = {"llm", "llm_agent", "multi_agent"}


def group_key(
    condition: str, model: str | None, harness: str | None = None, player_type: str | None = None
) -> str:
    label = f"{short_condition(condition)} {short_model_name(model)}"
    if player_type and player_type not in _AUTONOMOUS_PLAYER_TYPES:
        label += f" ({player_type})"
    if harness:
        label += f" [{harness}]"  # keep claude-code vs pi runs of the same model distinct
    return label


def assign_group_styles(groups: list[str]) -> dict[str, dict]:
    unique = sorted(set(groups))
    return {
        g: {
            "color": GROUP_COLORS[i % len(GROUP_COLORS)],
            "marker": MARKERS[i % len(MARKERS)],
        }
        for i, g in enumerate(unique)
    }


def total_tokens(token_usage: dict | None) -> int:
    if not token_usage:
        return 0
    total = 0
    for usage in token_usage.values():
        total += usage.get("input_tokens", 0)
        total += usage.get("output_tokens", 0)
        total += usage.get("cache_creation_input_tokens", 0)
        total += usage.get("cache_read_input_tokens", 0)
    return total


def parse_seed_range(s: str) -> tuple[int, int]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected LO,HI (e.g. 100,105)")
    try:
        lo, hi = int(parts[0].strip()), int(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seed values must be integers") from exc
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


# ── Data fetching ──────────────────────────────────────────────────────────


async def fetch_data(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    await Tortoise.init(db_url=get_db_url(), modules={"models": ["alignsim.src.persistence.models"]})
    try:
        return await _fetch_data_inner(args)
    finally:
        await Tortoise.close_connections()


async def _fetch_data_inner(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    if args.run_ids:
        qs = RunModel.filter(id__in=args.run_ids)
    else:
        qs = RunModel.filter(finished_at__isnull=False)
        if args.commit:
            from functools import reduce
            from operator import or_
            from tortoise.expressions import Q
            q = reduce(or_, (Q(engine_commit__startswith=c) for c in args.commit))
            qs = qs.filter(q)
        if args.condition:
            cond = args.condition.lower()
            if cond in ("c2", "condition2"):
                cond = "condition2"
            elif cond in ("c3", "condition3"):
                cond = "condition3"
            qs = qs.filter(condition=cond)
        if args.model:
            qs = qs.filter(model__icontains=args.model)
        if args.thinking:
            qs = qs.filter(thinking=args.thinking)
        if args.seeds:
            lo, hi = args.seeds
            qs = qs.filter(seed__gte=lo, seed__lte=hi)

    has_filters = any([args.run_ids, args.commit, args.condition, args.model, args.thinking, args.seeds])
    limit = args.num_runs if args.num_runs is not None else (200 if has_filters else 10)
    runs = await qs.order_by("-started_at").limit(limit)

    run_info = []
    for r in runs:
        gk = group_key(r.condition, r.model, r.harness, r.player_type)
        run_info.append({
            "id": str(r.id),
            "model": r.model,
            "condition": r.condition,
            "harness": r.harness,
            "player_type": r.player_type,
            "thinking": r.thinking,
            "seed": r.seed,
            "group": gk,
            "label": f"{gk} s{r.seed}",
            "turns_played": r.turns_played or 0,
            "final_mrr": r.final_mrr or 0,
            "final_runway_turns": r.final_runway_turns or 0,
            "score_composite": r.score_composite,
            "score_mrr": r.score_mrr,
            "score_churn": r.score_churn,
            "score_runway": r.score_runway,
            "score_pareto": r.score_pareto,
            "function_scores": r.function_scores or {},
            "alignment_scores": r.alignment_scores or {},
            "token_usage": r.token_usage,
            "total_tokens": total_tokens(r.token_usage),
            "engine_commit": r.engine_commit,
            "started_at": r.started_at,
        })

    # Facet by reasoning level only when it varies across the set, so a single-level pull stays clean
    # but a mixed pull (or comparing thinking on/off) splits into separate groups/series.
    if len({r["thinking"] for r in run_info if r["thinking"]}) > 1:
        for r in run_info:
            r["group"] += f" t={r['thinking'] or '?'}"
            r["label"] = f"{r['group']} s{r['seed']}"

    ids = [r["id"] for r in run_info]
    snaps = await TurnSnapshotModel.filter(run_id__in=ids).order_by("turn")

    snap_data = []
    for s in snaps:
        snap_data.append({
            "run_id": str(s.run_id),
            "turn": s.turn,
            "mrr": s.mrr,
            "budget": s.budget,
            "runway_turns": s.runway_turns,
            "tech_debt_level": s.tech_debt_level,
            "active_customers": s.active_customers,
            "pipeline_customers": s.pipeline_customers,
            "capacity_used": s.capacity_used,
            "capacity_available": s.capacity_available,
        })

    return run_info, snap_data


# ── Time series ────────────────────────────────────────────────────────────


def build_individual_series(
    run_info: list[dict],
    snap_data: list[dict],
    metric: str,
) -> dict[str, dict[int, float]]:
    id_to_label = {r["id"]: r["label"] for r in run_info}
    series: dict[str, dict[int, float]] = {r["label"]: {} for r in run_info}
    for s in snap_data:
        label = id_to_label.get(s["run_id"])
        if label:
            series[label][s["turn"]] = s[metric]
    return series


def build_aggregated_series(
    run_info: list[dict],
    snap_data: list[dict],
    metric: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns {group: (turns_array, means_array, stds_array)}."""
    group_ids: dict[str, set[str]] = defaultdict(set)
    for r in run_info:
        group_ids[r["group"]].add(r["id"])

    result = {}
    for grp, ids in group_ids.items():
        turn_vals: dict[int, list[float]] = defaultdict(list)
        for s in snap_data:
            if s["run_id"] in ids:
                turn_vals[s["turn"]].append(s[metric])

        if not turn_vals:
            continue
        turns = sorted(turn_vals.keys())
        means = np.array([np.mean(turn_vals[t]) for t in turns])
        stds = np.array([np.std(turn_vals[t]) for t in turns])
        result[grp] = (np.array(turns), means, stds)

    return result


def plot_line_individual(
    ax: plt.Axes,
    series: dict[str, dict[int, float]],
    run_info: list[dict],
    styles: dict[str, dict],
    ylabel: str,
    title: str,
    fmt_func: Callable[..., str] | None = None,
) -> None:
    label_to_group = {r["label"]: r["group"] for r in run_info}
    plotted_groups: set[str] = set()
    for label, data in series.items():
        turns = sorted(data.keys())
        vals = [data[t] for t in turns]
        grp = label_to_group.get(label, label)
        style = styles.get(grp, {"color": "#999", "marker": "o"})
        legend_label = grp if grp not in plotted_groups else None
        plotted_groups.add(grp)
        ax.plot(
            turns, vals,
            color=style["color"], linewidth=1.5,
            marker=style["marker"], markersize=2, alpha=0.7,
            label=legend_label,
        )
    ax.set_xlabel("Turn")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if fmt_func:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_func))


def plot_line_aggregated(
    ax: plt.Axes,
    agg: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    styles: dict[str, dict],
    ylabel: str,
    title: str,
    fmt_func: Callable[..., str] | None = None,
) -> None:
    for grp, (turns, means, stds) in agg.items():
        style = styles.get(grp, {"color": "#999", "marker": "o"})
        ax.plot(turns, means, color=style["color"], linewidth=2, label=grp)
        ax.fill_between(turns, means - stds, means + stds, color=style["color"], alpha=0.15)
    ax.set_xlabel("Turn")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if fmt_func:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_func))


def page_time_series(
    run_info: list[dict],
    snap_data: list[dict],
    styles: dict[str, dict],
) -> plt.Figure:
    aggregate = len(run_info) > AGGREGATE_THRESHOLD
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ("mrr", "MRR ($)", "MRR Progression", lambda x, _: f"${x:,.0f}"),
        ("runway_turns", "Runway (turns)", "Runway Remaining", None),
        ("budget", "Budget ($M)", "Cash Position", None),
        ("tech_debt_level", "Tech Debt Level", "Tech Debt", None),
        ("active_customers", "Active Customers", "Active Customers", None),
        ("capacity_used", "Capacity Used", "Capacity Utilization", None),
    ]

    for idx, (metric, ylabel, title, fmt) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        if aggregate:
            agg = build_aggregated_series(run_info, snap_data, metric)
            if metric == "budget":
                agg = {g: (t, m / 1e6, s / 1e6) for g, (t, m, s) in agg.items()}
                ylabel = "Budget ($M)"
            plot_line_aggregated(ax, agg, styles, ylabel, title, fmt)
        else:
            series = build_individual_series(run_info, snap_data, metric)
            if metric == "budget":
                series = {lab: {t: v / 1e6 for t, v in d.items()} for lab, d in series.items()}
                ylabel = "Budget ($M)"
            plot_line_individual(ax, series, run_info, styles, ylabel, title, fmt)

        if metric == "budget":
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)

    n = len(run_info)
    groups = sorted(set(r["group"] for r in run_info))
    mode = "aggregated" if aggregate else "individual"
    fig.suptitle(
        f"AlignSim Time Series — {n} runs, {len(groups)} group(s) ({mode})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ── Scatter / Pareto ──────────────────────────────────────────────────────


def pareto_frontier_2d(
    xs: list[float], ys: list[float],
) -> tuple[list[float], list[float]]:
    """Points on the Pareto frontier (higher is better on both axes)."""
    paired = sorted(zip(xs, ys), key=lambda p: -p[0])
    front_x: list[float] = []
    front_y: list[float] = []
    max_y = float("-inf")
    for x, y in paired:
        if y > max_y:
            front_x.append(x)
            front_y.append(y)
            max_y = y
    return front_x, front_y


def plot_scatter_ax(
    ax: plt.Axes,
    run_info: list[dict],
    x_key: str,
    y_key: str,
    styles: dict[str, dict],
    xlabel: str,
    ylabel: str,
    title: str,
    show_pareto: bool = True,
) -> None:
    plotted_groups: set[str] = set()
    all_x: list[float] = []
    all_y: list[float] = []
    for r in run_info:
        xv, yv = r.get(x_key), r.get(y_key)
        if xv is None or yv is None:
            continue
        grp = r["group"]
        style = styles.get(grp, {"color": "#999", "marker": "o"})
        legend_label = grp if grp not in plotted_groups else None
        plotted_groups.add(grp)
        ax.scatter(
            xv, yv,
            color=style["color"], marker=style["marker"],
            s=60, alpha=0.7, edgecolors="white", linewidths=0.5,
            label=legend_label,
        )
        all_x.append(xv)
        all_y.append(yv)

    if show_pareto and len(all_x) >= 3:
        fx, fy = pareto_frontier_2d(all_x, all_y)
        if len(fx) >= 2:
            order = sorted(range(len(fx)), key=lambda i: fx[i])
            fx = [fx[i] for i in order]
            fy = [fy[i] for i in order]
            ax.plot(fx, fy, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Pareto frontier")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def page_scatter(run_info: list[dict], styles: dict[str, dict]) -> plt.Figure:
    """Placeholder scatter axes — swap pairs as analysis needs evolve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_scatter_ax(
        axes[0], run_info,
        "score_mrr", "score_churn", styles,
        "MRR Score", "Churn Score", "MRR vs Churn Score",
    )
    plot_scatter_ax(
        axes[1], run_info,
        "score_composite", "score_pareto", styles,
        "Composite Score", "Pareto Score", "Composite vs Pareto Score",
    )

    fig.suptitle(
        "AlignSim Scatter Analysis",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


# ── Distributions ──────────────────────────────────────────────────────────


def page_distributions(run_info: list[dict], styles: dict[str, dict]) -> plt.Figure:
    groups = sorted(set(r["group"] for r in run_info))
    group_runs = {g: [r for r in run_info if r["group"] == g] for g in groups}

    score_keys = [
        ("score_composite", "Composite Score"),
        ("score_mrr", "MRR Score"),
        ("score_churn", "Churn Score"),
        ("score_runway", "Runway Score"),
        ("score_pareto", "Pareto Score"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (key, title) in enumerate(score_keys):
        ax = axes[idx // 3, idx % 3]
        data: list[list[float]] = []
        labels: list[str] = []
        colors: list[str] = []
        for g in groups:
            vals = [r[key] for r in group_runs[g] if r[key] is not None]
            if vals:
                data.append(vals)
                labels.append(g)
                colors.append(styles[g]["color"])

        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=30)

    # Function scores — grouped bar with error bars
    ax = axes[1, 2]
    x = np.arange(len(FUNC_ORDER))
    width = 0.7 / max(len(groups), 1)
    for i, g in enumerate(groups):
        means = []
        stds = []
        for f in FUNC_ORDER:
            vals = [r["function_scores"].get(f, 0.0) for r in group_runs[g]]
            means.append(float(np.mean(vals)) if vals else 0.0)
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        ax.bar(
            x + i * width, means, width,
            yerr=stds, label=g, color=styles[g]["color"],
            edgecolor="white", linewidth=0.5, alpha=0.8, capsize=3,
        )
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels([f.capitalize() for f in FUNC_ORDER])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_title("Function Scores (mean ± std)", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "AlignSim Score Distributions",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ── Console summary ───────────────────────────────────────────────────────


def _fmt_score(vals: list[float]) -> str:
    if not vals:
        return "—"
    mu = np.mean(vals)
    if len(vals) > 1:
        return f"{mu:.3f}±{np.std(vals):.3f}"
    return f"{mu:.4f}"


def _fmt_tokens(vals: list[int]) -> str:
    if not vals:
        return "—"
    avg = np.mean(vals)
    if avg > 1e6:
        return f"{avg / 1e6:.1f}M"
    if avg > 1e3:
        return f"{avg / 1e3:.0f}K"
    return f"{avg:.0f}"


def print_summary(run_info: list[dict]) -> None:
    groups = sorted(set(r["group"] for r in run_info))
    group_runs = {g: [r for r in run_info if r["group"] == g] for g in groups}

    print(f"\n{'Group':<22} {'Runs':>4}  {'Composite':>13}  {'MRR':>13}  {'Churn':>13}  {'Runway':>13}  {'Tokens':>8}")
    print("─" * 100)

    for g in groups:
        runs = group_runs[g]
        n = len(runs)
        composites = [r["score_composite"] for r in runs if r["score_composite"] is not None]
        mrrs = [r["score_mrr"] for r in runs if r["score_mrr"] is not None]
        churns = [r["score_churn"] for r in runs if r["score_churn"] is not None]
        runways = [r["score_runway"] for r in runs if r["score_runway"] is not None]
        tokens = [r["total_tokens"] for r in runs if r["total_tokens"] > 0]

        print(
            f"{g:<22} {n:>4}  "
            f"{_fmt_score(composites):>13}  "
            f"{_fmt_score(mrrs):>13}  "
            f"{_fmt_score(churns):>13}  "
            f"{_fmt_score(runways):>13}  "
            f"{_fmt_tokens(tokens):>8}"
        )

    commits = set(r.get("engine_commit", "")[:8] for r in run_info if r.get("engine_commit"))
    if commits:
        print(f"\nEngine commit(s): {', '.join(sorted(commits))}")
    print()


# ── PDF output ─────────────────────────────────────────────────────────────


def _alignment_metric(run: dict, key: str) -> float | None:
    """Pull a metric's normalized score from the alignment_scores nested dict."""
    val = (run.get("alignment_scores") or {}).get(key)
    if isinstance(val, dict):
        return val.get("score")
    return None


def page_alignment(run_info: list[dict], styles: dict[str, dict]) -> plt.Figure:
    """Page 4: Layer 2 alignment metrics + composite-vs-pareto scatter.

    Left: grouped bar chart of the five hidden alignment metrics (mean ± std per
    group). Right: scatter of Layer 1 composite vs Layer 2 alignment_pareto —
    the core research question (high goal attainment + high alignment quality).
    """
    groups = sorted(set(r["group"] for r in run_info))
    group_runs: dict[str, list[dict]] = defaultdict(list)
    for r in run_info:
        group_runs[r["group"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: grouped bars per alignment metric (mean ± std)
    ax = axes[0]
    x = np.arange(len(ALIGNMENT_METRIC_ORDER))
    width = 0.7 / max(len(groups), 1)
    for i, g in enumerate(groups):
        means: list[float] = []
        stds: list[float] = []
        for key in ALIGNMENT_METRIC_ORDER:
            vals = [s for s in (_alignment_metric(r, key) for r in group_runs[g]) if s is not None]
            means.append(float(np.mean(vals)) if vals else 0.0)
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        ax.bar(
            x + i * width, means, width,
            yerr=stds, label=g, color=styles[g]["color"],
            edgecolor="white", linewidth=0.5, alpha=0.8, capsize=3,
        )
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels([k.replace("_", "\n") for k in ALIGNMENT_METRIC_ORDER], fontsize=9)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Alignment Metrics (mean ± std)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # Right: composite (Layer 1) vs alignment_pareto (Layer 2)
    ax = axes[1]
    for g in groups:
        xs = []
        ys = []
        for r in group_runs[g]:
            comp = r.get("score_composite")
            ap = _alignment_metric(r, "alignment_pareto")
            if comp is None or ap is None:
                continue
            xs.append(comp)
            ys.append(ap)
        if xs:
            ax.scatter(xs, ys, label=g, color=styles[g]["color"],
                       marker=styles[g]["marker"], s=60, alpha=0.75,
                       edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Layer 1 Composite (goal attainment)")
    ax.set_ylabel("Layer 2 alignment_pareto (geometric mean)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Goal Attainment vs Alignment Quality", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.2)
    if groups:
        ax.legend(fontsize=8)

    fig.suptitle("AlignSim Layer 2: Alignment Metrics", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def page_tokens(run_info: list[dict], styles: dict[str, dict]) -> plt.Figure:
    """Token usage: total per run by group + mean composition by group.

    Left: box plot of total tokens per run (cost spread per group). Right: mean
    tokens by component on a log axis — cache reads dwarf the rest (~100×), so a
    log scale is the only way to compare input/output/cache-write against them.
    """
    groups = sorted(set(r["group"] for r in run_info))
    group_runs = {g: [r for r in run_info if r["group"] == g] for g in groups}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: total tokens per run, box plot by group
    ax = axes[0]
    data: list[list[int]] = []
    labels: list[str] = []
    colors: list[str] = []
    for g in groups:
        vals = [r["total_tokens"] for r in group_runs[g] if r["total_tokens"] > 0]
        if vals:
            data.append(vals)
            labels.append(g)
            colors.append(styles[g]["color"])
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("Total tokens per run")
    ax.set_title("Total Token Usage by Group", fontweight="bold", fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=30)

    # Right: mean tokens by component (grouped bars, log scale)
    ax = axes[1]
    x = np.arange(len(TOKEN_COMPONENT_ORDER))
    width = 0.7 / max(len(groups), 1)
    for i, g in enumerate(groups):
        means: list[float] = []
        for key, _ in TOKEN_COMPONENT_ORDER:
            vals = [
                sum(u.get(key, 0) for u in (r["token_usage"] or {}).values())
                for r in group_runs[g]
            ]
            means.append(float(np.mean(vals)) if vals else 0.0)
        ax.bar(
            x + i * width, means, width,
            label=g, color=styles[g]["color"],
            edgecolor="white", linewidth=0.5, alpha=0.85,
        )
    ax.set_yscale("log")
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels([lbl for _, lbl in TOKEN_COMPONENT_ORDER])
    ax.set_ylabel("Mean tokens per run (log)")
    ax.set_title("Token Composition by Group (mean)", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("AlignSim Token Usage", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def make_pdf(run_info: list[dict], snap_data: list[dict], output: Path) -> None:
    groups_list = [r["group"] for r in run_info]
    styles = assign_group_styles(groups_list)

    group_runs: dict[str, list[dict]] = defaultdict(list)
    for r in run_info:
        group_runs[r["group"]].append(r)

    has_scores = sum(1 for r in run_info if r["score_composite"] is not None) >= 2
    has_multi = any(len(v) >= 2 for v in group_runs.values())
    has_alignment = sum(
        1 for r in run_info if _alignment_metric(r, "alignment_pareto") is not None
    ) >= 1
    has_tokens = sum(1 for r in run_info if r["total_tokens"] > 0) >= 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output)) as pdf:
        fig1 = page_time_series(run_info, snap_data, styles)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        if has_scores:
            fig2 = page_scatter(run_info, styles)
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

        if has_scores and has_multi:
            fig3 = page_distributions(run_info, styles)
            pdf.savefig(fig3, bbox_inches="tight")
            plt.close(fig3)

        if has_alignment:
            fig4 = page_alignment(run_info, styles)
            pdf.savefig(fig4, bbox_inches="tight")
            plt.close(fig4)

        if has_tokens:
            fig5 = page_tokens(run_info, styles)
            pdf.savefig(fig5, bbox_inches="tight")
            plt.close(fig5)

    pages = (
        1 + int(has_scores) + int(has_scores and has_multi)
        + int(has_alignment) + int(has_tokens)
    )
    print(f"Saved {pages}-page PDF to {output}")


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AlignSim run metrics from the database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --commit 3e46d9 --condition c2
  %(prog)s --commit 3e46d9 --seeds 100,104 --model opus
  %(prog)s --run-ids UUID1 UUID2 UUID3
  %(prog)s -n 20 --condition c3""",
    )

    filt = parser.add_argument_group("filters")
    filt.add_argument(
        "--commit", nargs="+", metavar="PREFIX",
        help="Engine commit prefix(es) to match. Pass multiple to OR them together.",
    )
    filt.add_argument("--condition", type=str, metavar="c2|c3", help="Filter by condition")
    filt.add_argument("--model", type=str, metavar="SUBSTR", help="Filter by model name (substring)")
    filt.add_argument(
        "--thinking", type=str, metavar="LEVEL",
        help="Filter by reasoning level (off|minimal|low|medium|high|xhigh)",
    )
    filt.add_argument(
        "--seeds", type=parse_seed_range, metavar="LO,HI",
        help="Seed range inclusive (e.g. 100,105)",
    )
    filt.add_argument(
        "--run-ids", nargs="+", metavar="UUID",
        help="Specific run UUIDs (overrides other filters)",
    )
    filt.add_argument(
        "-n", "--num-runs", type=int, default=None,
        help="Max runs to fetch (default: 10 without filters, 200 with)",
    )

    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output PDF path (default: auto-generated in results/)",
    )

    args = parser.parse_args()
    if args.output is None:
        args.output = str(_build_output_path(args))
    return args


def _build_output_path(args: argparse.Namespace) -> Path:
    parts = ["plot"]
    if args.condition:
        parts.append(args.condition.lower())
    if args.model:
        parts.append(args.model.lower())
    if args.thinking:
        parts.append(f"t-{args.thinking.lower()}")
    if args.commit:
        parts.append("-".join(c[:8] for c in args.commit))
    if args.seeds:
        parts.append(f"s{args.seeds[0]}-{args.seeds[1]}")
    if args.run_ids:
        parts.append(f"{len(args.run_ids)}runs")
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return RESULTS_DIR / ("_".join(parts) + ".pdf")


def main() -> None:
    args = parse_args()
    run_info, snap_data = asyncio.run(fetch_data(args))

    if not run_info:
        print("No runs found.", file=sys.stderr)
        sys.exit(1)

    print_summary(run_info)
    make_pdf(run_info, snap_data, Path(args.output))


if __name__ == "__main__":
    main()
