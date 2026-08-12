#!/usr/bin/env python3
"""Compare single-player (C2) against its multi-player counterparts (C3/C4a/C4b).

Where plot_runs.py plots trajectories for a set of runs, this script answers a
head-to-head question: holding model + seed + scenario + turns constant (and, by
default, engine commit too), does the single agent compete with, lose to, or
outperform the same model playing the multi-agent conditions?

Runs are matched into "cells" keyed on (model, seed, scenario, max_turns[, commit]).
A cell contributes to the comparison only when it contains both the baseline
(C2) and at least one multi-player condition. Within a cell, repeated runs are
averaged. We compare on three shared-goal outcomes: final_mrr, final_runway_turns,
and churn. Churn uses the score_churn retention rate (max(0, 1 - avg_churn_rate),
higher = better) since RunModel carries no raw final_churn; it is an individual
per-goal sub-score, distinct from the composite/pareto scores, which are
deliberately avoided (they are known-unreliable).

Matching modes:
  --match commit  (default) engine_commit is part of the cell key → clean
                  apples-to-apples, mechanics held constant.
  --match loose   ignore commit → more cells, but confounded by engine drift.

Output:
  Console — a win/tie/loss scorecard for the baseline vs each multi condition.
  PDF     — Page 1: paired scatters (baseline x-axis vs multi y-axis) per metric
                    (MRR, runway, churn) with a y=x diagonal; points below the
                    line are baseline wins.
            Page 2: win/tie/loss bars per condition and metric.

Examples:
  compare_conditions.py
  compare_conditions.py --commit 99b41e6 --model sonnet
  compare_conditions.py --match loose --scenario seed_stage --turns 48
  compare_conditions.py --multi c3 --mrr-tol 2000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alignsim.src.persistence.database import get_db_url
from alignsim.src.persistence.models import RunModel

from tortoise import Tortoise

# ── Convictional palette ─────────────────────────────────────────────────────
# Pulled from the app theme (app/web/app/styles/daisyui.css) so charts read as
# ours: warm-paper canvas, blue primary, the signature "decision" green, gold,
# and red, on a warm-dark neutral ink.

BRAND = {
    "primary": "#2b7fff",   # blue — baseline / single-player hero
    "decision": "#4d6847",  # muted forest green — Convictional's signature
    "amber": "#d08700",     # gold
    "error": "#e7000b",     # red
    "success": "#5ea500",   # green (positive outcomes)
    "ink": "#292524",       # warm near-black
    "muted": "#a8a29e",     # warm gray — gridlines, diagonal, ties
    "paper": "#fffdf5",     # figure background
    "surface": "#f7f3ea",   # axes background
}

# Outcome colors for the win/tie/loss view (win = baseline better).
OUTCOME_COLORS = {"win": BRAND["success"], "tie": BRAND["muted"], "loss": BRAND["error"]}

# Per-condition accent (used to tint the multi axis label / panel).
CONDITION_COLORS = {
    "condition2": BRAND["primary"],
    "condition3": BRAND["decision"],
    "condition4a": BRAND["amber"],
    "condition4b": BRAND["error"],
}

# Model -> point color, cycled from the brand accents.
_MODEL_CYCLE = [BRAND["primary"], BRAND["decision"], BRAND["amber"], BRAND["error"], BRAND["ink"]]

CONDITION_ALIASES = {
    "c2": "condition2", "condition2": "condition2",
    "c3": "condition3", "condition3": "condition3",
    "c4a": "condition4a", "condition4a": "condition4a",
    "c4b": "condition4b", "condition4b": "condition4b",
}
CLEAN_REASONS = {"max_turns_reached", "bankruptcy"}
# score_churn is defined as max(0, 1 - avg_churn_rate), so a valid retention score
# lives in [0, 1]. Runs above this ceiling reflect buggy/stale scoring and are dropped
# entirely (not counted in any metric — MRR, runway, or churn).
MAX_VALID_CHURN_SCORE = 1.0


# ── Helpers ──────────────────────────────────────────────────────────────────


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
        "deepseek/deepseek-v4-flash": "DeepSeek v4",
        "z-ai/glm-5.2": "GLM 5.2",
    }
    return replacements.get(model, model)


def short_condition(condition: str) -> str:
    """Compact code (C2/C3/...) for console, markdown, and output filenames."""
    return {
        "condition2": "C2", "condition3": "C3",
        "condition4a": "C4a", "condition4b": "C4b",
    }.get(condition, condition[:6].upper())


def condition_label(condition: str) -> str:
    """Essay-facing substrate name (no 'C2'/'condition' jargon) for the PDF plots."""
    return {
        "condition2": "Single-player",
        "condition3": "Multi-player (chat)",
        "condition4a": "Multi-player (channels)",
        "condition4b": "Multi-player (posts + goals)",
    }.get(condition, condition)


def apply_brand_style() -> None:
    """Warm-paper matplotlib theme evoking the Convictional app surfaces."""
    plt.rcParams.update({
        "figure.facecolor": BRAND["paper"],
        "savefig.facecolor": BRAND["paper"],
        "axes.facecolor": BRAND["surface"],
        "axes.edgecolor": BRAND["muted"],
        "axes.labelcolor": BRAND["ink"],
        "axes.titlecolor": BRAND["ink"],
        "text.color": BRAND["ink"],
        "xtick.color": BRAND["ink"],
        "ytick.color": BRAND["ink"],
        "grid.color": BRAND["muted"],
        "font.size": 10,
    })


def assign_colors(values: list[str]) -> dict:
    """Map distinct values to colors: brand accents for a few, tab20 for many (e.g. commits)."""
    unique = sorted(set(values))
    if len(unique) <= len(_MODEL_CYCLE):
        return {v: _MODEL_CYCLE[i] for i, v in enumerate(unique)}
    cmap = plt.get_cmap("tab20")
    return {v: cmap(i % 20) for i, v in enumerate(unique)}


def assign_blue_gradient(ordered: list[str]) -> dict:
    """Shades of blue across an ordered list (e.g. commits in dev order): light → dark."""
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: BRAND["primary"]}
    cmap = plt.get_cmap("Blues")
    return {v: cmap(0.35 + 0.57 * i / (len(ordered) - 1)) for i, v in enumerate(ordered)}


def parse_seed_range(s: str) -> tuple[int, int]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected LO,HI (e.g. 100,105)")
    try:
        lo, hi = int(parts[0].strip()), int(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seed values must be integers") from exc
    return (lo, hi) if lo <= hi else (hi, lo)


def resolve_condition(name: str) -> str:
    key = name.lower()
    if key not in CONDITION_ALIASES:
        raise argparse.ArgumentTypeError(f"Unknown condition '{name}' (valid: c2, c3, c4a, c4b)")
    return CONDITION_ALIASES[key]


# ── Data fetching ────────────────────────────────────────────────────────────


async def fetch_runs(args: argparse.Namespace, conditions: list[str]) -> list[dict]:
    await Tortoise.init(db_url=get_db_url(), modules={"models": ["alignsim.src.persistence.models"]})
    try:
        qs = RunModel.filter(finished_at__isnull=False, condition__in=conditions)
        qs = qs.exclude(player_type="human")
        if args.commit:
            from functools import reduce
            from operator import or_
            from tortoise.expressions import Q
            q = reduce(or_, (Q(engine_commit__startswith=c) for c in args.commit))
            qs = qs.filter(q)
        if args.model:
            qs = qs.filter(model__icontains=args.model)
        if args.scenario:
            qs = qs.filter(scenario_name=args.scenario)
        if args.turns is not None:
            qs = qs.filter(max_turns=args.turns)
        if args.seeds:
            lo, hi = args.seeds
            qs = qs.filter(seed__gte=lo, seed__lte=hi)
        runs = await qs.limit(args.num_runs)
    finally:
        await Tortoise.close_connections()

    return [{
        "model": r.model,
        "condition": r.condition,
        "seed": r.seed,
        "scenario": r.scenario_name,
        "max_turns": r.max_turns,
        "commit": (r.engine_commit or "none")[:9],
        "started_at": r.started_at,
        "turns_played": r.turns_played or 0,
        "reason": r.game_over_reason or "NULL",
        "final_mrr": r.final_mrr,
        "final_runway_turns": r.final_runway_turns,
        "score_mrr": r.score_mrr,
        "score_churn": r.score_churn,
        "score_runway": r.score_runway,
    } for r in runs]


# ── Matching & comparison ────────────────────────────────────────────────────


def cell_key(run: dict, match_commit: bool) -> tuple:
    base = (run["model"], run["seed"], run["scenario"], run["max_turns"])
    return base + ((run["commit"],) if match_commit else ())


def build_cells(runs: list[dict], match_commit: bool) -> dict[tuple, dict[str, list[dict]]]:
    cells: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        cells[cell_key(r, match_commit)][r["condition"]].append(r)
    return cells


def _mean(runs: list[dict], field: str) -> float | None:
    vals = [r[field] for r in runs if r.get(field) is not None]
    return float(np.mean(vals)) if vals else None


def classify(delta: float | None, tol: float) -> str | None:
    """win = baseline higher by more than tol; loss = lower; else tie."""
    if delta is None:
        return None
    if delta > tol:
        return "win"
    if delta < -tol:
        return "loss"
    return "tie"


def compare(
    cells: dict[tuple, dict[str, list[dict]]],
    baseline: str,
    multi: list[str],
    mrr_tol: float,
    runway_tol: float,
    churn_tol: float,
) -> dict[str, dict]:
    """For each multi condition, gather matched-cell outcomes vs the baseline."""
    out: dict[str, dict] = {
        mc: {"mrr": {"win": 0, "tie": 0, "loss": 0},
             "runway": {"win": 0, "tie": 0, "loss": 0},
             "churn": {"win": 0, "tie": 0, "loss": 0},
             "pairs": [], "n": 0}
        for mc in multi
    }
    for key, by_cond in cells.items():
        if baseline not in by_cond:
            continue
        b_mrr = _mean(by_cond[baseline], "final_mrr")
        b_rw = _mean(by_cond[baseline], "final_runway_turns")
        b_churn = _mean(by_cond[baseline], "score_churn")
        model = key[0]
        for mc in multi:
            if mc not in by_cond:
                continue
            m_mrr = _mean(by_cond[mc], "final_mrr")
            m_rw = _mean(by_cond[mc], "final_runway_turns")
            m_churn = _mean(by_cond[mc], "score_churn")
            rec = out[mc]
            rec["n"] += 1
            if (c := classify(None if None in (b_mrr, m_mrr) else b_mrr - m_mrr, mrr_tol)):
                rec["mrr"][c] += 1
            if (c := classify(None if None in (b_rw, m_rw) else b_rw - m_rw, runway_tol)):
                rec["runway"][c] += 1
            if (c := classify(None if None in (b_churn, m_churn) else b_churn - m_churn, churn_tol)):
                rec["churn"][c] += 1
            rec["pairs"].append({
                "model": model,
                "commit": by_cond[baseline][0]["commit"],
                "b_mrr": b_mrr, "m_mrr": m_mrr,
                "b_rw": b_rw, "m_rw": m_rw,
                "b_churn": b_churn, "m_churn": m_churn,
            })
    return out


# ── Console summary ──────────────────────────────────────────────────────────


def print_scorecard(results: dict[str, dict], baseline: str, args: argparse.Namespace) -> None:
    b = short_condition(baseline)
    match = "same-commit" if args.match == "commit" else "commit-agnostic"
    clean = "clean only" if not args.include_incomplete else "incl. incomplete"
    print(f"\nBaseline {b} vs multi-player  ·  {match}  ·  {clean}"
          f"  ·  tol: MRR ±${args.mrr_tol:,.0f}, runway ±{args.runway_tol:g}t, churn ±{args.churn_tol:g}")
    print(f"(W = {b} outperforms · T = within tolerance · L = {b} loses)\n")
    header = (f"{'vs':6} {'cells':>5}   {'final MRR  W/T/L':>18}   {'final runway  W/T/L':>21}"
              f"   {'churn (retention)  W/T/L':>25}")
    print(header)
    print("─" * len(header))
    for mc, rec in results.items():
        if rec["n"] == 0:
            continue
        m = rec["mrr"]; r = rec["runway"]; c = rec["churn"]
        mrr = f"{m['win']}/{m['tie']}/{m['loss']}"
        rw = f"{r['win']}/{r['tie']}/{r['loss']}"
        churn = f"{c['win']}/{c['tie']}/{c['loss']}"
        print(f"{short_condition(mc):6} {rec['n']:>5}   {mrr:>18}   {rw:>21}   {churn:>25}")
    print()


# ── PDF: paired scatters ─────────────────────────────────────────────────────


def _scatter_panel(
    ax: plt.Axes,
    pairs: list[dict],
    b_key: str,
    m_key: str,
    color_map: dict,
    color_key: str,
    baseline_lbl: str,
    multi_lbl: str,
    title: str,
    value_kind: str,
) -> None:
    xs = [p[b_key] for p in pairs if p[b_key] is not None and p[m_key] is not None]
    ys = [p[m_key] for p in pairs if p[b_key] is not None and p[m_key] is not None]
    cols = [color_map[p[color_key]] for p in pairs if p[b_key] is not None and p[m_key] is not None]
    if not xs:
        ax.text(0.5, 0.5, "no paired data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontweight="bold", fontsize=10)
        return

    lo = min(0, min(xs), min(ys))
    hi = max(max(xs), max(ys)) * 1.08
    ax.plot([lo, hi], [lo, hi], color=BRAND["muted"], linestyle="--", linewidth=1.2, zorder=1)
    ax.fill_between([lo, hi], [lo, lo], [hi, hi], color=BRAND["success"], alpha=0.05, zorder=0)
    ax.scatter(xs, ys, c=cols, s=64, alpha=0.85, edgecolors=BRAND["paper"], linewidths=0.7, zorder=3)

    ax.text(0.97, 0.05, f"{baseline_lbl} better ▸", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=BRAND["muted"], style="italic")
    ax.text(0.03, 0.95, f"◂ {multi_lbl} better", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=BRAND["muted"], style="italic")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(baseline_lbl)
    ax.set_ylabel(multi_lbl)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.25)
    if value_kind == "currency":
        fmt = lambda v, _: f"${v/1000:.0f}k"
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
    elif value_kind == "score":
        fmt = lambda v, _: f"{v*100:.0f}%"
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
    else:  # "turns"
        ax.axhline(0, color=BRAND["error"], linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color=BRAND["error"], linestyle=":", linewidth=0.8, alpha=0.5)


def page_scatters(results: dict[str, dict], baseline: str, color_map: dict,
                  color_key: str, label_fn, show_legend: bool) -> plt.Figure:
    active = [(mc, rec) for mc, rec in results.items() if rec["n"] > 0]
    n = len(active)
    fig, axes = plt.subplots(n, 3, figsize=(19, 5.6 * n), squeeze=False)
    b_lbl = condition_label(baseline)
    for row, (mc, rec) in enumerate(active):
        m_lbl = condition_label(mc)
        _scatter_panel(axes[row][0], rec["pairs"], "b_mrr", "m_mrr", color_map, color_key,
                       b_lbl, m_lbl, f"Final MRR — {b_lbl} vs {m_lbl}  (n={rec['n']})", "currency")
        _scatter_panel(axes[row][1], rec["pairs"], "b_rw", "m_rw", color_map, color_key,
                       b_lbl, m_lbl, f"Final runway — {b_lbl} vs {m_lbl}  (n={rec['n']})", "turns")
        _scatter_panel(axes[row][2], rec["pairs"], "b_churn", "m_churn", color_map, color_key,
                       b_lbl, m_lbl, f"Churn (retention) — {b_lbl} vs {m_lbl}  (n={rec['n']})", "score")

    if show_legend:
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c,
                              markeredgecolor=BRAND["paper"], markersize=9, label=label_fn(v))
                   for v, c in color_map.items()]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01), title="model")
    fig.suptitle("Single-player vs multi-player — per matched cell",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99) if show_legend else (0, 0, 1, 0.99))
    return fig


# ── PDF: win/tie/loss bars ───────────────────────────────────────────────────


def page_scorecard(results: dict[str, dict], baseline: str) -> plt.Figure:
    active = [(mc, rec) for mc, rec in results.items() if rec["n"] > 0]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), squeeze=False)
    b_lbl = condition_label(baseline)
    for col, (metric, title) in enumerate([("mrr", "Final MRR"), ("runway", "Final runway"),
                                           ("churn", "Churn (retention)")]):
        ax = axes[0][col]
        labels = [condition_label(mc) for mc, _ in active]
        y = np.arange(len(active))
        wins = [rec[metric]["win"] for _, rec in active]
        ties = [rec[metric]["tie"] for _, rec in active]
        losses = [rec[metric]["loss"] for _, rec in active]
        ax.barh(y, wins, color=OUTCOME_COLORS["win"], label=f"{b_lbl} wins")
        ax.barh(y, ties, left=wins, color=OUTCOME_COLORS["tie"], label="tie")
        ax.barh(y, losses, left=np.add(wins, ties), color=OUTCOME_COLORS["loss"], label=f"{b_lbl} loses")
        for i, (w, t, ll) in enumerate(zip(wins, ties, losses)):
            if w:
                ax.text(w / 2, i, str(w), ha="center", va="center", color=BRAND["paper"], fontsize=9)
            if t:
                ax.text(w + t / 2, i, str(t), ha="center", va="center", color=BRAND["ink"], fontsize=9)
            if ll:
                ax.text(w + t + ll / 2, i, str(ll), ha="center", va="center", color=BRAND["paper"], fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([f"vs {lbl}" for lbl in labels])
        ax.invert_yaxis()
        ax.set_xlabel("matched cells")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.2, axis="x")
        if col == 0:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{b_lbl} head-to-head scorecard", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ── PDF assembly ─────────────────────────────────────────────────────────────


def make_pdf(results: dict[str, dict], baseline: str, output: Path,
             color_key: str, color_map: dict, label_fn, show_legend: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output)) as pdf:
        fig1 = page_scatters(results, baseline, color_map, color_key, label_fn, show_legend)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2 = page_scorecard(results, baseline)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)
    print(f"Saved 2-page PDF to {output}")


# ── Markdown: per-model breakdown ────────────────────────────────────────────


def write_breakdown_md(results: dict[str, dict], baseline: str,
                       args: argparse.Namespace, path: Path) -> None:
    """Write a per-model win/tie/loss table (baseline vs each multi condition)."""
    b = short_condition(baseline)
    match = "same-commit" if args.match == "commit" else "commit-agnostic (pooled across commits)"
    lines = [
        f"# {b} vs multi-player — win/tie/loss by model",
        "",
        f"- Matching: **{match}**",
        f"- Tie bands: MRR ±${args.mrr_tol:,.0f}/mo, runway ±{args.runway_tol:g} turns, retention ±{args.churn_tol:g}",
        f"- **W** = {b} (single-player) outperforms · **T** = within tolerance · **L** = {b} loses",
        "- *Cells* counts matched (model, seed, scenario, turns) groups; repeated runs are averaged per cell.",
        "",
    ]
    for mc, rec in results.items():
        pairs = rec["pairs"]
        if not pairs:
            continue
        lines += [f"## {b} vs {short_condition(mc)}", "",
                  "| Model | Cells | Final MRR (W/T/L) | Final runway (W/T/L) | Retention (W/T/L) |",
                  "|---|---:|:---:|:---:|:---:|"]
        by_model: dict[str, list[dict]] = defaultdict(list)
        for p in pairs:
            by_model[p["model"]].append(p)
        tot_n = 0
        tot_mrr = {"win": 0, "tie": 0, "loss": 0}
        tot_rw = {"win": 0, "tie": 0, "loss": 0}
        tot_churn = {"win": 0, "tie": 0, "loss": 0}
        for model in sorted(by_model):
            ps = by_model[model]
            mrr = {"win": 0, "tie": 0, "loss": 0}
            rw = {"win": 0, "tie": 0, "loss": 0}
            churn = {"win": 0, "tie": 0, "loss": 0}
            for p in ps:
                cm = classify(None if None in (p["b_mrr"], p["m_mrr"]) else p["b_mrr"] - p["m_mrr"], args.mrr_tol)
                if cm:
                    mrr[cm] += 1
                cr = classify(None if None in (p["b_rw"], p["m_rw"]) else p["b_rw"] - p["m_rw"], args.runway_tol)
                if cr:
                    rw[cr] += 1
                cc = classify(None if None in (p["b_churn"], p["m_churn"]) else p["b_churn"] - p["m_churn"], args.churn_tol)
                if cc:
                    churn[cc] += 1
            tot_n += len(ps)
            for k in mrr:
                tot_mrr[k] += mrr[k]
                tot_rw[k] += rw[k]
                tot_churn[k] += churn[k]
            lines.append(f"| {short_model_name(model)} | {len(ps)} | "
                         f"{mrr['win']}/{mrr['tie']}/{mrr['loss']} | {rw['win']}/{rw['tie']}/{rw['loss']} | "
                         f"{churn['win']}/{churn['tie']}/{churn['loss']} |")
        lines.append(f"| **All models** | **{tot_n}** | "
                     f"**{tot_mrr['win']}/{tot_mrr['tie']}/{tot_mrr['loss']}** | "
                     f"**{tot_rw['win']}/{tot_rw['tie']}/{tot_rw['loss']}** | "
                     f"**{tot_churn['win']}/{tot_churn['tie']}/{tot_churn['loss']}** |")
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"Saved model breakdown to {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single-player (C2) vs multi-player (C3/C4a/C4b) on matched cells.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s
  %(prog)s --commit 99b41e6 --model sonnet
  %(prog)s --match loose --scenario seed_stage --turns 48
  %(prog)s --multi c3 --mrr-tol 2000""",
    )
    filt = parser.add_argument_group("filters")
    filt.add_argument("--commit", nargs="+", metavar="PREFIX",
                      help="Engine commit prefix(es) to match. Pass multiple to OR them.")
    filt.add_argument("--model", type=str, metavar="SUBSTR", help="Filter by model name (substring)")
    filt.add_argument("--scenario", type=str, default="seed_stage",
                      help="Scenario name (default: seed_stage; pass '' for all)")
    filt.add_argument("--turns", type=int, default=48,
                      help="max_turns to match (default: 48; pass -1 for all)")
    filt.add_argument("--seeds", type=parse_seed_range, metavar="LO,HI",
                      help="Seed range inclusive (e.g. 100,105)")
    filt.add_argument("-n", "--num-runs", type=int, default=5000, help="Max runs to fetch")

    comp = parser.add_argument_group("comparison")
    comp.add_argument("--baseline", type=resolve_condition, default="condition2",
                      help="Baseline (single-player) condition (default: c2)")
    comp.add_argument("--multi", type=resolve_condition, nargs="+",
                      default=["condition3", "condition4a", "condition4b"],
                      help="Multi-player conditions to compare against (default: c3 c4a c4b)")
    comp.add_argument("--match", choices=["commit", "loose"], default="commit",
                      help="commit: hold engine_commit constant (default); loose: ignore it")
    comp.add_argument("--include-incomplete", action="store_true",
                      help="Include runs that did not terminate cleanly (NULL game_over_reason).")
    comp.add_argument("--mrr-tol", type=float, default=1000.0,
                      help="MRR tie band in $/mo (default: 1000)")
    comp.add_argument("--runway-tol", type=float, default=2.0,
                      help="Runway tie band in turns (default: 2.0)")
    comp.add_argument("--churn-tol", type=float, default=0.02,
                      help="Churn retention-score tie band, 0-1 scale (default: 0.02)")
    comp.add_argument("--color-by", choices=["model", "commit"], default="model",
                      help="Scatter point color dimension (default: model). Use 'commit' to show "
                           "the single-vs-multi gap holding across development.")

    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output PDF path (default: auto-generated in results/)")

    args = parser.parse_args()
    if args.scenario == "":
        args.scenario = None
    if args.turns is not None and args.turns < 0:
        args.turns = None
    if args.output is None:
        args.output = str(_build_output_path(args))
    return args


def _build_output_path(args: argparse.Namespace) -> Path:
    results_dir = Path(__file__).resolve().parent.parent / "results"
    parts = ["compare", short_condition(args.baseline).lower(),
             "vs", "-".join(short_condition(m).lower() for m in args.multi), args.match]
    if args.model:
        parts.append(args.model.lower())
    if args.commit:
        parts.append("-".join(c[:8] for c in args.commit))
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return results_dir / ("_".join(parts) + ".pdf")


def main() -> None:
    args = parse_args()
    apply_brand_style()

    conditions = [args.baseline, *args.multi]
    runs = asyncio.run(fetch_runs(args, conditions))
    if not args.include_incomplete:
        runs = [r for r in runs if r["reason"] in CLEAN_REASONS]

    buggy = [r for r in runs if r["score_churn"] is not None and r["score_churn"] > MAX_VALID_CHURN_SCORE]
    if buggy:
        print(f"Excluded {len(buggy)} run(s) with score_churn > {MAX_VALID_CHURN_SCORE:g} "
              f"(buggy retention scoring).", file=sys.stderr)
        runs = [r for r in runs if r["score_churn"] is None or r["score_churn"] <= MAX_VALID_CHURN_SCORE]

    if not runs:
        print("No runs found for the given filters.", file=sys.stderr)
        sys.exit(1)

    cells = build_cells(runs, match_commit=(args.match == "commit"))
    results = compare(cells, args.baseline, args.multi, args.mrr_tol, args.runway_tol, args.churn_tol)

    if not any(rec["n"] for rec in results.values()):
        print("No matched cells (need baseline + a multi condition sharing "
              "model/seed/scenario/turns" + ("/commit" if args.match == "commit" else "") + ").",
              file=sys.stderr)
        sys.exit(1)

    print_scorecard(results, args.baseline, args)
    color_key = args.color_by
    values = sorted({p[color_key] for rec in results.values() for p in rec["pairs"]})
    if color_key == "commit":
        # order commits by earliest run so the blue gradient reads oldest → most recent
        first_seen: dict[str, object] = {}
        for r in runs:
            c, t = r["commit"], r.get("started_at")
            if c in values and t is not None and (c not in first_seen or t < first_seen[c]):
                first_seen[c] = t
        ordered = sorted([c for c in values if c in first_seen], key=lambda c: first_seen[c])
        ordered += [c for c in values if c not in first_seen]
        color_map = assign_blue_gradient(ordered)
        label_fn, show_legend = (lambda v: (v or "none")[:7]), False
    else:
        color_map = assign_colors(values)
        label_fn, show_legend = short_model_name, True
    make_pdf(results, args.baseline, Path(args.output), color_key, color_map, label_fn, show_legend)
    write_breakdown_md(results, args.baseline, args, Path(args.output).with_suffix(".md"))


if __name__ == "__main__":
    main()
