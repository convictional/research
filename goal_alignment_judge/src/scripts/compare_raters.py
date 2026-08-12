"""Compare two goal alignment rating CSVs — overlap, agreement, and distributions.

Produces a standalone HTML report showing:
1. Content overlap: which (goal, content) pairs are in each dataset
2. Rating distributions: pinned/neutral/deleted per dataset and per goal
3. Agreement on overlapping items: Cohen's kappa, confusion matrix, per-goal breakdown
4. LLM score comparison: alignment_score correlation and signal agreement on overlap

Usage via CLI:
    make run_experiment ARGS="goal_alignment_judge compare_raters \
        --csv-a input/goal_alignments_adam.csv \
        --csv-b input/goal_alignments_rated.csv"
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def load_ratings_csv(path: Path) -> pd.DataFrame:
    """Load a ratings CSV and derive human_action from pinned_by_id/deleted_at."""
    df = pd.read_csv(path)

    def _classify(row: pd.Series) -> str:
        if pd.notna(row.get("pinned_by_id")) and str(row.get("pinned_by_id", "")).strip():
            return "pinned"
        if pd.notna(row.get("deleted_at")) and str(row.get("deleted_at", "")).strip():
            return "deleted"
        return "neutral"

    df["human_action"] = df.apply(_classify, axis=1)
    df["goal_id"] = df["goal_id"].astype(str)
    df["content_source_url"] = df["content_source_url"].astype(str)
    df["pair_key"] = df["goal_id"] + "|" + df["content_source_url"]
    return df


def compute_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Compute (goal_id, content_source_url) overlap between two datasets."""
    pairs_a = set(df_a["pair_key"])
    pairs_b = set(df_b["pair_key"])
    common = pairs_a & pairs_b
    only_a = pairs_a - pairs_b
    only_b = pairs_b - pairs_a

    # Per-goal breakdown
    goals_a = set(df_a["goal_id"])
    goals_b = set(df_b["goal_id"])
    all_goals = sorted(goals_a | goals_b)

    # Get goal titles from whichever df has them
    goal_titles = {}
    for df in (df_a, df_b):
        for _, row in df[["goal_id", "goal_title"]].drop_duplicates().iterrows():
            goal_titles[str(row["goal_id"])] = str(row["goal_title"])

    per_goal = []
    for gid in all_goals:
        g_pairs_a = {pk for pk in pairs_a if pk.startswith(gid + "|")}
        g_pairs_b = {pk for pk in pairs_b if pk.startswith(gid + "|")}
        g_common = g_pairs_a & g_pairs_b
        per_goal.append({
            "goal_id": gid,
            "goal_title": goal_titles.get(gid, gid[:8]),
            "n_a": len(g_pairs_a),
            "n_b": len(g_pairs_b),
            "n_common": len(g_common),
            "only_a": len(g_pairs_a - g_pairs_b),
            "only_b": len(g_pairs_b - g_pairs_a),
        })

    return {
        "n_a": len(pairs_a),
        "n_b": len(pairs_b),
        "n_common": len(common),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "common_pairs": common,
        "per_goal": per_goal,
    }


def compute_agreement(df_a: pd.DataFrame, df_b: pd.DataFrame, common_pairs: set[str]) -> dict:
    """Compute rating agreement on overlapping items."""
    a_common = df_a[df_a["pair_key"].isin(common_pairs)].set_index("pair_key")
    b_common = df_b[df_b["pair_key"].isin(common_pairs)].set_index("pair_key")

    # Align on common pairs
    common_keys = sorted(common_pairs)
    labels_a = [a_common.loc[k, "human_action"] for k in common_keys]
    labels_b = [b_common.loc[k, "human_action"] for k in common_keys]

    # Cohen's kappa
    kappa = cohen_kappa_score(labels_a, labels_b)

    # Confusion matrix (rows = A, cols = B)
    classes = ["pinned", "neutral", "deleted"]
    cm = confusion_matrix(labels_a, labels_b, labels=classes)

    # Simple agreement rate
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    agreement_rate = agree / len(common_keys) if common_keys else 0

    # Per-goal kappa
    a_common_df = a_common.loc[common_keys]
    b_common_df = b_common.loc[common_keys]

    goal_ids = sorted(set(a_common_df["goal_id"]))
    goal_titles = {}
    for df in (df_a, df_b):
        for _, row in df[["goal_id", "goal_title"]].drop_duplicates().iterrows():
            goal_titles[str(row["goal_id"])] = str(row["goal_title"])

    per_goal = []
    for gid in goal_ids:
        mask_a = a_common_df["goal_id"] == gid
        mask_b = b_common_df["goal_id"] == gid
        ga = a_common_df[mask_a]["human_action"].tolist()
        gb = b_common_df[mask_b]["human_action"].tolist()
        if len(ga) < 2:
            g_kappa = float("nan")
        else:
            try:
                g_kappa = cohen_kappa_score(ga, gb)
            except ValueError:
                g_kappa = float("nan")
        g_agree = sum(1 for a, b in zip(ga, gb) if a == b)
        per_goal.append({
            "goal_id": gid,
            "goal_title": goal_titles.get(gid, gid[:8]),
            "n": len(ga),
            "agree": g_agree,
            "accuracy": g_agree / len(ga) if ga else 0,
            "kappa": g_kappa,
        })

    # Transition matrix: how do labels change from A to B?
    transitions = {}
    for a, b in zip(labels_a, labels_b):
        key = f"{a}→{b}"
        transitions[key] = transitions.get(key, 0) + 1

    return {
        "kappa": kappa,
        "agreement_rate": agreement_rate,
        "n_agree": agree,
        "n_total": len(common_keys),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "per_goal": per_goal,
        "transitions": transitions,
    }


def compute_score_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, common_pairs: set[str]) -> dict:
    """Compare LLM-generated alignment_score and signal on overlapping items."""
    a_common = df_a[df_a["pair_key"].isin(common_pairs)].set_index("pair_key")
    b_common = df_b[df_b["pair_key"].isin(common_pairs)].set_index("pair_key")

    common_keys = sorted(common_pairs)
    scores_a = [float(a_common.loc[k, "alignment_score"]) for k in common_keys]
    scores_b = [float(b_common.loc[k, "alignment_score"]) for k in common_keys]

    # Spearman correlation
    if len(scores_a) > 2:
        corr, pval = spearmanr(scores_a, scores_b)
    else:
        corr, pval = float("nan"), float("nan")

    # Score differences
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / len(diffs) if diffs else 0
    abs_diffs = [abs(d) for d in diffs]
    mean_abs_diff = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0

    # Signal agreement
    signals_a = [str(a_common.loc[k, "signal"]) for k in common_keys]
    signals_b = [str(b_common.loc[k, "signal"]) for k in common_keys]
    signal_agree = sum(1 for a, b in zip(signals_a, signals_b) if a == b)

    signal_classes = ["strong", "medium", "weak"]
    signal_cm = confusion_matrix(signals_a, signals_b, labels=signal_classes)

    return {
        "spearman_corr": corr,
        "spearman_pval": pval,
        "mean_diff": mean_diff,
        "mean_abs_diff": mean_abs_diff,
        "n": len(common_keys),
        "signal_agreement": signal_agree / len(common_keys) if common_keys else 0,
        "signal_confusion": signal_cm.tolist(),
        "signal_classes": signal_classes,
    }


def compute_distributions(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    """Compute action distributions overall and per-goal."""
    def _dist(df: pd.DataFrame) -> dict:
        counts = df["human_action"].value_counts().to_dict()
        total = len(df)
        return {
            "counts": counts,
            "total": total,
            "pcts": {k: v / total * 100 for k, v in counts.items()} if total else {},
        }

    goal_titles = {}
    for df in (df_a, df_b):
        for _, row in df[["goal_id", "goal_title"]].drop_duplicates().iterrows():
            goal_titles[str(row["goal_id"])] = str(row["goal_title"])

    all_goals = sorted(set(df_a["goal_id"]) | set(df_b["goal_id"]))
    per_goal = []
    for gid in all_goals:
        ga = df_a[df_a["goal_id"] == gid]
        gb = df_b[df_b["goal_id"] == gid]
        per_goal.append({
            "goal_id": gid,
            "goal_title": goal_titles.get(gid, gid[:8]),
            "a": _dist(ga) if len(ga) else {"counts": {}, "total": 0, "pcts": {}},
            "b": _dist(gb) if len(gb) else {"counts": {}, "total": 0, "pcts": {}},
        })

    return {
        "overall_a": _dist(df_a),
        "overall_b": _dist(df_b),
        "name_a": name_a,
        "name_b": name_b,
        "per_goal": per_goal,
    }


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "—"


def _bar(value: float, max_val: float = 100, width: int = 120) -> str:
    """Inline CSS bar."""
    pct = min(value / max_val * 100, 100) if max_val else 0
    return f'<div style="background:#4a90d9;height:16px;width:{pct * width / 100:.0f}px;border-radius:3px;display:inline-block"></div>'


def build_html_report(
    name_a: str,
    name_b: str,
    overlap: dict,
    agreement: dict,
    scores: dict,
    distributions: dict,
) -> str:
    """Render all analysis into a standalone HTML report."""
    sections = []

    # --- Summary ---
    sections.append(f"""
    <div class="card summary">
        <h2>Summary</h2>
        <div class="stat-grid">
            <div class="stat"><span class="stat-value">{overlap['n_a']}</span><span class="stat-label">{name_a} items</span></div>
            <div class="stat"><span class="stat-value">{overlap['n_b']}</span><span class="stat-label">{name_b} items</span></div>
            <div class="stat"><span class="stat-value">{overlap['n_common']}</span><span class="stat-label">Common pairs</span></div>
            <div class="stat"><span class="stat-value">{overlap['n_only_a']}</span><span class="stat-label">Only in {name_a}</span></div>
            <div class="stat"><span class="stat-value">{overlap['n_only_b']}</span><span class="stat-label">Only in {name_b}</span></div>
            <div class="stat"><span class="stat-value">{agreement['kappa']:.3f}</span><span class="stat-label">Cohen's kappa</span></div>
            <div class="stat"><span class="stat-value">{agreement['agreement_rate']:.1%}</span><span class="stat-label">Agreement rate</span></div>
            <div class="stat"><span class="stat-value">{scores['spearman_corr']:.3f}</span><span class="stat-label">Score Spearman r</span></div>
        </div>
    </div>""")

    # --- Content Overlap per Goal ---
    rows = ""
    for g in overlap["per_goal"]:
        rows += f"""<tr>
            <td>{g['goal_title']}</td><td class="num">{g['n_a']}</td><td class="num">{g['n_b']}</td>
            <td class="num">{g['n_common']}</td><td class="num">{g['only_a']}</td><td class="num">{g['only_b']}</td>
        </tr>"""
    sections.append(f"""
    <div class="card">
        <h2>Content Overlap by Goal</h2>
        <table><thead><tr><th>Goal</th><th>{name_a}</th><th>{name_b}</th><th>Common</th><th>Only {name_a}</th><th>Only {name_b}</th></tr></thead>
        <tbody>{rows}</tbody></table>
    </div>""")

    # --- Action Distributions ---
    actions = ["pinned", "neutral", "deleted"]
    dist_rows = ""
    for g in distributions["per_goal"]:
        cells = f"<td>{g['goal_title']}</td>"
        for action in actions:
            ca = g["a"]["counts"].get(action, 0)
            cb = g["b"]["counts"].get(action, 0)
            pa = g["a"]["pcts"].get(action, 0)
            pb = g["b"]["pcts"].get(action, 0)
            diff_class = ""
            if ca != cb:
                diff_class = ' class="diff"'
            cells += f"<td{diff_class}>{ca} ({pa:.0f}%)</td><td{diff_class}>{cb} ({pb:.0f}%)</td>"
        dist_rows += f"<tr>{cells}</tr>"

    da = distributions["overall_a"]
    db = distributions["overall_b"]
    total_row = "<td><strong>Total</strong></td>"
    for action in actions:
        ca = da["counts"].get(action, 0)
        cb = db["counts"].get(action, 0)
        pa = da["pcts"].get(action, 0)
        pb = db["pcts"].get(action, 0)
        total_row += f"<td><strong>{ca} ({pa:.0f}%)</strong></td><td><strong>{cb} ({pb:.0f}%)</strong></td>"

    sections.append(f"""
    <div class="card">
        <h2>Action Distributions</h2>
        <table><thead><tr>
            <th>Goal</th>
            <th>Pinned ({name_a})</th><th>Pinned ({name_b})</th>
            <th>Neutral ({name_a})</th><th>Neutral ({name_b})</th>
            <th>Deleted ({name_a})</th><th>Deleted ({name_b})</th>
        </tr></thead>
        <tbody>{dist_rows}<tr>{total_row}</tr></tbody></table>
    </div>""")

    # --- Agreement Confusion Matrix ---
    cm = agreement["confusion_matrix"]
    classes = agreement["classes"]
    cm_rows = ""
    for i, cls_a in enumerate(classes):
        cells = f"<td><strong>{cls_a}</strong></td>"
        for j, cls_b in enumerate(classes):
            val = cm[i][j]
            highlight = ' class="diag"' if i == j else (' class="critical"' if abs(i - j) == 2 else "")
            cells += f"<td{highlight}>{val}</td>"
        cm_rows += f"<tr>{cells}</tr>"

    sections.append(f"""
    <div class="card">
        <h2>Agreement: Action Confusion Matrix</h2>
        <p>Rows = {name_a}, Columns = {name_b}. Diagonal = agreement. Corners = critical disagreements (pinned↔deleted).</p>
        <table class="cm"><thead><tr><th>{name_a} \\ {name_b}</th>{"".join(f"<th>{c}</th>" for c in classes)}</tr></thead>
        <tbody>{cm_rows}</tbody></table>
        <p>Cohen's kappa = <strong>{agreement['kappa']:.3f}</strong> | Agreement = <strong>{agreement['n_agree']}/{agreement['n_total']}</strong> ({agreement['agreement_rate']:.1%})</p>
    </div>""")

    # --- Per-goal Agreement ---
    goal_rows = ""
    for g in sorted(agreement["per_goal"], key=lambda x: x["accuracy"]):
        kappa_str = f"{g['kappa']:.3f}" if not pd.isna(g["kappa"]) else "—"
        goal_rows += f"""<tr>
            <td>{g['goal_title']}</td><td class="num">{g['n']}</td>
            <td class="num">{g['agree']}/{g['n']}</td><td class="num">{g['accuracy']:.1%}</td>
            <td class="num">{kappa_str}</td>
        </tr>"""
    sections.append(f"""
    <div class="card">
        <h2>Per-Goal Agreement</h2>
        <table><thead><tr><th>Goal</th><th>n</th><th>Agree</th><th>Accuracy</th><th>Kappa</th></tr></thead>
        <tbody>{goal_rows}</tbody></table>
    </div>""")

    # --- LLM Score Comparison ---
    sig_cm = scores["signal_confusion"]
    sig_classes = scores["signal_classes"]
    sig_rows = ""
    for i, cls_a in enumerate(sig_classes):
        cells = f"<td><strong>{cls_a}</strong></td>"
        for j in range(len(sig_classes)):
            val = sig_cm[i][j]
            highlight = ' class="diag"' if i == j else ""
            cells += f"<td{highlight}>{val}</td>"
        sig_rows += f"<tr>{cells}</tr>"

    sections.append(f"""
    <div class="card">
        <h2>LLM Alignment Score Comparison (Overlapping Items)</h2>
        <p>Both datasets were scored by the same LLM prompt. Differences indicate non-determinism or context changes.</p>
        <div class="stat-grid">
            <div class="stat"><span class="stat-value">{scores['spearman_corr']:.3f}</span><span class="stat-label">Spearman r (p={scores['spearman_pval']:.2e})</span></div>
            <div class="stat"><span class="stat-value">{scores['mean_diff']:+.4f}</span><span class="stat-label">Mean score diff (A−B)</span></div>
            <div class="stat"><span class="stat-value">{scores['mean_abs_diff']:.4f}</span><span class="stat-label">Mean |score diff|</span></div>
            <div class="stat"><span class="stat-value">{scores['signal_agreement']:.1%}</span><span class="stat-label">Signal agreement</span></div>
        </div>
        <h3>Signal Confusion Matrix</h3>
        <p>Rows = {name_a}, Columns = {name_b}</p>
        <table class="cm"><thead><tr><th>{name_a} \\ {name_b}</th>{"".join(f"<th>{c}</th>" for c in sig_classes)}</tr></thead>
        <tbody>{sig_rows}</tbody></table>
    </div>""")

    # --- Assemble ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Rater Comparison: {name_a} vs {name_b}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1100px; margin: 20px auto; padding: 0 20px; background: #f5f5f5; color: #333; }}
    h1 {{ color: #222; border-bottom: 2px solid #4a90d9; padding-bottom: 8px; }}
    .card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }}
    .stat {{ text-align: center; padding: 12px; background: #f8f9fa; border-radius: 6px; }}
    .stat-value {{ display: block; font-size: 1.5em; font-weight: bold; color: #4a90d9; }}
    .stat-label {{ display: block; font-size: 0.85em; color: #666; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 0.9em; }}
    th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; }}
    th {{ background: #f0f4f8; font-weight: 600; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .diag {{ background: #e8f5e9; font-weight: bold; }}
    .critical {{ background: #ffebee; font-weight: bold; }}
    .diff {{ background: #fff8e1; }}
    .cm {{ max-width: 400px; }}
</style>
</head>
<body>
<h1>Rater Comparison: {name_a} vs {name_b}</h1>
{"".join(sections)}
</body>
</html>"""
    return html


def compare_raters(csv_a: Path, csv_b: Path, output: Path) -> Path:
    """Run the full comparison and write the HTML report."""
    name_a = csv_a.stem
    name_b = csv_b.stem

    df_a = load_ratings_csv(csv_a)
    df_b = load_ratings_csv(csv_b)

    overlap = compute_overlap(df_a, df_b)
    common_pairs = overlap["common_pairs"]

    agreement = compute_agreement(df_a, df_b, common_pairs)
    scores = compute_score_comparison(df_a, df_b, common_pairs)
    distributions = compute_distributions(df_a, df_b, name_a, name_b)

    html = build_html_report(name_a, name_b, overlap, agreement, scores, distributions)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"Report written to {output}")

    # Console summary
    print(f"\n  {name_a}: {overlap['n_a']} items | {name_b}: {overlap['n_b']} items")
    print(f"  Common: {overlap['n_common']} | Only {name_a}: {overlap['n_only_a']} | Only {name_b}: {overlap['n_only_b']}")
    print(f"  Cohen's kappa: {agreement['kappa']:.3f} | Agreement: {agreement['agreement_rate']:.1%}")
    print(f"  Score Spearman r: {scores['spearman_corr']:.3f} | Signal agreement: {scores['signal_agreement']:.1%}")

    return output
