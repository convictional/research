"""Plot training set size vs test F1 variance from ablation Study 5.

Usage:
    cd experiments
    PYTHONPATH=. uv run python goal_alignment_judge/src/scripts/plot_ablation_variance.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SUMMARIES = [
    "goal_alignment_judge/output/results/ablation_train_size/summary_20260412_211531.json",
    "goal_alignment_judge/output/results/ablation_train_size/summary_20260414_194054.json",
]
OUTPUT = Path("goal_alignment_judge/output/eda/ablation_study5_variance.png")

FULL_SIZES = {"adam": 67, "matt": 63}
COLORS = {"adam": "#4a90d9", "matt": "#e07b54"}
SIZES_ORDER = [10, 20, 35, 50, "full"]


def load_data():
    all_data = []
    for path in SUMMARIES:
        p = Path(path)
        if p.exists():
            all_data.extend(json.loads(p.read_text()))
    return all_data


def main():
    all_data = load_data()
    by_key = defaultdict(list)
    for r in all_data:
        size = r["train_size_subsampled"]
        full = FULL_SIZES[r["rater"]]
        label = "full" if size == full else size
        by_key[(r["rater"], label)].append(r["test_macro_f1"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("GEPA Test Macro F1 vs Training Set Size\n(each point = one GEPA run, cache cleared between runs)", fontsize=12)

    for ax, rater in zip(axes, ["adam", "matt"]):
        color = COLORS[rater]
        full_size = FULL_SIZES[rater]
        x_positions = []
        x_labels = []

        for i, size_label in enumerate(SIZES_ORDER):
            key = (rater, size_label)
            if key not in by_key:
                continue
            values = by_key[key]
            n = len(values)
            jitter = np.random.RandomState(42).uniform(-0.12, 0.12, n)
            xs = [i + j for j in jitter]

            # Scatter individual runs
            ax.scatter(xs, values, color=color, alpha=0.7, s=60, zorder=3)

            # Mean line
            mean = np.mean(values)
            ax.hlines(mean, i - 0.3, i + 0.3, colors=color, linewidths=2.5, zorder=4)

            # Std band
            std = np.std(values)
            ax.fill_between([i - 0.3, i + 0.3], mean - std, mean + std, alpha=0.15, color=color)

            label = f"{full_size}*" if size_label == "full" else str(size_label)
            x_positions.append(i)
            x_labels.append(label)

        ax.axhline(0.70, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="0.70 target")
        ax.axhline(0.50, color="lightgray", linestyle=":", linewidth=1, alpha=0.7, label="Multi-rater baseline")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Training set size (* = full)")
        ax.set_ylabel("Test macro F1")
        ax.set_ylim(0.20, 0.90)
        ax.set_title(f"{rater.capitalize()}'s ratings")
        ax.grid(axis="y", alpha=0.3)

        # Annotate n per point
        for i, size_label in enumerate(SIZES_ORDER):
            key = (rater, size_label)
            if key not in by_key:
                continue
            n = len(by_key[key])
            ax.text(i, 0.23, f"n={n}", ha="center", fontsize=8, color="gray")

    # Legend
    target_line = mpatches.Patch(color="gray", label="0.70 target (dashed)")
    baseline_line = mpatches.Patch(color="lightgray", label="Multi-rater baseline (dotted)")
    dot = mpatches.Patch(color="steelblue", alpha=0.7, label="Individual run")
    mean_line = mpatches.Patch(color="steelblue", label="Mean ± 1 std")
    fig.legend(handles=[dot, mean_line, target_line, baseline_line],
               loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
