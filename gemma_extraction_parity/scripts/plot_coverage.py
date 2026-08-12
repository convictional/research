"""Generate the Sonnet <-> Gemma per-prompt coverage figure for EXPERIMENT.md.

Data is the 3-trial mean and [min-max] range per prompt, transcribed from the
9 parity reports generated 2026-06-09 (see output/trial_{1,2,3}/ and the
per-prompt table in EXPERIMENT.md). Ordered by descending S->G mean.

Usage (from experiments/):
    PYTHONPATH=. uv run python gemma_extraction_parity/scripts/plot_coverage.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROMPTS = ["q5 bio", "q13 sales", "q11 my work", "q6 research fn", "q2 design", "q9 papers", "q1 analytics", "q7 intern"]

SG_MEAN = np.array([93, 88, 85, 73, 71, 71, 67, 59])
SG_MIN = np.array([86, 76, 75, 73, 62, 65, 61, 56])
SG_MAX = np.array([100, 100, 93, 74, 88, 78, 78, 60])

GS_MEAN = np.array([50, 52, 60, 65, 65, 75, 69, 67])
GS_MIN = np.array([44, 42, 54, 59, 58, 65, 64, 45])
GS_MAX = np.array([53, 66, 63, 73, 69, 82, 78, 100])

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREY = "#666666"

x = np.arange(len(PROMPTS))
fig, ax = plt.subplots(figsize=(11, 5.5))

bars = ax.bar(
    x,
    SG_MEAN,
    width=0.58,
    color=BLUE,
    alpha=0.85,
    yerr=[SG_MEAN - SG_MIN, SG_MAX - SG_MEAN],
    error_kw={"ecolor": "#2A4A7A", "elinewidth": 1.2, "capsize": 4, "capthick": 1.2},
    label="Sonnet → Gemma",
    zorder=2,
)

ax.errorbar(
    x,
    GS_MEAN,
    yerr=[GS_MEAN - GS_MIN, GS_MAX - GS_MEAN],
    fmt="o-",
    color=ORANGE,
    linewidth=2,
    markersize=7,
    capsize=4,
    capthick=1.2,
    label="Gemma → Sonnet",
    zorder=3,
)

ax.axhline(80, color=GREY, linestyle="--", linewidth=1.2, zorder=1)
ax.text(len(PROMPTS) - 0.45, 81, "80% target (S→G)", color=GREY, fontsize=9, ha="right")

for xi, mean in zip(x, SG_MEAN):
    ax.text(xi, 33, f"{mean}%", ha="center", va="bottom", color="white", fontsize=9.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(PROMPTS, fontsize=10)
ax.set_ylim(30, 105)
ax.set_ylabel("Coverage (%)", fontsize=11)
ax.set_title(
    "Sonnet ↔ Gemma coverage by prompt\n3-trial mean; whiskers = trial min–max; ordered by descending S→G",
    fontsize=12,
    pad=12,
)
ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", ncol=2, fontsize=10, framealpha=0.95)

fig.tight_layout()
out = __file__.rsplit("/scripts/", 1)[0] + "/figures/sonnet_gemma_coverage.png"
fig.savefig(out, dpi=200)
print(f"Saved {out}")
