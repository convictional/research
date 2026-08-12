"""Reports: load run artifacts, compute sections, render summary.md."""

from geo_analyzer.reports.funnel import FunnelTier, compute_funnel, render_sparkline
from geo_analyzer.reports.goals import GoalEvaluation, GoalStatus, evaluate_goal
from geo_analyzer.reports.grounded_gap import GroundedGapRow, compute_grounded_gaps
from geo_analyzer.reports.html import render_html
from geo_analyzer.reports.loader import (
    latest_run_id,
    list_run_ids,
    read_scores_jsonl,
)
from geo_analyzer.reports.multi_run import MultiRunRow, compute_multi_run_trends
from geo_analyzer.reports.summary import SummaryInputs, render_summary
from geo_analyzer.reports.topline import TopLineRow, compute_topline
from geo_analyzer.reports.worst_prompts import WorstPromptRow, compute_worst_prompts

__all__ = [
    "FunnelTier",
    "GoalEvaluation",
    "GoalStatus",
    "GroundedGapRow",
    "MultiRunRow",
    "SummaryInputs",
    "TopLineRow",
    "WorstPromptRow",
    "compute_funnel",
    "compute_grounded_gaps",
    "compute_multi_run_trends",
    "compute_topline",
    "compute_worst_prompts",
    "evaluate_goal",
    "latest_run_id",
    "list_run_ids",
    "read_scores_jsonl",
    "render_html",
    "render_sparkline",
    "render_summary",
]
