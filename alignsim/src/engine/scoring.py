"""Goal attainment scoring functions.

Two-layer scoring (primary goals + function sub-goals). Each layer's headline is the
`composite` — the **geometric mean** of its scores: a single 0 zeroes it (a goal cannot
be ignored), a weak leg drags the whole score down, and — because sub-scores are uncapped
above 1 — exceeding a target can lift the composite above par. `pareto` (min) is still
computed and logged for continuity but is superseded by the composite.
Sub-scores are uncapped: 1.0 = hit target, >1.0 = exceeded, <1.0 = fell short.
"""

import math

from alignsim.src.models.entities import CustomerStage, FeatureStatus, ProcessProjectStatus
from alignsim.src.models.game_state import GameState
from alignsim.src.models.goals import GoalAttainmentScore, PrimaryGoal


def compute_goal_attainment(state: GameState, goal: PrimaryGoal) -> GoalAttainmentScore:
    """Compute uncapped goal attainment across primary and function layers."""
    # Primary scores (uncapped — 1.0 = par)
    mrr_score = state.resources.mrr / goal.mrr_target if goal.mrr_target > 0 else 1.0

    # Churn → retention rate: 1 - avg_churn_rate, bounded [0,1]
    avg_churn_rate = _compute_avg_churn_rate(state)
    churn_score = max(0.0, 1.0 - avg_churn_rate)

    # Runway → log scale capped at 4× target.
    # Par = 1.0 at target; ~1.585 at 2× target; caps at ~2.322 at 4× target.
    # Handles runway_turns=999 (profitable) via the 4× cap.
    if goal.min_runway_turns > 0:
        bounded_runway = min(max(0.0, state.resources.runway_turns), 4 * goal.min_runway_turns)
        runway_score = math.log(1 + bounded_runway / goal.min_runway_turns) / math.log(2)
    else:
        runway_score = 1.0
    runway_score = max(0.0, runway_score)

    primary_scores = [mrr_score, churn_score, runway_score]
    composite = _geometric_mean(primary_scores)
    pareto_score = min(primary_scores)  # retained for continuity; superseded by composite

    # Function sub-goal scores (uncapped)
    function_scores = {}
    for sub_goal in getattr(goal, "sub_goals", []):
        actual = _extract_metric(state, sub_goal.metric)
        attainment = actual / sub_goal.target_value if sub_goal.target_value > 0 else 1.0
        function_scores[sub_goal.role] = round(attainment, 4)

    func_values = list(function_scores.values())
    function_composite = _geometric_mean(func_values)
    function_pareto = min(func_values) if func_values else 0.0

    return GoalAttainmentScore(
        mrr_score=round(mrr_score, 4),
        churn_score=round(churn_score, 4),
        runway_score=round(runway_score, 4),
        composite=round(composite, 4),
        pareto_score=round(pareto_score, 4),
        function_scores=function_scores,
        function_composite=round(function_composite, 4),
        function_pareto=round(function_pareto, 4),
        final_mrr=state.resources.mrr,
        avg_churn_rate=round(avg_churn_rate, 4),
        final_runway_turns=round(state.resources.runway_turns, 2),
        final_turn=state.turn,
    )


def _geometric_mean(scores: list[float]) -> float:
    """Geometric mean of non-negative scores; 0 if any score is 0 or the list is empty.

    Replaces the old additive composite: a single 0 zeroes the result (a goal cannot be
    ignored), a weak leg drags the whole score down, and — because sub-scores are uncapped
    above 1 — exceeding a target can still lift the composite above par (1.0).
    """
    if not scores:
        return 0.0
    product = 1.0
    for s in scores:
        product *= s
    return product ** (1.0 / len(scores))


def _extract_metric(state: GameState, metric: str) -> float:
    """Extract a metric value from game state for sub-goal evaluation."""
    if metric == "features_shipped_solid_plus":
        return float(sum(
            1 for f in state.features.values()
            if f.status in (FeatureStatus.shipped_solid, FeatureStatus.shipped_polished)
        ))
    elif metric == "pipeline_velocity":
        turns = max(state.turn - 1, 1)
        return state.total_customers_closed / turns
    elif metric == "avg_customer_health":
        active = [c for c in state.customers.values() if c.stage == CustomerStage.customer]
        if not active:
            return 0.0
        return sum(c.health for c in active) / len(active)
    elif metric == "marketing_leads_generated":
        count = 0
        for record in state.turn_history:
            count += sum(1 for e in record.events if e.startswith("inbound_lead:"))
        return float(count)
    elif metric == "process_projects_completed":
        return float(sum(
            1 for p in state.process_projects.values()
            if p.status == ProcessProjectStatus.completed
        ))
    elif metric == "tech_debt_control":
        return max(0.0, 1.0 - state.tech_debt.level / 15.0)
    raise ValueError(f"Unknown sub-goal metric: {metric!r}. Check RoleSubGoal configuration in scenario.")


def _compute_avg_churn_rate(state: GameState) -> float:
    """Cumulative churn rate: total churned / total customers ever active."""
    total_churned = sum(state.churn_history) if state.churn_history else 0
    current_active = sum(1 for c in state.customers.values() if c.stage == CustomerStage.customer)
    total_ever_active = current_active + total_churned
    if total_ever_active == 0:
        return 0.0
    return total_churned / total_ever_active
