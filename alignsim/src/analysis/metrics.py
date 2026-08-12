"""Post-game metric computation and analysis."""

from alignsim.src.models.actions import (
    AnalysisScopeAction,
    BuildAction,
    DiscoverAction,
    FixBugsAction,
    HireAction,
    InfrastructureAction,
    MarketAction,
    MarketSupportAction,
    OpsAnalysisAction,
    SellAction,
    SupportAction,
)
from alignsim.src.models.game_state import GameState
from alignsim.src.models.goals import GoalAttainmentScore


def compute_game_metrics(state: GameState, score: GoalAttainmentScore) -> dict:
    """Compute detailed metrics from a completed game for analysis."""
    return {
        "score": score.model_dump(),
        "turns_played": state.turn - 1,
        "game_over_reason": state.game_over_reason,
        "trajectory": _compute_trajectory(state),
        "action_distribution": _compute_action_distribution(state),
        "validation_stats": _compute_validation_stats(state),
        "customer_outcomes": _compute_customer_outcomes(state),
        "feature_outcomes": _compute_feature_outcomes(state),
    }


def _compute_trajectory(state: GameState) -> dict:
    """MRR, runway, and churn trajectory across turns."""
    return {
        "mrr_by_turn": [r.mrr for r in state.turn_history],
        "runway_by_turn": [round(r.runway_turns, 1) for r in state.turn_history],
        "budget_by_turn": [r.budget for r in state.turn_history],
        "churn_by_turn": state.churn_history,
        "bugs_injected_by_turn": [r.bugs_injected for r in state.turn_history],
        "bugs_fixed_by_turn": [r.bugs_fixed for r in state.turn_history],
        "capacity_used_by_turn": [r.capacity_used for r in state.turn_history],
    }


def _compute_action_distribution(state: GameState) -> dict:
    """How capacity was allocated across action types over the game."""
    totals: dict[str, int] = {
        "build": 0, "fix_bugs": 0, "infrastructure": 0,
        "sell": 0, "discover": 0, "support": 0,
        "market": 0, "market_support": 0, "hire_count": 0,
        "ops_analysis": 0, "analysis_scope": 0,
    }

    for record in state.turn_history:
        for action in record.actions_valid:
            if isinstance(action, BuildAction):
                totals["build"] += action.capacity
            elif isinstance(action, FixBugsAction):
                totals["fix_bugs"] += action.capacity
            elif isinstance(action, InfrastructureAction):
                totals["infrastructure"] += action.capacity
            elif isinstance(action, SellAction):
                totals["sell"] += action.capacity
            elif isinstance(action, DiscoverAction):
                totals["discover"] += action.capacity
            elif isinstance(action, SupportAction):
                totals["support"] += action.capacity
            elif isinstance(action, MarketAction):
                totals["market"] += action.capacity
            elif isinstance(action, MarketSupportAction):
                totals["market_support"] += action.capacity
            elif isinstance(action, OpsAnalysisAction):
                totals["ops_analysis"] += action.capacity
            elif isinstance(action, AnalysisScopeAction):
                totals["analysis_scope"] += action.capacity
            elif isinstance(action, HireAction):
                totals["hire_count"] += 1

    total_capacity = sum(v for k, v in totals.items() if k != "hire_count")
    percentages = {}
    if total_capacity > 0:
        for k, v in totals.items():
            if k != "hire_count":
                percentages[k] = round(v / total_capacity * 100, 1)

    return {
        "capacity_totals": totals,
        "capacity_percentages": percentages,
        "total_capacity_used": total_capacity,
        "total_capacity_available": sum(r.capacity_available for r in state.turn_history),
    }


def _compute_validation_stats(state: GameState) -> dict:
    """How many actions were rejected and why."""
    total_submitted = sum(len(r.actions_submitted) for r in state.turn_history)
    total_valid = sum(len(r.actions_valid) for r in state.turn_history)
    total_rejected = sum(len(r.actions_rejected) for r in state.turn_history)

    rejection_reasons: dict[str, int] = {}
    for record in state.turn_history:
        for rejection in record.actions_rejected:
            # Bucket by first word of reason
            bucket = rejection.reason.split(":")[0] if ":" in rejection.reason else rejection.reason[:50]
            rejection_reasons[bucket] = rejection_reasons.get(bucket, 0) + 1

    return {
        "total_submitted": total_submitted,
        "total_valid": total_valid,
        "total_rejected": total_rejected,
        "rejection_rate": round(total_rejected / max(total_submitted, 1) * 100, 1),
        "rejection_reasons": rejection_reasons,
    }


def _compute_customer_outcomes(state: GameState) -> dict:
    """Final state of all customers."""
    outcomes: dict[str, int] = {}
    for customer in state.customers.values():
        stage = customer.stage.value
        outcomes[stage] = outcomes.get(stage, 0) + 1

    total_churned = sum(state.churn_history)
    active = [c for c in state.customers.values() if c.stage.value == "customer"]
    avg_health = sum(c.health for c in active) / len(active) if active else 0

    return {
        "stage_distribution": outcomes,
        "total_churned": total_churned,
        "active_count": len(active),
        "avg_health": round(avg_health, 1),
        "total_deal_value_closed": sum(
            c.deal_value for c in state.customers.values()
            if c.stage.value == "customer"
        ),
    }


def _compute_feature_outcomes(state: GameState) -> dict:
    """Final state of all features."""
    status_counts: dict[str, int] = {}
    for feature in state.features.values():
        status = feature.status.value
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "status_distribution": status_counts,
        "total_bugs_injected": sum(r.bugs_injected for r in state.turn_history),
        "total_bugs_fixed": sum(r.bugs_fixed for r in state.turn_history),
        "unresolved_bugs": sum(1 for b in state.bugs if not b.is_resolved),
        "final_tech_debt": round(state.tech_debt.level, 2),
        "tech_debt_category": state.tech_debt.category,
    }
