"""GameInspector: read-only wrapper enforcing information boundaries.

The agent's tools route through this class. It exposes only
player-visible data — never rubric weights, hidden feature_needs
scores, dealbreakers, churn_drivers, or discovery_difficulty.
"""

from __future__ import annotations

import math

from alignsim.src.engine import customer_logic
from alignsim.src.engine.game import GameEngine
from alignsim.src.models.entities import CustomerStage, FeatureStatus, QualityLevel
from alignsim.src.models.scenario import CalibrationParams


class GameInspector:
    """Read-only view into game state, respecting information boundaries."""

    def __init__(self, engine: GameEngine) -> None:
        self._engine = engine
        self._rejection_history: list[dict] = []

    @property
    def _state(self):
        return self._engine.state

    @property
    def _calibration(self):
        return self._engine.scenario.calibration

    # -- Mutations (called by harness after engine.step) --

    def record_rejections(self, turn: int, rejections: list[dict]) -> None:
        """Append rejection records from a completed turn."""
        for rej in rejections:
            self._rejection_history.append({"turn": turn, **rej})

    # -- Query tools --

    def get_customer_details(self, customer_id: str) -> dict:
        customer = self._state.customers.get(customer_id)
        if customer is None:
            return {"error": f"Customer {customer_id} not found"}
        if not customer.is_visible:
            return {"error": f"Customer {customer_id} is not yet discovered"}

        result: dict = {
            "id": customer.id,
            "size": customer.size,
            "segment": customer.segment.value,
            "stage": customer.stage.value,
            "engagement": customer.engagement.value,
            "known_needs": customer.known_needs,
            "deal_value": customer.deal_value,
            "competitive_pressure": round(customer.competitive_pressure, 2),
        }

        # Pricing info for in_deal
        if customer.stage == CustomerStage.in_deal and customer.last_proposed_price is not None:
            result["last_proposed_price"] = customer.last_proposed_price

        # Timeline only visible for pipeline customers
        if customer.stage in (
            CustomerStage.prospect, CustomerStage.qualified, CustomerStage.in_deal
        ):
            result["timeline_remaining"] = customer.timeline

        # Health only visible for active customers
        if customer.stage == CustomerStage.customer:
            result["health"] = round(customer.health, 1)
            if len(customer.health_history) >= 2:
                diff = customer.health - customer.health_history[-2]
                if diff > 0.5:
                    result["health_trend"] = "improving"
                elif diff < -0.5:
                    result["health_trend"] = "declining"
                else:
                    result["health_trend"] = "stable"
            else:
                result["health_trend"] = "stable"
            if customer.onboarding_turns_remaining > 0:
                result["onboarding_remaining"] = customer.onboarding_turns_remaining

        # Min sell capacity per valid action
        if customer.stage in (
            CustomerStage.lead, CustomerStage.prospect,
            CustomerStage.qualified, CustomerStage.in_deal,
        ):
            valid_actions = {
                "lead": ["outbound"],
                "prospect": ["outbound", "demo"],
                "qualified": ["demo"],
                "in_deal": ["proposal", "negotiate"],
            }
            min_caps = {}
            for sa in valid_actions.get(customer.stage.value, []):
                min_caps[sa] = customer_logic.compute_sell_minimum_capacity(
                    customer, sa, self._calibration
                )
            result["min_sell_capacity"] = min_caps

        return result

    def get_feature_status(self, feature_id: str) -> dict:
        feature = self._state.features.get(feature_id)
        if feature is None:
            return {"error": f"Feature {feature_id} not found"}

        # Check blocked_by
        blocked_by = None
        for dep_id in feature.depends_on:
            dep = self._state.features.get(dep_id)
            if dep and dep.status in (FeatureStatus.not_started, FeatureStatus.in_progress):
                blocked_by = dep_id
                break

        # Estimate completion turns (reuse observer logic)
        est_completion = None
        if feature.status == FeatureStatus.in_progress and feature.current_target:
            remaining_pct = 100.0 - feature.progress
            cost = feature.cost.get(feature.current_target.value, 0)
            if cost > 0:
                remaining_cost = cost * (remaining_pct / 100.0)
                effective_per_turn = min(
                    self._calibration.build_optimal_capacity,
                    cost * (self._calibration.build_max_progress_pct / 100.0),
                )
                est_completion = max(1, int(math.ceil(remaining_cost / effective_per_turn)))
                min_turns = max(2, math.ceil(cost * self._calibration.build_min_turns_factor))
                remaining_min = max(0, min_turns - feature.turns_worked)
                est_completion = max(est_completion, remaining_min)

        return {
            "id": feature.id,
            "name": feature.name,
            "description": feature.description,
            "cost": feature.cost,
            "depends_on": feature.depends_on,
            "status": feature.status.value,
            "progress": round(feature.progress, 1),
            "current_target": feature.current_target.value if feature.current_target else None,
            "turns_worked": feature.turns_worked,
            "blocked_by": blocked_by,
            "est_completion_turns": est_completion,
        }

    def list_bugs(self) -> list[dict]:
        return [
            {
                "id": bug.id,
                "severity": bug.severity.value,
                "feature_id": bug.feature_id,
                "turn_injected": bug.turn_injected,
                "turns_unresolved": bug.turns_unresolved,
                "affected_customers": bug.affected_customers,
            }
            for bug in self._state.bugs
            if not bug.is_resolved
        ]

    def get_rejection_history(self) -> list[dict]:
        return list(self._rejection_history)

    # -- Compute tools --

    def compute_maturity(self) -> dict:
        shipped = [
            f for f in self._state.features.values()
            if f.status in _SHIPPED_STATUSES
        ]
        if not shipped:
            return {
                "maturity_score": 0.0,
                "total_shipped": 0,
                "polished_count": 0,
                "solid_count": 0,
                "mvp_count": 0,
                "features": [],
            }

        polished = sum(1 for f in shipped if f.status == FeatureStatus.shipped_polished)
        solid = sum(1 for f in shipped if f.status == FeatureStatus.shipped_solid)
        mvp = sum(1 for f in shipped if f.status == FeatureStatus.shipped_mvp)
        maturity = (polished * 1.0 + solid * 0.6) / len(shipped)

        return {
            "maturity_score": round(maturity, 3),
            "total_shipped": len(shipped),
            "polished_count": polished,
            "solid_count": solid,
            "mvp_count": mvp,
            "features": [
                {"feature_id": f.id, "name": f.name, "status": f.status.value}
                for f in shipped
            ],
        }

    def estimate_satisfaction(self, customer_id: str) -> dict:
        customer = self._state.customers.get(customer_id)
        if customer is None:
            return {"error": f"Customer {customer_id} not found"}
        if not customer.is_visible:
            return {"error": f"Customer {customer_id} is not yet discovered"}

        satisfaction = customer_logic.compute_rubric_satisfaction(
            customer, self._state.features
        )
        dealbreakers_met = customer_logic.has_dealbreakers_met(
            customer, self._state.features
        )
        threshold = customer.close_threshold if customer.close_threshold > 0 else self._calibration.min_rubric_for_close

        return {
            "customer_id": customer_id,
            "satisfaction_score": round(satisfaction, 3),
            "dealbreakers_met": dealbreakers_met,
            "min_for_close": self._calibration.min_rubric_for_close,
            "gap": round(max(0, threshold - satisfaction), 3),
            "can_close": satisfaction >= threshold and dealbreakers_met,
        }

    def simulate_maturity_change(self, feature_id: str, target_quality: str) -> dict:
        feature = self._state.features.get(feature_id)
        if feature is None:
            return {"error": f"Feature {feature_id} not found"}

        try:
            target_q = QualityLevel(target_quality)
        except ValueError:
            return {"error": f"Invalid quality '{target_quality}'. Use mvp, solid, or polished."}

        # Current maturity
        current = self.compute_maturity()
        current_score = current["maturity_score"]

        # Hypothetical: what if this feature were at target_quality?
        target_status = {
            QualityLevel.mvp: FeatureStatus.shipped_mvp,
            QualityLevel.solid: FeatureStatus.shipped_solid,
            QualityLevel.polished: FeatureStatus.shipped_polished,
        }[target_q]

        # Count shipped features in the hypothetical scenario
        shipped = []
        for f in self._state.features.values():
            if f.id == feature_id:
                shipped.append(target_status)
            elif f.status in _SHIPPED_STATUSES:
                shipped.append(f.status)

        if not shipped:
            return {
                "feature_id": feature_id,
                "target_quality": target_quality,
                "current_maturity": current_score,
                "hypothetical_maturity": 0.0,
                "delta": 0.0,
            }

        polished = sum(1 for s in shipped if s == FeatureStatus.shipped_polished)
        solid = sum(1 for s in shipped if s == FeatureStatus.shipped_solid)
        hyp_maturity = (polished * 1.0 + solid * 0.6) / len(shipped)

        return {
            "feature_id": feature_id,
            "target_quality": target_quality,
            "current_maturity": round(current_score, 3),
            "hypothetical_maturity": round(hyp_maturity, 3),
            "delta": round(hyp_maturity - current_score, 3),
        }

    def compute_capacity_cost(self, actions: list[dict]) -> dict:
        pools = {
            "engineering": {"used": 0, "available": self._state.resources.eng_capacity},
            "sales": {"used": 0, "available": self._state.resources.sales_capacity},
            "support": {"used": 0, "available": self._state.resources.support_capacity},
            "marketing": {"used": 0, "available": self._state.resources.marketing_capacity},
            "ops": {"used": 0, "available": self._state.resources.ops_capacity},
        }

        pool_map = {
            "build": "engineering",
            "fix_bugs": "engineering",
            "infrastructure": "engineering",
            "sell": "sales",
            "discover": "sales",
            "market_support": "sales",
            "support": "support",
            "market": "marketing",
            "ops_project": "ops",
            "ops_analysis": "ops",
            "hire": None,
            "sustain_hire": None,
            "ops_project_support": None,
            "analysis_scope": None,
        }

        func_to_pool = {
            "engineering": "engineering", "sales": "sales",
            "cs": "support", "marketing": "marketing", "ops": "ops",
        }

        hire_capacity_cost = self._calibration.hire_capacity_cost if self._calibration else CalibrationParams().hire_capacity_cost
        warnings: list[str] = []

        for action in actions:
            if not isinstance(action, dict):
                warnings.append(
                    f"malformed action ignored (expected a JSON object, got {type(action).__name__}): {str(action)[:60]}"
                )
                continue
            action_type = action.get("action_type", "")
            pool_name = pool_map.get(action_type)
            if pool_name is None:
                if action_type == "hire":
                    fn = action.get("hiring_function", "")
                    fn_pool = func_to_pool.get(fn)
                    if fn_pool:
                        pools[fn_pool]["used"] += hire_capacity_cost
                    continue
                if action_type == "sustain_hire":
                    hire_id = action.get("hire_id", "")
                    hire = next((h for h in self._state.pending_hires if h.id == hire_id), None)
                    if hire:
                        fn_pool = func_to_pool.get(hire.hiring_function)
                        if fn_pool:
                            pools[fn_pool]["used"] += hire_capacity_cost
                    else:
                        warnings.append(f"sustain_hire: hire {hire_id} not found")
                    continue
                if action_type == "ops_project_support":
                    pid = action.get("project_id", "")
                    project = self._state.process_projects.get(pid)
                    if project:
                        tfn_pool = func_to_pool.get(project.target_function)
                        if tfn_pool:
                            pools[tfn_pool]["used"] += action.get("capacity", 0)
                    else:
                        warnings.append(f"ops_project_support: project {pid} not found")
                    continue
                if action_type == "analysis_scope":
                    # Scope co-investment draws from the requesting team's pool (cs -> support).
                    tfn_pool = func_to_pool.get(action.get("target_function", ""))
                    if tfn_pool:
                        pools[tfn_pool]["used"] += action.get("capacity", 0)
                    else:
                        warnings.append(
                            f"analysis_scope: unknown target_function {action.get('target_function')}"
                        )
                    continue
                warnings.append(f"Unknown action_type: {action_type}")
                continue

            capacity = action.get("capacity", 0)
            pools[pool_name]["used"] += capacity

            # Check sell minimum capacity
            if action_type == "sell":
                cid = action.get("customer_id", "")
                sa = action.get("sell_action", "")
                customer = self._state.customers.get(cid)
                if customer:
                    min_cap = customer_logic.compute_sell_minimum_capacity(
                        customer, sa, self._calibration
                    )
                    if capacity < min_cap:
                        warnings.append(
                            f"sell {cid} ({sa}): capacity {capacity} < minimum {min_cap} "
                            f"(customer size {customer.size})"
                        )

        over_limit = []
        for pool_name, pool in pools.items():
            pool["over"] = max(0, pool["used"] - pool["available"])
            if pool["over"] > 0:
                over_limit.append(
                    f"{pool_name}: {pool['used']} used > {pool['available']} available"
                )

        return {
            **pools,
            "any_over_limit": len(over_limit) > 0,
            "over_limit_details": over_limit,
            "warnings": warnings,
        }

    # -- Game status --

    def get_status(self) -> dict:
        goal = self._engine.scenario.primary_goal
        mrr = self._state.resources.mrr
        mrr_pct = (mrr / goal.mrr_target) * 100 if goal.mrr_target > 0 else 0
        turns_remaining = self._state.max_turns - self._state.turn + 1

        return {
            "turn": self._state.turn,
            "max_turns": self._state.max_turns,
            "turns_remaining": turns_remaining,
            "game_over": self._state.game_over,
            "game_over_reason": self._state.game_over_reason,
            "mrr": mrr,
            "mrr_target": goal.mrr_target,
            "mrr_pct": round(mrr_pct, 1),
            "mrr_needed": goal.mrr_target - mrr,
            "runway_turns": round(self._state.resources.runway_turns, 1),
            "min_runway_turns": goal.min_runway_turns,
            "budget": self._state.resources.budget,
            "tech_debt": self._state.tech_debt.category,
        }


_SHIPPED_STATUSES = {
    FeatureStatus.shipped_mvp,
    FeatureStatus.shipped_solid,
    FeatureStatus.shipped_polished,
}
