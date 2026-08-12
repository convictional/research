"""Wraps GameEngine for interactive web play."""

from alignsim.src.engine.customer_logic import compute_rubric_satisfaction, compute_sell_minimum_capacity, has_dealbreakers_met
from alignsim.src.engine.game import GameEngine
from alignsim.src.models.actions import (
    BuildAction,
    DiscoverAction,
    FireAction,
    FixBugsAction,
    GameAction,
    HireAction,
    InfrastructureAction,
    MarketAction,
    OpsProjectAction,
    OpsProjectSupportAction,
    QualityLevel,
    SellAction,
    SupportAction,
    SustainHireAction,
    TurnActions,
)
from alignsim.src.models.entities import Customer, CustomerStage, Feature, FeatureStatus
from alignsim.src.models.observations import TurnObservation
from alignsim.src.models.scenario import ScenarioDefinition
from alignsim.src.scenarios.playtest import create_playtest_scenario
from alignsim.src.scenarios.seed_stage import create_seed_stage_scenario

SCENARIO_FACTORIES = {
    "playtest": create_playtest_scenario,
    "seed_stage": create_seed_stage_scenario,
}


class GameSession:
    """Manages a single interactive game."""

    def __init__(self, seed: int = 42, max_turns: int = 48, scenario: str = "playtest"):
        factory = SCENARIO_FACTORIES.get(scenario)
        if factory is None:
            raise ValueError(f"Unknown scenario '{scenario}'. Available: {', '.join(SCENARIO_FACTORIES)}")
        self.scenario = factory(seed=seed)
        self.scenario.max_turns = max_turns
        self.engine = GameEngine(self.scenario)
        self.observation = self.engine.get_initial_observation()
        self.last_events: list[str] = []
        self.game_over = False
        self.game_over_reason: str | None = None
        self.score = None

    @property
    def turn(self) -> int:
        return self.engine.state.turn

    @property
    def state(self):
        return self.engine.state

    def get_context(self) -> dict:
        """Build the full context dict for the turn template."""
        state = self.engine.state
        obs = self.observation
        d = obs.global_dashboard

        # Pipeline customers with satisfaction info
        pipeline = []
        for c in state.customers.values():
            if not c.is_visible:
                continue
            if c.stage in (CustomerStage.churned, CustomerStage.lost):
                continue
            satisfaction = compute_rubric_satisfaction(c, state.features)
            dealbreakers_met = has_dealbreakers_met(c, state.features)
            feature_needs_detail = []
            for fid, scores in c.feature_needs.items():
                f = state.features.get(fid)
                feature_needs_detail.append({
                    "feature_id": fid,
                    "feature_name": f.name if f else fid,
                    "status": f.status.value if f else "unknown",
                    "scores": scores,
                    "shipped": f.status.value.startswith("shipped") if f else False,
                })
            # Compute min sell capacity per action type
            min_sell_caps = {}
            valid_actions_for_stage = {
                "lead": ["outbound"],
                "prospect": ["outbound", "demo"],
                "qualified": ["demo"],
                "in_deal": ["proposal", "negotiate"],
            }
            for sa in valid_actions_for_stage.get(c.stage.value, []):
                min_sell_caps[sa] = compute_sell_minimum_capacity(c, sa, self.scenario.calibration)

            pipeline.append({
                "id": c.id,
                "stage": c.stage.value,
                "engagement": c.engagement.value,
                "deal_value": c.deal_value,
                "known_needs": c.known_needs,
                "timeline": c.timeline,
                "timeline_active": c.timeline_active,
                "timeline_resets": c.timeline_resets,
                "health": round(c.health, 1),
                "competitive_pressure": round(c.competitive_pressure, 2),
                "satisfaction": round(satisfaction, 3),
                "dealbreakers_met": dealbreakers_met,
                "dealbreakers": c.dealbreakers,
                "feature_needs": feature_needs_detail,
                "size": c.size,
                "segment": c.segment.value,
                "is_customer": c.stage == CustomerStage.customer,
                "min_sell_capacity": min_sell_caps,
            })

        # Features with dependency + demand info
        features = []
        for f in state.features.values():
            demanded_by = []
            for c in state.customers.values():
                if c.is_visible and f.id in c.feature_needs:
                    demanded_by.append(c.id)
            deps_met = all(
                state.features[dep].status.value.startswith("shipped")
                for dep in f.depends_on if dep in state.features
            )
            features.append({
                "id": f.id,
                "name": f.name,
                "description": f.description,
                "status": f.status.value,
                "progress": round(f.progress, 1),
                "cost": f.cost,
                "depends_on": f.depends_on,
                "deps_met": deps_met,
                "demanded_by": demanded_by,
                "current_target": f.current_target.value if f.current_target else None,
            })

        # Bugs
        bugs = []
        for b in state.bugs:
            if not b.is_resolved:
                bugs.append({
                    "id": b.id,
                    "severity": b.severity.value,
                    "feature_id": b.feature_id,
                    "turns_unresolved": b.turns_unresolved,
                    "affected_customers": b.affected_customers,
                })

        # Maturity score (global)
        shipped = [f for f in state.features.values()
                   if f.status in (FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished)]
        if shipped:
            polished = sum(1 for f in shipped if f.status == FeatureStatus.shipped_polished)
            solid = sum(1 for f in shipped if f.status == FeatureStatus.shipped_solid)
            maturity = (polished * 1.0 + solid * 0.6) / len(shipped)
        else:
            maturity = 0.0

        # Goal progress
        goal = self.scenario.primary_goal
        mrr_pct = (d.mrr / goal.mrr_target) * 100 if goal.mrr_target else 0

        # Customer details (for inline detail panel)
        customer_details = {}
        for c in state.customers.values():
            if not c.is_visible:
                continue
            if c.stage in (CustomerStage.churned, CustomerStage.lost):
                continue
            detail = self.get_customer_detail(c.id)
            if detail:
                customer_details[c.id] = detail

        # Turn history
        history = self.get_history()

        return {
            "turn": d.turn,
            "mrr": d.mrr,
            "mrr_target": goal.mrr_target,
            "mrr_pct": round(mrr_pct, 1),
            "runway": d.runway_turns,
            "min_runway": goal.min_runway_turns,
            "capacity": d.capacity_available,
            "eng_capacity": d.eng_capacity,
            "sales_capacity": d.sales_capacity,
            "support_capacity": d.support_capacity,
            "marketing_capacity": d.marketing_capacity,
            "ops_capacity": d.ops_capacity,
            "sales_momentum": d.sales_momentum,
            "ops_observation": {
                "available_projects": obs.ops.available_projects,
                "active_projects": obs.ops.active_projects,
                "completed_projects": obs.ops.completed_projects,
                "active_bonuses": obs.ops.active_bonuses,
            },
            "pending_hires": [
                {
                    "id": h.id,
                    "target_function": h.target_function,
                    "hiring_function": h.hiring_function,
                    "turns_remaining": h.turns_remaining,
                    "capacity_on_arrival": h.capacity_bonus,
                    "is_cross_function": h.is_cross_function,
                    "phase": "active" if h.active_turns_completed < h.active_turns_required else "auto",
                    "active_turns_completed": h.active_turns_completed,
                    "active_turns_required": h.active_turns_required,
                    "needs_sustain": h.active_turns_completed < h.active_turns_required,
                }
                for h in state.pending_hires
            ],
            "budget": state.resources.budget,
            "debt_level": d.debt_level,
            "debt_value": round(state.tech_debt.level, 2),
            "active_customers": d.active_customers,
            "maturity": round(maturity, 2),
            "max_turns": self.scenario.max_turns,
            "turns_remaining": self.scenario.max_turns - d.turn + 1,
            "pipeline": sorted(pipeline, key=lambda x: _stage_order(x["stage"])),
            "features": features,
            "bugs": bugs,
            "events": self.last_events,
            "game_over": self.game_over,
            "game_over_reason": self.game_over_reason,
            "score": self.score,
            "bug_backlog": d.bug_backlog,
            "customer_details": customer_details,
            "history": history,
            "min_rubric": self.scenario.calibration.min_rubric_for_close,
        }

    def submit_actions(self, actions: list[GameAction]) -> dict:
        """Submit actions for the current turn. Returns events from resolution."""
        turn_actions = TurnActions(turn=self.turn, actions=actions)
        result, next_obs = self.engine.step(turn_actions)

        self.last_events = result.record.events
        valid_count = len(result.validation.valid_actions)
        rejected = result.validation.rejected_actions

        if next_obs:
            self.observation = next_obs

        if result.game_over:
            self.game_over = True
            self.game_over_reason = result.game_over_reason
            self.score = self.engine.get_final_score()

        return {
            "valid_count": valid_count,
            "rejected": [{"action": str(r.action), "reason": r.reason} for r in rejected],
            "events": result.record.events,
        }

    def get_history(self) -> list[dict]:
        """Build turn-by-turn history for the history page."""
        history = []
        for record in self.state.turn_history:
            actions_summary = []
            for a in record.actions_valid:
                actions_summary.append(_describe_action(a))
            history.append({
                "turn": record.turn,
                "mrr": record.mrr,
                "runway": round(record.runway_turns, 1),
                "budget": record.budget,
                "capacity_used": record.capacity_used,
                "capacity_available": record.capacity_available,
                "bugs_injected": record.bugs_injected,
                "bugs_fixed": record.bugs_fixed,
                "churn_count": record.churn_count,
                "actions": actions_summary,
                "events": record.events,
                "rejected": [{"reason": r.reason} for r in record.actions_rejected],
            })
        return history

    def get_customer_detail(self, customer_id: str) -> dict | None:
        """Build detailed view for a single customer."""
        state = self.state
        c = state.customers.get(customer_id)
        if c is None or not c.is_visible:
            return None

        satisfaction = compute_rubric_satisfaction(c, state.features)
        dealbreakers_met = has_dealbreakers_met(c, state.features)

        # Satisfaction breakdown
        feature_coverage_score, price_score, maturity_score, support_score = _compute_satisfaction_breakdown(
            c, state.features
        )

        # Feature needs detail
        feature_needs = []
        for fid, scores in c.feature_needs.items():
            f = state.features.get(fid)
            shipped_quality = None
            if f and f.status.value.startswith("shipped"):
                shipped_quality = f.status.value.replace("shipped_", "")
            feature_needs.append({
                "feature_id": fid,
                "feature_name": f.name if f else fid,
                "status": f.status.value if f else "unknown",
                "scores": scores,
                "shipped": f.status.value.startswith("shipped") if f else False,
                "shipped_quality": shipped_quality,
                "current_score": scores.get(shipped_quality, 0) if shipped_quality else 0,
            })

        # Action history from turn records
        action_history = []
        for record in state.turn_history:
            for a in record.actions_valid:
                if _action_targets_customer(a, customer_id):
                    action_history.append({
                        "turn": record.turn,
                        "action": _describe_action(a),
                    })
            for event in record.events:
                if customer_id in event:
                    action_history.append({
                        "turn": record.turn,
                        "action": f"[event] {event}",
                    })

        return {
            "id": c.id,
            "size": c.size,
            "segment": c.segment.value,
            "stage": c.stage.value,
            "engagement": c.engagement.value,
            "deal_value": c.deal_value,
            "timeline": c.timeline,
            "timeline_active": c.timeline_active,
            "timeline_resets": c.timeline_resets,
            "health": round(c.health, 1),
            "health_history": [round(h, 1) for h in c.health_history],
            "competitive_pressure": round(c.competitive_pressure, 2),
            "known_needs": c.known_needs,
            "dealbreakers": c.dealbreakers,
            "dealbreakers_met": dealbreakers_met,
            "is_customer": c.stage == CustomerStage.customer,
            "is_visible": c.is_visible,
            "satisfaction": round(satisfaction, 3),
            "satisfaction_breakdown": {
                "feature_coverage": round(feature_coverage_score, 3),
                "price": round(price_score, 3),
                "maturity": round(maturity_score, 3),
                "support": round(support_score, 3),
            },
            "rubric_weights": {
                "feature_coverage": c.rubric.feature_coverage,
                "price": c.rubric.price,
                "maturity": c.rubric.maturity,
                "support": c.rubric.support,
            },
            "feature_needs": feature_needs,
            "action_history": action_history,
            "turns_below_churn": c.turns_below_churn_threshold,
            "turns_above_expansion": c.turns_above_expansion_threshold,
            "onboarding_remaining": c.onboarding_turns_remaining,
        }


def _compute_satisfaction_breakdown(
    customer: Customer, features: dict[str, Feature]
) -> tuple[float, float, float, float]:
    """Compute individual rubric component scores (not weighted)."""
    # Feature coverage
    feature_coverage_score = 0.0
    if customer.feature_needs:
        total_needs = len(customer.feature_needs)
        needs_met = 0
        total_satisfaction = 0.0
        for fid, quality_scores in customer.feature_needs.items():
            f = features.get(fid)
            if f is None:
                continue
            if f.status == FeatureStatus.shipped_mvp and "mvp" in quality_scores:
                needs_met += 1
                total_satisfaction += quality_scores["mvp"]
            elif f.status == FeatureStatus.shipped_solid and "solid" in quality_scores:
                needs_met += 1
                total_satisfaction += quality_scores["solid"]
            elif f.status == FeatureStatus.shipped_polished and "polished" in quality_scores:
                needs_met += 1
                total_satisfaction += quality_scores["polished"]
        breadth = needs_met / total_needs
        depth = total_satisfaction / needs_met if needs_met > 0 else 0.0
        feature_coverage_score = breadth * 0.4 + depth * 0.6

    price_score = 0.5 + (customer.size * 0.1)

    shipped = [f for f in features.values()
               if f.status in (FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished)]
    if shipped:
        polished = sum(1 for f in shipped if f.status == FeatureStatus.shipped_polished)
        solid = sum(1 for f in shipped if f.status == FeatureStatus.shipped_solid)
        maturity_score = (polished * 1.0 + solid * 0.6) / len(shipped)
    else:
        maturity_score = 0.0

    support_score = customer.health / 10.0

    return feature_coverage_score, price_score, maturity_score, support_score


def _describe_action(action: GameAction) -> str:
    """Human-readable description of an action."""
    if isinstance(action, BuildAction):
        return f"Build {action.feature_id} → {action.quality.value} ({action.capacity} cap)"
    elif isinstance(action, SellAction):
        return f"Sell {action.customer_id} [{action.sell_action}] ({action.capacity} cap)"
    elif isinstance(action, SupportAction):
        return f"Support {action.customer_id} [{action.support_action}] ({action.capacity} cap)"
    elif isinstance(action, FixBugsAction):
        target = action.bug_id or "auto"
        return f"Fix bugs [{target}] ({action.capacity} cap)"
    elif isinstance(action, InfrastructureAction):
        return f"Infrastructure ({action.capacity} cap)"
    elif isinstance(action, DiscoverAction):
        targets = f" [{','.join(action.target_features)}]" if action.target_features else ""
        return f"Discover{targets} ({action.capacity} cap)"
    elif isinstance(action, MarketAction):
        return f"Market [{action.channel}] ({action.capacity} cap)"
    elif isinstance(action, HireAction):
        if action.hiring_function != action.target_function:
            return f"Hire {action.target_function} (cross from {action.hiring_function})"
        return f"Hire {action.target_function}"
    elif isinstance(action, FireAction):
        return f"Fire {action.function}"
    elif isinstance(action, OpsProjectAction):
        return f"Ops Project {action.project_id} ({action.capacity} cap)"
    elif isinstance(action, OpsProjectSupportAction):
        return f"Ops Support {action.project_id} ({action.capacity} cap)"
    return str(action)


def _action_targets_customer(action: GameAction, customer_id: str) -> bool:
    """Check if an action targets a specific customer."""
    if isinstance(action, (SellAction, SupportAction)):
        return action.customer_id == customer_id
    return False


def parse_action(form_data: dict) -> GameAction | None:
    """Parse a single action from form data."""
    action_type = form_data.get("action_type")
    capacity = int(form_data.get("capacity", 0))

    if action_type == "build":
        return BuildAction(
            feature_id=form_data["feature_id"],
            quality=QualityLevel(form_data["quality"]),
            capacity=capacity,
        )
    elif action_type == "fix_bugs":
        bug_id = form_data.get("bug_id") or None
        return FixBugsAction(bug_id=bug_id, capacity=capacity)
    elif action_type == "infrastructure":
        return InfrastructureAction(capacity=capacity)
    elif action_type == "sell":
        return SellAction(
            customer_id=form_data["customer_id"],
            sell_action=form_data["sell_action"],
            capacity=capacity,
        )
    elif action_type == "discover":
        raw = form_data.get("target_features", "")
        if isinstance(raw, list):
            target_features = [f.strip() for f in raw if f.strip()]
        else:
            target_features = [f.strip() for f in raw.split(",") if f.strip()] if raw else []
        return DiscoverAction(target_features=target_features, capacity=capacity)
    elif action_type == "support":
        return SupportAction(
            customer_id=form_data["customer_id"],
            support_action=form_data["support_action"],
            capacity=capacity,
        )
    elif action_type == "market":
        return MarketAction(
            channel=form_data["channel"],
            capacity=capacity,
        )
    elif action_type == "hire":
        return HireAction(
            hiring_function=form_data["hiring_function"],
            target_function=form_data["target_function"],
        )
    elif action_type == "sustain_hire":
        return SustainHireAction(hire_id=form_data["hire_id"])
    elif action_type == "fire":
        return FireAction(function=form_data["function"])
    elif action_type == "ops_project":
        return OpsProjectAction(project_id=form_data["project_id"], capacity=capacity)
    elif action_type == "ops_project_support":
        return OpsProjectSupportAction(project_id=form_data["project_id"], capacity=capacity)
    return None


def _stage_order(stage: str) -> int:
    order = {"customer": 0, "in_deal": 1, "qualified": 2, "prospect": 3, "lead": 4}
    return order.get(stage, 5)
