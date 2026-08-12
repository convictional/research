"""Action validation: checks legality of submitted actions against game state."""

from pydantic import BaseModel, Field

from alignsim.src.models.actions import (
    AnalysisScopeAction,
    BuildAction,
    DiscoverAction,
    FireAction,
    FixBugsAction,
    GameAction,
    HireAction,
    InfrastructureAction,
    MarketAction,
    MarketSupportAction,
    OpsAnalysisAction,
    OpsProjectAction,
    OpsProjectSupportAction,
    SellAction,
    SupportAction,
    SustainHireAction,
    TurnActions,
)
from alignsim.src.engine.customer_logic import compute_sell_minimum_capacity
from alignsim.src.engine.ops_logic import compute_maintenance_cost
from alignsim.src.models.entities import CustomerStage, FeatureStatus, ProcessProjectStatus
from alignsim.src.engine.customer_generator import filter_hidden_by_features
from alignsim.src.models.game_state import ActionRejection, GameState
from alignsim.src.models.scenario import CalibrationParams, CustomerGeneratorConfig


class ValidationResult(BaseModel):
    valid_actions: list[GameAction] = Field(default_factory=list)
    rejected_actions: list[ActionRejection] = Field(default_factory=list)
    total_capacity_used: int = 0


class ActionValidator:
    def __init__(
        self,
        state: GameState,
        calibration: CalibrationParams | None = None,
        generator_config: CustomerGeneratorConfig | None = None,
    ):
        self.state = state
        self.calibration = calibration
        self.generator_config = generator_config

    def validate(self, turn_actions: TurnActions) -> ValidationResult:
        result = ValidationResult()

        # Per-pool remaining capacity
        pools = {
            "engineering": self.state.resources.eng_capacity,
            "sales": self.state.resources.sales_capacity,
            "support": self.state.resources.support_capacity,
            "marketing": self.state.resources.marketing_capacity,
            "ops": self.state.resources.ops_capacity,
        }

        # Phase 0: Pre-commit sustain_hire capacity (takes priority over all other actions)
        sustain_actions = [a for a in turn_actions.actions if isinstance(a, SustainHireAction)]
        other_actions = [a for a in turn_actions.actions if not isinstance(a, SustainHireAction)]

        hire_cap_cost = self.calibration.hire_capacity_cost if self.calibration else 3
        func_to_pool = {
            "engineering": "engineering", "sales": "sales",
            "cs": "support", "marketing": "marketing", "ops": "ops",
        }

        for action in sustain_actions:
            hire = next((h for h in self.state.pending_hires if h.id == action.hire_id), None)
            if hire is None:
                result.rejected_actions.append(ActionRejection(
                    action=action, reason=f"No pending hire with id {action.hire_id}",
                ))
                continue
            if hire.active_turns_completed >= hire.active_turns_required:
                result.rejected_actions.append(ActionRejection(
                    action=action, reason=f"Hire {action.hire_id} is already in auto-phase (no sustain needed)",
                ))
                continue

            pool = func_to_pool.get(hire.hiring_function, "engineering")
            if hire_cap_cost > pools[pool]:
                result.rejected_actions.append(ActionRejection(
                    action=action,
                    reason=f"Insufficient {pool} capacity for sustain: needs {hire_cap_cost}, only {pools[pool]} remaining",
                ))
                continue

            result.valid_actions.append(action)
            pools[pool] -= hire_cap_cost

        # Running shared-budget tracker for budgeted market actions (events/content). Keeps
        # the runway constraint honest: a budgeted market action that would drive budget
        # below zero is rejected. Outbound (budget_cost_per_capacity == 0) is never gated.
        budget_remaining = self.state.resources.budget

        # Phase 1: Validate remaining actions with reduced pools
        for action in other_actions:
            capacity_cost = self._get_capacity_cost(action)
            pool = self._get_pool(action)

            pool_remaining = pools.get(pool, 999)

            if capacity_cost > pool_remaining:
                result.rejected_actions.append(ActionRejection(
                    action=action,
                    reason=f"Insufficient {pool} capacity: needs {capacity_cost}, only {pool_remaining} remaining",
                ))
                continue

            rejection_reason = self._check_action_legality(action)
            if rejection_reason:
                result.rejected_actions.append(ActionRejection(action=action, reason=rejection_reason))
                continue

            # Marketing budget gate (budgeted channels only)
            if isinstance(action, MarketAction):
                market_cost = self._get_market_budget_cost(action)
                if market_cost > budget_remaining:
                    result.rejected_actions.append(ActionRejection(
                        action=action,
                        reason=(
                            f"Insufficient budget for {action.channel} marketing: needs "
                            f"{market_cost}, only {budget_remaining} remaining"
                        ),
                    ))
                    continue
                budget_remaining -= market_cost

            result.valid_actions.append(action)

            if pool in pools:
                pools[pool] -= capacity_cost

        result.total_capacity_used = (
            (self.state.resources.eng_capacity - pools["engineering"])
            + (self.state.resources.sales_capacity - pools["sales"])
            + (self.state.resources.support_capacity - pools["support"])
            + (self.state.resources.marketing_capacity - pools["marketing"])
            + (self.state.resources.ops_capacity - pools["ops"])
        )
        return result

    def _get_market_budget_cost(self, action: MarketAction) -> int:
        """Shared-budget cost of a market action = capacity * channel budget_cost_per_capacity."""
        if self.calibration is None:
            return 0
        profile = self.calibration.channel_profiles.get(action.channel)
        if profile is None:
            return 0
        return action.capacity * profile.budget_cost_per_capacity

    def _get_capacity_cost(self, action: GameAction) -> int:
        if isinstance(action, HireAction):
            return self.calibration.hire_capacity_cost if self.calibration else 0
        if isinstance(action, (FireAction, SustainHireAction)):
            return 0  # sustain handled in pre-commit phase; fire is budget-only
        return action.capacity

    def _get_pool(self, action: GameAction) -> str:
        """Map action type to its capacity pool."""
        if isinstance(action, (BuildAction, FixBugsAction, InfrastructureAction)):
            return "engineering"
        elif isinstance(action, (SellAction, DiscoverAction, MarketSupportAction)):
            # market_support is Sales co-investing in Marketing's campaign — draws from sales.
            return "sales"
        elif isinstance(action, SupportAction):
            return "support"
        elif isinstance(action, MarketAction):
            return "marketing"
        elif isinstance(action, OpsProjectAction):
            return "ops"
        elif isinstance(action, OpsAnalysisAction):
            # The Ops side of an analysis handshake draws from the ops pool.
            return "ops"
        elif isinstance(action, AnalysisScopeAction):
            # The scope co-investment draws from the REQUESTING team's pool (cs -> support).
            func_to_pool = {
                "engineering": "engineering", "sales": "sales",
                "cs": "support", "marketing": "marketing",
            }
            return func_to_pool.get(action.target_function, "engineering")
        elif isinstance(action, OpsProjectSupportAction):
            # Support actions draw from the TARGET function's pool
            project = self.state.process_projects.get(action.project_id)
            if project is None:
                return "engineering"  # will fail legality check
            func_to_pool = {
                "engineering": "engineering", "sales": "sales",
                "support": "support", "marketing": "marketing",
            }
            return func_to_pool.get(project.target_function, "engineering")
        elif isinstance(action, HireAction):
            # Hire costs capacity from the hiring_function pool (not the target)
            func_to_pool = {
                "engineering": "engineering", "sales": "sales",
                "cs": "support", "marketing": "marketing", "ops": "ops",
            }
            return func_to_pool.get(action.hiring_function, "engineering")
        elif isinstance(action, FireAction):
            return "none"  # fire costs no capacity; budget-only cost handled in resolver
        return "engineering"

    def _check_action_legality(self, action: GameAction) -> str | None:
        if isinstance(action, BuildAction):
            return self._validate_build(action)
        elif isinstance(action, SellAction):
            return self._validate_sell(action)
        elif isinstance(action, SupportAction):
            return self._validate_support(action)
        elif isinstance(action, FixBugsAction):
            return self._validate_fix_bugs(action)
        elif isinstance(action, DiscoverAction):
            return self._validate_discover(action)
        elif isinstance(action, HireAction):
            return self._validate_hire(action)
        elif isinstance(action, FireAction):
            return self._validate_fire(action)
        elif isinstance(action, OpsProjectAction):
            return self._validate_ops_project(action)
        elif isinstance(action, OpsProjectSupportAction):
            return self._validate_ops_project_support(action)
        elif isinstance(action, OpsAnalysisAction):
            return self._validate_ops_analysis(action)
        elif isinstance(action, (MarketAction, InfrastructureAction, SustainHireAction, AnalysisScopeAction)):
            return None  # Always valid if capacity available
        return None

    def _validate_build(self, action: BuildAction) -> str | None:
        feature = self.state.features.get(action.feature_id)
        if feature is None:
            return f"Feature {action.feature_id} does not exist"

        # Check dependencies are met (shipped at any quality)
        shipped_statuses = {FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished}
        for dep_id in feature.depends_on:
            dep = self.state.features.get(dep_id)
            if dep is None or dep.status not in shipped_statuses:
                return f"Dependency {dep_id} not met for feature {action.feature_id}"

        # Check target quality is higher than current
        quality_order = {FeatureStatus.not_started: 0, FeatureStatus.in_progress: 0,
                         FeatureStatus.shipped_mvp: 1, FeatureStatus.shipped_solid: 2,
                         FeatureStatus.shipped_polished: 3}
        target_order = {"mvp": 1, "solid": 2, "polished": 3}
        current_order = quality_order.get(feature.status, 0)
        desired_order = target_order.get(action.quality.value, 0)
        if desired_order <= current_order:
            return f"Feature {action.feature_id} already at or above {action.quality.value} quality"

        return None

    def _validate_sell(self, action: SellAction) -> str | None:
        customer = self.state.customers.get(action.customer_id)
        if customer is None:
            return f"Customer {action.customer_id} does not exist"
        if not customer.is_visible:
            return f"Customer {action.customer_id} has not been discovered"

        # Check stage is valid for the sell action
        valid_actions = {
            CustomerStage.lead: {"outbound"},
            CustomerStage.prospect: {"outbound", "demo"},
            CustomerStage.qualified: {"demo"},
            CustomerStage.in_deal: {"proposal", "negotiate"},
        }
        allowed = valid_actions.get(customer.stage, set())
        if action.sell_action not in allowed:
            return (
                f"Cannot perform {action.sell_action} on customer {action.customer_id} "
                f"in stage {customer.stage.value}"
            )

        # Check minimum capacity based on customer size
        if self.calibration is not None:
            min_cap = compute_sell_minimum_capacity(customer, action.sell_action, self.calibration)
            if action.capacity < min_cap:
                return (
                    f"Sell {action.sell_action} on {action.customer_id} (size {customer.size}) "
                    f"requires minimum {min_cap} capacity, got {action.capacity}"
                )

        # Pricing validation
        if action.proposed_deal_value is not None:
            if action.sell_action not in ("proposal", "negotiate"):
                return (
                    f"proposed_deal_value is only valid for proposal/negotiate, "
                    f"not {action.sell_action}"
                )
            if action.proposed_deal_value <= 0:
                return "proposed_deal_value must be positive"

        if action.sell_action == "negotiate" and not customer.has_received_proposal:
            return (
                f"Cannot negotiate with {action.customer_id} before submitting a proposal"
            )

        return None

    def _validate_support(self, action: SupportAction) -> str | None:
        customer = self.state.customers.get(action.customer_id)
        if customer is None:
            return f"Customer {action.customer_id} does not exist"
        if customer.stage != CustomerStage.customer:
            return f"Customer {action.customer_id} is not an active customer (stage: {customer.stage.value})"
        return None

    def _validate_fix_bugs(self, action: FixBugsAction) -> str | None:
        if action.bug_id is not None:
            bug = next((b for b in self.state.bugs if b.id == action.bug_id and not b.is_resolved), None)
            if bug is None:
                return f"Bug {action.bug_id} does not exist or is already resolved"
        else:
            # Check there are any unresolved bugs
            if not any(not b.is_resolved for b in self.state.bugs):
                return "No unresolved bugs to fix"
        return None

    def _validate_discover(self, action: DiscoverAction) -> str | None:
        shipped_statuses = {FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished}
        effective_targets = action.target_features or [
            fid for fid, f in self.state.features.items() if f.status in shipped_statuses
        ]

        if action.target_features:
            if not any(
                self.state.features.get(fid) and self.state.features[fid].status in shipped_statuses
                for fid in action.target_features
            ):
                return "At least one target feature must be shipped"

        if self.generator_config is not None:
            return None

        hidden = [c for c in self.state.customers.values() if not c.is_visible]
        matching = filter_hidden_by_features(hidden, effective_targets)
        if not matching:
            return "No hidden customers matching target features remaining to discover"
        return None

    def _validate_hire(self, action: HireAction) -> str | None:
        hire_cost = self.state.resources.capacity_per_turn * (self.calibration.hire_budget_cost_multiplier if self.calibration else 2)
        if self.state.resources.budget < hire_cost:
            return f"Insufficient budget for hiring: needs {hire_cost}, have {self.state.resources.budget}"
        return None

    def _validate_fire(self, action: FireAction) -> str | None:
        func_to_pool = {
            "engineering": self.state.resources.eng_capacity,
            "sales": self.state.resources.sales_capacity,
            "cs": self.state.resources.support_capacity,
            "marketing": self.state.resources.marketing_capacity,
            "ops": self.state.resources.ops_capacity,
        }
        current_capacity = func_to_pool.get(action.function, 0)
        if current_capacity <= 0:
            return f"Cannot fire from {action.function}: no capacity remaining"
        return None

    def _validate_ops_project(self, action: OpsProjectAction) -> str | None:
        project = self.state.process_projects.get(action.project_id)
        if project is None:
            return f"Process project {action.project_id} does not exist"

        if project.status == ProcessProjectStatus.completed:
            # Completed project can be re-run. Determine mode by checking for active bonus.
            active_bonus = next(
                (b for b in self.state.active_process_bonuses if b.project_id == action.project_id),
                None,
            )
            if active_bonus is not None:
                # Maintenance mode: single-turn refresh; scaled capacity required
                required = compute_maintenance_cost(active_bonus)
                if action.capacity < required:
                    return (
                        f"Maintenance refresh of {action.project_id} requires {required} ops capacity "
                        f"({int(compute_maintenance_cost(active_bonus) / project.ops_capacity_cost * 100)}% "
                        f"of original {project.ops_capacity_cost}), got {action.capacity}"
                    )
            else:
                # Net-new re-run (bonus lapsed): full capacity required
                if action.capacity < project.ops_capacity_cost:
                    return (
                        f"Re-run of lapsed project {action.project_id} requires full "
                        f"{project.ops_capacity_cost} ops capacity, got {action.capacity}"
                    )
            return None

        # Starting an AVAILABLE project: enforce tech-tree prerequisites (single source of truth,
        # symmetric with _validate_build's dependency gate). in_progress projects already passed
        # this gate when they were started, and a completed project's prereqs were necessarily met.
        if project.status == ProcessProjectStatus.available:
            for prereq_id in project.prerequisites:
                prereq = self.state.process_projects.get(prereq_id)
                if prereq is None or prereq.status != ProcessProjectStatus.completed:
                    return (
                        f"Process project {action.project_id} is locked: prerequisite "
                        f"{prereq_id} not completed"
                    )

        if action.capacity < project.ops_capacity_cost:
            return (
                f"Process project {action.project_id} ({project.size.value}) requires "
                f"{project.ops_capacity_cost} ops capacity, got {action.capacity}"
            )
        return None

    def _validate_ops_project_support(self, action: OpsProjectSupportAction) -> str | None:
        project = self.state.process_projects.get(action.project_id)
        if project is None:
            return f"Process project {action.project_id} does not exist"
        if project.status != ProcessProjectStatus.in_progress:
            return f"Process project {action.project_id} is not in progress (status: {project.status.value})"
        return None

    def _validate_ops_analysis(self, action: OpsAnalysisAction) -> str | None:
        required = self.calibration.analysis_ops_capacity_cost if self.calibration else 2
        if action.capacity < required:
            return (
                f"Ops analysis ({action.analysis_type} for {action.target_function}) requires "
                f"{required} ops capacity, got {action.capacity}"
            )
        return None
