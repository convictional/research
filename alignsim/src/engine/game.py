"""GameEngine: the top-level turn orchestrator.

Owns the game state, RNG, and coordinates validation, resolution, and observation.
The harness interacts exclusively through this class.
"""

import random

from alignsim.src.engine.alignment_scoring import compute_alignment_scores
from alignsim.src.engine.observer import ObservationGenerator
from alignsim.src.engine.resolver import TurnResolver
from alignsim.src.engine.scoring import compute_goal_attainment
from alignsim.src.engine.validator import ActionValidator, ValidationResult
from alignsim.src.models.actions import TurnActions
from alignsim.src.models.entities import Bug, BugSeverity, CustomerStage
from alignsim.src.models.game_state import GameState, ResourcePool, TechDebt, TurnRecord
from alignsim.src.models.goals import GoalAttainmentScore
from alignsim.src.models.observations import TurnObservation
from alignsim.src.models.scenario import ScenarioDefinition


class TurnResult:
    """Result of a single turn step."""

    def __init__(
        self,
        turn: int,
        validation: ValidationResult,
        record: TurnRecord,
        game_over: bool,
        game_over_reason: str | None,
    ):
        self.turn = turn
        self.validation = validation
        self.record = record
        self.game_over = game_over
        self.game_over_reason = game_over_reason


class GameEngine:
    """Turn-based game engine. The harness calls get_initial_observation() then step() each turn."""

    def __init__(self, scenario: ScenarioDefinition):
        self.scenario = scenario
        self.rng = random.Random(scenario.seed)
        self.generator_config = scenario.generator_config

        # Initialize game state from scenario
        self.state = GameState(
            turn=1,
            max_turns=scenario.max_turns,
            seed=scenario.seed,
            customers={c.id: c.model_copy(deep=True) for c in scenario.customers},
            features={f.id: f.model_copy(deep=True) for f in scenario.features},
            competitors={c.id: c.model_copy(deep=True) for c in scenario.competitors},
            resources=ResourcePool(
                capacity_per_turn=scenario.financials.capacity_per_turn,
                eng_capacity=scenario.financials.eng_capacity,
                sales_capacity=scenario.financials.sales_capacity,
                support_capacity=scenario.financials.support_capacity,
                marketing_capacity=scenario.financials.marketing_capacity,
                ops_capacity=scenario.financials.ops_capacity,
                budget=scenario.financials.starting_budget,
                base_cost_per_turn=scenario.financials.base_cost_per_turn,
                mrr=scenario.financials.starting_mrr,
            ),
            tech_debt=TechDebt(level=0.0),
            process_projects={p.id: p.model_copy(deep=True) for p in scenario.process_projects},
        )
        self.state.initial_tech_debt = self.state.tech_debt.level

        # Compute initial runway using full cost model (team + overhead + maintenance)
        team_cost = scenario.financials.capacity_per_turn * scenario.calibration.team_cost_per_capacity
        maintenance = sum(
            f.maintenance_cost for f in self.state.features.values()
            if f.status.value.startswith("shipped")
        )
        total_cost = team_cost + scenario.financials.base_cost_per_turn + maintenance
        if total_cost > self.state.resources.mrr:
            burn = total_cost - self.state.resources.mrr
            self.state.resources.runway_turns = self.state.resources.budget / burn if burn > 0 else 999
        else:
            self.state.resources.runway_turns = 999

        # Inject any initial bugs from scenario
        for i, bug_desc in enumerate(scenario.initial_bugs):
            parts = bug_desc.split(":")
            if len(parts) >= 3:
                severity = BugSeverity(parts[0])
                feature_id = parts[1]
                affected = parts[2].split(",") if parts[2] else []
                self.state.bugs.append(Bug(
                    id=f"BUG_{self.state.next_bug_id:03d}",
                    severity=severity,
                    feature_id=feature_id,
                    turn_injected=0,
                    affected_customers=affected,
                ))
                self.state.next_bug_id += 1

    def get_initial_observation(self) -> TurnObservation:
        """Generate the first turn's observation."""
        observer = ObservationGenerator(self.state, self.scenario.calibration)
        return observer.generate()

    def step(self, actions: TurnActions) -> tuple[TurnResult, TurnObservation | None]:
        """Process one turn: validate actions, resolve, generate next observation.

        Returns (TurnResult, next_observation). next_observation is None if game is over.
        """
        # Snapshot debt level before resolution (for observer debt_delta)
        debt_before = self.state.tech_debt.level

        # Validate
        validator = ActionValidator(self.state, self.scenario.calibration, self.generator_config)
        validation = validator.validate(actions)

        # Resolve
        features_dict = {f.id: f for f in self.scenario.features}
        resolver = TurnResolver(
            self.state, self.scenario.calibration, self.rng,
            self.generator_config, features_dict,
        )
        record = resolver.resolve(validation.valid_actions)
        record.actions_submitted = list(actions.actions)
        record.actions_rejected = validation.rejected_actions

        # Build result
        result = TurnResult(
            turn=self.state.turn,
            validation=validation,
            record=record,
            game_over=self.state.game_over,
            game_over_reason=self.state.game_over_reason,
        )

        # Advance turn
        self.state.turn += 1

        # Generate next observation if game continues
        next_obs = None
        if not self.state.game_over:
            observer = ObservationGenerator(self.state, self.scenario.calibration, prev_debt_level=debt_before)
            next_obs = observer.generate(turn_record=record)

        return result, next_obs

    def get_final_score(self) -> GoalAttainmentScore:
        """Compute goal attainment score based on current state.

        Layer 2 alignment scores are populated here and stored on the score
        object, but must be stripped from any player-facing serialization
        (use score_to_player_dict).
        """
        score = compute_goal_attainment(self.state, self.scenario.primary_goal)
        score.alignment_scores = compute_alignment_scores(self.state, self.scenario.calibration)
        return score

    def is_game_over(self) -> bool:
        return self.state.game_over

    def get_scenario_info(self) -> dict:
        """Return public scenario information for the harness (no hidden stats)."""
        visible_customers = [
            {
                "id": c.id,
                "size": c.size,
                "segment": c.segment.value,
                "stage": c.stage.value,
                "engagement": c.engagement.value,
                "known_needs": c.known_needs,
                "deal_value": c.deal_value,
            }
            for c in self.state.customers.values()
            if c.is_visible
        ]

        features = [
            {
                "id": f.id,
                "name": f.name,
                "description": f.description,
                "cost": f.cost,
                "depends_on": f.depends_on,
                "status": f.status.value,
                "progress": f.progress,
            }
            for f in self.state.features.values()
        ]

        return {
            "name": self.scenario.name,
            "description": self.scenario.description,
            "max_turns": self.scenario.max_turns,
            "primary_goal": {
                "mrr_target": self.scenario.primary_goal.mrr_target,
                "max_churn_rate": self.scenario.primary_goal.max_churn_rate,
                "min_runway_turns": self.scenario.primary_goal.min_runway_turns,
                "target_turn": self.scenario.primary_goal.target_turn,
            },
            "starting_mrr": self.state.resources.mrr,
            "starting_budget": self.state.resources.budget,
            "capacity_per_turn": self.state.resources.capacity_per_turn,
            "eng_capacity": self.state.resources.eng_capacity,
            "sales_capacity": self.state.resources.sales_capacity,
            "support_capacity": self.state.resources.support_capacity,
            "marketing_capacity": self.state.resources.marketing_capacity,
            "ops_capacity": self.state.resources.ops_capacity,
            "base_cost_per_turn": self.state.resources.base_cost_per_turn,
            "sell_base_costs": {
                "outbound": self.scenario.calibration.sell_base_cost_outbound,
                "demo": self.scenario.calibration.sell_base_cost_demo,
                "proposal": self.scenario.calibration.sell_base_cost_proposal,
                "negotiate": self.scenario.calibration.sell_base_cost_negotiate,
            },
            "visible_customers": visible_customers,
            "features": features,
            "competitors": [c.id for c in self.state.competitors.values()],
            "process_projects": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "size": p.size.value,
                    "ops_capacity_cost": p.ops_capacity_cost,
                    "target_function": p.target_function,
                    "duration_turns": p.duration_turns,
                    "prerequisites": p.prerequisites,
                }
                for p in self.state.process_projects.values()
            ],
        }

    def get_state_summary(self) -> dict:
        """Return a summary of current game state (public info only) for harness context."""
        active_customers = sum(
            1 for c in self.state.customers.values() if c.stage == CustomerStage.customer
        )
        pipeline_customers = sum(
            1 for c in self.state.customers.values()
            if c.is_visible and c.stage in (CustomerStage.lead, CustomerStage.prospect, CustomerStage.qualified, CustomerStage.in_deal)
        )

        return {
            "turn": self.state.turn,
            "mrr": self.state.resources.mrr,
            "budget": self.state.resources.budget,
            "runway_turns": round(self.state.resources.runway_turns, 1),
            "capacity_per_turn": self.state.resources.capacity_per_turn,
            "eng_capacity": self.state.resources.eng_capacity,
            "sales_capacity": self.state.resources.sales_capacity,
            "support_capacity": self.state.resources.support_capacity,
            "marketing_capacity": self.state.resources.marketing_capacity,
            "ops_capacity": self.state.resources.ops_capacity,
            "active_customers": active_customers,
            "pipeline_customers": pipeline_customers,
            "tech_debt": self.state.tech_debt.category,
            "unresolved_bugs": sum(1 for b in self.state.bugs if not b.is_resolved),
            "features_shipped": sum(
                1 for f in self.state.features.values()
                if f.status.value.startswith("shipped")
            ),
            "sales_momentum": round(self.state.sales_momentum, 3),
            "active_process_bonuses": len(self.state.active_process_bonuses),
        }
