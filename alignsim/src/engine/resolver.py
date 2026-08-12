"""Turn resolution: processes validated actions against game state.

Resolution order is deterministic and documented:
0. Hiring (sustain, cancel unsustained, new hires, tick pending, apply arrivals) + Firing
1. Engineering (build, bugs, infra, debt)
2. Sales (pipeline advancement, with momentum + process bonuses)
3. CS (health updates, with process bonuses)
4. Marketing (log investment, with process bonuses)
5. Discovery (reveal customers)
6. Ops Projects (process improvement work + bonus lifecycle)
7. Competitive events
8. Bug injection (with process bonuses)
8b. Emergent-need lifecycle + injection (SAME slot as bug injection — strictly AFTER CS)
9. Financial (revenue, costs, runway)
10. Metrics (churn rate, history, sales momentum update)
"""

import random

from alignsim.src.engine import analysis_logic, customer_logic, market_logic, ops_logic, product_logic
from alignsim.src.engine.customer_logic import compute_pricing_modifier, compute_sandbagged_price
from alignsim.src.engine.customer_generator import (
    filter_hidden_by_features,
    generate_discovery_candidates,
    generate_inbound_candidates,
)
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
)
from alignsim.src.models.entities import Customer, CustomerStage, Engagement, FeatureStatus, ProcessProjectStatus, QualityLevel
from alignsim.src.models.game_state import ActiveProcessBonus, GameState, PendingHire, TurnRecord
from alignsim.src.models.scenario import CalibrationParams, CustomerGeneratorConfig

# Map an analysis action's engine target_function name to the agent/pool function name used to
# key pending_analyses (and the requesting team's capacity pool). Mirrors condition3_filters'
# ENGINE_TO_AGENT_FUNCTION (cs -> support) but defined locally to avoid a harness import cycle.
_ANALYSIS_TARGET_TO_AGENT_FN: dict[str, str] = {
    "engineering": "engineering",
    "sales": "sales",
    "cs": "support",
    "marketing": "marketing",
}
# Per-pool capacity-used record attribute for the requesting team's scope co-investment.
_ANALYSIS_TARGET_TO_USED_ATTR: dict[str, str] = {
    "engineering": "eng_capacity_used",
    "sales": "sales_capacity_used",
    "cs": "support_capacity_used",
    "marketing": "marketing_capacity_used",
}


def _apply_reveal_awareness(
    customer: Customer,
    awareness: dict[str, float],
    calibration: CalibrationParams,
    rng: random.Random,
) -> None:
    """Apply awareness-on-reveal engagement + timeline effects to a freshly-revealed customer.

    market_logic.compute_awareness_reveal is pure (returns values); the resolver owns the customer
    mutation. Used at every reveal site (inbound + discovery, handwritten + generated).
    """
    engagement, timeline_bonus = market_logic.compute_awareness_reveal(
        market_logic.compute_awareness_score(customer, awareness), calibration, rng,
    )
    customer.engagement = engagement
    if timeline_bonus > 0:
        customer.timeline += timeline_bonus
        customer.timeline_original = customer.timeline


class TurnResolver:
    def __init__(
        self,
        state: GameState,
        calibration: CalibrationParams,
        rng: random.Random,
        generator_config: CustomerGeneratorConfig | None = None,
        features_dict: dict | None = None,
    ):
        self.state = state
        self.calibration = calibration
        self.rng = rng
        self.generator_config = generator_config
        self.features_dict = features_dict or {}

    def resolve(self, valid_actions: list[GameAction]) -> TurnRecord:
        record = TurnRecord(
            turn=self.state.turn,
            actions_valid=valid_actions,
            capacity_available=self.state.resources.capacity_per_turn,
        )

        # Feature IDs that received build progress this turn — read by the emergent-need
        # lifecycle to pause the bleed/expiry clock while Eng is actively building. Populated
        # in _resolve_engineering (step 1), consumed in _resolve_cs (step 3) and
        # _resolve_emergent_needs (step 8b).
        self._features_built_this_turn: set[str] = set()

        # Marketing budget spend accrued in _resolve_marketing (step 4), deducted in
        # _resolve_financial (step 9) alongside team_cost/overhead/maintenance.
        self._marketing_budget_spend: int = 0

        # Clear the 1-turn analysis buffer: any results stashed last turn have already been
        # delivered to this turn's observation. Re-populated in _resolve_analyses (step 6b).
        self.state.pending_analyses = {}

        # Categorize actions by type
        build_actions = [a for a in valid_actions if isinstance(a, BuildAction)]
        fix_bug_actions = [a for a in valid_actions if isinstance(a, FixBugsAction)]
        infra_actions = [a for a in valid_actions if isinstance(a, InfrastructureAction)]
        sell_actions = [a for a in valid_actions if isinstance(a, SellAction)]
        support_actions = [a for a in valid_actions if isinstance(a, SupportAction)]
        discover_actions = [a for a in valid_actions if isinstance(a, DiscoverAction)]
        market_actions = [a for a in valid_actions if isinstance(a, MarketAction)]
        market_support_actions = [a for a in valid_actions if isinstance(a, MarketSupportAction)]
        hire_actions = [a for a in valid_actions if isinstance(a, HireAction)]
        sustain_hire_actions = [a for a in valid_actions if isinstance(a, SustainHireAction)]
        fire_actions = [a for a in valid_actions if isinstance(a, FireAction)]
        ops_project_actions = [a for a in valid_actions if isinstance(a, OpsProjectAction)]
        ops_support_actions = [a for a in valid_actions if isinstance(a, OpsProjectSupportAction)]
        ops_analysis_actions = [a for a in valid_actions if isinstance(a, OpsAnalysisAction)]
        analysis_scope_actions = [a for a in valid_actions if isinstance(a, AnalysisScopeAction)]

        # 0. Hiring (pre-committed work) and firing
        self._resolve_hiring(hire_actions, sustain_hire_actions, record)
        self._resolve_fire(fire_actions, record)

        # 1. Engineering
        self._resolve_engineering(build_actions, fix_bug_actions, infra_actions, record)

        # 2. Sales
        self._resolve_sales(sell_actions, record)

        # 3. CS
        self._resolve_cs(support_actions, record)

        # 4. Marketing (incl. Sales-gated pipeline progression via market_support co-investment)
        self._resolve_marketing(market_actions, market_support_actions, record)

        # 5. Discovery
        self._resolve_discovery(discover_actions, record)

        # 6. Ops Projects
        self._resolve_ops_projects(ops_project_actions, ops_support_actions, record)

        # 6b. Ops cross-functional analyses (same-turn co-invest handshake). Reads prior-turn
        # turn_history + current visible state; output only affects next-turn observation.
        self._resolve_analyses(ops_analysis_actions, analysis_scope_actions, record)

        # 7. Competitive events
        self._resolve_competitive_events(record)

        # 7b. Competitor pricing events (random injection)
        self._resolve_competitor_pricing_events(record)

        # 8. Bug injection
        self._resolve_bug_injection(record)

        # 8b. Emergent-need lifecycle + injection. MUST run after _resolve_cs (step 3) so a
        # need injected on turn T cannot be health_checked until T+1 (discovery gate).
        self._resolve_emergent_needs(record)

        # 9. Financial
        self._resolve_financial(record)

        # 10. Metrics and turn advancement
        self._resolve_metrics(record)

        # Track capacity used per pool
        for a in valid_actions:
            if isinstance(a, (BuildAction, FixBugsAction, InfrastructureAction)):
                record.eng_capacity_used += a.capacity
            elif isinstance(a, (SellAction, DiscoverAction, MarketSupportAction)):
                # market_support co-investment draws from the Sales pool.
                record.sales_capacity_used += a.capacity
            elif isinstance(a, SupportAction):
                record.support_capacity_used += a.capacity
            elif isinstance(a, MarketAction):
                record.marketing_capacity_used += a.capacity
            elif isinstance(a, OpsProjectAction):
                record.ops_capacity_used += a.capacity
            elif isinstance(a, OpsAnalysisAction):
                # The Ops side of an analysis handshake draws from the ops pool.
                record.ops_capacity_used += a.capacity
            elif isinstance(a, AnalysisScopeAction):
                # The scope co-investment draws from the REQUESTING team's pool.
                attr = _ANALYSIS_TARGET_TO_USED_ATTR.get(a.target_function)
                if attr:
                    setattr(record, attr, getattr(record, attr) + a.capacity)
            elif isinstance(a, OpsProjectSupportAction):
                # Support draws from the target function's pool
                project = self.state.process_projects.get(a.project_id)
                if project:
                    func_to_attr = {
                        "engineering": "eng_capacity_used",
                        "sales": "sales_capacity_used",
                        "support": "support_capacity_used",
                        "marketing": "marketing_capacity_used",
                    }
                    attr = func_to_attr.get(project.target_function)
                    if attr:
                        setattr(record, attr, getattr(record, attr) + a.capacity)
            elif isinstance(a, HireAction):
                # Hire capacity cost tracked against hiring_function pool (not target)
                hire_cap = self.calibration.hire_capacity_cost if self.calibration else 0
                func_to_attr = {
                    "engineering": "eng_capacity_used", "sales": "sales_capacity_used",
                    "cs": "support_capacity_used", "marketing": "marketing_capacity_used",
                    "ops": "ops_capacity_used",
                }
                attr = func_to_attr.get(a.hiring_function, "eng_capacity_used")
                setattr(record, attr, getattr(record, attr) + hire_cap)
            elif isinstance(a, SustainHireAction):
                hire_cap = self.calibration.hire_capacity_cost if self.calibration else 3
                hire = next((h for h in self.state.pending_hires if h.id == a.hire_id), None)
                if hire:
                    func_to_attr = {
                        "engineering": "eng_capacity_used", "sales": "sales_capacity_used",
                        "cs": "support_capacity_used", "marketing": "marketing_capacity_used",
                        "ops": "ops_capacity_used",
                    }
                    attr = func_to_attr.get(hire.hiring_function, "eng_capacity_used")
                    setattr(record, attr, getattr(record, attr) + hire_cap)

        record.capacity_used = (
            record.eng_capacity_used + record.sales_capacity_used
            + record.support_capacity_used + record.marketing_capacity_used
            + record.ops_capacity_used
        )

        return record

    def _resolve_engineering(
        self,
        build_actions: list[BuildAction],
        fix_bug_actions: list[FixBugsAction],
        infra_actions: list[InfrastructureAction],
        record: TurnRecord,
    ) -> None:
        # Build features
        build_capacity_by_quality: dict[str, int] = {}
        shipped_statuses = {FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished}
        for action in build_actions:
            feature = self.state.features.get(action.feature_id)
            if feature is None:
                continue

            was_shipped = feature.status in shipped_statuses

            # If starting a new build/upgrade target, reset progress tracking
            if feature.current_target != action.quality:
                feature.turns_worked = 0
                feature.progress = 0.0

            new_progress, new_status = product_logic.apply_build_progress(
                feature, action.capacity, action.quality, self.calibration,
            )

            new_shipped = new_status in shipped_statuses

            # Increment turns worked
            if action.capacity > 0:
                feature.turns_worked += 1
                # Mark feature as actively built this turn (pauses emergent-need bleed/clock)
                self._features_built_this_turn.add(feature.id)

            feature.progress = new_progress

            if was_shipped and not new_shipped:
                # Upgrade in progress — keep shipped status, just track target and progress
                feature.current_target = action.quality
            else:
                # New build completing/progressing, or upgrade completing
                feature.status = new_status
                feature.current_target = action.quality if new_status == FeatureStatus.in_progress else None

            if new_shipped:
                if was_shipped:
                    record.events.append(f"feature_upgraded:{feature.id}:{new_status.value}")
                else:
                    record.events.append(f"feature_shipped:{feature.id}:{new_status.value}")
                feature.turns_worked = 0

            quality = action.quality.value
            build_capacity_by_quality[quality] = build_capacity_by_quality.get(quality, 0) + action.capacity

        # Fix bugs
        for action in fix_bug_actions:
            if action.bug_id:
                bug = next((b for b in self.state.bugs if b.id == action.bug_id and not b.is_resolved), None)
                if bug and product_logic.compute_bug_fix_progress(bug, action.capacity):
                    bug.is_resolved = True
                    record.bugs_fixed += 1
                    record.events.append(
                        f"bug_fixed:{bug.id}:{bug.severity.value}:{bug.feature_id}"
                    )
            else:
                # Auto-target: fix highest severity unresolved bugs
                remaining_capacity = action.capacity
                unresolved = sorted(
                    [b for b in self.state.bugs if not b.is_resolved],
                    key=lambda b: {"critical": 0, "major": 1, "minor": 2}[b.severity.value],
                )
                for bug in unresolved:
                    if remaining_capacity <= 0:
                        break
                    if product_logic.compute_bug_fix_progress(bug, remaining_capacity):
                        bug.is_resolved = True
                        record.bugs_fixed += 1
                        record.events.append(
                            f"bug_fixed:{bug.id}:{bug.severity.value}:{bug.feature_id}"
                        )
                        remaining_capacity -= {"critical": 4, "major": 2, "minor": 1}[bug.severity.value]

        # Infrastructure
        infra_capacity = sum(a.capacity for a in infra_actions)
        if infra_capacity > 0:
            record.events.append(f"infrastructure_work:capacity={infra_capacity}")

        # Tech debt
        debt_delta = product_logic.compute_tech_debt_delta(
            build_capacity_by_quality, infra_capacity, self.calibration,
        )
        self.state.tech_debt.level = max(0.0, self.state.tech_debt.level + debt_delta)

    def _resolve_sales(self, sell_actions: list[SellAction], record: TurnRecord) -> None:
        # Track capacity per customer for engagement updates
        sell_capacity_by_customer: dict[str, int] = {}
        for action in sell_actions:
            sell_capacity_by_customer[action.customer_id] = (
                sell_capacity_by_customer.get(action.customer_id, 0) + action.capacity
            )

        # Look up process bonuses for sales
        sales_conversion_bonus = ops_logic.get_active_bonus(
            self.state.active_process_bonuses, "sales", "conversion_rate",
        )

        for action in sell_actions:
            customer = self.state.customers.get(action.customer_id)
            if customer is None:
                continue

            if not customer.timeline_active:
                customer.timeline_active = True
                record.events.append(f"timeline_started:{customer.id}")

            # Determine effective proposed price for proposal/negotiate
            pricing_modifier = 1.0
            effective_price: int | None = None
            if action.sell_action in ("proposal", "negotiate"):
                effective_price = (
                    action.proposed_deal_value
                    or customer.last_proposed_price
                    or customer.deal_value
                )
                if customer.desired_price_point > 0:
                    pricing_modifier = compute_pricing_modifier(
                        effective_price, customer.desired_price_point, self.calibration,
                    )

            satisfaction = customer_logic.compute_rubric_satisfaction(customer, self.state.features)
            probability = customer_logic.compute_conversion_probability(
                customer, action.sell_action, satisfaction, self.calibration,
                capacity_allocated=action.capacity,
                sales_momentum=self.state.sales_momentum,
                process_bonus=sales_conversion_bonus,
                pricing_modifier=pricing_modifier,
            )

            new_stage = customer_logic.advance_pipeline_stage(
                customer, action.sell_action, probability, self.state.features,
                self.calibration, self.rng,
            )

            # Update pricing state for proposal/negotiate
            if action.sell_action in ("proposal", "negotiate") and effective_price is not None:
                customer.last_proposed_price = effective_price
                customer.has_received_proposal = True

            if new_stage is not None:
                old_stage = customer.stage
                customer.stage = new_stage
                customer.turns_in_current_stage = 0

                if new_stage == CustomerStage.customer:
                    # Deal closed — MRR uses the agreed price
                    closing_price = effective_price or customer.deal_value
                    customer.deal_value = closing_price
                    customer.health = self.calibration.new_customer_starting_health
                    customer.onboarding_turns_remaining = self.calibration.new_customer_onboarding_turns
                    customer.timeline_active = False
                    self.state.resources.mrr += customer.deal_value
                    record.events.append(f"deal_won:{customer.id}")
                else:
                    record.events.append(f"stage_advanced:{customer.id}:{old_stage.value}->{new_stage.value}")
            elif (
                action.sell_action in ("proposal", "negotiate")
                and effective_price is not None
                and customer.desired_price_point > 0
                and effective_price > customer.desired_price_point
            ):
                # Failed proposal/negotiate where price was too high — emit sandbagged feedback
                sandbagged = compute_sandbagged_price(
                    customer.desired_price_point, self.calibration, self.rng,
                )
                record.events.append(
                    f"pricing_feedback:{customer.id}:proposed={effective_price}:indicated={sandbagged}"
                )

        # Update engagement for all visible pipeline customers
        for customer in self.state.customers.values():
            if not customer.is_visible or customer.stage == CustomerStage.customer:
                continue
            capacity = sell_capacity_by_customer.get(customer.id, 0)
            customer.engagement = customer_logic.update_engagement(customer, capacity, self.calibration)

        # Timeline tick — only ticks once activated by a sell action
        for customer in self.state.customers.values():
            if customer.timeline_active:
                customer.timeline -= 1
                if customer_logic.check_timeline_expiry(customer):
                    customer.stage = CustomerStage.lead
                    customer.turns_in_current_stage = 0
                    customer.timeline = customer.timeline_original
                    customer.timeline_active = False
                    customer.timeline_resets += 1
                    customer.engagement = Engagement.cold
                    record.events.append(
                        f"timeline_expired_reset:{customer.id}:resets={customer.timeline_resets}"
                    )

    def _resolve_cs(self, support_actions: list[SupportAction], record: TurnRecord) -> None:
        # Bucket support actions per customer: total capacity (drives the baseline attention
        # curve) plus per-verb capacity (drives the specialty effects).
        cs_capacity_by_customer: dict[str, int] = {}
        capacity_by_verb: dict[str, dict[str, int]] = {}
        for action in support_actions:
            cid = action.customer_id
            cs_capacity_by_customer[cid] = cs_capacity_by_customer.get(cid, 0) + action.capacity
            verb_caps = capacity_by_verb.setdefault(cid, {})
            verb_caps[action.support_action] = verb_caps.get(action.support_action, 0) + action.capacity

        # Look up process bonuses for support
        health_bonus = ops_logic.get_active_bonus(
            self.state.active_process_bonuses, "support", "health_delta_bonus",
        )

        # Update health for all active customers
        for customer in self.state.customers.values():
            if customer.stage != CustomerStage.customer:
                continue

            cid = customer.id
            cs_capacity = cs_capacity_by_customer.get(cid, 0)
            verb_caps = capacity_by_verb.get(cid, {})

            # --- health_check: the ONLY way CS learns this customer's emergent needs and
            # churn drivers (CS discovery gate, parallel to Sales' price discovery). ---
            if "health_check" in verb_caps:
                customer.churn_drivers_revealed = True
                for need in self.state.emergent_needs:
                    if (
                        need.customer_id == cid
                        and not need.is_revealed
                        and not need.is_met
                        and not need.is_expired
                    ):
                        need.is_revealed = True
                        record.events.append(
                            f"emergent_need_revealed:{need.id}:{cid}:{need.feature_id}"
                        )

            # Emergent needs bleeding this turn: not met, and not paused by active build.
            # turns_unmet supplies the magnitude inside compute_health_delta.
            bleeding_needs = [
                n for n in self.state.emergent_needs
                if n.customer_id == cid
                and not n.is_met
                and n.feature_id not in self._features_built_this_turn
            ]

            delta = customer_logic.compute_health_delta(
                customer, self.state.bugs, cs_capacity, self.calibration,
                unmet_emergent_needs=bleeding_needs,
            )
            # Apply process bonus from Support Automation
            delta += health_bonus

            # --- onboard: extra health attention during the onboarding window only. ---
            if "onboard" in verb_caps and customer.onboarding_turns_remaining > 0:
                delta += self.calibration.onboard_health_bonus

            # --- churn_intervention: costly stochastic save, only below a health threshold. ---
            if "churn_intervention" in verb_caps:
                if (
                    verb_caps["churn_intervention"] >= self.calibration.churn_intervention_min_capacity
                    and customer.health < self.calibration.churn_intervention_health_threshold
                ):
                    if self.rng.random() < self.calibration.churn_intervention_success_prob:
                        delta += self.calibration.churn_intervention_health_recovery
                        record.events.append(f"churn_intervention:{cid}:success")
                    else:
                        record.events.append(f"churn_intervention:{cid}:failed")

            customer.health = max(0.0, min(10.0, customer.health + delta))
            customer.health_history.append(customer.health)

            # Onboarding tick — onboard verb accelerates the window by an extra decrement.
            if customer.onboarding_turns_remaining > 0:
                decrement = 1
                if "onboard" in verb_caps:
                    decrement += self.calibration.onboard_acceleration
                customer.onboarding_turns_remaining = max(0, customer.onboarding_turns_remaining - decrement)

            # Churn tracking
            if customer.health < self.calibration.churn_health_threshold:
                customer.turns_below_churn_threshold += 1
            else:
                customer.turns_below_churn_threshold = 0

            # Expansion tracking
            if customer.health > self.calibration.expansion_health_threshold:
                customer.turns_above_expansion_threshold += 1
            else:
                customer.turns_above_expansion_threshold = 0

            # Check churn
            if customer_logic.check_churn(customer, self.calibration):
                customer.stage = CustomerStage.churned
                customer.turns_in_current_stage = 0
                self.state.resources.mrr -= customer.deal_value
                record.churn_count += 1
                record.events.append(f"churn:{customer.id}")

            # Check expansion
            elif customer_logic.check_expansion(customer, self.calibration):
                increase = int(customer.deal_value * self.calibration.expansion_deal_value_increase)
                customer.deal_value += increase
                self.state.resources.mrr += increase
                customer.turns_above_expansion_threshold = 0
                record.events.append(f"expansion:{customer.id}:+{increase}")

    def _resolve_marketing(
        self,
        market_actions: list[MarketAction],
        market_support_actions: list[MarketSupportAction],
        record: TurnRecord,
    ) -> None:
        total_marketing = sum(a.capacity for a in market_actions)
        self.state.marketing_history.append(total_marketing)

        # --- Awareness accrual: schedule lagged/spread increments per channel, accumulate
        #     shared-budget spend (deducted later in _resolve_financial). ---
        budget_spend = 0
        for action in market_actions:
            profile = self.calibration.channel_profiles.get(action.channel)
            if profile is None:
                continue
            targets = market_logic.effective_awareness_targets(action.target_features, self.state.features)
            self.state.pending_awareness.extend(
                market_logic.schedule_awareness(profile, targets, action.capacity, self.state.turn)
            )
            budget_spend += action.capacity * profile.budget_cost_per_capacity
        self._marketing_budget_spend = budget_spend

        # --- Mature pending awareness whose land_turn has arrived ---
        matured, remaining = market_logic.mature_pending_awareness(
            self.state.pending_awareness, self.state.turn,
        )
        self.state.pending_awareness = remaining
        matured_features: set[str] = set()
        for entry in matured:
            self.state.awareness[entry.feature_id] = (
                self.state.awareness.get(entry.feature_id, 0.0) + entry.amount
            )
            matured_features.add(entry.feature_id)
        for fid in sorted(matured_features):
            # Marketing-only analysis event (filtered to the marketing role only).
            record.events.append(f"awareness_built:{fid}")

        # --- Decay every awareness stock; drop entries below epsilon ---
        self.state.awareness = market_logic.decay_awareness(
            self.state.awareness, self.calibration.awareness_decay, self.calibration.awareness_epsilon,
        )

        # --- Inbound leads: COUNT from the existing formula; awareness only shapes which
        #     leads are favoured and what state they arrive in (quality, not access). ---
        marketing_bonus = ops_logic.get_active_bonus(
            self.state.active_process_bonuses, "marketing", "marketing_effectiveness",
        )

        num_inbound = market_logic.compute_inbound_leads(
            self.state.marketing_history, self.calibration,
            process_bonus=marketing_bonus,
        )
        # Customers revealed via inbound THIS turn — eligible for new-lead progression below.
        revealed_inbound: list = []
        if num_inbound > 0:
            shipped_statuses = {FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished}
            shipped_features = [
                fid for fid, f in self.state.features.items() if f.status in shipped_statuses
            ]

            # Reveal handwritten hidden customers first — weighted toward those whose needs
            # include high-awareness features (feature-bias), arriving warm via awareness.
            hidden = [c for c in self.state.customers.values() if not c.is_visible]
            revealed = 0
            for _ in range(min(num_inbound, len(hidden))):
                if not hidden:
                    break
                customer = self._weighted_choice_by_awareness(hidden)
                customer.is_visible = True
                customer.turns_in_current_stage = 0
                _apply_reveal_awareness(customer, self.state.awareness, self.calibration, self.rng)
                hidden.remove(customer)
                record.events.append(f"inbound_lead:{customer.id}")
                revealed_inbound.append(customer)
                revealed += 1

            # Generate additional inbound if pool exhausted + generator config present
            remaining_inbound = num_inbound - revealed
            if self.generator_config and remaining_inbound > 0:
                gen_count = min(remaining_inbound, self.generator_config.max_candidates_per_inbound)
                candidates = generate_inbound_candidates(
                    shipped_features, self.generator_config,
                    self.features_dict, self.rng,
                    self.state.next_generated_customer_id, gen_count,
                    awareness=self.state.awareness,
                    awareness_bias=self.calibration.inbound_awareness_bias,
                )
                for cust in candidates:
                    cust.is_visible = True
                    cust.turns_in_current_stage = 0
                    _apply_reveal_awareness(cust, self.state.awareness, self.calibration, self.rng)
                    self.state.customers[cust.id] = cust
                    record.events.append(f"inbound_lead:{cust.id}")
                    revealed_inbound.append(cust)
                self.state.next_generated_customer_id += len(candidates)

        # --- Competitive radar (marketing-only passive signal). Deterministic per seed;
        #     surfaced only in the marketing obs / marketing event stream. ---
        radar_signals = market_logic.scan_competitor_radar(
            self.state.competitors, self.state.customers, self.state.awareness,
            self.state.turn, self.calibration, self.rng,
        )
        for signal in radar_signals:
            record.events.append(f"competitor_radar:{signal}")

        # --- Sales-gated pipeline progression (Marketing<->Sales co-investment) ---
        # Sales co-invests (market_support) in a budget channel's SAME-TURN campaign to buy
        # one-stage pipeline progression (capped at in_deal). Self-limiting: needs budget AND
        # the scarce sales pool. Mis-timed co-investment (no matching campaign) is wasted.
        if market_support_actions:
            self._resolve_pipeline_progression(market_actions, market_support_actions, revealed_inbound, record)

    def _resolve_pipeline_progression(
        self,
        market_actions: list[MarketAction],
        market_support_actions: list[MarketSupportAction],
        revealed_inbound: list,
        record: TurnRecord,
    ) -> None:
        budget_channels = ("content", "events")

        # 1. Tally Sales collab capacity per channel + events target customers.
        collab: dict[str, int] = {}
        events_targets: list[str] = []
        for action in market_support_actions:
            collab[action.channel] = collab.get(action.channel, 0) + action.capacity
            if action.channel == "events" and action.target_customer_id is not None:
                events_targets.append(action.target_customer_id)

        # 2. Tally Marketing capacity per budget channel run THIS turn.
        marketing_by_channel: dict[str, int] = {}
        for action in market_actions:
            if action.channel in budget_channels:
                marketing_by_channel[action.channel] = (
                    marketing_by_channel.get(action.channel, 0) + action.capacity
                )

        # 3. Unmatched check: Sales co-invested but no matching campaign ran → wasted capacity.
        matched: dict[str, int] = {}
        for channel in budget_channels:
            collab_cap = collab.get(channel, 0)
            if collab_cap <= 0:
                continue
            mkt_cap = marketing_by_channel.get(channel, 0)
            if mkt_cap <= 0:
                record.events.append(f"market_support_unmatched:{channel}")
            else:
                matched[channel] = mkt_cap

        if not matched:
            return

        # 4. New-lead progression (content + events): advance newly-revealed inbound one stage.
        #    Prefer the higher-prob events channel when both are funded+matched.
        preferred = "events" if "events" in matched else next(iter(matched), None)
        if preferred is not None:
            mkt_cap = matched[preferred]
            collab_cap = collab[preferred]
            for customer in revealed_inbound:
                next_stage = market_logic.next_pipeline_stage_capped(customer.stage)
                if next_stage is None:
                    continue
                if market_logic.roll_pipeline_progression(
                    preferred, mkt_cap, collab_cap, self.calibration, self.rng,
                ):
                    from_stage = customer.stage
                    customer.stage = next_stage
                    customer.turns_in_current_stage = 0
                    record.events.append(
                        f"pipeline_progression:{customer.id}:{from_stage.value}->{next_stage.value}"
                    )

        # 5. Existing-customer progression (events only): push named pipeline customers one stage.
        if "events" in matched:
            mkt_cap = matched["events"]
            collab_cap = collab["events"]
            for cid in events_targets:
                customer = self.state.customers.get(cid)
                if customer is None or not customer.is_visible:
                    continue
                next_stage = market_logic.next_pipeline_stage_capped(customer.stage)
                if next_stage is None:
                    continue
                if market_logic.roll_pipeline_progression(
                    "events", mkt_cap, collab_cap, self.calibration, self.rng,
                ):
                    from_stage = customer.stage
                    customer.stage = next_stage
                    customer.turns_in_current_stage = 0
                    record.events.append(
                        f"pipeline_progression:{customer.id}:{from_stage.value}->{next_stage.value}"
                    )

        # 6. Echo the matched co-investment handshake (visible to sales + marketing).
        for channel, mkt_cap in matched.items():
            record.events.append(
                f"market_support:{channel}:capacity={collab[channel]}:matched={mkt_cap}"
            )

    def _weighted_choice_by_awareness(self, customers: list) -> object:
        """Pick a customer weighted toward high-awareness feature needs (inbound bias)."""
        weights = [
            market_logic.awareness_lead_weight(
                c, self.state.awareness, self.calibration.inbound_awareness_bias,
            )
            for c in customers
        ]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(customers)
        r = self.rng.random() * total
        cumulative = 0.0
        for customer, weight in zip(customers, weights):
            cumulative += weight
            if r <= cumulative:
                return customer
        return customers[-1]

    def _resolve_discovery(self, discover_actions: list[DiscoverAction], record: TurnRecord) -> None:
        discovery_bonus = ops_logic.get_active_bonus(
            self.state.active_process_bonuses, "sales", "discovery_bonus",
        )
        shipped_statuses = {FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished}

        for action in discover_actions:
            effective_targets = action.target_features or [
                fid for fid, f in self.state.features.items() if f.status in shipped_statuses
            ]
            shipped_features = [
                fid for fid, f in self.state.features.items() if f.status in shipped_statuses
            ]

            # Step 1: handwritten hidden candidates matching target features
            hidden = [c for c in self.state.customers.values() if not c.is_visible]
            matching = filter_hidden_by_features(hidden, effective_targets)

            # Step 2: roll discovery on handwritten matches
            discovered_ids = market_logic.discover_customers(
                action.capacity, matching, None, self.rng,
                process_bonus=discovery_bonus,
            )
            for cid in discovered_ids:
                customer = self.state.customers[cid]
                customer.is_visible = True
                customer.turns_in_current_stage = 0
                _apply_reveal_awareness(customer, self.state.awareness, self.calibration, self.rng)
                record.events.append(f"discovered:{cid}")

            remaining_capacity = action.capacity - len(matching)

            # Step 3: generate additional candidates if config present + capacity remains
            if self.generator_config and remaining_capacity > 0:
                gen_count = min(remaining_capacity, self.generator_config.max_candidates_per_discover)
                candidates = generate_discovery_candidates(
                    effective_targets, shipped_features, self.generator_config,
                    self.features_dict, self.rng,
                    self.state.next_generated_customer_id, gen_count,
                )
                gen_discovered_ids = market_logic.discover_customers(
                    remaining_capacity, candidates, None, self.rng,
                    process_bonus=discovery_bonus,
                )
                for cid in gen_discovered_ids:
                    cust = next(c for c in candidates if c.id == cid)
                    cust.is_visible = True
                    cust.turns_in_current_stage = 0
                    _apply_reveal_awareness(cust, self.state.awareness, self.calibration, self.rng)
                    self.state.customers[cid] = cust
                    record.events.append(f"discovered:{cid}")
                self.state.next_generated_customer_id += len(candidates)

    def _resolve_ops_projects(
        self,
        ops_actions: list[OpsProjectAction],
        support_actions: list[OpsProjectSupportAction],
        record: TurnRecord,
    ) -> None:
        # First: handle maintenance refreshes (completed + active bonus) and start new/re-runs
        for action in ops_actions:
            project = self.state.process_projects.get(action.project_id)
            if project is None:
                continue

            if project.status == ProcessProjectStatus.completed:
                active_bonus = next(
                    (b for b in self.state.active_process_bonuses if b.project_id == project.id),
                    None,
                )
                if active_bonus is not None:
                    # Maintenance: single-turn action — reset bonus to full.
                    # Set +1 so that after tick_bonus_durations runs later this same
                    # _resolve_ops_projects call, it lands at bonus_duration_turns.
                    active_bonus.turns_remaining = active_bonus.bonus_duration_turns + 1
                    record.events.append(
                        f"ops_project_refresh:{project.id}:{project.name}:bonus_reset_to_full"
                    )
                else:
                    # Net-new re-run (bonus lapsed): reset investment, restart project
                    project.target_team_capacity_invested = 0
                    project.progress_turns = 0
                    project.status = ProcessProjectStatus.in_progress
                    record.events.append(f"ops_project_restarted:{project.id}:{project.name}")
                continue  # maintenance handled above; net-new re-run is now in_progress and
                          # WILL be picked up by the Third loop this same turn — this is intentional.
                          # A 1-turn project (duration_turns=1) will therefore complete immediately
                          # on the re-run turn, which is the correct behaviour.

            if project.status == ProcessProjectStatus.available:
                project.status = ProcessProjectStatus.in_progress
                record.events.append(f"ops_project_started:{project.id}:{project.name}")

        # Second: process target team support investments
        for action in support_actions:
            project = self.state.process_projects.get(action.project_id)
            if project is None or project.status != ProcessProjectStatus.in_progress:
                continue
            project.target_team_capacity_invested += action.capacity
            record.events.append(
                f"ops_project_support:{project.id}:capacity={action.capacity}:"
                f"total={project.target_team_capacity_invested}"
            )

        # Third: advance progress on in_progress projects and check completions
        for action in ops_actions:
            project = self.state.process_projects.get(action.project_id)
            if project is None or project.status != ProcessProjectStatus.in_progress:
                continue

            if action.capacity >= project.ops_capacity_cost:
                # Use compute_project_progress to check completion before incrementing
                # (the function checks progress_turns + 1 internally)
                completes = ops_logic.compute_project_progress(project, action.capacity)
                project.progress_turns += 1

                if completes:
                    project.status = ProcessProjectStatus.completed
                    project.completed_turn = self.state.turn

                    bonus_value = ops_logic.compute_process_bonus(project, self.rng)
                    # Remove any stale bonus for this project before adding the fresh one
                    self.state.active_process_bonuses = [
                        b for b in self.state.active_process_bonuses if b.project_id != project.id
                    ]
                    self.state.active_process_bonuses.append(ActiveProcessBonus(
                        project_id=project.id,
                        bonus_type=project.bonus_type,
                        bonus_value=bonus_value,
                        target_function=project.target_function,
                        turns_remaining=project.bonus_duration_turns,
                        bonus_duration_turns=project.bonus_duration_turns,
                        original_ops_capacity_cost=project.ops_capacity_cost,
                        # Apply the global floor scale once, here — keeps compute_effective_bonus
                        # a pure function of the bonus (no calibration in the hot path).
                        permanent_floor_fraction=(
                            project.permanent_floor_fraction * self.calibration.permanent_floor_scale
                        ),
                    ))
                    record.events.append(
                        f"ops_project_completed:{project.id}:{project.name}:"
                        f"bonus={bonus_value:.3f}:{project.bonus_type}"
                    )

        # Tick active bonus durations (spike decay; floored bonuses pin at 0, others removed)
        self.state.active_process_bonuses = ops_logic.tick_bonus_durations(
            self.state.active_process_bonuses,
        )

    def _resolve_analyses(
        self,
        ops_analysis_actions: list[OpsAnalysisAction],
        analysis_scope_actions: list[AnalysisScopeAction],
        record: TurnRecord,
    ) -> None:
        """Resolve the symmetric, same-turn cross-functional analysis handshake.

        Both an OpsAnalysisAction and an AnalysisScopeAction with the SAME
        (target_function, analysis_type) must be submitted this turn. Co-presence is the gate
        (direction-agnostic):
          * matched keys -> compute the analysis from observable history (pure, no RNG), stash
            under pending_analyses[requester_agent_fn] for next-turn delivery, emit an
            ops_analysis:{tf}:{at} echo.
          * keys present on exactly one side -> wasted (capacity already debited in the accounting
            pass), emit analysis_unmatched:{tf}:{at}.
        """
        if not ops_analysis_actions and not analysis_scope_actions:
            return

        ops_keys = {(a.target_function, a.analysis_type) for a in ops_analysis_actions}
        scope_keys = {(a.target_function, a.analysis_type) for a in analysis_scope_actions}

        # Symmetric difference = unmatched (one side submitted alone, or mismatched type).
        for tf, at in sorted(ops_keys ^ scope_keys):
            record.events.append(f"analysis_unmatched:{tf}:{at}")

        # Intersection = matched. Sorted for determinism (no RNG anywhere in analysis).
        for tf, at in sorted(ops_keys & scope_keys):
            result = analysis_logic.compute_analysis(at, self.state, self.calibration)
            result["target_function"] = tf
            agent_fn = _ANALYSIS_TARGET_TO_AGENT_FN.get(tf, tf)
            self.state.pending_analyses.setdefault(agent_fn, []).append(result)
            record.events.append(f"ops_analysis:{tf}:{at}")

    def _resolve_hiring(
        self,
        hire_actions: list[HireAction],
        sustain_actions: list[SustainHireAction],
        record: TurnRecord,
    ) -> None:
        # Step 1: Process sustains
        sustained_ids: set[str] = set()
        for action in sustain_actions:
            hire = next((h for h in self.state.pending_hires if h.id == action.hire_id), None)
            if hire is None:
                continue
            hire.active_turns_completed += 1
            sustained_ids.add(hire.id)
            record.events.append(
                f"hire_sustained:{hire.id}:{hire.target_function}"
                f":active_{hire.active_turns_completed}/{hire.active_turns_required}"
            )

        # Step 2: Cancel unsustained active-phase hires
        cancelled = []
        for hire in self.state.pending_hires:
            if hire.active_turns_completed < hire.active_turns_required and hire.id not in sustained_ids:
                cancelled.append(hire)
                record.events.append(
                    f"hire_cancelled:{hire.id}:{hire.target_function}:missed_sustain"
                    f":was_{hire.active_turns_completed}/{hire.active_turns_required}"
                )
        for hire in cancelled:
            self.state.pending_hires.remove(hire)

        # Step 3: New hires
        for action in hire_actions:
            hire_cost = self.state.resources.capacity_per_turn * self.calibration.hire_budget_cost_multiplier
            self.state.resources.budget -= hire_cost

            cross = action.hiring_function != action.target_function
            if cross:
                delay = round(self.calibration.hire_arrival_delay * self.calibration.cross_hire_delay_multiplier)
                capacity = round(self.calibration.hire_capacity_bonus * self.calibration.cross_hire_capacity_factor)
            else:
                delay = self.calibration.hire_arrival_delay
                capacity = self.calibration.hire_capacity_bonus

            active_required = delay // 2
            hire_id = f"H{self.state.next_hire_id}"
            self.state.next_hire_id += 1

            self.state.pending_hires.append(PendingHire(
                id=hire_id,
                target_function=action.target_function,
                hiring_function=action.hiring_function,
                turns_remaining=delay,
                onboarding_turns_remaining=self.calibration.hire_onboarding_turns,
                capacity_bonus=capacity,
                is_cross_function=cross,
                active_turns_required=active_required,
                active_turns_completed=1,
            ))
            cross_tag = f":cross_hire_from_{action.hiring_function}" if cross else ""
            record.events.append(
                f"hire_started:{hire_id}:{action.target_function}{cross_tag}"
                f":arrives_in_{delay}_turns:capacity_{capacity}"
                f":active_phase_{active_required}_turns"
            )

        # Step 4: Tick pending hires and apply arrivals
        arrived = []
        for hire in self.state.pending_hires:
            hire.turns_remaining -= 1
            if hire.turns_remaining <= 0:
                arrived.append(hire)

        for hire in arrived:
            self.state.pending_hires.remove(hire)
            bonus = hire.capacity_bonus
            if hire.target_function == "engineering":
                self.state.resources.eng_capacity += bonus
            elif hire.target_function == "sales":
                self.state.resources.sales_capacity += bonus
            elif hire.target_function == "cs":
                self.state.resources.support_capacity += bonus
            elif hire.target_function == "marketing":
                self.state.resources.marketing_capacity += bonus
            elif hire.target_function == "ops":
                self.state.resources.ops_capacity += bonus
            self.state.resources.capacity_per_turn += bonus
            record.events.append(f"hire_arrived:{hire.id}:{hire.target_function}:+{bonus}_capacity")

    def _resolve_fire(self, fire_actions: list[FireAction], record: TurnRecord) -> None:
        for action in fire_actions:
            # Severance cost: fire_severance_turns × per-capacity-unit team cost
            severance = self.calibration.fire_severance_turns * self.calibration.team_cost_per_capacity
            self.state.resources.budget -= severance

            # Remove up to one hire-unit of capacity (or remainder if less)
            hire_unit = self.calibration.hire_capacity_bonus
            func_to_attr = {
                "engineering": "eng_capacity",
                "sales": "sales_capacity",
                "cs": "support_capacity",
                "marketing": "marketing_capacity",
                "ops": "ops_capacity",
            }
            attr = func_to_attr.get(action.function)
            if attr:
                current = getattr(self.state.resources, attr)
                actual_cut = min(hire_unit, current)
                setattr(self.state.resources, attr, current - actual_cut)
                self.state.resources.capacity_per_turn = max(0, self.state.resources.capacity_per_turn - actual_cut)
                record.events.append(f"fire:{action.function}:-{actual_cut}_capacity:severance_{severance}")

    def _resolve_competitive_events(self, record: TurnRecord) -> None:
        events = market_logic.fire_competitive_events(self.state.competitors, self.state.turn)

        for event in events:
            record.events.append(f"competitive:{event.event_type}:{event.description}")

            # Apply competitive pressure to affected customers
            for cid in event.affected_customers:
                customer = self.state.customers.get(cid)
                if customer is None:
                    continue
                customer.competitive_pressure = market_logic.apply_competitive_pressure(
                    customer, [event],
                )

                # If customer is in pipeline and competitor launches relevant feature,
                # check if competitor wins the deal
                if customer.stage == CustomerStage.in_deal:
                    competitor_satisfaction = sum(event.rubric_impact.values()) / max(len(event.rubric_impact), 1)
                    player_satisfaction = customer_logic.compute_rubric_satisfaction(
                        customer, self.state.features,
                    )
                    if market_logic.check_competitor_deal_win(
                        customer, competitor_satisfaction, player_satisfaction,
                    ):
                        customer.stage = CustomerStage.lost
                        customer.turns_in_current_stage = 0
                        record.events.append(
                            f"deal_lost:{customer.id}:competitor_won:{event.description}"
                        )

    def _resolve_competitor_pricing_events(self, record: TurnRecord) -> None:
        eligible = [
            c for c in self.state.customers.values()
            if c.stage == CustomerStage.in_deal and c.has_received_proposal
        ]
        if not eligible:
            return

        num_events = product_logic.poisson_sample(
            self.calibration.pricing_competitor_event_lambda, self.rng,
        )

        for _ in range(num_events):
            if not eligible:
                break
            customer = self.rng.choice(eligible)

            discount = (
                self.calibration.pricing_competitor_offer_discount
                + self.rng.uniform(
                    -self.calibration.pricing_competitor_offer_jitter,
                    self.calibration.pricing_competitor_offer_jitter,
                )
            )
            competitor_offer = int(customer.desired_price_point * (1.0 - discount))

            original_engagement = customer.engagement
            customer.engagement = Engagement.warm
            competitor_conversion = customer_logic.compute_conversion_probability(
                customer, "proposal",
                satisfaction=self.calibration.pricing_competitor_assumed_satisfaction,
                calibration=self.calibration,
            )
            customer.engagement = original_engagement

            if self.rng.random() < competitor_conversion:
                customer.stage = CustomerStage.lost
                customer.turns_in_current_stage = 0
                eligible.remove(customer)
                record.events.append(
                    f"deal_lost:{customer.id}:competitor_won:competitor_pricing"
                )
            else:
                customer.competitive_pressure += self.calibration.pricing_competitor_pressure_boost
                record.events.append(
                    f"competitor_pricing:{customer.id}:offer={competitor_offer}"
                )

    def _resolve_bug_injection(self, record: TurnRecord) -> None:
        shipped_features = [
            f for f in self.state.features.values()
            if f.status in (FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished)
        ]

        # Look up process bonus for bug rate reduction
        bug_rate_bonus = ops_logic.get_active_bonus(
            self.state.active_process_bonuses, "engineering", "bug_rate_reduction",
        )

        new_bugs = product_logic.inject_bugs(
            self.state.tech_debt.level,
            shipped_features,
            self.calibration,
            self.state.customers,
            self.state.next_bug_id,
            self.state.turn,
            self.rng,
            bug_rate_reduction=bug_rate_bonus,
        )

        for bug in new_bugs:
            self.state.bugs.append(bug)
            record.bugs_injected += 1
            affected_str = ",".join(bug.affected_customers) if bug.affected_customers else ""
            record.events.append(
                f"bug_injected:{bug.id}:{bug.severity.value}:{bug.feature_id}:{affected_str}"
            )

        self.state.next_bug_id += len(new_bugs)

        # Tick unresolved bug timers
        for bug in self.state.bugs:
            if not bug.is_resolved:
                bug.turns_unresolved += 1

    def _resolve_emergent_needs(self, record: TurnRecord) -> None:
        """Emergent-need lifecycle (met / pause / bleed-tick / expire) + injection.

        CRITICAL ORDERING: this runs in the same slot as bug injection (step 8b), strictly
        AFTER _resolve_cs (step 3). A need injected on turn T therefore cannot be revealed by
        a health_check until T+1 — same-turn discovery is structurally impossible. Do NOT move
        this before _resolve_cs; a regression test guards this ordering.

        The bleed itself is applied in compute_health_delta during _resolve_cs; this method
        only advances the unmet clock and runs the state transitions.
        """
        shipped_statuses = {
            FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished,
        }
        grace = self.calibration.emergent_need_grace_turns
        expiry = self.calibration.emergent_need_expiry_turns

        # --- Lifecycle for existing needs ---
        for need in self.state.emergent_needs:
            if need.is_met or need.is_expired:
                continue

            customer = self.state.customers.get(need.customer_id)
            if customer is None or customer.stage != CustomerStage.customer:
                continue  # churned/lost — the need is moot

            feature = self.state.features.get(need.feature_id)

            # 1. Met — feature shipped at any quality. Apply one-time health bonus.
            if feature is not None and feature.status in shipped_statuses:
                need.is_met = True
                customer.health = max(
                    0.0, min(10.0, customer.health + self.calibration.emergent_need_met_health_bonus),
                )
                record.events.append(
                    f"emergent_need_met:{need.id}:{need.customer_id}:{need.feature_id}"
                )
                continue

            # Still within the grace window — no clock, no bleed, no expiry.
            if self.state.turn - need.turn_injected < grace:
                continue

            # 2. Paused — feature received build progress this turn: clock + bleed both halt.
            if need.feature_id in self._features_built_this_turn:
                continue

            # 3. Bleeding — advance the unmet clock (bleed magnitude read in compute_health_delta).
            need.turns_unmet += 1

            # 4. Expired — convert to an (informational-only, v1) churn driver. The bleed
            # continues; turns_unmet freezes here since expired needs skip this loop hereafter.
            if need.turns_unmet >= expiry:
                need.is_expired = True
                customer.churn_drivers[need.feature_id] = self.calibration.emergent_need_churn_driver_weight
                record.events.append(f"emergent_need_expired:{need.id}:{need.customer_id}")

        # --- Injection (mirrors bug injection; hidden ground truth) ---
        new_needs = product_logic.inject_emergent_needs(
            self.calibration,
            self.state.customers,
            self.state.features,
            self.state.emergent_needs,
            self.state.next_emergent_need_id,
            self.state.turn,
            self.rng,
        )
        for need in new_needs:
            self.state.emergent_needs.append(need)
            # Internal/analysis ONLY — emergent_need_injected must NEVER reach any agent
            # stream (that would bypass the health_check discovery gate). No reveal here.
            record.events.append(
                f"emergent_need_injected:{need.id}:{need.customer_id}:{need.feature_id}"
            )
        self.state.next_emergent_need_id += len(new_needs)

    def _resolve_financial(self, record: TurnRecord) -> None:
        # Revenue from active customers
        revenue = self.state.resources.mrr

        # Costs: team cost scales with capacity, plus base overhead, plus maintenance
        team_cost = self.state.resources.capacity_per_turn * self.calibration.team_cost_per_capacity
        base_overhead = self.state.resources.base_cost_per_turn
        maintenance = sum(
            f.maintenance_cost for f in self.state.features.values()
            if f.status in (FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished)
        )
        recurring_cost = team_cost + base_overhead + maintenance

        # Marketing budget spend (events/content cost shared runway; outbound is free). This is a
        # ONE-TIME cost, not recurring burn — it hits the budget once but must be excluded from the
        # runway projection below, else a lump-sum campaign manufactures a phantom one-turn runway
        # cliff (it cratered then recovered), scaring agents off the mechanic.
        marketing_spend = self._marketing_budget_spend
        if marketing_spend > 0:
            record.events.append(f"marketing_spend:{marketing_spend}")
        total_cost = recurring_cost + marketing_spend

        # Net income — the one-time spend is applied to cash here, once.
        net = revenue - total_cost
        self.state.resources.budget += net

        # Update runway — projects ONGOING burn only (recurring cost vs revenue). One-time spend
        # already lowered the budget numerator; keeping it out of the denominator avoids the cliff.
        if recurring_cost > revenue and recurring_cost > 0:
            burn_per_turn = recurring_cost - revenue
            self.state.resources.runway_turns = self.state.resources.budget / burn_per_turn if burn_per_turn > 0 else 999
        else:
            self.state.resources.runway_turns = 999  # profitable, infinite runway

        record.mrr = self.state.resources.mrr
        record.runway_turns = self.state.resources.runway_turns
        record.budget = self.state.resources.budget

        # Check bankruptcy
        if self.state.resources.budget < 0:
            self.state.game_over = True
            self.state.game_over_reason = "bankruptcy"
            record.events.append("game_over:bankruptcy")

    def _resolve_metrics(self, record: TurnRecord) -> None:
        # Track churn history
        self.state.churn_history.append(record.churn_count)

        # Count deals closed this turn and update sales momentum
        deals_closed = sum(1 for e in record.events if e.startswith("deal_won:"))
        self.state.total_customers_closed += deals_closed

        active_customer_count = sum(
            1 for c in self.state.customers.values() if c.stage == CustomerStage.customer
        )
        shipped_feature_count = sum(
            1 for f in self.state.features.values()
            if f.status in (FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid, FeatureStatus.shipped_polished)
        )
        # Get lagged marketing investment (if history long enough)
        lag = self.calibration.marketing_lag_turns
        lagged_marketing = (
            self.state.marketing_history[-lag] if len(self.state.marketing_history) >= lag else 0
        )

        self.state.sales_momentum = customer_logic.compute_sales_momentum_update(
            current_momentum=self.state.sales_momentum,
            deals_closed_this_turn=deals_closed,
            active_customer_count=active_customer_count,
            shipped_feature_count=shipped_feature_count,
            marketing_investment_lagged=lagged_marketing,
            calibration=self.calibration,
        )

        # Increment turns_in_current_stage for all customers
        for customer in self.state.customers.values():
            customer.turns_in_current_stage += 1

        # Record to history
        self.state.turn_history.append(record)

        # Check if game should end
        if self.state.turn >= self.state.max_turns:
            self.state.game_over = True
            self.state.game_over_reason = "max_turns_reached"
