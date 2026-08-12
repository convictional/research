"""Observation generation: converts full game state into role-specific views."""

import math

from alignsim.src.engine import customer_logic, ops_logic
from alignsim.src.models.entities import BugSeverity, CustomerStage, FeatureStatus, ProcessProjectStatus
from alignsim.src.models.game_state import GameState, TurnRecord
from alignsim.src.models.scenario import CalibrationParams
from alignsim.src.models.observations import (
    BugReport,
    CSObservation,
    CustomerHealthReport,
    CustomerPipelineStatus,
    DealEvent,
    FeatureProgressReport,
    GlobalDashboard,
    OpsObservation,
    ProductEngObservation,
    SalesObservation,
    TurnObservation,
)


class ObservationGenerator:
    def __init__(self, state: GameState, calibration: CalibrationParams | None = None, prev_debt_level: float | None = None):
        self.state = state
        self.calibration = calibration
        self._prev_debt_level = prev_debt_level

    def generate(self, turn_record: TurnRecord | None = None) -> TurnObservation:
        return TurnObservation(
            global_dashboard=self._generate_global_dashboard(),
            sales=self._generate_sales_observation(turn_record),
            product_eng=self._generate_product_eng_observation(turn_record),
            cs=self._generate_cs_observation(),
            ops=self._generate_ops_observation(),
            # God-view copy of the 1-turn analysis buffer; C3 routes it per-function downstream.
            analyses_received={fn: list(results) for fn, results in self.state.pending_analyses.items()},
        )

    def _generate_global_dashboard(self) -> GlobalDashboard:
        s = self.state
        active_customers = sum(
            1 for c in s.customers.values() if c.stage == CustomerStage.customer
        )

        churn_this_turn = [
            c.id for c in s.customers.values()
            if c.stage == CustomerStage.churned
            and c.turns_in_current_stage == 0
        ]
        churn_reasons = {}
        for cid in churn_this_turn:
            customer = s.customers[cid]
            # Determine primary churn reason
            unresolved_bugs = [b for b in s.bugs if cid in b.affected_customers and not b.is_resolved]
            if unresolved_bugs:
                churn_reasons[cid] = f"unresolved_bug_{unresolved_bugs[0].feature_id}"
            elif customer.competitive_pressure > 0.5:
                churn_reasons[cid] = "competitive_pressure"
            else:
                churn_reasons[cid] = "low_health"

        new_leads = [
            c.id for c in s.customers.values()
            if c.is_visible and c.stage == CustomerStage.lead and c.turns_in_current_stage == 0
        ]

        bug_backlog = {}
        for bug in s.bugs:
            if not bug.is_resolved:
                sev = bug.severity.value
                bug_backlog[sev] = bug_backlog.get(sev, 0) + 1

        last_capacity_used = 0
        if s.turn_history:
            last_capacity_used = s.turn_history[-1].capacity_used

        hire_cap = self.calibration.hire_capacity_cost if self.calibration else CalibrationParams().hire_capacity_cost
        func_to_pool = {
            "engineering": "eng_capacity", "sales": "sales_capacity",
            "cs": "support_capacity", "marketing": "marketing_capacity",
            "ops": "ops_capacity",
        }

        pending_hires = []
        sustain_committed: dict[str, int] = {}
        for h in s.pending_hires:
            needs_sustain = h.active_turns_completed < h.active_turns_required
            pending_hires.append({
                "id": h.id,
                "target_function": h.target_function,
                "hiring_function": h.hiring_function,
                "turns_remaining": h.turns_remaining,
                "capacity_on_arrival": h.capacity_bonus,
                "is_cross_function": h.is_cross_function,
                "phase": "active" if needs_sustain else "auto",
                "active_turns_completed": h.active_turns_completed,
                "active_turns_required": h.active_turns_required,
                "needs_sustain": needs_sustain,
            })
            if needs_sustain:
                pool = func_to_pool.get(h.hiring_function, "eng_capacity")
                sustain_committed[pool] = sustain_committed.get(pool, 0) + hire_cap

        return GlobalDashboard(
            turn=s.turn,
            mrr=s.resources.mrr,
            pipeline_value=self._compute_pipeline_value(),
            active_customers=active_customers,
            churn_this_turn=churn_this_turn,
            churn_reasons=churn_reasons,
            new_leads_this_turn=new_leads,
            debt_level=s.tech_debt.category,
            bug_backlog=bug_backlog,
            runway_turns=round(s.resources.runway_turns, 1),
            capacity_available=s.resources.capacity_per_turn - sum(sustain_committed.values()),
            eng_capacity=s.resources.eng_capacity - sustain_committed.get("eng_capacity", 0),
            sales_capacity=s.resources.sales_capacity - sustain_committed.get("sales_capacity", 0),
            support_capacity=s.resources.support_capacity - sustain_committed.get("support_capacity", 0),
            marketing_capacity=s.resources.marketing_capacity - sustain_committed.get("marketing_capacity", 0),
            ops_capacity=s.resources.ops_capacity - sustain_committed.get("ops_capacity", 0),
            sales_momentum=round(s.sales_momentum, 3),
            capacity_used_last_turn=last_capacity_used,
            pending_hires=pending_hires,
        )

    def _generate_sales_observation(self, turn_record: TurnRecord | None) -> SalesObservation:
        pipeline = []
        for c in self.state.customers.values():
            if not c.is_visible:
                continue
            if c.stage in (CustomerStage.churned, CustomerStage.lost, CustomerStage.customer):
                continue
            # Compute minimum sell capacity per action type
            min_caps = {}
            if self.calibration is not None:
                valid_actions_for_stage = {
                    "lead": ["outbound"],
                    "prospect": ["outbound", "demo"],
                    "qualified": ["demo"],
                    "in_deal": ["proposal", "negotiate"],
                }
                for sa in valid_actions_for_stage.get(c.stage.value, []):
                    min_caps[sa] = customer_logic.compute_sell_minimum_capacity(c, sa, self.calibration)

            # Pricing feedback from turn record
            pricing_fb = None
            if turn_record is not None and c.stage.value == "in_deal":
                for event_str in turn_record.events:
                    if event_str.startswith(f"pricing_feedback:{c.id}:"):
                        pricing_fb = event_str
                        break

            pipeline.append(CustomerPipelineStatus(
                customer_id=c.id,
                size=c.size,
                stage=c.stage.value,
                engagement=c.engagement.value,
                interest=self._infer_interest(c),
                known_needs=c.known_needs,
                deal_value=c.deal_value,
                timeline_remaining=c.timeline if c.timeline_active and c.timeline > 0 else None,
                timeline_resets=c.timeline_resets if c.timeline_resets > 0 else None,
                competitor_bidding=self._get_competitor_bidding(c),
                min_sell_capacity=min_caps,
                last_proposed_price=c.last_proposed_price,
                pricing_feedback=pricing_fb,
            ))

        deals = self._extract_deal_events(turn_record)

        # Competitor pricing events (non-steal pressure events)
        competitor_pricing_events: list[str] = []
        if turn_record is not None:
            for event_str in turn_record.events:
                if event_str.startswith("competitor_pricing:"):
                    competitor_pricing_events.append(event_str)

        # Pipeline summary
        prospect_count = sum(1 for p in pipeline if p.stage in ("lead", "prospect", "qualified"))
        in_deal_count = sum(1 for p in pipeline if p.stage == "in_deal")
        est_value = sum(
            self.state.customers[p.customer_id].deal_value * 12
            for p in pipeline if p.stage == "in_deal"
        )

        return SalesObservation(
            pipeline=pipeline,
            deals_this_turn=deals,
            competitor_pricing_events=competitor_pricing_events,
            pipeline_summary=f"{prospect_count} prospects, {in_deal_count} in-deal, est_close_value={est_value:,}",
        )

    def _generate_product_eng_observation(self, turn_record: TurnRecord | None) -> ProductEngObservation:
        features = []
        for f in self.state.features.values():
            blocked_by = None
            for dep_id in f.depends_on:
                dep = self.state.features.get(dep_id)
                if dep and dep.status in (FeatureStatus.not_started, FeatureStatus.in_progress):
                    blocked_by = dep_id
                    break

            est_completion = None
            if f.status == FeatureStatus.in_progress and f.current_target:
                remaining_pct = 100.0 - f.progress
                cost = f.cost.get(f.current_target.value, 0)
                if cost > 0:
                    remaining_cost = cost * (remaining_pct / 100.0)
                    if self.calibration:
                        effective_per_turn = min(
                            self.calibration.build_optimal_capacity,
                            cost * (self.calibration.build_max_progress_pct / 100.0),
                        )
                        est_completion = max(1, int(math.ceil(remaining_cost / effective_per_turn)))
                        min_turns = max(2, math.ceil(cost * self.calibration.build_min_turns_factor))
                        remaining_min = max(0, min_turns - f.turns_worked)
                        est_completion = max(est_completion, remaining_min)
                    else:
                        est_completion = max(1, int(remaining_cost / 8))

            # Derive capacity invested/needed from progress and current target cost
            target_key = f.current_target.value if f.current_target else None
            capacity_needed = f.cost.get(target_key, 0) if target_key else 0
            capacity_invested = round(f.progress / 100.0 * capacity_needed) if capacity_needed > 0 else 0

            features.append(FeatureProgressReport(
                feature_id=f.id,
                name=f.name,
                status=f.status.value,
                progress=round(f.progress, 1),
                capacity_invested=capacity_invested,
                capacity_needed=capacity_needed,
                est_completion_turns=est_completion,
                blocked_by=blocked_by,
            ))

        bugs_this_turn = self._extract_bug_events(turn_record)

        # Compute debt delta from previous turn's debt level
        debt_delta = 0.0
        if self._prev_debt_level is not None:
            debt_delta = self.state.tech_debt.level - self._prev_debt_level

        # Feature requests from pipeline
        feature_requests: dict[str, int] = {}
        for c in self.state.customers.values():
            if not c.is_visible or c.stage == CustomerStage.customer:
                continue
            for need in c.known_needs:
                feature_requests[need] = feature_requests.get(need, 0) + 1

        return ProductEngObservation(
            features=features,
            bugs_this_turn=bugs_this_turn,
            debt_delta=round(debt_delta, 2),
            feature_requests_from_pipeline=feature_requests,
        )

    def _generate_cs_observation(self) -> CSObservation:
        health_reports = []
        at_risk = []
        onboarding = []

        for c in self.state.customers.values():
            if c.stage != CustomerStage.customer:
                continue

            trend = "stable"
            if len(c.health_history) >= 2:
                diff = c.health - c.health_history[-2] if len(c.health_history) >= 2 else 0
                if diff > 0.5:
                    trend = "improving"
                elif diff < -0.5:
                    trend = "declining"

            # Emergent needs for this customer. Only REVEALED (via health_check), unmet,
            # unexpired needs are ever surfaced — never read unrevealed needs or feature_needs.
            unmet_needs = [
                n for n in self.state.emergent_needs
                if n.customer_id == c.id and not n.is_met and not n.is_expired
            ]
            revealed_need_features = sorted(n.feature_id for n in unmet_needs if n.is_revealed)
            has_unrevealed_bleed = any(n.turns_unmet > 0 and not n.is_revealed for n in unmet_needs)

            cause = None
            if c.health < 5:
                unresolved = [b for b in self.state.bugs if c.id in b.affected_customers and not b.is_resolved]
                if unresolved:
                    # Coarse cause shared elsewhere (bug events) — stays auto-visible.
                    cause = f"bug_in_{unresolved[0].feature_id}"
                elif c.competitive_pressure > 0.3:
                    # Coarse cause shared elsewhere (competitive events) — stays auto-visible.
                    cause = "competitive_pressure"
                elif revealed_need_features:
                    # A health_check has diagnosed the specific unmet need(s) (see emergent_needs).
                    cause = "unmet_feature_need"
                elif has_unrevealed_bleed:
                    # Hidden emergent need is bleeding health — run a health_check to diagnose it.
                    cause = "undiagnosed_decline"
                else:
                    cause = "general_decline"

            expansion_signal = c.turns_above_expansion_threshold >= 3

            report = CustomerHealthReport(
                customer_id=c.id,
                health=round(c.health, 1),
                health_trend=trend,
                cause=cause,
                onboarding_remaining=c.onboarding_turns_remaining if c.onboarding_turns_remaining > 0 else None,
                expansion_signal=expansion_signal,
                emergent_needs=revealed_need_features,
                churn_drivers=(dict(c.churn_drivers) if c.churn_drivers_revealed else None),
            )
            health_reports.append(report)

            if c.health < 5:
                at_risk.append(c.id)
            if c.onboarding_turns_remaining > 0:
                onboarding.append(c.id)

        churned = [
            c.id for c in self.state.customers.values()
            if c.stage == CustomerStage.churned and c.turns_in_current_stage == 0
        ]

        avg_health = 0.0
        active = [c for c in self.state.customers.values() if c.stage == CustomerStage.customer]
        if active:
            avg_health = sum(c.health for c in active) / len(active)

        return CSObservation(
            customer_health=health_reports,
            churned_this_turn=churned,
            at_risk=at_risk,
            avg_customer_health=round(avg_health, 1),
            onboarding_in_progress=onboarding,
        )

    def _generate_ops_observation(self) -> OpsObservation:
        s = self.state
        available = []
        active = []
        completed = []

        for project in s.process_projects.values():
            info = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "size": project.size.value,
                "ops_capacity_cost": project.ops_capacity_cost,
                "target_function": project.target_function,
                "duration_turns": project.duration_turns,
                "prerequisites": project.prerequisites,
            }
            if project.status == ProcessProjectStatus.available:
                # Keep locked projects VISIBLE so the agent can plan the tree; surface which
                # incomplete prerequisites block them.
                incomplete = [
                    pid for pid in project.prerequisites
                    if (p := s.process_projects.get(pid)) is None
                    or p.status != ProcessProjectStatus.completed
                ]
                info["locked"] = bool(incomplete)
                info["locked_by"] = incomplete
                available.append(info)
            elif project.status == ProcessProjectStatus.in_progress:
                info["progress_turns"] = project.progress_turns
                info["target_team_capacity_invested"] = project.target_team_capacity_invested
                active.append(info)
            elif project.status == ProcessProjectStatus.completed:
                info["completed_turn"] = project.completed_turn
                info["target_team_capacity_invested"] = project.target_team_capacity_invested
                # Indicate whether a re-run would be maintenance or net-new
                has_active_bonus = any(b.project_id == project.id for b in s.active_process_bonuses)
                if not has_active_bonus:
                    info["re_run_available"] = True
                    info["re_run_mode"] = "net_new"
                    info["re_run_ops_cost"] = project.ops_capacity_cost
                completed.append(info)

        active_bonuses = []
        for b in s.active_process_bonuses:
            degradation_pct = ops_logic.compute_degradation_pct(b)
            effective_bonus = ops_logic.compute_effective_bonus(b)
            effectiveness = self._describe_bonus_effectiveness(effective_bonus, b.bonus_type)
            permanent_floor = round(b.bonus_value * b.permanent_floor_fraction, 3)
            active_bonuses.append({
                "project_id": b.project_id,
                "target_function": b.target_function,
                "effectiveness": effectiveness,
                "turns_remaining": b.turns_remaining,
                "degradation_pct": round(degradation_pct * 100),   # spike decay only; floor persists below
                "effective_bonus": round(effective_bonus, 3),      # includes the permanent floor
                # A bonus floored at 0 still delivers permanent_floor — surface it so the spike's
                # "100% degraded" reading isn't mistaken for worthless.
                "permanent_floor": permanent_floor,
                "is_permanent": permanent_floor > 0,
                "maintenance_cost": ops_logic.compute_maintenance_cost(b),  # ops capacity to refresh the spike now
            })

        return OpsObservation(
            available_projects=available,
            active_projects=active,
            completed_projects=completed,
            active_bonuses=active_bonuses,
        )

    @staticmethod
    def _describe_bonus_effectiveness(bonus_value: float, bonus_type: str) -> str:
        """Convert a numeric bonus into a qualitative description."""
        _BONUS_LABELS = {
            "conversion_rate": "improving sales conversion",
            "bug_rate_reduction": "reducing bug frequency",
            "health_delta_bonus": "boosting customer health",
            "marketing_effectiveness": "improving marketing reach",
            "build_efficiency": "improving engineering throughput",
            "discovery_bonus": "improving customer discovery",
        }
        label = _BONUS_LABELS.get(bonus_type, f"improving {bonus_type}")

        if bonus_value >= 0.15:
            return f"highly effective — {label}"
        elif bonus_value >= 0.08:
            return f"moderately effective — {label}"
        else:
            return f"somewhat effective — {label}"

    # --- Helpers ---

    def _compute_pipeline_value(self) -> int:
        total = 0
        for c in self.state.customers.values():
            if c.stage in (CustomerStage.lead, CustomerStage.prospect, CustomerStage.qualified, CustomerStage.in_deal):
                total += c.deal_value * 12  # annualized
        return total

    def _infer_interest(self, customer) -> str:
        if customer.engagement.value == "hot":
            return "high"
        elif customer.engagement.value == "warm":
            return "medium"
        return "low"

    def _get_competitor_bidding(self, customer) -> str | None:
        if customer.competitive_pressure > 0.3:
            return "competitor_active"
        return None

    def _extract_deal_events(self, turn_record: TurnRecord | None) -> list[DealEvent]:
        if turn_record is None:
            return []
        events = []
        for event_str in turn_record.events:
            if event_str.startswith("deal_won:"):
                parts = event_str.split(":")
                if len(parts) >= 2:
                    cid = parts[1]
                    customer = self.state.customers.get(cid)
                    events.append(DealEvent(
                        customer_id=cid,
                        event_type="win",
                        deal_value=customer.deal_value if customer else 0,
                    ))
            elif event_str.startswith("deal_lost:"):
                parts = event_str.split(":")
                cid = parts[1] if len(parts) >= 2 else ""
                reason = parts[2] if len(parts) >= 3 else ""
                lost_to = parts[3] if len(parts) >= 4 else None
                events.append(DealEvent(
                    customer_id=cid,
                    event_type="loss",
                    reason=reason,
                    lost_to=lost_to,
                ))
        return events

    def _extract_bug_events(self, turn_record: TurnRecord | None) -> list[BugReport]:
        if turn_record is None:
            return []
        reports = []
        for event_str in turn_record.events:
            if event_str.startswith("bug_injected:"):
                parts = event_str.split(":")
                if len(parts) >= 4:
                    reports.append(BugReport(
                        bug_id=parts[1],
                        severity=parts[2],
                        feature_id=parts[3],
                        event_type="injected",
                        affected_customers=parts[4].split(",") if len(parts) >= 5 and parts[4] else [],
                    ))
            elif event_str.startswith("bug_fixed:"):
                parts = event_str.split(":")
                if len(parts) >= 4:
                    reports.append(BugReport(
                        bug_id=parts[1],
                        severity=parts[2],
                        feature_id=parts[3],
                        event_type="fixed",
                    ))
        return reports
