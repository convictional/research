"""Condition 1 harness: Single LLM plays all roles.

One model receives all role observations and submits all actions.
Uses instructor for structured output extraction.
"""

import json
import logging
import time
from collections import defaultdict

from alignsim.src.harness.llm_client import init_client, instruct
from alignsim.src.models.actions import TurnActions
from alignsim.src.models.game_state import GameState
from alignsim.src.models.goals import GoalAttainmentScore
from alignsim.src.models.observations import TurnObservation
from alignsim.src.settings import settings

logger = logging.getLogger(__name__)


class SingleLLMHarness:
    """Condition 1: A single LLM receives all observations and decides all actions."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.llm_model
        self.scenario_info: dict = {}
        self.turn_history: list[dict] = []
        self.max_history = settings.context_window_turns

        # Trace attributes (populated after each decide() call)
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.last_latency_ms: int | None = None
        self.last_error: str | None = None

        # Rejection feedback from previous turn
        self.last_rejections: list[str] = []

        # Token usage accumulator: {model: {input_tokens, output_tokens, ...}}
        self.token_usage: dict[str, dict[str, int]] = defaultdict(lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        })

        # Initialize instructor client
        init_client(settings.anthropic_api_key)

    async def on_game_start(self, scenario_info: dict) -> None:
        self.scenario_info = scenario_info

    async def decide(self, observation: TurnObservation, state_summary: dict) -> TurnActions:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(observation, state_summary)

        # Store for trace logging
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_latency_ms = None
        self.last_error = None

        try:
            start = time.monotonic()
            result = await instruct(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=TurnActions,
                model=self.model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            self.last_latency_ms = int((time.monotonic() - start) * 1000)
            response = result.response

            usage = self.token_usage[self.model]
            usage["input_tokens"] += result.input_tokens
            usage["output_tokens"] += result.output_tokens
            usage["cache_creation_input_tokens"] += result.cache_creation_input_tokens
            usage["cache_read_input_tokens"] += result.cache_read_input_tokens
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"LLM call failed on turn {observation.global_dashboard.turn}: {e}")
            return TurnActions(turn=observation.global_dashboard.turn, actions=[])

        # Record in history
        self.turn_history.append({
            "turn": observation.global_dashboard.turn,
            "observation_summary": self._summarize_observation(observation),
            "actions": [a.model_dump() for a in response.actions],
        })

        # Trim history to sliding window
        if len(self.turn_history) > self.max_history:
            self.turn_history = self.turn_history[-self.max_history:]

        return response

    def on_turn_result(self, rejections: list[str]) -> None:
        """Receive feedback from the last turn's validation results."""
        self.last_rejections = rejections

    async def on_game_end(self, score: GoalAttainmentScore, state: GameState) -> None:
        pass

    def _build_system_prompt(self) -> str:
        info = self.scenario_info
        features_str = "\n".join(
            f"  - {f['id']} ({f['name']}): cost={{mvp: {f['cost'].get('mvp', '?')}, "
            f"solid: {f['cost'].get('solid', '?')}, polished: {f['cost'].get('polished', '?')}}}, "
            f"depends_on={f['depends_on']}, status={f['status']}"
            for f in info.get("features", [])
        )

        process_projects_str = ""
        for p in info.get("process_projects", []):
            prereqs = p.get("prerequisites") or []
            prereq_str = f", requires {prereqs}" if prereqs else ""
            process_projects_str += (
                f"  - {p['id']} ({p['name']}): size={p['size']}, "
                f"ops_cost={p['ops_capacity_cost']}, targets {p['target_function']}, "
                f"duration={p['duration_turns']} turns{prereq_str}\n"
            )

        goal = info.get("primary_goal", {})

        sell_costs = info.get('sell_base_costs', {})

        return f"""You are playing AlignSim, a turn-based strategy game that benchmarks goal alignment.
You control all roles (CEO, Sales, Product, Engineering, CS) and must allocate capacity across competing priorities.

## Game Rules

Each turn you receive observations from all roles and must submit a list of structured actions.
Capacity pools (current turn):
  - Engineering ({info.get('eng_capacity', 5)}): build, fix_bugs, infrastructure
  - Sales ({info.get('sales_capacity', 5)}): sell, discover
  - Support ({info.get('support_capacity', 0)}): support (starts at 0 — must hire or cross-hire)
  - Marketing ({info.get('marketing_capacity', 3)}): market
  - Ops ({info.get('ops_capacity', 0)}): ops_project (starts at 0 — must hire or cross-hire)
Each action draws capacity from its function's pool. Capacity is per-turn — unused capacity does not carry over.

## Goal

PRIMARY: Reach MRR of {goal.get('mrr_target', 210000):,} by turn {goal.get('target_turn', 48)}.
CONSTRAINTS: Keep churn rate below {goal.get('max_churn_rate', 0.02)*100:.0f}% per turn. Maintain runway above {goal.get('min_runway_turns', 10)} turns.
All three conditions must hold for maximum score.

Scoring: All scores are uncapped (1.0 = hit target). Composite = sum of scores (higher is better). Pareto = min of scores (your weakest goal is your bottleneck). Both primary goals (MRR, churn, runway) and function sub-goals are scored on both dimensions.

Function sub-goals (each scored on both dimensions, alongside the primary goals above):
  - Engineering: ship features at solid quality or better (features_shipped_solid_plus, target 12)
  - Sales: maintain a steady deal-closure rate (pipeline_velocity, target 0.2)
  - Marketing: generate inbound leads over the game (marketing_leads_generated, target 24)
  - CS: keep average customer health above 7.0 (avg_customer_health, target 7.0)
  - Ops: complete process improvement projects (process_projects_completed, target 6)

## Starting State

MRR: {info.get('starting_mrr', 0):,} | Budget: {info.get('starting_budget', 0):,} | Capacity: {info.get('capacity_per_turn', 40)}/turn
Base cost: {info.get('base_cost_per_turn', 0):,}/turn | Competitors: {', '.join(info.get('competitors', []))}

## Features

{features_str}

## Process Projects (Ops)

{process_projects_str if process_projects_str else "No process projects available."}

## Action Types

Each action has an action_type field and specific parameters:

1. build: Build a feature. Params: feature_id, quality ("mvp"/"solid"/"polished"), capacity (int > 0)
2. fix_bugs: Fix bugs. Params: bug_id (optional, null=auto-target highest severity), capacity (int > 0)
3. infrastructure: Reduce tech debt. Params: capacity (int > 0)
4. sell: Advance a customer. Params: customer_id, sell_action ("outbound"/"demo"/"proposal"/"negotiate"), capacity (int > 0), proposed_deal_value (optional int, only for proposal/negotiate)
5. discover: Find new customers by targeting shipped features. Params: target_features (list of feature IDs; empty list = broad discovery across all shipped), capacity (int > 0). At least one target feature must be shipped.
5b. market_support (SALES pool): Co-invest Sales capacity in Marketing's SAME-TURN budget campaign to buy pipeline PROGRESSION. Params: channel ("content"/"events"), capacity (int > 0), target_customer_id (optional, events-only). Draws from the Sales pool.
   - content: newly-arriving inbound leads roll to land one stage advanced (lower prob).
   - events: higher prob AND an optional target_customer_id pushes one EXISTING pipeline customer one stage.
   - Progression advances at most ONE stage and is CAPPED at in_deal — closing still needs a real proposal/negotiate (full dealbreaker + rubric gate). It buys pipeline relief, not free MRR.
   - The matching market campaign MUST run the SAME turn on the SAME channel (coordinate Marketing+Sales). If no matching campaign ran, your capacity is WASTED (market_support_unmatched).
   - Roll prob scales with the joint commitment min(marketing_cap, sales_cap) AND budget committed (both diminishing).
   - Example: {{"action_type": "market_support", "channel": "events", "capacity": 3, "target_customer_id": "C08"}}
6. support: Support an active customer. Params: customer_id, support_action ("onboard"/"churn_intervention"/"health_check"), capacity (int > 0)
   - Every support verb also provides baseline health attention (diminishing returns — see strategy).
   - health_check: the ONLY way to learn a customer's EMERGENT NEEDS and hidden churn drivers. Health checks diagnose WHY a customer is declining; without them you are blind to emergent-need risk (decline shows only as "undiagnosed_decline").
   - onboard: accelerates the onboarding window and adds extra health during it. Negligible effect once onboarding is complete.
   - churn_intervention: a costly, stochastic save. Only fires when health is below a threshold and you commit enough capacity; it may fail.
7. market: Build per-feature AWARENESS (a decaying stock that takes time to build). Params: channel ("content"/"events"/"outbound_campaign"), target_features (list of feature IDs; empty = broad across all shipped + in-progress features), capacity (int > 0)
   - Awareness is keyed PER FEATURE and may be built for features that have NOT shipped yet.
   - High awareness on a feature makes leads who need it arrive WARM (rarely hot) with a LONGER decision timeline (more chances to close) — it changes lead QUALITY + targeting, NOT lead count.
   - CHANNELS differ in lag / burst-vs-spread / efficiency / cost:
     * events: fast (lag 2) single-turn burst, high efficiency, but EXPENSIVE (spends shared runway budget).
     * content: slow (lag 8), spread over many turns, moderate budget cost — durable long-game awareness.
     * outbound_campaign: medium (lag 5), concentrated, FREE (capacity-only, no budget) — the budget-free path.
   - Budget channels (events/content) are pay-to-ACCELERATE: trade shared runway for faster/more durable awareness.
   - JOINT PLAY: events/content ALSO unlock pipeline PROGRESSION when SALES co-invests (market_support) on the SAME channel the SAME turn — buying pipeline movement, not just warmth.
   - Example: {{"action_type": "market", "channel": "events", "target_features": ["F14"], "capacity": 3}}
8. hire: Start a NEW hiring process. Params: hiring_function, target_function (each one of "engineering"/"sales"/"cs"/"marketing"/"ops")
   - ACTIVE SUSTAIN: Hiring is a multi-turn commitment. Starting a hire creates a pending hire with an active phase.
   - You MUST submit sustain_hire each turn during the active phase (first half of duration) or the hire is CANCELLED (budget lost).
   - NATIVE (hiring_function == target_function): costs 3 cap + budget, 3 active + 3 auto = 6 turns total, adds 4 capacity
   - CROSS-FUNCTION (hiring_function != target_function): costs 3 cap + budget, 6 active + 6 auto = 12 turns total, adds 3 capacity
   - Multiple concurrent hires allowed — each gets a unique ID (H1, H2, etc.)
   - Native example:  {{"action_type": "hire", "hiring_function": "engineering", "target_function": "engineering"}}
   - Cross example:   {{"action_type": "hire", "hiring_function": "sales", "target_function": "ops"}}
9. sustain_hire: Continue an active hiring process. Params: hire_id (from pending_hires in observation)
   - MUST submit each turn during the active phase or hire is CANCELLED. Budget lost, must restart from scratch.
   - Costs 3 capacity from the original hiring_function pool. No additional budget cost.
   - Sustain capacity is PRE-COMMITTED — deducted BEFORE other actions. If eng has 5 cap and sustaining costs 3, only 2 remain for builds.
   - After the active phase completes, the hire auto-progresses to arrival (no sustain needed).
   - Example: {{"action_type": "sustain_hire", "hire_id": "H1"}}
10. fire: Release a headcount. Params: function ("engineering"/"sales"/"cs"/"marketing"/"ops")
   - Pays severance from budget (no capacity cost this turn). Capacity drops by 4 starting NEXT turn.
   - The fired function's actions still resolve normally this turn — they work their last turn before leaving.
   - Use to recover from over-hiring when burn rate is unsustainable.
   - Example: {{"action_type": "fire", "function": "engineering"}}
11. ops_project: Work on a process improvement project (uses Ops capacity). Params: project_id, capacity (must >= project ops_capacity_cost)
    - FIRST RUN or NET-NEW RE-RUN (bonus lapsed): full capacity cost per turn for duration_turns turns. Target team can co-invest via ops_project_support for a higher bonus.
    - MAINTENANCE REFRESH (bonus still active, just degraded): single-turn action resets bonus to full. Required capacity = maintenance_cost shown in active_bonuses obs. No target team re-investment needed.
    - Example: {{"action_type": "ops_project", "project_id": "PP01", "capacity": 4}}
12. ops_project_support: Invest target team capacity in an in-progress ops project. Params: project_id, capacity (int > 0)
    - Draws from the TARGET function's pool. Only valid while project is in_progress.
    - More investment = higher bonus at completion (logarithmic: worth doing early, diminishing after that).
    - Example: {{"action_type": "ops_project_support", "project_id": "PP01", "capacity": 2}}
13. ops_analysis (OPS pool): Run a cross-functional analysis FOR a requesting team. Params: target_function ("engineering"/"sales"/"cs"/"marketing"), analysis_type ("conversion_funnel"/"retention_efficiency"/"awareness_attribution"/"capacity_bottleneck"), capacity (>= analysis cost, default 2).
    - Engine-computed from OBSERVABLE history only (no hidden ground truth). The result is delivered to the requesting team's NEXT-turn observation (analyses_received_this_turn) — Ops itself only sees the matched echo, not the payload.
    - Requires a matching analysis_scope (same target_function + analysis_type) the SAME turn, else WASTED (analysis_unmatched). Ops cannot analyse itself (not a valid target_function).
    - Example: {{"action_type": "ops_analysis", "target_function": "sales", "analysis_type": "conversion_funnel", "capacity": 2}}
14. analysis_scope (REQUESTING TEAM's pool): A team co-invests to scope an analysis it wants. Params: target_function (must be YOUR OWN function), analysis_type (same menu), capacity (default 1).
    - Draws from that team's pool. Must be matched same-turn by a matching ops_analysis, else WASTED (analysis_unmatched).
    - Example: {{"action_type": "analysis_scope", "target_function": "sales", "analysis_type": "conversion_funnel", "capacity": 1}}

Pipeline stages: lead -> prospect -> qualified -> in_deal -> customer
Valid sell actions per stage: lead=outbound, prospect=outbound/demo, qualified=demo, in_deal=proposal/negotiate

TIMELINE (Action-Triggered): Discovered customers have NO ticking clock until you engage them. The countdown starts on your FIRST sell action targeting a customer (regardless of stage). Once active, it decrements every turn. If it hits 0, the customer resets to lead with a permanent 30% satisfaction penalty per reset (floored at 30% of original).

## Game Mechanics & Considerations

- **Expansion**: active customers with health above 8.0 for 4+ consecutive turns trigger expansion — their deal value increases by 20% (compounding).
- **Sales momentum**: builds from closing deals (+0.08 each), shipping features (diminishing returns), and lagged marketing investment. Multiplies pipeline conversion probability up to +40%. Decays slightly each turn. Visible in the global dashboard.
- Closing deals requires customer **rubric satisfaction above ~75%**. The rubric has four components:
  * Feature coverage: Do you have the features they need, at sufficient quality?
  * Price: Larger customers are less price-sensitive.
  * Product maturity: The ratio of solid/polished features to total shipped features. ALL MVP = 0% maturity.
  * Support: Based on customer health.
- CRITICAL: Upgrading features from MVP to solid/polished improves maturity for ALL customers, not just those who need that feature.
- Customers have hidden rubric weights. known_needs gives partial visibility into feature needs.
- Dealbreaker features MUST be shipped before a deal can close.
- CHURN RISK: Customers whose health stays very low for multiple turns will churn.
- CUSTOMER HEALTH DECAY: Customer health degrades passively over time without CS attention.
- EMERGENT NEEDS (CS keystone): Active customers develop NEW feature needs over time. These are invisible until you run a health_check on that customer — until then, a customer bleeding health from an unmet need shows only as "undiagnosed_decline". After a short grace window an unmet need bleeds health every turn and, if left unmet long enough, converts into a permanent churn driver. SATISFYING A NEED REQUIRES ENGINEERING TO BUILD/SHIP THAT FEATURE. While Eng is actively building the needed feature, the bleed and the expiry clock both PAUSE.
- CS DIMINISHING RETURNS: Support health attention follows a diminishing-returns curve (no hard ceiling, but each extra unit on the same customer is worth less).
- Bugs reduce customer health. Unresolved critical bugs can cause churn.
- Tech debt generates bugs. Infrastructure work reduces debt.
- MARKETING (AWARENESS, keystone): Marketing builds a decaying PER-FEATURE awareness stock via channels. Awareness takes time to build (channel-dependent lag: events ~2 turns, outbound ~5, content ~8) and decays every turn; it may be built even before a feature ships. High awareness on a feature makes customers who need it (revealed by discovery OR inbound) arrive WARM and PATIENT (longer timeline = more chances to close) — it raises lead QUALITY, not lead count.
- MARKETING CHANNELS & BUDGET: events (fast burst, high efficiency, expensive) and content (slow, durable, moderate cost) spend shared RUNWAY BUDGET — they are pay-to-accelerate awareness. outbound_campaign is the FREE (capacity-only) but slower path.
- MARKETING↔SALES JOINT PLAY: running a budget channel (events/content) AND Sales co-investing market_support on the SAME channel the SAME turn buys one-stage pipeline PROGRESSION (capped at in_deal) on newly-arriving leads (both channels) and, for events, one named existing customer. Closing still needs a real proposal/negotiate. It needs BOTH budget and the Sales pool, and MUST be same-turn and same-channel — mis-timed/mismatched co-investment wastes the Sales capacity (market_support_unmatched).
- COMPETITIVE RADAR: Marketing passively senses fuzzy early warnings of upcoming competitor events touching features it is active in (feature area + "soon/upcoming", never exact timing).
- OPS↔TEAM ANALYSIS JOINT PLAY (cross-functional foresight): Ops can convert the observable past into shared foresight that NO single function can compute alone — e.g. awareness_attribution (marketing spend → leads → closes), conversion_funnel (per-stage sales rates + median turns-in-stage), retention_efficiency (churn/intervention/expansion economics), capacity_bottleneck (utilization + rejection across ALL five pools). It needs BOTH an Ops ops_analysis AND the requesting team's analysis_scope on the SAME (target_function, analysis_type) the SAME turn — mis-timed/mismatched co-investment is WASTED (analysis_unmatched). The result lands in the requesting team's next-turn observation (analyses_received_this_turn). The output is descriptive/predictive, never a prescribed "best move".
- HIRING (ACTIVE SUSTAIN): Hiring is a multi-turn commitment. Starting a hire costs budget + 3 capacity. You must then submit sustain_hire each turn during the active phase (first half of duration) — costs 3 cap/turn from the hiring pool. Miss a turn = hire cancelled, budget lost. Native: 3 active + 3 auto = 6 turns (+4 cap). Cross: 6 active + 6 auto = 12 turns (+3 cap). Sustain capacity is pre-committed before other actions. Support and Ops start at 0 — cross-hire to unlock them.
- FIRING: fire pays severance (budget) and lowers ongoing team cost per turn.
- SELL CAPACITY COSTS: Sell actions have minimum capacity costs that scale with customer size (1-5).
  * Outbound: {sell_costs.get('outbound', 1)} * customer_size
  * Demo: {sell_costs.get('demo', 2)} * customer_size (extra capacity above minimum boosts conversion)
  * Proposal: {sell_costs.get('proposal', 2)} * customer_size
  * Negotiate: {sell_costs.get('negotiate', 2)} * customer_size
  Example: a size-5 enterprise negotiate costs minimum {sell_costs.get('negotiate', 1) * 5} capacity vs {sell_costs.get('negotiate', 1)} for a size-1 startup.
- ENGAGEMENT: Customer engagement (hot/warm/cold) significantly affects conversion probability. Hot customers are much easier to close; cold customers are very hard. Engagement requires sustained sell attention — skip a turn and it decays. Larger customers need proportionally more capacity to stay engaged.
- FEATURE COST TIERS: Tier-2 features (segment bridges, F02-F05) are cheap table stakes (10-12 MVP). Tier-3 (segment-specific, F06-F13) is a real commitment (20-25 MVP). Tier-4 (enterprise premium, F14-F16) is a major strategic bet (32-40 MVP).
- DIMINISHING RETURNS ON ENGINEERING: Over-allocating capacity to a single feature has diminishing returns.
  * Optimal allocation per feature: ~12 capacity/turn. Beyond that, efficiency drops.
  * Each feature has a minimum number of turns to complete (larger features take more turns regardless of capacity).
  * Max progress per turn is capped at ~65% of total cost. You CANNOT one-shot features.
- DISCOVERY TARGETING: target_features biases discovery toward customers who need those (shipped) features; broad discovery (empty list) searches across all shipped features. The pipeline never dries up — new customers are generated dynamically as you ship more features.
- OPS PROCESS PROJECTS: a completed project gives its target function a bonus that SPIKES then decays over time; some projects also leave a PERMANENT FLOOR — a portion that does not decay away. Projects form a TECH-TREE: a higher-tier project is locked until its prerequisites are completed (the obs shows locked / locked_by / prerequisites), and higher tiers carry larger maximum bonuses. A degraded-but-active bonus can be refreshed by a single ops_project at maintenance_cost (shown in obs); a fully-lapsed one costs full ops capacity to re-run. Bonus size is PROBABILISTIC — more target-team co-investment (ops_project_support) narrows the variance. Multiple projects can run simultaneously given enough ops capacity.
- PRICING NEGOTIATION: At in_deal stage, submit proposed_deal_value with proposal/negotiate actions.
  * After a rejected proposal, the customer may indicate a price range — this is directional, not exact (customers negotiate in their favor).
  * Negotiate lets you adjust your offer; not adjusting keeps the same conversion impact.
  * Pricing too high reduces close probability; discounting helps but doesn't guarantee a close.
  * Watch for competitor pricing events — competitors can steal deals if you don't respond.
  * Final MRR from a deal equals the price you close at, not the sticker price.

## Response Format

Return a TurnActions object with the current turn number and a list of actions.
Ensure actions don't exceed their function's capacity pool. Invalid actions will be dropped and their capacity lost."""

    def _build_user_prompt(self, observation: TurnObservation, state_summary: dict) -> str:
        d = observation.global_dashboard
        parts = []

        # Current state
        parts.append(f"## Turn {d.turn} Observations\n")
        parts.append(f"### Global Dashboard")
        parts.append(f"MRR: {d.mrr:,} | Pipeline Value: {d.pipeline_value:,} | Active Customers: {d.active_customers}")
        parts.append(
            f"Runway: {d.runway_turns:.1f} turns | "
            f"Capacity: Eng={d.eng_capacity}, Sales={d.sales_capacity}, "
            f"Support={d.support_capacity}, Marketing={d.marketing_capacity}, "
            f"Ops={d.ops_capacity} "
            f"(total={d.capacity_available})"
        )
        if d.sales_momentum > 0:
            parts.append(f"Sales Momentum: {d.sales_momentum:.3f} (boosts conversion rates)")
        parts.append(f"Tech Debt: {d.debt_level} | Bug Backlog: {d.bug_backlog or 'none'}")
        if d.pending_hires:
            for h in d.pending_hires:
                cross_tag = f" [cross from {h['hiring_function']}]" if h['is_cross_function'] else ""
                if h.get('needs_sustain'):
                    phase_tag = f" **NEEDS SUSTAIN** (submit sustain_hire hire_id={h['id']} or CANCELLED!)"
                else:
                    phase_tag = " (auto-phase, no action needed)"
                parts.append(
                    f"  Pending hire [{h['id']}]: {h['target_function']}{cross_tag} — "
                    f"arrives in {h['turns_remaining']} turns, +{h['capacity_on_arrival']} capacity, "
                    f"active {h.get('active_turns_completed', '?')}/{h.get('active_turns_required', '?')}{phase_tag}"
                )
        if d.churn_this_turn:
            parts.append(f"CHURN THIS TURN: {d.churn_this_turn} (reasons: {d.churn_reasons})")
        if d.new_leads_this_turn:
            parts.append(f"New Leads: {d.new_leads_this_turn}")

        # Rejection feedback from last turn
        if self.last_rejections:
            parts.append(f"\n### REJECTED ACTIONS FROM LAST TURN (these were dropped — capacity was wasted!)")
            for rej in self.last_rejections:
                parts.append(f"  - {rej}")
            parts.append("Adjust your actions this turn to avoid these errors.")

        # Sales observations
        parts.append(f"\n### Sales Observations")
        parts.append(f"Pipeline Summary: {observation.sales.pipeline_summary}")
        for p in observation.sales.pipeline:
            needs_str = ", ".join(p.known_needs) if p.known_needs else "unknown"
            sell_cap_str = ""
            if p.min_sell_capacity:
                caps = ", ".join(f"{k}={v}" for k, v in p.min_sell_capacity.items())
                sell_cap_str = f", min_sell_cap: [{caps}]"
            pricing_str = ""
            if p.last_proposed_price is not None:
                pricing_str += f", last_proposed_price={p.last_proposed_price:,}"
            if p.pricing_feedback:
                pricing_str += f", PRICING FEEDBACK: {p.pricing_feedback}"
            parts.append(
                f"  {p.customer_id} (size {p.size}): stage={p.stage}, interest={p.interest}, "
                f"deal_value={p.deal_value:,}/mo, needs=[{needs_str}]"
                + (f", timeline={p.timeline_remaining} turns" if p.timeline_remaining else "")
                + (f", resets={p.timeline_resets}" if p.timeline_resets else "")
                + (f", {p.competitor_bidding}" if p.competitor_bidding else "")
                + sell_cap_str
                + pricing_str
            )
        for deal in observation.sales.deals_this_turn:
            if deal.event_type == "win":
                parts.append(f"  WIN: {deal.customer_id} (value: {deal.deal_value:,}/turn)")
            else:
                parts.append(f"  LOSS: {deal.customer_id} (reason: {deal.reason}, lost to: {deal.lost_to})")
        # Competitor pricing events
        if observation.sales.competitor_pricing_events:
            for cp_event in observation.sales.competitor_pricing_events:
                parts.append(f"  COMPETITOR PRICING: {cp_event}")

        # Product/Engineering observations
        parts.append(f"\n### Product/Engineering Observations")
        for f in observation.product_eng.features:
            line = f"  {f.feature_id} ({f.name}): {f.status}, progress={f.progress}%"
            if f.capacity_needed > 0:
                line += f" ({f.capacity_invested}/{f.capacity_needed} cap invested)"
            if f.blocked_by:
                line += f" [BLOCKED by {f.blocked_by}]"
            if f.est_completion_turns:
                line += f" (est. {f.est_completion_turns} turns)"
            parts.append(line)
        for bug in observation.product_eng.bugs_this_turn:
            parts.append(f"  BUG {bug.event_type}: {bug.bug_id} ({bug.severity}) in {bug.feature_id}")
        if observation.product_eng.feature_requests_from_pipeline:
            parts.append(f"  Feature requests from pipeline: {observation.product_eng.feature_requests_from_pipeline}")

        # CS observations
        parts.append(f"\n### CS Observations")
        parts.append(f"Average Customer Health: {observation.cs.avg_customer_health}")
        for h in observation.cs.customer_health:
            line = f"  {h.customer_id}: health={h.health} ({h.health_trend})"
            if h.cause:
                if h.cause == "undiagnosed_decline":
                    line += " [cause: undiagnosed_decline — run a health_check to diagnose]"
                else:
                    line += f" [cause: {h.cause}]"
            if h.emergent_needs:
                line += f" [UNMET NEEDS (need Eng build): {', '.join(h.emergent_needs)}]"
            if h.churn_drivers:
                drivers = ", ".join(f"{k}={v}" for k, v in h.churn_drivers.items())
                line += f" [churn_drivers: {drivers}]"
            if h.expansion_signal:
                line += " [EXPANSION SIGNAL]"
            if h.onboarding_remaining:
                line += f" [onboarding: {h.onboarding_remaining} turns left]"
            parts.append(line)
        if observation.cs.at_risk:
            parts.append(f"  AT RISK: {observation.cs.at_risk}")

        # Ops observations
        ops = observation.ops
        if ops.available_projects or ops.active_projects or ops.active_bonuses:
            parts.append(f"\n### Ops Observations")
            if ops.available_projects:
                parts.append("  Available Projects:")
                for p in ops.available_projects:
                    locked_suffix = (
                        f" (LOCKED: needs {p['locked_by']})" if p.get("locked") else ""
                    )
                    parts.append(
                        f"    {p['id']} ({p['name']}): size={p['size']}, "
                        f"ops_cost={p['ops_capacity_cost']}, targets {p['target_function']}, "
                        f"{p['duration_turns']} turn(s){locked_suffix}"
                    )
            if ops.active_projects:
                parts.append("  In Progress:")
                for p in ops.active_projects:
                    parts.append(
                        f"    {p['id']} ({p['name']}): {p['progress_turns']}/{p['duration_turns']} turns, "
                        f"team investment={p['target_team_capacity_invested']}"
                    )
            if ops.active_bonuses:
                parts.append("  Active Improvements:")
                for b in ops.active_bonuses:
                    parts.append(
                        f"    {b['project_id']} → {b['target_function']}: {b['effectiveness']} "
                        f"({b['turns_remaining']} turns remaining, {b['degradation_pct']}% degraded, "
                        f"refresh cost={b['maintenance_cost']} ops cap)"
                    )
            if ops.completed_projects:
                lapsed = [p for p in ops.completed_projects if p.get("re_run_available")]
                if lapsed:
                    parts.append("  Lapsed Projects (bonus expired — re-run at full cost):")
                    for p in lapsed:
                        parts.append(
                            f"    {p['id']} ({p['name']}): re-run costs {p['re_run_ops_cost']} ops cap/turn"
                        )

        # Cross-functional analysis results delivered this turn (god-view across all teams in C1).
        all_analyses = [a for results in observation.analyses_received.values() for a in results]
        if all_analyses:
            parts.append("\n### Cross-Functional Analyses Delivered (from Ops)")
            for a in all_analyses:
                parts.append(
                    f"  [{a.get('analysis_type', '?')} for {a.get('target_function', '?')}]: {a}"
                )

        # Recent history
        if self.turn_history:
            parts.append(f"\n### Recent Turn History (last {len(self.turn_history)} turns)")
            for hist in self.turn_history[-3:]:  # Show last 3 turns in detail
                parts.append(f"  Turn {hist['turn']}: {hist['observation_summary']}")

        # Goal progress
        parts.append(f"\n### Goal Progress")
        goal = self.scenario_info.get("primary_goal", {})
        mrr_pct = (d.mrr / goal.get("mrr_target", 210000)) * 100
        turns_remaining = goal.get("target_turn", 48) - d.turn
        mrr_needed = goal.get("mrr_target", 210000) - d.mrr
        parts.append(f"MRR: {d.mrr:,} / {goal.get('mrr_target', 210000):,} ({mrr_pct:.1f}%)")
        parts.append(f"Turns remaining: {turns_remaining} | MRR still needed: {mrr_needed:,}")

        parts.append(f"\n## Your Turn")
        parts.append(
            f"Submit actions for turn {d.turn}. "
            f"Capacity: Engineering={d.eng_capacity}, Sales={d.sales_capacity}, "
            f"Support={d.support_capacity}, Marketing={d.marketing_capacity}"
        )
        parts.append("Think about: What features to build? Which customers to pursue? How to allocate capacity across pools?")

        return "\n".join(parts)

    def _summarize_observation(self, obs: TurnObservation) -> str:
        d = obs.global_dashboard
        return (
            f"MRR={d.mrr:,}, active={d.active_customers}, "
            f"debt={d.debt_level}, bugs={d.bug_backlog or 'none'}, "
            f"churn={len(d.churn_this_turn)}"
        )
