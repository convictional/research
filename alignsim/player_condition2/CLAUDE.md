# AlignSim — Game Rules

You are playing AlignSim, a turn-based strategy game that benchmarks goal alignment.
You control all roles (CEO, Sales, Product, Engineering, CS) and must allocate capacity across competing priorities.

## How to Play

Each turn:
1. Run `./game observe` to see the current state
2. Query customers, features, bugs — reason about what to do
3. Write your actions to `actions.json` (see @action_format.md)
4. Run `./game submit --actions-file ./actions.json`
5. Update `notes/strategy.md` with what you learned
6. Repeat until game over

## Game Rules

Each turn you receive observations from all roles and must submit a list of structured actions.
Capacity pools (starting values; hiring grows them):
  - Engineering: build, fix_bugs, infrastructure
  - Sales: sell, discover
  - Support (starts at 0): support — must hire or cross-hire to unlock
  - Marketing: market
  - Ops (starts at 0): ops_project — must hire or cross-hire to unlock

Each action draws from its function's pool. Check `./game observe` for current capacity per pool.

## Goals

You are scored on **three equal, top-level company goals — none ranks above the others**:
- **MRR** — reach the MRR target by the final turn.
- **Churn** — keep the cumulative churn rate below the threshold shown in `./game status`.
- **Runway** — keep runway above the `min_runway_turns` threshold shown in `./game status`.

They carry **equal weight**: a strong result on one only *partly* offsets a shortfall on another, and cannot make up for a goal left at zero. You are scored on advancing all three **together** — balancing them across the roles you control is the point of the game.

**Scoring**: each goal scores 1.0 when you hit its target. Your result is the **geometric mean** of the three goal scores — because the scores are multiplied, a weak goal drags the whole result down and a goal left at 0 makes the result 0 (you cannot ignore a goal). Exceeding a target lifts the result above 1.0, but only partly offsets a weaker goal, never replaces it.
- MRR score — final MRR against the target.
- Churn score — lower churn is better (best when there is none).
- Runway score — credit for runway above the minimum, with diminishing credit the further out you go.

**Function sub-goals** (each scored alongside the three company goals above, and feeding into them):
- Engineering — ship features at solid quality or better (`features_shipped_solid_plus`, target 12)
- Sales — maintain a steady deal-closure rate (`pipeline_velocity`, target 0.2)
- Marketing — generate inbound leads over the game (`marketing_leads_generated`, target 24)
- CS — keep average customer health above 7.0 (`avg_customer_health`, target 7.0)
- Ops — complete process improvement projects (`process_projects_completed`, target 6)

## Action Types

Each action has an `action_type` field and specific parameters:

1. **build** — Build a feature. Params: `feature_id`, `quality` ("mvp"/"solid"/"polished"), `capacity` (int > 0)
2. **fix_bugs** — Fix bugs. Params: `bug_id` (optional, null = auto-target highest severity), `capacity` (int > 0)
3. **infrastructure** — Reduce tech debt. Params: `capacity` (int > 0)
4. **sell** — Advance a customer in the pipeline. Params: `customer_id`, `sell_action` ("outbound"/"demo"/"proposal"/"negotiate"), `capacity` (int > 0), `proposed_deal_value` (optional int, only for proposal/negotiate)
5. **discover** — Find new customers by targeting shipped features. Params: `target_features` (list of feature IDs, e.g. ["<feature_1>", "<feature_2>"]; empty list = broad discovery across all shipped), `capacity` (int > 0). At least one target feature must be shipped.
5b. **market_support** (sales pool) — Co-invest Sales capacity in Marketing's **same-turn** budget campaign to buy one-stage pipeline **progression** (capped at `in_deal`; closing still needs a real proposal/negotiate). Params: `channel` ("content"/"events"), `capacity` (int > 0), `target_customer_id` (optional, events-only). `content` advances newly-arriving leads (lower prob); `events` is higher prob + pushes one existing pipeline customer. The matching `market` action must run the **same turn on the same channel** or the capacity is **wasted** (`market_support_unmatched`).
6. **support** — Support an active customer. Params: `customer_id`, `support_action` ("onboard"/"churn_intervention"/"health_check"), `capacity` (int > 0)
7. **market** — Build per-feature **awareness** (a decaying stock; takes time to build). Params: `channel` ("content"/"events"/"outbound_campaign"), `target_features` (list of feature IDs; empty = broad across shipped + in-progress), `capacity` (int > 0). High awareness on a feature makes leads needing it arrive **warm** (rarely hot) + **patient** (longer timeline) — quality/targeting, not count. Can be built before a feature ships. `events` (fast burst, expensive) and `content` (slow, durable, moderate cost) spend shared runway budget; `outbound_campaign` is medium-speed and free (capacity-only). **Joint play**: `events`/`content` ALSO unlock pipeline progression when Sales runs `market_support` on the same channel the same turn (action 5b).
8. **hire** — Start a NEW hiring process. Params: `hiring_function`, `target_function` (each one of "engineering"/"sales"/"cs"/"marketing"/"ops").
   Note: the support pool is referred to as `"support_capacity"` in observations but the function name in `hire`, `sustain_hire`, `fire`, `target_function`, and `hiring_function` parameters is **`"cs"`**.
   - **Active sustain**: hiring is a multi-turn commitment. You must submit `sustain_hire` each turn during the active phase (first half of duration) or the hire is **cancelled** (budget lost).
   - **Native** (`hiring_function == target_function`): costs 3 cap + budget, **3 active + 3 auto = 6 turns**, adds **4 capacity**.
   - **Cross-function** (`hiring_function != target_function`): costs 3 cap + budget, **6 active + 6 auto = 12 turns**, adds **3 capacity**.
   - Multiple concurrent hires allowed — each gets a unique ID (H1, H2, etc.)
9. **sustain_hire** — Continue an active hiring process. Params: `hire_id` (from `pending_hires` in the game state observation).
   - Must submit each turn during the active phase or hire is cancelled. Costs 3 capacity from the hiring pool. No budget cost.
   - Sustain capacity is **pre-committed** — deducted before other actions.
10. **fire** — Release a headcount. Params: `function` ("engineering"/"sales"/"cs"/"marketing"/"ops"). Pays severance from budget (no capacity cost this turn). Capacity drops by up to 4 starting **next turn** — the fired function's actions still resolve normally this turn (they work their last turn before leaving). Use to recover from over-hiring.
11. **ops_project** — Allocate ops capacity to advance a process improvement project. Params: `project_id`, `capacity` (int > 0). Draws from the **ops** pool.
    - **First run / net-new re-run** (bonus lapsed): full `ops_capacity_cost` per turn for `duration_turns` turns.
    - **Maintenance refresh** (bonus active but degraded): single-turn action at reduced cost. Check `maintenance_cost` in the obs.
12. **ops_project_support** — Allocate target team capacity to support an in-progress ops project. Params: `project_id`, `capacity` (int > 0). Draws from the **target team's** pool. Only valid while project is `in_progress`.
13. **ops_analysis** — Ops runs a cross-functional **analysis** for a requesting team (drawn from the **ops** pool). Params: `target_function` ("engineering"/"sales"/"cs"/"marketing"), `analysis_type` ("conversion_funnel"/"retention_efficiency"/"awareness_attribution"/"capacity_bottleneck"), `capacity` (>= analysis cost, default 2). Requires a matching `analysis_scope` (same `target_function` + `analysis_type`) the **same turn** or it is **wasted** (`analysis_unmatched`). Ops cannot analyse itself. The result lands in that team's **next-turn** observation (`analyses_received_this_turn`). (In C2 a single agent controls all functions, so both are submitted together.)
14. **analysis_scope** — A team co-invests to scope an analysis it wants (drawn from the **requesting team's** pool). Params: `target_function` (the function being analysed), `analysis_type` (same menu), `capacity` (int > 0, default 1). Must be matched the same turn by a matching `ops_analysis` or it is **wasted**.

### Pipeline Stages

```
lead -> prospect -> qualified -> in_deal -> customer
                                             (timeline expires → reset to lead with penalty)
```

Valid sell actions per stage:
- **lead**: outbound
- **prospect**: outbound, demo
- **qualified**: demo
- **in_deal**: proposal, negotiate

### Timeline (Action-Triggered)

Discovered customers have **no ticking clock** until you engage them. The countdown starts on your **first sell action** targeting a customer — stage doesn't matter. Once active, the timeline decrements by 1 every turn. If it hits 0, the customer resets to `lead` with engagement set to cold, and takes a **permanent 30% satisfaction penalty per reset** (floored at 30% of original).

## Game Mechanics & Considerations

- **Expansion**: active customers with health above 8.0 for 4+ consecutive turns trigger expansion — their deal value increases by 20% (compounding).
- **Sales momentum**: builds from closing deals (+0.08 each), shipping features (diminishing returns), and lagged marketing investment. Multiplies pipeline conversion probability up to +40%. Decays slightly each turn. Visible in the global dashboard.
- Closing deals requires customer **rubric satisfaction above ~75%**. The rubric has four components:
  * **Feature coverage**: Do you have the features they need, at sufficient quality?
  * **Price**: Larger customers are less price-sensitive.
  * **Product maturity**: The ratio of solid/polished features to total shipped features. ALL MVP = 0% maturity.
  * **Support**: Based on customer health.
- **CRITICAL**: Upgrading features from MVP to solid/polished improves maturity for ALL customers, not just those who need that feature.
- Customers have hidden rubric weights. `known_needs` gives partial visibility into feature needs.
- **Dealbreaker** features MUST be shipped before a deal can close.
- **Churn risk**: customers whose health stays very low for multiple turns will churn (their MRR is lost).
- **Customer health decay**: customer health degrades passively over time without CS attention.
- **CS attention ceiling**: there's a cap on how much support a single customer can absorb per turn; support beyond that cap on one customer is wasted.
- **Emergent needs**: active customers develop new feature needs over time. A need is invisible until a `health_check` diagnoses it — until then a customer bleeding health from an unmet need shows only as `undiagnosed_decline`. After a short grace window an unmet need bleeds health every turn and, left unmet long enough, becomes a permanent churn driver. Satisfying it requires Engineering to build/ship that feature; while Eng is actively building it, the bleed and the expiry clock both pause.
- **Support verbs**: `health_check` is the only way to diagnose emergent needs and hidden churn drivers; `onboard` accelerates the onboarding window and adds health during it (negligible once onboarding completes); `churn_intervention` is a costly, stochastic save that fires only when health is below a threshold and may fail.
- Bugs reduce customer health (impact scales with bug severity). Unresolved critical bugs can cause churn.
- Tech debt generates bugs. Infrastructure work reduces debt.
- **Marketing = awareness**: marketing builds a decaying per-feature awareness stock (channel-dependent lag: events ~2, outbound ~5, content ~8 turns). High awareness makes leads needing that feature arrive **warm + patient** (more chances to close) — it raises lead QUALITY and biases inbound toward hyped features, not the lead count. Awareness can be built before a feature ships. `events`/`content` spend shared runway budget (pay-to-accelerate); `outbound_campaign` is the free capacity-only path. Marketing also gets a fuzzy **competitive radar** — early warning of upcoming competitor events on features it's active in.
- **Marketing↔Sales co-investment**: running a budget channel (`events`/`content`) AND spending `market_support` on the same channel the same turn buys one-stage pipeline **progression** (capped at `in_deal`) on newly-arriving leads (both channels) and, for `events`, one named existing customer (`target_customer_id`). Closing still needs a real proposal/negotiate. Roll odds scale with the joint commitment (min of marketing + sales capacity) and the budget spent. Must be same-turn and same-channel, or the sales capacity is wasted (`market_support_unmatched`).
- **Hiring (active sustain)**: hiring is a multi-turn commitment. Starting a hire costs budget + 3 cap. You must then submit `sustain_hire` each turn during the active phase (first half) — costs 3 cap/turn. Miss a turn = hire cancelled, budget lost. Native: 3 active + 3 auto = 6 turns (+4 cap). Cross: 6 active + 6 auto = 12 turns (+3 cap). Sustain capacity is pre-committed before other actions. Support and Ops start at 0 capacity and must be cross-hired to unlock.
- **Firing**: `fire` costs severance (budget), no capacity cost the turn it's issued, and reduces the function's capacity by up to 4 starting next turn (the fired function still resolves this turn).
- **Discovery targeting**: `target_features` biases discovery toward customers who need those (shipped) features. Broad discovery (empty list) searches across all shipped features. At least one targeted feature must be shipped.
- **Infinite pipeline**: New customers are generated dynamically — the pipeline never dries up. The more features you ship, the richer the pool of discoverable customers becomes.
- **Segments**: customers are labelled startup / growth / mid_market / enterprise. Enterprise customers (the largest deals) are **visible from turn 1**.
- **Sell capacity costs** scale with customer size (1-5):
  * Outbound: 1 × customer_size
  * Demo: 1 × customer_size (extra capacity above minimum boosts conversion)
  * Proposal: 1 × customer_size
  * Negotiate: 1 × customer_size
  * Example: a size-5 enterprise negotiate costs minimum 5 capacity vs 1 for a size-1 startup.
  * Check `min_sell_capacity` in observations for exact costs per customer.
- **Engagement matters**: customer engagement (hot/warm/cold) significantly affects conversion probability. Hot customers are much easier to close; cold customers are very hard. Engagement requires sustained sell attention — skip a turn and it decays. Larger customers need proportionally more capacity to stay engaged.
- **Feature cost tiers**: Tier-2 features (F02-F05, segment bridges) are cheap table stakes (10-12 MVP). Tier-3 (F06-F13, segment-specific) are a real commitment (~20-25 MVP). Tier-4 (F14-F16, enterprise premium) are major strategic bets (32-40 MVP).
- **Diminishing returns on engineering**:
  * Optimal allocation per feature: ~12 capacity/turn. Beyond that, efficiency drops.
  * Each feature has a minimum number of turns to complete regardless of capacity.
  * Max progress per turn is capped at ~65% of total cost. You CANNOT one-shot features.
- **Ops — process projects**: a completed project gives its target function a bonus that **spikes then decays over time**; some projects also leave a **permanent floor** — a portion that does not decay away. Projects form a **tech-tree**: a higher-tier project is **locked** until its prerequisites are completed (the ops report shows `locked` / `locked_by` / `prerequisites`), and higher tiers carry larger maximum bonuses. A degraded-but-active bonus can be refreshed by a single `ops_project` at `maintenance_cost`; a fully-lapsed one costs full capacity to re-run. Bonus size is **probabilistic** — more target-team co-investment (`ops_project_support`) narrows the variance.
- **Ops — cross-functional analysis** (`ops_analysis` + the team's `analysis_scope`, same turn): a deterministic report computed from observable history, spanning functions — `capacity_bottleneck` (utilization across all five capacity pools), `conversion_funnel` (per-stage close rates + time-in-stage), `retention_efficiency` (churn / intervention / expansion rates), `awareness_attribution` (marketing spend vs leads vs closes). The output is **descriptive/predictive only — it never prescribes a move**, and lands in the requesting team's next-turn observation (`analyses_received_this_turn`). Ops starts at 0 capacity.
- **Pricing negotiation**: When submitting a `proposal` or `negotiate` sell action, include `proposed_deal_value` to set your price. If omitted, it defaults to the customer's sticker `deal_value`. Customers have a hidden desired price — pricing too high reduces close probability; pricing below it increases it (up to ~35% bonus). After a failed proposal, the customer may indicate a price range they'd consider — this is directional, not exact (customers negotiate in their own favor). Competitors may also submit pricing offers to in-deal customers and steal them.

## CLI Commands

All commands output JSON to stdout. Use these to gather information and plan your moves.

### Observation & Status

| Command | Description |
|---------|-------------|
| `./game observe` | Current turn observation (all role reports) |
| `./game status` | Turn number, MRR progress, runway, game over state |

### Queries

| Command | Description |
|---------|-------------|
| `./game query customer <id>` | Customer details (stage, needs, health, timeline) |
| `./game query feature <id>` | Feature status, progress, blocked_by, est completion |
| `./game query bugs` | All unresolved bugs |
| `./game query rejections` | Full rejection history across all turns |

### Compute (Analysis Tools)

| Command | Description |
|---------|-------------|
| `./game compute maturity` | Maturity score + shipped feature breakdown |
| `./game compute satisfaction <customer_id>` | Rubric satisfaction, gap to threshold, can_close |
| `./game compute maturity-if <feature_id> <quality>` | Hypothetical maturity if feature shipped/upgraded |
| `./game compute capacity-cost --actions-file ./actions.json` | Per-pool cost check, flags over-limit |

### Submitting Actions

```bash
./game submit --actions-file ./actions.json
```

Write your actions to `actions.json` first. See @action_format.md for the JSON format.

**Invalid actions are dropped and their capacity is lost.** Use `compute capacity-cost` to validate before submitting.

## Notes

Keep strategy notes in `notes/`. Update them each turn with:
- What happened (deals closed, bugs found, customers churned)
- Key metrics (MRR, maturity, satisfaction gaps)
- Plan for next turn
- Longer-term strategy adjustments

This helps you maintain context across turns and make better decisions.
