# Goal Alignment Benchmark: AlignSim

## What Is This

AlignSim is a turn-based strategy game designed to benchmark goal alignment. It is not a simulation of a SaaS company — it is a game whose mechanics are *derived from* SaaS business dynamics, in the same way that Civilization's mechanics are derived from real-world geopolitics without pretending to be a literal simulation of history, or the way MIT's Beer Game uses a simplified supply chain to reveal fundamental truths about coordination failure.

The game is played by 1-5 players (AI agents, humans, or hybrids) over 48 turns. Players allocate finite capacity across competing priorities, process an information stream of structured observations, and attempt to achieve a set of hierarchical goals with inherent tensions. The game is interacted with entirely through structured, programmatic actions — no natural language exchange with the game engine. Natural language coordination happens between players through whatever substrate is being tested (Convictional, Slack, shared filesystem, etc.).

The benchmark measures: given the same game, the same goals, the same starting conditions, and the same information stream, does the coordination substrate affect goal attainment?

## Why a Game, Not a Simulation

Initially we thought about financial fund simulation models as a realistic financial environment with synthetic market data and LLM-authored narrative. That approach works because financial markets are exogenous — the world doesn't react to the fund's decisions. A SaaS business doesn't have that property. Customer behavior changes based on what you build and sell. That reactive world state is harder to simulate credibly. We decided against this given our lack of domain knowledge which would inevitably lead to non-believable scenarios and unrealistic mechanics.

Rather than trying to build a high-fidelity SaaS simulation (which is arguably its own research project), we lean into game design. The game's rules are transparent, inspectable, and arguable on their own terms. The question is not "is this a realistic SaaS company?" but "is this a fair game where goal alignment is genuinely hard?" That's a much cleaner question to answer, and it's the one that matters for the benchmark.

The game-first approach also eliminates several problems:

- **No fiction to maintain.** Features are "Lightning" and "Cascade," not "Snowflake integration." Customers are `C01` through `C40` with stat blocks, not elaborate personas. No LLM will break the fourth wall because there's no fourth wall to break.
- **No training data contamination.** The model can't apply memorized SaaS playbook heuristics because the features are abstract. "Should we build Lightning or Cascade?" can only be answered by processing the game state — there's no prior knowledge about what Lightning does.
- **Cheap scenario generation.** Scenarios are parameterized stat blocks, not authored narratives. Generate hundreds procedurally, validate that the distributions produce interesting games, and discard the degenerate ones.
- **Clean publishability.** A research paper describes game rules, not a fictional company narrative that readers must evaluate for plausibility. Rules are inspectable and forkable.

## Game Design

### Entities

All entities are generic and stat-driven. Names are simple identifiers or short evocative labels (for readability in agent coordination) without domain-specific meaning.

**Customers** (`C01` through `C40+`)

Visible stats (revealed when a customer becomes known):
- `size`: 1-5 (abstract scale, correlates with deal value)
- `segment`: A, B, C, D (abstract groupings — customers in the same segment tend to have similar needs)
- `stage`: lead → prospect → qualified → in-deal → customer → churning
- `engagement`: cold / warm / hot
- `known_needs`: list of features they've expressed interest in (partial view of true needs)
- `deal_value`: capacity units of revenue per turn if closed

Hidden stats (drive game outcomes, never directly revealed):
- `rubric`: weighted scorecard — `{feature_coverage: 0.4, price: 0.2, maturity: 0.2, support: 0.2}` with weights varying per customer
- `feature_needs`: full list of features that satisfy the feature_coverage rubric weight, with per-feature satisfaction scores at each quality level (MVP/solid/polished)
- `dealbreakers`: features without which close probability is zero
- `timeline`: turns until the customer's decision window closes (they buy, or they leave the pipeline)
- `churn_drivers`: for existing customers — which features/quality levels their retention depends on
- `discovery_difficulty`: how much discovery effort is needed to reveal this customer

**Features** (`F01: Lightning`, `F02: Cascade`, `F03: Vortex`, etc.)

Visible stats:
- `cost`: capacity units to build at each quality level — e.g. `{mvp: 8, solid: 15, polished: 25}`
- `depends_on`: list of prerequisite features
- `status`: not_started / in_progress / shipped_mvp / shipped_solid / shipped_polished
- `progress`: percentage complete toward current target quality level
- `description`: brief generic description of what the feature does in game terms (e.g. "Enables segment B integrations" — enough for agents to reason about, abstract enough to prevent domain heuristic shortcuts)

Hidden stats:
- `customer_impact`: map of customer IDs to rubric satisfaction scores at each quality level — e.g. `{C03: {mvp: 0.6, solid: 0.8, polished: 0.95}, C07: {mvp: 0.3, solid: 0.5, polished: 0.7}}`
- `bug_rate_modifier`: how much this feature contributes to bug injection when shipped (higher for complex features)
- `maintenance_cost`: ongoing capacity cost per turn once shipped

**Competitors** (`Comp_Alpha`, `Comp_Beta`)

Not player-controlled. Follow a pre-defined event schedule:
- Launch features at defined turns (satisfying rubric weights for specific customers)
- Change pricing at defined turns
- Win specific deals from the pipeline if their offering scores higher on a customer's rubric at the decision point

Competitive events appear in the information stream as structured observations.

### Resources

**Capacity.** The team has a fixed pool of capacity units per turn (e.g. 40 units), allocated across functions:

- `engineering`: build features, fix bugs, infrastructure work
- `sales`: advance pipeline (outbound, demos, proposals, negotiations)
- `cs`: retain customers (onboarding, support, churn intervention)
- `marketing`: demand generation (increases inbound discovery rate with lag)
- `discovery`: reveal new customers from the hidden pool

Capacity is the fundamental scarce resource. Every allocation decision is a tradeoff.

**Budget.** A financial resource measured in abstract currency. Revenue comes in from closed customers (deal_value per turn). Costs are: base team cost per turn, hiring costs, and any special actions. If budget (runway) drops below zero, the game ends in failure regardless of other goal attainment. This creates a survival constraint that tensions against growth investment.

### Turn Structure

Each turn represents one week. A turn proceeds in phases:

**Phase 1: Observe.** Players receive structured observations appropriate to their role (see Information Stream below). This is the only way players learn about the game state.

**Phase 2: Coordinate.** Players discuss strategy, share information, and decide on actions. This phase happens in the coordination substrate being tested (Convictional, Slack, etc.). The game engine is not involved — this is purely inter-player communication. Players can take as much time as they need (no real-time pressure — satisfying the discrete time step criterion).

**Phase 3: Act.** Players submit structured actions to the game engine:

```
build(F03, quality=mvp, capacity=4)
build(F07, quality=solid, capacity=6)
fix_bugs(capacity=3)
infrastructure(capacity=2)
sell(C12, action=demo, capacity=2)
sell(C08, action=proposal, capacity=2)
discover(segment=B, capacity=2)
support(C05, action=churn_intervention, capacity=2)
market(channel=content, capacity=2)
hire(function=engineering)
```

Total capacity allocated must not exceed the pool. Actions are validated by the engine (can't demo a customer you haven't discovered, can't build a feature whose dependencies aren't met, etc.).

**Phase 4: Resolve.** The game engine processes all actions deterministically:

- Features progress based on engineering capacity allocated
- Customer stages advance or stall based on sales actions + rubric scores against current product state
- Customer health changes based on bugs, CS attention, and product quality
- Bugs are injected based on technical debt level and shipped feature bug rates (seeded RNG)
- Discovery reveals new customers from the hidden pool
- Marketing investment updates the inbound lead generation pipeline (lagged effect)
- Competitive events fire per the pre-defined schedule
- Revenue is collected from active customers, costs are deducted, runway is updated
- New observations are generated for the next turn

**Phase 5: Report.** The engine produces a turn summary (visible to all roles) and role-specific observations for the next turn.

### Information Stream

Each turn, players receive structured observations. Information is asymmetric — each role sees different things, creating the coordination problem.

**Global dashboard (all roles see):**
```
turn: 14
mrr: 105,000
pipeline_value: 820,000
active_customers: 18
churn_this_turn: 1 (C03, reason: unresolved_bug_F01)
new_leads_this_turn: 2 (C29, C30)
debt_level: medium
bug_backlog: 4 (2 major, 2 minor)
runway: 14 months
capacity_available: 40 units
```

**Sales observations (Head of Sales):**
```
C12: demo_complete, interest=high, needs=[F03, F08], timeline=6_turns
C08: proposal_under_review, competitor_Alpha_also_bidding
C15: outbound_contact, response=warm, segment=B
C22: went_cold, no_response_3_turns
pipeline_summary: 8 prospects, 3 in-deal, est_close_value=280K
win_this_turn: C19 (deal_value=4200/turn)
loss_this_turn: C11 (reason=missing_F05, went_to=Comp_Beta)
```

**Product/Engineering observations (Head of Product, Head of Engineering):**
```
F03: progress=75%, est_completion=2_turns at current allocation
F07: progress=30%, blocked_by=F09 (not started)
bug_injected: major bug in F01 (affects C05, C08, C14)
bug_fixed: minor bug in F02
debt_delta: +2 (velocity exceeded sustainable rate)
infrastructure_impact: maintenance costs reduced by 1 unit/turn
feature_requests_from_pipeline: F03 (3 prospects), F05 (2 prospects), F11 (1 prospect)
```

**CS observations (Head of CS):**
```
C05: health=declining (was 7, now 5), cause=bug_in_F01, severity=major
C14: health=stable (8), usage_growing, expansion_signal
C03: churned (health reached 0, unresolved major bug for 4 turns)
C08: health=at_risk (6), cause=competitor_Alpha_offering_better_F08_equivalent
onboarding: C19 started, est_ramp=4_turns
avg_customer_health: 7.2 (was 7.5 last turn)
```

**Cross-functional signals (the alignment problem):**
These are observations that are visible to one role but relevant to another. The game doesn't route them automatically — that's the coordination substrate's job. Examples:

- Sales sees `C12 needs F03` → Product needs to know this to prioritize
- CS sees `C05 declining due to bug in F01` → Engineering needs to know to prioritize the fix
- Engineering sees `feature_requests_from_pipeline: F03 (3 prospects)` → but this is an aggregate; Sales knows *which* prospects and how much they're worth
- Sales sees `C08: competitor_Alpha_also_bidding` → Product needs to know to evaluate whether to accelerate competing feature work

**The game does not tell players what information is important.** It presents observations. The players must decide what matters, who needs to know, and what to do about it. That's goal alignment.

### Goal System

**Primary goal (game-level win condition):**
"Reach MRR of 210,000 by turn 48 while maintaining churn rate below 2% per turn and runway above 10 months."

All three conditions must hold at turn 48 for a passing score. The compound structure means you can't just grow at all costs (churn constraint) or just preserve cash (growth target).

**Role-level sub-goals (create tension):**

CEO:
- Reach MRR target
- Maintain financial constraints
- Keep team capacity utilization above 85%

Head of Sales:
- Close 15 new customers by turn 48
- Maintain average deal size above 3,000/turn
- Keep pipeline coverage above 3x target

Head of Product:
- Ship features F01-F06 by turn 36
- Maintain debt level below "high"
- Reduce bug backlog to zero critical bugs

Head of Engineering:
- Deliver 35+ capacity units of feature/infra work per turn
- Keep debt level below "high"
- Average bug fix time below 3 turns

Head of CS:
- Maintain average customer health above 7.0
- Achieve net revenue retention above 110%
- Onboard new customers within 4 turns

These sub-goals are **in tension by design:**
- Sales wants features shipped fast (even at MVP) to close deals → conflicts with Engineering's quality goals and Product's debt management
- CS wants bug fixes prioritized → conflicts with Product's feature shipping goals
- CEO wants high capacity utilization → conflicts with Engineering's sustainable pace goals
- Sales wants to pursue large deals → may conflict with CS's retention priorities if the team gets stretched
- Product wants to ship the strategic features (F01-F06) → Sales may want different features that close immediate pipeline

### Difficulty Tuning

The game's difficulty is controlled by a small number of parameters that can be adjusted per scenario:

- **Capacity pool size** (more capacity = easier)
- **Customer rubric stringency** (how many features/what quality needed to close)
- **Competitive pressure** (how often and how aggressively competitors act)
- **Bug injection rate** (how quickly technical debt creates problems)
- **Goal target aggressiveness** (how ambitious the MRR target is relative to the addressable market)
- **Information completeness** (how much of a customer's needs are revealed through `known_needs` vs. requiring inference from behavior)

These parameters allow us to create scenarios ranging from "achievable with basic coordination" to "requires excellent alignment to have any chance."

## Scenario Generation

Scenarios are procedurally generated with seeded RNG, then validated. The generation process:

**Step 1: Generate market parameters.** Sample from calibrated distributions:
- Total addressable market: 40-60 customers
- Initially visible: 8-12 (mix of stages)
- Segment distribution: 4 segments with 25% ± 10% each
- Deal value distribution: log-normal centered on parameters derived from ICP-scale SaaS benchmarks

**Step 2: Generate customer rubrics.** For each customer:
- Sample rubric weights from a Dirichlet distribution (ensuring each customer has a distinct priority mix)
- Assign feature needs based on segment (customers in the same segment tend to need similar features, with variation)
- Set dealbreakers (0-2 per customer, sampled from the feature set)
- Set timeline (sampled from a distribution calibrated to real sales cycle lengths)
- Set churn drivers for existing customers

**Step 3: Generate tech tree.** 12-18 features with:
- Costs sampled from calibrated distributions (real engineering velocity data)
- Dependency graph generated to ensure interesting tradeoffs (no single path is optimal)
- Customer impact scores derived from rubrics (ensuring features matter to specific customer clusters)
- Bug rate modifiers correlated with feature complexity

**Step 4: Generate competitive schedule.** 2 competitors with:
- Feature launch events at defined turns (designed to create time pressure)
- Pricing changes that affect specific customer segments
- Deal wins that remove pipeline customers (if the competitor's offering scores higher)

**Step 5: Set initial conditions.** Starting MRR, existing customer base with health scores, current product state (some features already shipped), pipeline state, team composition, runway.

**Step 6: Validate.** Run the scenario in single-LLM mode 50+ times with a simple heuristic agent. Check that:
- Outcomes have meaningful variance (different strategies produce different results)
- The game is not trivially solvable (no single obvious strategy dominates)
- The game is not impossibly hard (a reasonable strategy can achieve partial goal attainment)
- The goal tensions actually manifest (you can't satisfy all sub-goals simultaneously without tradeoffs)

Scenarios that pass validation are included in the benchmark suite.

## Calibration Parameters

The game's mechanics are calibrated to real SaaS dynamics. These parameters are transparent and adjustable — anyone can argue with them and fork the game with different assumptions.

**Pipeline conversion (base rates, modified by rubric score):**
- Lead → Prospect: 20% per turn with sales attention
- Prospect → Qualified: 50% per turn with sales attention
- Qualified → In-deal: 40% per turn with demo action
- In-deal → Closed-won: 25% per turn with proposal + rubric satisfaction > 0.7
- All rates modified by customer rubric score: `base_rate * rubric_satisfaction`
- Zero conversion if dealbreaker features are missing

**Engineering velocity:**
- Base capacity: 40 units/turn (representing ~10 person team)
- MVP feature: 6-12 units. Solid: 12-20 units. Polished: 20-30 units.
- Bug fix: 1-4 units depending on severity
- Infrastructure work: 3-8 units per meaningful improvement

**Technical debt and bugs:**
- Debt accumulation: `+1 per 10 units of feature work at MVP, +0.5 at solid, +0.2 at polished`
- Debt reduction: `-1 per 5 units of infrastructure work`
- Bug injection: `poisson(lambda = debt_level * 0.3)` per turn (seeded)
- Bug severity: 20% critical, 40% major, 40% minor
- Unresolved critical bugs: customer health drops 2/turn. Major: drops 1/turn. Minor: drops 0.5/turn.

**Customer health and churn:**
- Health scale: 0-10
- Starting health for new customers: 8 (after onboarding complete)
- Onboarding duration: 3-5 turns with CS attention
- Health drivers: bug impact (negative), CS attention (positive), feature quality (positive), competitive alternatives (negative)
- Churn trigger: health drops below 3 for 2 consecutive turns
- Expansion trigger: health above 8 for 4 consecutive turns → deal_value increases 20%

**Marketing and discovery:**
- Marketing investment → inbound leads: 8-12 turn lag, rate = `base_inbound + (marketing_units * 0.3)` leads per turn
- Active discovery: `discovery_capacity / customer_discovery_difficulty` probability of revealing a customer per turn
- Discovery reveals customer visible stats only; hidden stats remain hidden but influence game mechanics

**Hiring:**
- Decision to hire: costs 2x capacity in budget for the turn
- New capacity arrives: 6 turns after decision
- Onboarding penalty: 50% effective capacity for first 4 turns after arrival

## Connecting to the Benchmark

### The Four Conditions

Same as the main benchmark doc:

**Condition 1: Single player.** One LLM sees all observations and submits all actions. No coordination needed. This is the baseline — tests raw strategic prioritization without coordination overhead or benefit.

**Condition 2: Single player with tools.** One LLM with an agentic harness (planning, memory, multi-step reasoning). Still one decision-maker, but with better cognitive scaffolding.

**Condition 3: Multi-player with generic tools.** 5 agents (one per role), each seeing only their role's observations. Coordinate through Slack + shared spreadsheet. Must share information and align on strategy through these generic tools.

**Condition 4: Multi-player with Convictional.** Same 5 agents, same role constraints. Coordinate through Convictional. Goals, sub-goals, and alignment signals flow through a purpose-built system.

### Measurement

**Primary metric: Goal attainment score.**
Weighted composite of:
- MRR target achievement: `min(final_mrr / target_mrr, 1.0)` — weight 0.5
- Churn constraint: `1.0 if avg_churn < 2%, scaled penalty otherwise` — weight 0.25
- Runway constraint: `1.0 if final_runway > 10mo, scaled penalty otherwise` — weight 0.25

**Secondary metrics:**

- **Decision quality (hindsight-optimal comparison).** Using ground truth (hidden rubrics), compute the optimal action sequence given full information. Measure the gap between actual and optimal. Decompose by decision type: feature prioritization, pipeline focus, capacity allocation.

- **Information routing efficiency.** How often did role A possess information that role B needed, and how quickly did it get there? Using the cross-functional signal ground truth, measure: signal detected → signal shared → signal acted on, with latency at each step.

- **Coordination overhead.** Total messages exchanged, total coordination time, decisions per unit of communication.

- **Sub-goal conflict resolution.** How many turns involved active tension between sub-goals? How were they resolved? Did the resolution lead to better or worse primary goal outcomes?

## The Beer Game Parallel

The MIT Beer Game is arguably the most influential business simulation ever created. It has four roles, one product, one number per turn (how many units to order), and the total game state fits on an index card. Yet it reveals a profound truth: coordination structure determines outcomes more than individual decision quality. CEOs of Fortune 50 companies perform no better than high school students.

AlignSim is the Beer Game for AI-mediated organizational decision-making. More roles, more dimensions, richer goal conflicts — but the same core hypothesis: that the *substrate through which agents coordinate* has a measurable effect on goal attainment, independent of the agents' individual capabilities.

If that hypothesis is validated — if Condition 4 consistently outperforms Condition 3, or if multi-agent conditions outperform single-agent conditions despite the coordination overhead — that's a publishable and commercially significant result.

## Open Questions

1. **Role count.** We've specified 5 roles (CEO, Sales, Product, Engineering, CS). Is that too many for the first version? The Beer Game works with 4. We could start with 3 (CEO, Build, Sell) and expand later. Fewer roles = simpler coordination = cleaner signal about whether coordination substrate matters.

2. **Turn count.** 48 turns (weekly for 12 months) might be too many for practical benchmarking. Each turn requires LLM inference for every agent. At 5 agents × 48 turns × 4 conditions × N scenarios, the compute budget adds up. Could compress to 24 turns (bi-weekly) or 12 turns (monthly) without losing the alignment dynamics.

3. **Observation window.** Should agents have memory of all past observations, or only the last N turns? Full memory favors models with large context windows. Windowed memory is more realistic (humans forget) and computationally cheaper, but might unfairly handicap strategies that require long-term pattern recognition.

4. **Partial observability vs. summarization.** Currently the CEO sees everything. Should the CEO instead see role-submitted summaries? This would make the CEO role depend on the quality of information shared upward — a more realistic and more interesting alignment problem. But it adds a summarization mechanic that could introduce noise.

5. **Scenario count for publication.** How many validated scenarios do we need for a credible benchmark? 10? 50? 100? Tradeoff between statistical power and generation/validation effort.

## Next Steps

1. **Design one complete scenario by hand.** Define all customers, features, competitive events, and initial conditions. This is the playtest scenario.
2. **Implement the game engine in SimPy.** Customer state machines, product state, capacity allocation, bug injection, turn resolution.
3. **Build the Condition 1 harness.** Single LLM playing all roles. Run the playtest scenario 50+ times to validate dynamics.
4. **Iterate on calibration.** Adjust parameters until the game produces interesting variance and non-trivial tradeoffs.
5. **Build the procedural scenario generator.** Parameterized generation from calibrated distributions.
6. **Build Condition 3 and 4 harnesses.** Multi-agent with Slack and Convictional respectively.
7. **Run the benchmark.** Compare conditions across validated scenarios.
