# Action JSON Format

Write your actions to `actions.json` as a **plain JSON array** of action objects.

## Template

```json
[
  { "action_type": "...", ... },
  { "action_type": "...", ... }
]
```

## Action Examples

**IDs**: Where an example shows a placeholder like `<feature>`, `<customer>`, or `<project>`, substitute the real ID from your observation, copied exactly — feature IDs have the form `F##`, customer IDs `C##`, and ops-project IDs `PP##`.

### build

```json
{ "action_type": "build", "feature_id": "<feature>", "quality": "solid", "capacity": 10 }
```

Quality options: `"mvp"`, `"solid"`, `"polished"`. Higher quality costs more capacity but improves maturity.

### fix_bugs

```json
{ "action_type": "fix_bugs", "bug_id": "BUG001", "capacity": 5 }
```

Set `bug_id` to `null` to auto-target the highest severity bug:
```json
{ "action_type": "fix_bugs", "bug_id": null, "capacity": 5 }
```

### infrastructure

```json
{ "action_type": "infrastructure", "capacity": 5 }
```

### sell

```json
{ "action_type": "sell", "customer_id": "<customer>", "sell_action": "outbound", "capacity": 3 }
```

Sell actions: `"outbound"`, `"demo"`, `"proposal"`, `"negotiate"`. Must match the customer's pipeline stage.

For `proposal` and `negotiate`, include `proposed_deal_value` to set your price:
```json
{ "action_type": "sell", "customer_id": "<customer>", "sell_action": "proposal", "capacity": 3, "proposed_deal_value": 2800 }
```
```json
{ "action_type": "sell", "customer_id": "<customer>", "sell_action": "negotiate", "capacity": 3, "proposed_deal_value": 2500 }
```
If omitted, defaults to the customer's sticker `deal_value`. Only valid on proposal/negotiate — rejected on outbound/demo.

### discover

```json
{ "action_type": "discover", "target_features": ["<feature>"], "capacity": 3 }
```

Target features: list of feature IDs to bias discovery toward. At least one must be shipped. Use an empty list for broad discovery across all shipped features:
```json
{ "action_type": "discover", "target_features": [], "capacity": 3 }
```

### market_support

```json
{ "action_type": "market_support", "channel": "events", "capacity": 3, "target_customer_id": "<customer>" }
```

Co-invest **Sales** capacity in Marketing's **same-turn** budget campaign (`content`/`events`) to buy one-stage pipeline **progression** (capped at `in_deal` — closing still needs a real `proposal`/`negotiate`). Draws from the sales pool. `content` advances newly-arriving inbound leads (lower prob); `events` is higher prob **and** an optional `target_customer_id` pushes one existing pipeline customer one stage. The matching `market` action must run the **same turn on the same channel** or the capacity is **wasted** (`market_support_unmatched`).

### support

```json
{ "action_type": "support", "customer_id": "<customer>", "support_action": "onboard", "capacity": 3 }
```

Support actions: `"onboard"`, `"churn_intervention"`, `"health_check"`.

### market

```json
{ "action_type": "market", "channel": "events", "target_features": ["<feature>"], "capacity": 3 }
```

Channels: `"content"`, `"events"`, `"outbound_campaign"`. `target_features` is a list of feature IDs (empty = broad across all shipped + in-progress features). Builds a decaying per-feature **awareness** stock that makes leads needing those features arrive warmer + more patient (quality, not count) — and can be built before a feature ships. `events`/`content` spend shared runway budget; `outbound_campaign` is capacity-only.

### hire

Start a new hiring process (active sustain — must submit `sustain_hire` each turn during active phase):

Native hire (same team recruits — 3 active + 3 auto turns):
```json
{ "action_type": "hire", "hiring_function": "engineering", "target_function": "engineering" }
```

Cross-function hire (6 active + 6 auto turns, delivers 3 capacity instead of 4):
```json
{ "action_type": "hire", "hiring_function": "sales", "target_function": "ops" }
```

Costs 3 from `hiring_function` pool plus budget on initiation. Each hire gets a unique ID (H1, H2, etc.). Multiple concurrent hires allowed.

### sustain_hire

Continue an active hiring process. Submit each turn during the active phase or the hire is **cancelled**.

```json
{ "action_type": "sustain_hire", "hire_id": "H1" }
```

Costs 3 capacity from the original hiring pool. No budget cost. Sustain capacity is pre-committed before other actions.

### fire

```json
{ "action_type": "fire", "function": "engineering" }
```

Functions: `"engineering"`, `"sales"`, `"cs"`, `"marketing"`, `"ops"`. Removes up to 4 capacity. Pays severance from budget (no capacity cost this turn).

### ops_project

First run or net-new re-run (bonus lapsed):
```json
{ "action_type": "ops_project", "project_id": "<project>", "capacity": 4 }
```

Maintenance refresh (bonus still active but degraded — single-turn, reduced cost from `maintenance_cost` in obs):
```json
{ "action_type": "ops_project", "project_id": "<project>", "capacity": 2 }
```

Must provide at least the project's `ops_capacity_cost` for first run, or the `maintenance_cost` shown in obs for a refresh.

### ops_project_support

```json
{ "action_type": "ops_project_support", "project_id": "<project>", "capacity": 3 }
```

Allocates target team capacity to support an in-progress ops project (change management). Draws from the target team's pool, not ops.

### ops_analysis

```json
{ "action_type": "ops_analysis", "target_function": "sales", "analysis_type": "capacity_bottleneck", "capacity": 2 }
```

Ops runs a cross-functional analysis for a requesting team (draws from the **ops** pool). `analysis_type`: `"conversion_funnel"`, `"retention_efficiency"`, `"awareness_attribution"`, `"capacity_bottleneck"`. Must be paired with a matching `analysis_scope` (same `target_function` + `analysis_type`) the **same turn**, or it is **wasted** (`analysis_unmatched`). Ops cannot analyse itself. The result arrives in the requesting team's **next-turn** observation under `analyses_received_this_turn`.

### analysis_scope

```json
{ "action_type": "analysis_scope", "target_function": "sales", "analysis_type": "capacity_bottleneck", "capacity": 1 }
```

A team co-invests (default 1 cap from its own pool) to scope the analysis it wants. Must be matched the same turn by a matching `ops_analysis`, or it is **wasted**.

## Full Turn Example

```json
[
  { "action_type": "build", "feature_id": "<feature_1>", "quality": "solid", "capacity": 10 },
  { "action_type": "build", "feature_id": "<feature_2>", "quality": "mvp", "capacity": 8 },
  { "action_type": "infrastructure", "capacity": 2 },
  { "action_type": "sell", "customer_id": "<customer_1>", "sell_action": "outbound", "capacity": 3 },
  { "action_type": "discover", "target_features": ["<feature_1>"], "capacity": 4 },
  { "action_type": "sell", "customer_id": "<customer_2>", "sell_action": "demo", "capacity": 3 },
  { "action_type": "support", "customer_id": "<customer_3>", "support_action": "health_check", "capacity": 3 },
  { "action_type": "market", "channel": "content", "capacity": 2 },
  { "action_type": "market", "channel": "events", "capacity": 3 },
  { "action_type": "sustain_hire", "hire_id": "H1" },
  { "action_type": "hire", "hiring_function": "engineering", "target_function": "engineering" },
  { "action_type": "fire", "function": "marketing" },
  { "action_type": "ops_project", "project_id": "<project>", "capacity": 4 },
  { "action_type": "ops_project_support", "project_id": "<project>", "capacity": 2 },
  { "action_type": "ops_analysis", "target_function": "sales", "analysis_type": "capacity_bottleneck", "capacity": 2 },
  { "action_type": "analysis_scope", "target_function": "sales", "analysis_type": "capacity_bottleneck", "capacity": 1 }
]
```
