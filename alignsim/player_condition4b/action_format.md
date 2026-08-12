# Action JSON Format

Write your actions to `actions.json` as a **plain JSON array** of action objects.
Only include action types allowed for your function (see CLAUDE.md for your capabilities).

## Template

```json
[
  { "action_type": "...", ... },
  { "action_type": "...", ... }
]
```

## Action Examples

**IDs**: Where an example shows a placeholder like `<feature>`, `<customer>`, or `<project>`, substitute the real ID from your observation, copied exactly — feature IDs have the form `F##`, customer IDs `C##`, and ops-project IDs `PP##`.

### build *(engineering only)*

```json
{ "action_type": "build", "feature_id": "<feature>", "quality": "solid", "capacity": 10 }
```

Quality options: `"mvp"`, `"solid"`, `"polished"`.

### fix_bugs *(engineering only)*

```json
{ "action_type": "fix_bugs", "bug_id": "BUG001", "capacity": 5 }
```

Set `bug_id` to `null` to auto-target the highest severity bug:
```json
{ "action_type": "fix_bugs", "bug_id": null, "capacity": 5 }
```

### infrastructure *(engineering only)*

```json
{ "action_type": "infrastructure", "capacity": 5 }
```

### sell *(sales only)*

```json
{ "action_type": "sell", "customer_id": "<customer>", "sell_action": "outbound", "capacity": 3 }
```

Sell actions: `"outbound"`, `"demo"`, `"proposal"`, `"negotiate"`. Must match the customer's pipeline stage.

For `proposal` and `negotiate`, include `proposed_deal_value` to set your price (defaults to the customer's sticker `deal_value` if omitted; rejected on outbound/demo):
```json
{ "action_type": "sell", "customer_id": "<customer>", "sell_action": "negotiate", "capacity": 3, "proposed_deal_value": 2500 }
```

### discover *(sales only)*

```json
{ "action_type": "discover", "target_features": ["<feature>"], "capacity": 3 }
```

`target_features` is a list of feature IDs to bias discovery toward (at least one must be shipped). Use an empty list for broad discovery across all shipped features:
```json
{ "action_type": "discover", "target_features": [], "capacity": 3 }
```

### market_support *(sales only)*

```json
{ "action_type": "market_support", "channel": "events", "capacity": 3, "target_customer_id": "<customer>" }
```

Co-invest Sales capacity (drawn from the **sales** pool) in Marketing's **same-turn** budget campaign to buy one-stage pipeline **progression** (capped at `in_deal` — closing still needs a real `proposal`/`negotiate`). `channel` is `"content"` or `"events"`. `content` = newly-arriving inbound leads roll to land one stage advanced (lower prob); `events` = higher prob **and** an optional `target_customer_id` pushes one existing pipeline customer one stage. The matching `market` campaign **must run the same turn on the same channel** (coordinate with Marketing via a Post) — otherwise your capacity is **wasted**.

### support *(support only)*

```json
{ "action_type": "support", "customer_id": "<customer>", "support_action": "onboard", "capacity": 3 }
```

Support actions: `"onboard"`, `"churn_intervention"`, `"health_check"`.

### market *(marketing only)*

```json
{ "action_type": "market", "channel": "events", "target_features": ["<feature>"], "capacity": 3 }
```

Channels: `"content"`, `"events"`, `"outbound_campaign"`. `target_features` is a list of feature IDs (empty = broad across all shipped + in-progress features). Builds a decaying per-feature **awareness** stock that makes leads needing those features arrive warmer + more patient (quality, not count). `events`/`content` spend shared runway budget; `outbound_campaign` is capacity-only.

### hire *(all functions — restricted to your pool)*

```json
{ "action_type": "hire", "hiring_function": "engineering", "target_function": "engineering" }
```

Cross-function hire (your pool pays, different team receives):
```json
{ "action_type": "hire", "hiring_function": "engineering", "target_function": "ops" }
```

Engine function names: engineering, sales, cs (for support), marketing, ops.

### sustain_hire *(all functions — only hires you initiated)*

```json
{ "action_type": "sustain_hire", "hire_id": "H1" }
```

Costs 3 capacity from the original hiring pool. Pre-committed before other actions.

### fire *(all functions — your function only)*

```json
{ "action_type": "fire", "function": "engineering" }
```

Use your engine function name: engineering, sales, cs, marketing, ops.

### ops_project *(ops only)*

```json
{ "action_type": "ops_project", "project_id": "<project>", "capacity": 4 }
```

### ops_project_support *(target function only)*

```json
{ "action_type": "ops_project_support", "project_id": "<project>", "capacity": 3 }
```

Only valid for the function the project targets, while the project is in progress.

### ops_analysis *(ops only)*

```json
{ "action_type": "ops_analysis", "target_function": "sales", "analysis_type": "capacity_bottleneck", "capacity": 2 }
```

Ops runs a cross-functional analysis for a requesting team. `analysis_type`: `"conversion_funnel"`, `"retention_efficiency"`, `"awareness_attribution"`, `"capacity_bottleneck"`. Must be matched the **same turn** by the requesting team's `analysis_scope` (same `target_function` + `analysis_type`) — coordinate via a Post — or it is **wasted** (`analysis_unmatched`). Ops cannot analyse itself. The result arrives in the requesting team's next-turn observation (`analyses_received_this_turn`).

### analysis_scope *(requesting function only)*

```json
{ "action_type": "analysis_scope", "target_function": "sales", "analysis_type": "capacity_bottleneck", "capacity": 1 }
```

Co-invest (default 1 cap from your own pool) to scope an analysis your function wants. `target_function` must be **your own** function. Must be matched the same turn by Ops's `ops_analysis`, or it is **wasted**.
