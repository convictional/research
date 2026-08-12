# Reference — anthropic-spend-report

## Why allocation, not a pricing table

The Admin API splits the data you want across two endpoints:

- **Cost API** (`/v1/organizations/cost_report`) — exact USD, but `group_by` is
  only `workspace_id` + `description`. **No `api_key_id`.**
- **Usage API** (`/v1/organizations/usage_report/messages`) — token counts,
  and *can* group by `api_key_id`, `model`, `context_window`, `service_tier`.

Neither gives dollars per key directly. So the script takes each Cost API token
line item — keyed by `(day, model, context_window, service_tier, token_type)` —
and splits its dollar `amount` across keys in proportion to each key's share of
the matching tokens from the Usage API. Because price is constant within one
`(model, context_window, service_tier, token_type)` bucket, the proportional
split is **exact**, and per-key dollars sum to the bill. No rates are stored, so
nothing drifts when Anthropic reprices, adds models, or discounts batch usage.

The rounding remainder on each cost line is given to the largest-token holder so
each line reconciles to the cent.

## Key-name → person / key_type

`lead_token(name)` extracts the person token of a key name:
- `claude_code_key_<first>.<last>_<suffix>` → `<first>.<last>` (drops the random suffix)
- anything else → the first `[-_./:@\s]`-delimited token (e.g. `adams-local-key` → `adams`)

`attribute()` maps that token via `ALIAS` (built from `PEOPLE`) to a person;
otherwise the exact key name via `BUCKET` to a category; the no-key Workbench
row to `console/workbench`; else `uncategorized`. `key_type()` is `claude_code`
when the name starts with the Claude Code prefix, `dev` for any other personal
key, `service` for non-person keys.

## Caveats

- **Priority / flex tiers are excluded by the Cost API** — that spend is not in
  `spend_usd` (the usage tokens still appear). For most orgs this is $0.
- **Workbench / Console usage has no API key** → it lands in the
  `console/workbench` bucket (the `api_key_id` is `null`). See the enhancement
  below to attribute it to people.
- **Default workspace** reports `workspace_id = null` (handled).
- **`unattributed`** (printed in the summary) = a token cost line with no
  matching usage row; normally $0.
- Data is fresh within ~5 min of a request; the API allows ~1 poll/min
  sustained. The script throttles (1s/call) and backs off on 429, but avoid
  back-to-back full re-runs.

## Testing

Two automated layers (stdlib only — no deps), plus one human check:

1. **Offline** (`tests/test_offline.py`, no API): the allocation invariants (parts
   sum to the line; nonzero remainder to the largest holder; every share
   non-negative; deterministic tie-break; unattributed when no usage), the
   `reconcile()` residual, month-window boundaries, month-arg validation, and token
   extraction. Run from the skill root:
   `cd experiments/anthropic-spend-report && python3 -m unittest discover -s tests`.
   Catches our own regressions (allocation / parsing edits).
2. **Live self-test** (`--self-test YYYY-MM`, ~2 API calls): pulls one month and
   asserts data came back, the parsed fields are present (catches API-shape drift),
   the allocation reconciles to the Cost API total, unattributed cost is under $1.00
   (normally $0), and every cost line's **implied $/Mtok** falls in a sane band
   (~$0.02–150) — which trips if `amount` ever silently changed units (e.g.
   cents↔dollars). No stored golden values; the cross-endpoint checks are
   self-validating. It's a *consistency* test, not proof of absolute correctness.

## Confirming the numbers are right (one-time, human — Layer 3)

Automated checks (Layers 1–2) and a run's own "unattributed $0.00" line prove only
that per-key dollars **reconcile to the Cost API total** — they do **not** prove that
total matches your **actual invoice** (the Cost API omits priority/flex-tier spend).
So an agent must not assume this step was done, must not call a pull "reconciled to
the bill" until it is, and should **offer to run it rather than skip it silently**.
Do this once when adopting the tool, and any time a figure is doubted:

1. Run the report for a single **closed** month, e.g.
   `python3 scripts/anthropic_spend.py --start 2025-01 --end 2025-01`.
2. Take the **"GRAND TOTAL"** the run prints — it's summed from the exact
   pre-rounding allocation, so it carries the cent-exact invariant. (Don't sum the
   CSV `spend_usd` column instead: each key is rounded to the cent independently, so
   the column total can drift a few cents. Add back the printed `unattributed` too.)
3. In the Console, open **Cost** (<https://platform.claude.com/settings/cost>) and
   select the same month, all workspaces, all models; compare the total. Cross-check
   the month's invoice if available.
4. GRAND TOTAL + unattributed should match the Console to the cent (the Cost API is
   that same source). A gap points to a date-boundary/filter mismatch or
   priority/flex-tier spend the Cost API omits (see Caveats) — reconcile before
   trusting per-person figures.

An agent running this skill can walk a user through steps 1–3 and interpret the result.

## Possible enhancements

- **Attribute the Workbench bucket to people.** The usage endpoint accepts
  `group_by[]=account_id` (the human's user-account ID, populated even when
  `api_key_id` is null). Pull that, then map account IDs → names via the
  Admin API members endpoint (`GET /v1/organizations/users`). Adds API calls.
- **Per-workspace view** — add `group_by[]=workspace_id` to the cost pull.
- **Externalize the roster** to `roster.json` if it changes often.

## Distribution

Self-contained folder, run via **Claude Code only** (terminal or Claude Desktop)
— see [README.md](README.md). Teammates either copy it into `~/.claude/skills/`
to install it as a skill, or run Claude Code directly in the folder.

**Not claude.ai:** its skills execute in a network-isolated code sandbox with no
outbound internet, so the `api.anthropic.com` calls can't run there (and an admin
key shouldn't go into a web chat). claude.ai is fine for analyzing an exported
`spend.csv`, just not for the pull.

If one-command team install becomes worth it later, this folder can be wrapped as
a Claude Code plugin (`.claude-plugin/plugin.json` + `marketplace.json` in a
plugin repo, installed with `/plugin`) — optional, not needed for now.
