---
name: anthropic-spend-report
description: Generate a per-person, per-month Anthropic API spend report for a Claude Console organization, splitting Claude Code vs developer vs service usage and reconciling exactly to the Cost API total. Use when someone asks for Anthropic/Claude API spend or cost by user, team, or API key, monthly usage cost, a cost/usage export, or "how much are we spending on Claude". Requires an organization Admin API key.
---

# Anthropic spend report

Pulls org usage + cost from the Anthropic **Admin API** and produces a CSV of
**spend per person per month**, split into `claude_code` / `dev` / `service`.
Dollars come from the Cost API and are allocated to API keys by token share, so
the per-key totals **reconcile exactly to the Cost API total** (no pricing table).
Matching that total to your *actual invoice* — which can include priority/flex-tier
spend the Cost API omits — is a one-time human check (Layer 3 below); don't
describe a pull as "reconciled to the bill" until that anchor is done.

## Prerequisite: an Admin API key

The script needs an **Admin API key** (`sk-ant-admin01-…`) — distinct from a
normal `sk-ant-api…` key. An org **admin** creates one at
<https://platform.claude.com/settings/admin-keys> (Console orgs have full Admin
API access; no scope picker). Then, in the shell the agent will run in:

```bash
export ANTHROPIC_ADMIN_KEY=sk-ant-admin01-...
```

If the user hasn't set it, stop and walk them through the link above — do not
proceed without it, and never echo the key value.

**Where it runs:** the pull needs outbound network access *and* the admin key, so
run it from **Claude Code or a local/CI shell** — not inside a claude.ai chat,
whose code sandbox has no internet egress (the API call would fail). On claude.ai,
use this skill only to analyze a CSV that was already exported elsewhere. Never
paste an admin key into a web chat.

## Quick start

```bash
python3 scripts/anthropic_spend.py                              # last 12 full months
python3 scripts/anthropic_spend.py --start 2024-01 --end 2026-06 --out spend.csv
```

Output CSV columns: `month, user, key_type, api_key_name, spend_usd,
uncached_input_tokens, cache_read_input_tokens, cache_creation_input_tokens,
output_tokens`. A per-person summary table prints to stderr. Pivot the CSV by
`user`, `key_type`, or `month` as the user asks. (`api_key_id` is used internally
for the allocation but intentionally not written out.)

## Verify it's working

Three layers. **Never silently skip Layer 2 or Layer 3.** Run them by default; if
there's a real reason to skip (rate limit, already-anchored invoice), say so and let
the user decide — don't just omit them and imply the numbers are confirmed.

1. **Offline tests — always, free, instant (no API/key).** Cover the cost→key
   allocation (reconciliation + remainder-to-largest + non-negativity + deterministic
   tie-break), month windows, month-arg validation, and token parsing. Run before
   trusting a pull and after any edit (from the skill root):

       cd experiments/anthropic-spend-report && python3 -m unittest discover -s tests

2. **Live self-test — run it on every real pull (2 API calls).** Pulls one month
   and asserts the endpoints still return the expected fields, the pull reconciles to
   the Cost API total, unattributed cost is negligible (< $1.00), and implied unit
   prices are sane (catches a silent cents↔dollars change). No stored numbers:

       python3 scripts/anthropic_spend.py --self-test 2026-05

   Run it **first**, before a large pull (both share the ~1-req/min limit — self-test
   then big pull, not two big pulls back-to-back). Skipping it is a decision for the
   user to make, not a silent default.

3. **Invoice reconciliation — the only proof the dollars are actually right
   (one-time per org, human-in-the-loop).** Layers 1–2 and a run's own
   "unattributed $0.00" line prove *internal* consistency with the Cost API — **not**
   that the Cost API matches your real invoice (it omits priority/flex-tier spend).
   **Do not assume this was ever done.** Unless the user has confirmed it's anchored
   for this org *in this conversation*, surface it and offer to walk them through
   REFERENCE.md → "Confirming the numbers are right." Never call a pull "reconciled
   to your bill" until this is done — say "reconciles to the Cost API total" instead.

## Workflow

> This lives in a protected repo — **never push to `main`**. Any code change (a
> fix, or a roster edit) goes through a **branch + PR**.

1. Run the offline tests (`cd experiments/anthropic-spend-report && python3 -m unittest
   discover -s tests`) — instant, no API. **If any fail, do not quietly fix them:** report the failure and ask the
   user how they want to proceed. An ops user will usually open an issue in the
   project's tracker; a maintainer may fix it on a branch + PR.
2. Confirm `ANTHROPIC_ADMIN_KEY` is set (see above).
3. **Run the live self-test** (`--self-test <recent-full-month>`, 2 API calls) before
   the real pull — Layer 2 above. **Do not skip it silently.** If you're skipping it
   (e.g. to protect the rate limit ahead of a huge pull), tell the user and get their
   OK first.
4. Pick the date range (default: last 12 full months). Stay ≤ a couple of years
   per run; the Admin API is rate-limited (~1 req/min sustained), so **don't re-run
   back-to-back** — the script throttles and retries, but repeated full passes 429.
5. Run the script. The CSV lands in the git-ignored `output/` folder next to the
   skill by default (override with `--out`).
6. Read back the stderr summary + CSV and answer the user's actual question
   (totals by person, Claude Code vs dev, month-over-month, top keys, etc.). When you
   report figures, say the pull **reconciles to the Cost API total** — **not**
   "reconciled to your bill" unless Layer 3 has been done for this org. If it hasn't,
   say so plainly and **offer to run the one-time invoice reconciliation** (Layer 3 /
   REFERENCE.md → "Confirming the numbers are right"); never present the numbers as
   invoice-confirmed when they're only Cost-API-consistent.
7. Check the **"⚠ could not be categorized"** list the run prints. For each flagged
   key that's clearly a person or known project, propose the fragment to add to
   `PEOPLE` / `BUCKET` in the ORG CONFIG — then **confirm with the user and make the
   edit on a branch + PR** (a roster add is low-risk, but `main` is still protected).

## Org configuration

`scripts/anthropic_spend.py` has an **ORG CONFIG** block near the top:
`PEOPLE` (person → key-name fragments), `BUCKET` (non-person key → category),
and `KEY_NAME_PREFIX_CLAUDE_CODE`. Edit these for your org / as people and keys
change. The mapping is heuristic from key names — review the summary before
trusting per-person totals.

## How it works & caveats

Methodology (cost-allocation join), known limitations (priority/flex tiers and
Workbench/no-key usage), and extension notes are in [REFERENCE.md](REFERENCE.md).
