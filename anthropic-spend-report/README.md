# anthropic-spend-report

A **Claude Code skill** that reports **Anthropic API spend per person per month**
for our Claude Console org, split into `claude_code` / `dev` / `service` usage.
Dollars come from the Admin Cost API and are allocated to API keys by token
share, so per-person totals **reconcile exactly to the Cost API total** (no pricing
table to maintain). Confirming that total against your actual invoice is a quick
one-time check — step 5 of the walkthrough below.

Runs locally via **Claude Code** (terminal or Claude Desktop). Not claude.ai —
see *Why not claude.ai* below.

## First time? Start-to-finish (~10 min)

New to this? Here's the whole path (you'll need Python 3 — standard library only,
nothing to `pip install`). Steps 1–2 are one-time setup; step 5 is a one-time trust
check; after that it's just step 4 whenever you want a report.

1. **Get an Admin API key.** You need an `sk-ant-admin01-…` key — *not* a normal
   `sk-ant-api…` key. If you're an org admin, create one at
   <https://platform.claude.com/settings/admin-keys> (takes a minute). If you're
   not an admin, ask one to make you one.

2. **Set it in your shell** — the same shell Claude Code runs in. Never paste it
   into a chat (it would persist in the conversation):

   ```bash
   export ANTHROPIC_ADMIN_KEY=sk-ant-admin01-...
   ```

3. **Get the skill** — either install it for every session or just open Claude Code
   in this folder. Both options (and a run-it-yourself command) are under
   *Use it* below.

4. **Ask for a report.** In Claude Code, say e.g. *"run the Anthropic spend report
   for last quarter."* It runs the offline tests, a quick 2-call live self-test,
   then the pull, and gives you a CSV plus a per-person summary. (Prefer to run it
   yourself? See the command under *Use it*.)

5. **Reconcile once against the Console — do this the first time.** The totals
   reconcile to the Cost API automatically, but that's not the same as your
   **invoice** until you check it once. Run a single closed month:

   ```bash
   python3 scripts/anthropic_spend.py --start 2025-01 --end 2025-01
   ```

   Take the printed **GRAND TOTAL**, open **Cost** in the Console
   (<https://platform.claude.com/settings/cost>) for that same month (all
   workspaces, all models), and confirm they match — they should agree to the cent.
   If they don't, see [REFERENCE.md](REFERENCE.md) → *Confirming the numbers are
   right* (usually a date-range or priority/flex-tier caveat). Claude Code can walk
   you through this.

6. **You're set.** Re-run for any range whenever you like, and keep the roster
   current as people and keys change (*Keep the roster current* below).

## Use it (pick one)

**A. Install as a Claude Code skill** (available in every session, incl. Claude Desktop):

```bash
cp -r anthropic-spend-report ~/.claude/skills/
```

Then in Claude Code (with the env var set) just ask, e.g.
*"run the Anthropic spend report for 2024-01 to 2026-06"* — it loads the skill
and runs it.

**B. Run Claude Code over this folder** (no install): open Claude Code in this
directory with the env var set and ask for the spend report. Or skip the agent
entirely and run it yourself:

```bash
python3 scripts/anthropic_spend.py --start 2024-01 --end 2026-06 --out spend.csv
```

Output: `spend.csv` (`month, user, key_type, api_key_name, spend_usd` + 4 token
columns) plus a per-person summary printed to the terminal. Defaults to the last
12 full months if no dates are given.

## Keep the roster current

Open `scripts/anthropic_spend.py` → the **ORG CONFIG** block (`PEOPLE`, `BUCKET`)
and add new people / keys. If a key shows up under `uncategorized`, it just needs
a fragment added there. Full methodology and caveats are in
[REFERENCE.md](REFERENCE.md); trigger/usage notes for the agent are in
[SKILL.md](SKILL.md).

## Why not claude.ai (web)?

claude.ai runs skills in a **network-isolated** code sandbox, so it can't reach
`api.anthropic.com` to pull the data — the script would fail on the API call.
And an admin key shouldn't be pasted into a web chat (it persists in the
conversation). So the pull runs **locally via Claude Code**. You can still upload
the resulting `spend.csv` to claude.ai afterward for ad-hoc analysis.
