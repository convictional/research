---
name: goals
description: View the shared goal tree, comment on goals, and create your own goals or sub-goals
allowed-tools: Bash
---

The Goals hierarchy is the team's shared, owned outcomes. It is **seeded with the game's real
goals**: the company MRR / churn / runway targets and one goal per function, each shown with an
**owner** (or unowned) and live status/progress. Reading it shows who owns what and where the
team stands.

View the goal tree:

!`./game goals`

Create your own goal or sub-goal by running this command yourself (via Bash) with real values.
Nest it under an existing goal with `--parent`, and assign an owner with `--owner` (one of
engineering/sales/support/marketing/ops). This authored coordination is up to you — this only
shows the command shape:

`./game goal create "<short goal title>" "<one line on what it means / why it matters>" --parent <goal-id> --owner <function>`

The company/function goals **auto-track their progress from game state** — you can't set their
numbers, but they are **discussion threads**: comment your plan, assessment, and progress on any
goal (`./game goal comment <goal-id> "<note>"`); a comment on a shared goal notifies the team like
a Post. See the goal-update skill for commenting and for updating goals you create.
