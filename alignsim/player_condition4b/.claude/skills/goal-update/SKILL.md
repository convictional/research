---
name: goal-update
description: Comment on any goal, or report status and progress on a goal you created
allowed-tools: Bash
---

Two ways to keep the goal tree current — run either yourself (via Bash) with real values:

**Comment on any goal** (including the seeded company/function goals) — attach your plan,
assessment, or progress as a note. The goal's own number stays computed from game state; your
note rides alongside and, on a shared goal, notifies the team like a Post:

`./game goal comment <goal-id> "<short plan / progress note>"`

**Update a goal you created** — set its status and progress directly. Status is one of
`on_track`, `at_risk`, `off_track`; progress is a number (1.0 = target met). Each update is
appended to the goal's history so the team can see how it moved:

`./game goal update <goal-id> on_track 0.5 "<short progress note>"`

The seeded MRR / churn / runway / function goals auto-track their progress from game state, so you
`comment` on them rather than `update` them — `update` is for goals you create.
