# Handoff — "Evaluation Strategies for the Convictional Feed" research experiment

**Written:** 2026-07-02 07:59 EDT, by Claude (working session with a colleague).
**Branch:** `evaluation_strategies_convictional_feed_research`
**Git state:** the notes are committed (`8c0a61c8c first pass of research`, `1634f6174 directory rename`); only `README.md` had uncommitted edits at handoff time. This handoff file is new.
**Purpose of this file:** let a future colleague (a Claude agent or a human) get fully up to speed on this experiment, become an expert in it, and pick up exactly where we left off.

---

## 0. Orientation — read this first

This experiment is a **research log** (a literature + best-practices review with analysis), **not a code experiment**. Its one job: figure out **how to evaluate / compare the algorithms behind the future "Convictional Feed"**, given that we're in a **cold-start** situation (few users, and we'll build the eval ourselves). Over this session we produced five polished notes (`detailed_experiment_notes/01_`–`05_`) that map the whole evaluation landscape, decide what fits our situation, and surface the one genuinely-hard open problem. **There is no feed code to read — the feed doesn't exist yet.** Your job continuing this is research + synthesis + (eventually) proposing a concrete eval-harness design for our setting.

**If you read nothing else:** skim `detailed_experiment_notes/03_comparison-methods-in-detail.md` (the core) and Section 3 + Section 6 of this handoff.

---

## 0.5 Map of the experiment & how to become an expert fast

**Directory layout** (all under `experiments/evaluation_strategies_for_feed_research/`):
```
README.md                     ← top-level orientation + external links (issue, post, human write-up doc)
detailed_experiment_notes/    ← THE domain expertise: polished, cited notes 01–05
  01_primer-how-recommender-feeds-are-evaluated.md
  02_does-evaluation-depend-on-the-algorithm.md
  03_comparison-methods-in-detail.md          ← the core
  04_context-and-timing-dependence.md
  05_modaic-confidence-for-llm-judges.md
claude_notes/                 ← handoffs for a future agent (this file lives here)
```

**This handoff is the *map*, not the territory** — the real domain expertise lives in the five notes. Fast path to expert:
1. Read **`01_`** (the landscape — makes everything else legible), then
2. **`03_`** (the core — every method in detail, with worked examples), then
3. **`02_`, `04_`, `05_`** (each stands alone).

The five notes also trace the **arc of questions we worked through**, which is itself a good learning path: *how are feeds evaluated? (01) → does the evaluation depend on which algorithm we build? (02) → the methods in detail (03) → does every method fit "comparing algorithm A vs B"? (03 §3) → the "right item, wrong moment" timing problem (04) → a real vendor's confidence claim, dissected (05).*

> Naming caveat: this experiment has its **own** `claude_notes/`. The wider repo has *other* `claude_notes/` folders (e.g. the older deep-research-variance work) that appear in memory — don't confuse them with this one.

---

## 1. The big picture — what is this, and where does it fit?

- **Convictional** is a team decision/collaboration platform (goals, decisions, posts, meetings, email). The **"Convictional Feed"** is a planned feature: an *algorithmic feed of recommended actions* that surface personalized next-steps to move users toward their and their org's goals (e.g. "review this proposal," "weigh in on this decision").
- The parent vision doc is **"Convictional Feed - Research"** by Adam (pull it via the Convictional MCP — see §7). It defines the goal and **three "lab-bench" ingredients** needed to research a feed algorithm:
  1. **Outcomes** — "given an event, what happened?" (the reward/scoring signal).
  2. **Dynamic Personal Profiles** — compressed, activity-driven user context (vs. today's static bios).
  3. **Evaluation Strategy** — how to tell if a feed algorithm is any good, in a cold-start setting.
- **This experiment IS ingredient #3, Evaluation Strategy.** The other two are separate threads.
- **Related research threads** (context, not ours to do here): Aryan's decision→goal mapping (judged by a panel against a rubric); Adam's goal-alignment LLM-as-judge work (DSPy, ~140 human-labeled pairs, Spearman correlation); and **Matt's deep-research report-variance study** — which matters most here because it's our **existing internal eval precedent**: a "human evals app" that does **blind pairwise comparison on a 5-point Likert scale**, two quality dimensions, dwell-time tracking, and point-share scoring.
- **External anchors** (all in `README.md`): GitHub issue #8417 (internal GitHub issue, not public); the Convictional discussion post `b23d3abb-705a-456a-8b40-496a9245de3d`; and the **human-authored write-up doc** (internal, not public) that a colleague maintains based on these notes.

---

## 2. The exact problem we're scoping

Design/choose a harness to **compare recommender-feed algorithms head-to-head** ("is algorithm A better than B?"), under three hard constraints:
1. **Cold-start** — few users, little interaction history, no ability to run big live A/B tests yet.
2. **We build it ourselves** — not evaluating tools-to-buy; we care about *methods we can implement*.
3. **Novel item type** — the feed recommends *actions* (proposals, decisions to make), not movies/songs/products, so off-the-shelf datasets and pretrained "AI users" don't know our catalog.

**What the feed's MVP looks like (i.e. what we're evaluating algorithms *for*):** per the parent doc and Eng, the first version is **proposal extraction + routing** — detect a "proposal" from team activity (grounded in Convictional's `event` table) and route it to the right person. The sharp open question Eng framed is *reliably identifying who has decision rights but no inbox visibility* (the Premium Seats post in §7 is the archetype: Roger and Becca had decision interest but no notification signal). Longer term the "action space" broadens across **read / write / execute** actions.

**Scope & non-goals (so you don't wander):** this experiment is *only* ingredient #3, **Evaluation Strategy** — **not** the Outcomes or Dynamic-Profiles ingredients, and **not** the feed algorithm itself. We are **not writing code** (it's a learnings log). Live-experiment methods (A/B, interleaving) are understood but **deferred** — no traffic yet. The deliverable is cited research notes building toward a proposed evaluation approach for our cold-start setting.

---

## 3. What we've established — the substance (the through-lines)

These are the load-bearing ideas that recur across the notes. Internalize these and you're most of the way to expert.

- **The three-layer lens (the central organizing idea — Matt likes this a lot).** Any comparison harness = three separable choices:
  1. **Yardstick** — *what* you score on (a relevance metric? clicks? a human's "was this good?"? estimated retention?).
  2. **Procedure** — *how* you collect judgments (replay logs? live A/B? human side-by-side? LLM judge?).
  3. **Decider** — the *stats* that crown a winner (significance test, Bradley-Terry, etc.).
  Most confusion about evaluation is really people arguing across different layers. Use this to keep discussions clear.
- **The pattern every big platform converges on:** rank items by a **"value model"** (weighted sum of predicted actions, with big *negative* weights for harm) — not by raw accuracy; then layer a **survey-measured "was this worth it?" signal** on top because pure behavioral proxies get gamed (clickbait); evaluate in a **staged funnel** (cheap offline → fast online → long A/B); and remember **offline metrics don't reliably predict online** (Netflix's $1M-Prize model was never shipped).
- **The comparison paradigm — what fits vs. doesn't.** Most methods (offline metrics, off-policy, simulation, A/B, interleaving, human pairwise/best-worst, LLM-judge, Bradley-Terry) are genuine head-to-head comparators. Three things are *not* and sit outside the harness: **long-term/global holdouts** (measure cumulative shipped impact, not A-vs-B), **behavioral testing** (a per-model safety gate), and **survey/surrogate-metric work** (that *builds a yardstick*, it isn't the comparison).
- **Cold-start reality (what's actually usable for us):**
  - **Viable now:** human-judgment comparators (pairwise or best-worst, aggregated with **Bradley-Terry + bootstrap confidence intervals**, with rater-quality weighting and gold/dwell checks) + an **LLM-as-judge** to scale it once calibrated against human labels; a **multi-axis rubric** (right-for-this-person vs. good-in-general); **behavioral/slice tests**; and **offline metric scoring** as a cheap regression guard once we have any logs.
  - **Build day-one or lose it forever:** if we ever want **off-policy evaluation** (estimate a new feed from old logs), the feed must **log its selection odds + inject some randomness from the start** — cannot be retrofitted.
  - **Later, once traffic exists:** interleaving (most sample-efficient) → A/B (copy the SRM/A-A/guardrail safety patterns) → long-term holdouts.
  - **Weak for us:** offline-metric-only verdicts (too little history) and AI-agent simulation (the LLM "users" won't know our novel actions).
- **Human pairwise judgment is the right *family* for cold-start — but do NOT anchor to our existing tool.** The field converges on pairwise human evaluation when you can't run big live tests, and Matt's deep-research **human-evals app** (blind pairwise, 5-pt Likert, dwell-time, point-share) is an existing internal example of it. **Explicit steer from Matt: do not assume we'll just extend that app — he wants us open to *different or better* methods** (e.g. best-worst scaling / MaxDiff, a multi-axis rubric, stronger aggregation like **Bradley-Terry** with bootstrap CIs, an **LLM-judge** to scale throughput). Treat the existing app as *evidence the pairwise family fits our cold-start*, not as a design we're committed to.
- **THE open problem (most novel, unsolved — likely the highest-value next work):** a recommendation's value is **timing/context-dependent** ("right item, wrong moment" — e.g. the same Spotify song is great in one mood and wrong in another), unlike a search result or report whose quality is stable. Judging a recommendation *after the fact* loses the moment. See §6.
- **Modaic case study (skeptical vendor teardown):** a startup claiming an "accurate confidence score" for LLM-judges. Their method (disclosed) = a **linear probe on the judge's hidden state**, trained on labels bootstrapped from self-consistency + a model council + humans. Verdict: the technique is **standard prior art** (their own repo says it's based on a Dec-2025 Meta paper), the "SOTA accuracy" claim is **unsubstantiated**, and it's **white-box only** (can't run on a Claude/GPT judge). The one reusable idea: **distill an expensive confidence signal into a cheap probe**.

---

## 4. The polished notes (index) — `detailed_experiment_notes/`

Each answers a specific question; read `03_` first for depth. (This mirrors the README but with the "why it matters" attached.)

1. **`01_primer-how-recommender-feeds-are-evaluated.md`** — the beginner-friendly landscape: value-model scoring (with a worked example), north-star vs. proxy metrics, offline vs. online, the cold-start toolkit, and the "we already have the right tool" insight. Start here if you're new to recommender evaluation.
2. **`02_does-evaluation-depend-on-the-algorithm.md`** — short: evaluation is *decoupled* from the algorithm's internal implementation but *coupled* to its output-type and how it selects/serves. Introduces the three-layer lens.
3. **`03_comparison-methods-in-detail.md`** — **the core doc.** Every comparison method with how-it-works + worked examples (NDCG, off-policy IPS/DR, interleaving, Bradley-Terry), pros/cons, who uses it, the fit/doesn't-fit analysis, and cold-start commentary.
4. **`04_context-and-timing-dependence.md`** — the "right item, wrong moment" problem: how academia vs. industry handle (or don't) context/timing in evaluation. Academic = context-tagged data + temporal splits; industry = mostly let live A/B absorb the moment; reconstructing a *subjective* past moment is openly unsolved.
5. **`05_modaic-confidence-for-llm-judges.md`** — the vendor case study above (Modaic / confidence for LLM-as-judge).

---

## 5. How to work on this (conventions & preferences — IMPORTANT)

These are hard-won this session; a colleague ignoring them will get corrected.

- **Write at a high-school level.** Matt is sharp on statistics/analytics (he authored the deep-research variance study) but is **not** a recommender-systems expert. So: **expand every acronym, define jargon on first use, lean on everyday analogies, and use small worked examples with real numbers.** This applies to both the notes and chat. (This is a standing, load-bearing preference.)
- **This is a learnings log, not code.** The deliverable is well-organized, well-cited notes — not an implementation. Don't go hunting for feed code; it doesn't exist.
- **Notes conventions:** numbered files in `detailed_experiment_notes/`; **in-text citations _and_ a grouped "Sources" section** with URLs + dates at the end; separate **Academic vs. Industry** where the distinction matters (theory vs. practice); apply a **skeptical framing** to vendor claims (disclosed method vs. marketing).
- **Calibrate research effort to the ask.** Don't reflexively launch another big research round once you already have enough — Matt will explicitly say when he wants more depth ("think very hard / use subagents"). When he does, the method that worked well: **fan out several parallel `general-purpose` subagents** doing web research, each told to return *dense, citation-preserving* findings; then **synthesize and translate to high-school level** yourself. Preserve citations through the whole pipeline.
- **Match response length to the question.** Short question → short answer. Big synthesis → structured + scannable.

> If you're a Claude agent: the two most load-bearing preferences here — high-school-level writing, and calibrating research effort (don't over-research once you have enough; fan-out-then-synthesize when Matt asks for depth) — are also stored in memory as `feedback_feed_research_explain_simply` and `feedback_real_research_for_expert_requests`.

---

## 6. Current state & where to pick up (prioritized open threads)

**Done:** the five notes (committed), the README (Matt maintains it), and this handoff. The map of "how recommender evaluation works and what fits our cold-start situation" is solid.

**Open — likely next work, roughly in priority order:**
1. **The timing/context problem (highest-value, most novel).** How do we evaluate a *context-dependent* recommendation after the fact without it dissolving into rater noise? Concrete leads from `04_`: **give the judge the context** (scenario-based rating — show when it fired, what was happening, the user's recent activity); **log the moment** with every recommendation (timestamp, which decisions/events were live, what the user had/hadn't seen) so context is at least partially observable; **capture an in-the-moment "useful right now?" signal**; and accept that the *subjective* slice (true mood/intent) is unobservable — which is what everyone else accepts too.
2. **Define "a good recommended action" → draft the rater rubric.** This is the highest-leverage cold-start artifact (our analog of Google's "Needs Met" + "Page Quality" axes). Everything in a human/LLM-judge harness hangs off it.
3. **Instrumentation/logging decisions.** Decide *now* to log selection propensities + context, so off-policy evaluation and context-conditioned analysis become possible later. (Can't retrofit.)
4. **Simulation / "alignsim" track.** Whether an LLM-agent user-simulator is worth standing up as a cheap *relative* screen — with the heavy caveat that AI users won't know our novel actions (see `03_` A3 and `04_`).
5. **Move from survey → proposal.** Eventually, turn the method map into a concrete proposed eval-harness design for the Convictional Feed's cold-start reality.

---

## 7. Key source documents & how to pull them (Convictional MCP)

The primary source material lives in Convictional, readable via the **`convictional` MCP server**. Usage pattern: **first** read the resource `resource://convictional/guide`, **then** use `search_content` (previews only) + `get_content` (full text; accepts a `content_id` UUID or a `source_id` like `gid://decide/<Type>/<uuid>`).

Documents worth pulling to get the full context:
- **"Convictional Feed - Research"** (the parent vision doc, by Adam) — `get_content(content_id="66894ca2-2396-4a20-8924-11fbbd78e119")`. The definitive framing (goal, 3 ingredients, MVP shaping).
- **"Claude Code Team Premium Seats"** post — `get_content(source_id="gid://decide/Post/bdddb1a3-cec5-4d3e-b0eb-9f2dabbd5618")`. The running example of an ideal feed item (a proposal→decision→implementation; Roger literally says "this is what I'd want a feed to surface"; illustrates the "decision rights but no inbox visibility" gap).
- **"Evaluating deep research report quality variance"** (Matt) — `get_content(content_id="532acfa2-6908-4eb9-b4f7-68605e41b725")`. Our existing internal eval precedent (the human-evals app: pairwise 5-pt Likert, dwell-time, point-share).
- **"Research Ideas"** (Adam) — `get_content(content_id="15edcf97-9c0c-4d46-a96a-76067ed6c9b5")`. Situates this within the broader research program (decision logs, outcomes, goal-alignment judge).
- **Human write-up doc** — internal, not public (a colleague's human-authored synthesis based on these notes).

---

## 8. Glossary (jargon used across the notes)

- **CARS** — Context-Aware Recommender Systems (context = time, mood, location, device, session…).
- **LLMaaJ / LLM-as-judge** — using an LLM to grade/score outputs.
- **Yardstick / Procedure / Decider** — the three layers of a comparison harness (§3).
- **NDCG, Recall@k, MRR, MAP** — top-N ranking metrics (see `03_` A1 for worked examples).
- **A/B test** — split users, show variant A vs. B, measure which does better live.
- **Interleaving** — blend two rankers' picks into one list for the *same* user; ~100× more sample-efficient than A/B for ranking.
- **Off-policy / counterfactual evaluation (IPS, SNIPS, Doubly-Robust, "replay")** — estimate a new algorithm's live performance from old logs; requires logged selection **propensities**.
- **Bradley-Terry** — turns pairwise "A beat B" judgments into a ranking with confidence intervals (what Chatbot Arena uses); fit via logistic regression.
- **CUPED** — variance-reduction trick that roughly halves the users an A/B test needs.
- **SRM (Sample Ratio Mismatch)** — chi-square check that your split came out as designed; catches broken experiments.
- **Conformal prediction** — wraps a model to give a distribution-free coverage *guarantee* (needs labeled calibration data).
- **AUROC / ECE** — how well a confidence score separates right from wrong / how *honest* it is (does "80% sure" mean right 80% of the time).
- **White-box vs. black-box** — needs the model's internals (hidden states/logprobs) vs. only its text output (an API judge like Claude is black-box).
- **In-situ / EMA (Ecological Momentary Assessment)** — collecting feedback *in the moment* rather than afterward.
- **Cold-start** — little/no user data yet (our situation).
