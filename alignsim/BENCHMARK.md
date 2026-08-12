# Goal Alignment Benchmark: Simulation-Based Evaluation

See [ALIGNSIM_IDEA.md](ALIGNSIM_IDEA.md) for the simulation scenario chosen.

## The Problem

Goal alignment — whether the actions an organization takes actually advance its stated objectives — is the central problem Convictional is trying to solve. To make progress on that problem, we need a way to measure it. And to measure it credibly, we need a benchmark: a repeatable evaluation protocol that the field can use to compare approaches.

The obvious first attempt is to build a labeled dataset of (goal, action) pairs, where human experts rate whether a given action is aligned with a given goal. This is the structure behind most NLP benchmarks — SQuAD for reading comprehension, MMLU for general knowledge, etc. Curate the pairs, collect expert labels, publish the dataset, let people benchmark against it.

We tried this. It doesn't work for goal alignment. Here's why.

## Why Ground Truth Labels Fail

### The Inter-Rater Disagreement Problem

In our internal experiments measuring goal alignment on a bipolar scale (-3 to +3), we observed an intraclass correlation coefficient (ICC) of approximately 0.54 on difficult cases — cases where alignment is ambiguous, contextual, or requires strategic judgment. Pairwise agreement between individual raters ranged from 0.40 to 0.83.

An ICC of 0.54 means that barely half the variance in ratings is attributable to actual differences between items. The rest is rater-specific interpretation. On the hard cases — which are precisely the cases that matter for a benchmark — experts simply do not agree on what "aligned" means.

This isn't a problem of rater quality or rubric design. It reflects a genuine property of goal alignment: reasonable people with full context can look at the same (goal, action) pair and reach different conclusions about whether that action advances that goal. Alignment is not a fact about the pair; it's a judgment that depends on assumptions about strategy, time horizon, second-order effects, and organizational context.

A benchmark built on fixed labels would enshrine one group's interpretation as "correct." Any system that happened to share that group's assumptions would score well; any system with a different but equally valid strategic worldview would score poorly. The benchmark would be measuring agreement with the labelers, not alignment quality.

### The Deeper Issue: Alignment Is Trajectory-Dependent

The ICC problem points to something more fundamental. The question "is this action aligned with this goal?" is not well-posed in isolation. Whether an action is aligned depends on what happens next — what other actions are taken, how the environment responds, what information arrives later.

Consider a concrete example: a company has a goal to increase revenue by 20% this quarter. An employee spends a week building an internal tool instead of doing direct sales work. Is that aligned? It depends entirely on whether the tool accelerates sales in weeks 2 through 12. You cannot label the (goal, action) pair without knowing the trajectory.

This means the right question is not "is this action aligned?" but "does acting on this information, over time, lead to better goal outcomes?" And you can only answer that by observing trajectories — by watching what happens when different prioritization strategies play out.

## The Simulation-As-Judge Approach

If alignment can only be measured by outcomes, we need an environment where we can observe outcomes. Specifically, we need an environment where:

- Goals and starting conditions are fixed and repeatable
- Multiple strategies can be tested against the same scenario
- Outcomes are unambiguous and measurable
- The environment is complex enough that goal alignment is non-trivial

This is a simulation.

In a simulation-based benchmark, "correctness" is not determined by human labels. It's determined by results. Did the team that prioritized signal X over signal Y actually achieve better goal outcomes? The simulation answers that question empirically, across hundreds or thousands of runs.

This sidesteps the ICC problem entirely. We don't need raters to agree on whether an action is aligned, because the simulation tells us whether it *was* aligned — by showing us what happened.

### What We're Actually Measuring

The benchmark measures **strategic prioritization under ambiguity**: given a set of organizational goals, a stream of information and decision opportunities, and constrained resources, which prioritization strategy leads to the best goal outcomes?

This is distinct from existing benchmarks in important ways:

- **Not task completion.** GDPBench and similar benchmarks ask "can you do this task well?" We ask "given 10 possible tasks, which ones should you do?"
- **Not instruction following.** We don't test whether a system does what it's told. We test whether it figures out what *should* be done.
- **Not single-turn.** Alignment unfolds over time. A single decision can only be evaluated in the context of the trajectory it produces.

## Connecting to Convictional

The simulation is the arena. The thing we're actually benchmarking is the **coordination substrate** — the system through which agents receive information, make sense of goals, communicate priorities, and decide what to do next.

### The Four Conditions

We propose starting with four experimental conditions, holding the simulation environment, goals, agent capabilities, and random seeds constant across all four:

**Condition 1: Single LLM.** One language model receives the full goal set and full information stream, and selects actions directly. No coordination needed — this is the baseline. It isolates raw prioritization capability from any coordination overhead or benefit.

**Condition 2: Agent harness.** A single LLM operates through a tool-use harness (e.g. Claude Code or similar), with access to the simulation's API. The agent can plan, execute multi-step reasoning, and interact with the simulation programmatically, but there is still a single decision-maker. This tests whether agentic scaffolding improves prioritization.

**Condition 3: Agent team with generic tools.** Multiple specialized agents (e.g. one per role in the simulation) coordinate through general-purpose tools — a shared Slack channel, a spreadsheet for goal tracking, a shared file system. Each agent has a constrained role and private information. They must communicate to align. This tests multi-agent coordination through tools not designed for goal alignment.

**Condition 4: Agent team using Convictional.** Same multi-agent setup, same roles and constraints, but the coordination substrate is Convictional. Goals, sub-goals, content, and alignment signals flow through Convictional's system. This tests whether a purpose-built goal alignment system produces measurably better outcomes than generic coordination tools.

### Why This Design Is Publishable

The factorial design means the benchmark is not "Convictional wins." It's "here is the effect size of different coordination structures on agent team goal attainment." That's a genuinely novel research question with implications beyond any single product.

The benchmark can also be run with human teams, human-AI hybrid teams, or different LLM providers — each a publishable extension.

**On the obvious conflict of interest.** This benchmark was designed by the vendor whose product is
condition 4, and a reader should discount it accordingly — no amount of careful design makes a
vendor-authored benchmark a neutral one. The mitigations we can offer are structural rather than
rhetorical: conditions 1–3 are the informative comparisons and stand on their own without
condition 4; the simulation, goals, agent capabilities and random seeds are held constant and are
in this repository; and the harness is built so a third party can substitute a different
coordination substrate for condition 4 and re-run it. If you want to know whether the result
holds, run it against your own tool — that is the only version of this benchmark whose answer
should carry weight.

## Simulation Environment Criteria

Any simulation used in this benchmark must satisfy the following:

1. **Goal-hierarchical complexity.** The environment must be rich enough to require goals and sub-goals, with natural tension between competing objectives. A flat action space with a single objective is insufficient.

2. **Business-adjacency.** The action and goal space must be believably or explicitly business-relevant. Participants (human or AI) should be making decisions that map to real organizational challenges: resource allocation, prioritization, risk management, information routing.

3. **Deterministic repeatability.** Starting conditions and goals must be consistently repeatable across runs. All stochastic elements must be controllable via random seeds so that different strategies can be compared against identical scenarios.

4. **Programmatic interaction.** The simulation must expose an API or equivalent interface for selecting actions. No GUI-only interaction, no pixel-level observation.

5. **Discrete time steps.** The simulation must not run in real-time. Steps are discrete, and the system under test can take as long as it needs between steps. This ensures that latency differences between coordination substrates (e.g. Slack API round-trips vs. direct function calls) do not confound the results.

6. **Open source.** The simulation environment must be open-source or freely available so that other researchers can reproduce results, extend the benchmark, and contribute new scenarios.

7. **Training data resistance.** The simulation must not use real-world historical data that is likely to appear in LLM training corpora. Optimal strategies should not be memorizable from pre-training. This means either procedurally generated scenarios, synthetic data, or sufficiently novel environment dynamics.
