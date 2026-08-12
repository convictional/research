# Dynamic Research Depth/Breadth

An experiment to dynamically adjust research tree depth and breadth based on topic complexity, with adaptive early branch termination. Instead of using fixed (depth=2, breadth=3) for all research, the system classifies topics by complexity and allocates more resources to harder questions — while letting the LLM prune unproductive branches early to control cost.

**Status**: Complete — 4 trials over ~2 weeks. Final version (v4) achieved 74% dynamic win rate on info quality across 42 blind ratings from 9 raters (70% excluding the researcher). Adaptive branch termination was the key differentiator, improving both info quality and style relative to earlier versions. Ready for production consideration.

**Links**: Experiment branch [internal branch, not public] | Draft PR #6622 [internal PR, not public]

## Hypothesis

Research topics vary widely in complexity — "What did I work on last week?" needs far less exploration than "How did we decide on the current ICP and what research led to that decision?" Fixed depth/breadth parameters either under-explore complex topics or waste resources on simple ones.

If we classify topic complexity and allocate proportional search budget, complex topics should produce better answers without degrading simple ones. And if we let the LLM decide per-branch whether to continue deeper, we can set generous ceilings without paying the cost of always reaching them.

## How it works

### 1. Complexity classification

Before research starts, the LLM classifies the topic as low, medium, or high complexity based on:
- **Temporal span** — days vs months vs quarters
- **Scope** — single person/event vs cross-team/cross-domain
- **Synthesis required** — factual retrieval vs analytical synthesis

Each level maps to (depth, breadth) parameters:

| Complexity | Depth | Breadth | Theoretical Max Queries |
|-----------|-------|---------|------------------------|
| Low | 2 | 3 | ~4 (same as production) |
| Medium | 3 | 4 | ~12 |
| High | 3 | 6 | ~18 |

Production defaults are (2, 3).

### 2. Tree-position aware prompts

Query generation and review prompts are told their position in the tree:
- **Depth 0**: Cast a wide net, cover distinct angles
- **Middle depths**: Build on learnings, fill gaps, follow promising threads
- **Final depth**: Targeted, highest-value gaps only

### 3. Adaptive early branch termination (v4)

After reviewing each query's results, the LLM returns `should_continue_researching` — a boolean indicating whether continuing deeper on that branch would meaningfully improve the answer. Max depth is a ceiling, not a target.

The review prompt receives cumulative stats (total queries run, total learnings collected) so the LLM can assess global coverage, not just its local branch.

### 4. Dynamic completion check

`Research.is_completed` was updated to handle early-terminated trees. When `use_dynamic_parameters` is enabled, it checks that all iterations are complete (at any depth), rather than requiring iterations to exist at the max depth level.

## Benchmark setup

6 research topics ranging from simple ("What did I work on last week?") to complex ("Our Ops team is writing a software capitalization analysis, can you help me understand what engineering worked on last quarter?"). Each topic runs through both baseline (production defaults) and dynamic arms. The benchmark script uses concurrent asyncio job execution matching production behavior.

Human evaluators compare responses in a blind A/B setup — the eval app randomizes whether the dynamic or baseline answer appears as Answer 1 or Answer 2.

### Topics

1. Can you recommend updates I can make to my bio based on what I've been working on?
2. What's the history of our design principles and how have they evolved?
3. How did we decide on the current ICP and what research led to that decision?
4. What did sofia work on this summer?
5. Our Ops team is writing a software capitalization analysis, can you help me understand what engineering worked on last quarter?
6. What did I work on last week? *(control — classified low, same params as baseline)*

## Trial history

### V1: Dynamic parameters (researcher eval, 5 cases)

**Changes**: LLM complexity classifier, dynamic (depth, breadth) mapping. Initial parameters: low (2,3), medium (2,4), high (3,4).

**Info quality**: Dynamic 3, Baseline 2.
**Style**: Dynamic 0, Baseline 2, Same 3.

Dynamic found more information on breadth-heavy topics (bio updates, ICP, eng quarter) but was consistently wordier and less focused. The tension: better at *finding* information but worse at *staying focused*.

### V2: Tree-position awareness (researcher eval, 5 cases)

**Changes**: Added tree-position awareness to query generation and review prompts (wide early, focused late). Light succinctness scaling in final synthesis prompt. Fixed benchmark baseline to use production defaults.

**Info quality**: Dynamic 3, Baseline 2 (unchanged).
**Style**: Dynamic 2, Baseline 3 (improved from 0-2).

Style gap narrowed — went from 0 dynamic wins to 2, including a strong win on the ICP case. Most cases felt very similar, with low-strength wins dominating. Hallucination risk emerged: dynamic's ICP case erroneously included pre-pivot content.

### V3: Parameter tuning (no eval)

Experimented with wider breadth configurations: low (3,3), medium (4,5), high (5,10). The high configuration caused the benchmark to hang — with inline job execution, each query spawns its own child iteration, producing 98 iterations and 144 queries for a single topic. Reduced to low (2,3), medium (3,4), high (3,6).

Also discovered a critical difference between inline and concurrent job execution: inline runs each query sequentially (every query spawns a child), while production runs queries concurrently (only the last-completing query spawns a child). Switched benchmark to asyncio execution to match production.

### V4: Adaptive branch termination (team eval, 9 raters, 5 cases)

**Changes**: Added `should_continue_researching` to review models. LLM decides per-branch whether to continue. Cumulative stats (total queries, total learnings) passed to review prompts for global coverage awareness. Fixed `Research.is_completed` to handle early-terminated trees. Switched benchmark to concurrent asyncio execution.

**Query counts (v4)**:

| Topic | Complexity | Baseline Q/I | Dynamic d/b | Dynamic Q/I |
|-------|-----------|-------------|-------------|-------------|
| Bio updates | medium | 4/2 | 3/4 | 7/3 |
| Design principles | high | 4/2 | 3/6 | 10/3 |
| ICP decision | high | 4/2 | 3/6 | 10/3 |
| Sofia's summer | medium | 4/2 | 3/4 | 7/3 |
| Eng quarter | high | 4/2 | 3/6 | 9/2 |
| Last week (control) | low | 4/2 | 2/3 | 4/2 |

Early termination working: eng quarter ran 9 queries instead of theoretical ~18 for (3,6).

## Results (v4, team eval)

42 info quality ratings from 9 blind raters across 5 cases. Rater 1 is the researcher — while blinded to answer positions like everyone else, they have deep topic familiarity and may be unconsciously biased. Results reported both ways.

### By case

| Case | Dynamic Wins | Baseline Wins | Avg Signed Strength |
|------|-------------|---------------|---------------------|
| Design principles | 7 | 2 | +3.0 |
| ICP decision | 5 | 2 | +2.7 |
| Bio updates | 6 | 3 | +1.7 |
| Sofia's summer | **8** | **0** | **+7.8** |
| Eng quarter | 5 | 4 | +3.0 |
| **Total** | **31** | **11** | **+3.4** |

**74% dynamic win rate** (70% excluding the researcher, where eng quarter becomes a 4-4 split).

When dynamic won, avg strength was 4.8. When baseline won, avg strength was 2.3 — dynamic wins are roughly 2x the magnitude of baseline wins.

### By rater

| Rater | Dynamic | Baseline | Cases Rated |
|-------|---------|----------|-------------|
| Rater 1 (researcher) | 5 | 0 | 5 |
| Rater 2 | 5 | 0 | 5 |
| Rater 3 | 4 | 1 | 5 |
| Rater 4 | 4 | 1 | 5 |
| Rater 5 | 4 | 1 | 5 |
| Rater 6 | 3 | 0 | 3 |
| Rater 7 | 3 | 2 | 5 |
| Rater 8 | 2 | 3 | 5 |
| Rater 9 | 1 | 3 | 4 |

6 of 9 raters favored dynamic on a majority of their cases.

### Across versions (info quality)

| Version | Researcher Dyn/Base | Team Dyn/Base | Team Win Rate |
|---------|-------------------|---------------|---------------|
| V1 | 3/2 | — | — |
| V2 | 3/2 | — | — |
| V4 | 5/0 | 31/11 (26/11 ex-researcher) | 74% (70% ex-researcher) |

## Key lessons

**What worked:**
- **Adaptive branch termination** was the biggest lever. V1/v2 showed marginal info gains (3-2) with consistent style losses. V4's early stopping produced 5-0 info (researcher) / 31-11 (team) while also reversing the style gap — less bloat from unproductive branches means more coherent synthesis.
- **Complexity classification** is reliable — the LLM consistently maps topics to reasonable buckets. The control case ("What did I work on last week?") was always classified low and got baseline-equivalent parameters.
- **Tree-position awareness** (v2) directionally improved style even though info stayed the same. It set up the foundation for v4's branch-level decision-making.
- **Concurrent benchmark execution** is essential. Inline execution causes every query to spawn a child (sees itself as the only completed query), producing wildly different tree shapes than production. Any benchmark must match production's job execution model.

**What didn't work / limitations:**
- **Static parameter tuning** (v3) is a dead end. The discrete low/medium/high buckets are too coarse — (5,10) explodes the tree while (2,3) may under-explore. Adaptive stopping solved this by making max depth a ceiling rather than a target.
- **5 cases is thin** — enough to see a directional signal but not enough for statistical significance. The eng quarter case was 5-4 (4-4 ex-researcher), and individual rater tendencies may reflect topic familiarity as much as answer quality.
- **Retrieval quality bounds the ceiling** — one rater noted the eng quarter case was "a miss" for both arms. When the underlying search doesn't surface the right content, more queries just add noise. Depth/breadth improvements compound with retrieval quality but can't substitute for it.
- **Hallucination risk scales with queries** — v2's ICP case pulled in pre-pivot content. More search results mean more opportunities to surface irrelevant material. Adaptive stopping mitigates this by pruning unproductive branches, but doesn't eliminate it.

## Code changes (on experiment branch)

All changes are gated behind `use_dynamic_parameters` so production behavior is unchanged.

| File | Change |
|------|--------|
| `app/jobs/research.py` | Complexity classifier, `should_continue_researching` on review models, cumulative stats in review, gated iteration spawning |
| `app/models/collaboration/content.py` | `is_completed` handles early-terminated trees |
| `app/prompts/research/assess_complexity.md.jinja` | Complexity classification prompt |
| `app/prompts/research/generate_queries.md.jinja` | Tree-position awareness |
| `app/prompts/research/review_results.md.jinja` | Coverage-aware follow-up strategy + should_continue |
| `app/prompts/research/review_no_results.md.jinja` | Should-continue guidance |
| `app/prompts/research/findings.md.jinja` | Succinctness scaling |
| `scripts/benchmark_dynamic_depth_research.py` | A/B comparison benchmark |

See the experiment branch [internal branch, not public] and draft PR #6622 [internal PR, not public] for full implementation details.
