# Gemini vs. Claude in Prod

**Author:** Adam McCabe

I tested whether Gemini 3 Pro can replace Anthropic Claude as the LLM provider across prod's three main AI-powered services: research reports, meeting summaries, and custom inbox views. This document covers how I ran the experiment, what I found, and what an engineering team would need to know before migrating.

> **Note on this published version.** The experiment ran against a live production
> deployment. Employee and customer names, verbatim meeting summaries, internal metrics, and
> links to private PRs and screen recordings have been removed. Where a real example was
> load-bearing for a finding, it has been replaced with a **synthetic example** that preserves
> the shape of the contrast being illustrated; those are labelled inline. The methodology and
> the model comparison are unchanged.
>
> **Sample sizes are small.** The headline research result rests on 39 ratings from 9 raters
> across 5 prompts, and the meeting result on 12 ratings from 6 raters across 2 meetings. These
> are directional signals from a small team dogfooding its own product, not a benchmark. Treat
> percentage splits as indicative only — a handful of votes moves them substantially.

## Background

Prod uses a single LLM provider (currently Anthropic Claude) for all AI features. The provider is configurable via `settings.llm_provider`, and all LLM calls flow through `infra/llm.py`. The question: if I switch that setting to Gemini, what breaks, what degrades, and what stays the same?

## How I Tested

### Infrastructure

I created a dedicated test environment (`ENV=geminiinsteadofclaude`) with the seed database as well as the OnboardingMailboxSync backfill to 200 of my most recent emails. This gave me a very apples-to-apples comparison against my production experience. For Research and Meeting summaries, both Gemini and Claude were run over the seed data, while for custom views, I ran them over my production inbox and a recently refreshed local inbox.

The `settings.llm_provider` enum controls which provider is used globally. I toggled this per-run in the benchmark script, and ran the app side-by-side (Gemini on the right, prod/Anthropic on the left) for the custom views comparison.

### Benchmark Script

`scripts/benchmark_llm_providers.py` runs the same prompts^1 through both providers using the full production pipeline (not mocks). It forces `JobRunner.INLINE` so the complete job chain (research -> iteration -> sub-queries -> final report, or transcript -> title -> summary) executes synchronously within one script run.

For research:
```bash
ENV=geminiinsteadofclaude make script ARGS="scripts/benchmark_llm_providers.py \
  --research-prompts scripts/benchmark_prompts.yaml \
  --user-email researcher@example.com"
```

For meeting summaries:
```bash
ENV=geminiinsteadofclaude make script ARGS="scripts/benchmark_llm_providers.py \
  --meeting-ids scripts/benchmark_meetings.yaml"
```

The script outputs JSON results and a CSV formatted for our blind A/B human evaluation app (`tmp/human_eval_meeting_cases.csv`). Evaluators see two answers in randomized order with no provider labels and pick which they prefer (or "same").

1. For the initial trials of Meeting Summaries and Research Reports, I ran the exact same prompts, but found Gemini to consistently under-perform (more on that below) and had to tune the prompt for Gemini. Not unexpected, but I was suprised at how different their behaviour is against the same prompt.

### Custom Views

No automated benchmark here — I ran the app locally with each provider and did a side-by-side screen recording comparison across three inbox views (daily driver, by goals, urgent/important). Same emails, same bio, same prompts. (The recording shows a real inbox and is not published.)

## Results

### Research Reports

**Synthetic benchmarks (8 prompts, automated):**

For this first trial, I used 8 Research prompts from a synthetic prompt set a colleague had built during earlier Topics and Communities work. I also used the baseline production prompt for Gemini in this trial to see how it would behave.

| Metric | Anthropic | Gemini |
|--------|-----------|--------|
| Mean latency | 78.6s | 122.2s |
| Source citations per response | 6-10 | 4-6 |
| Gap flagging | Frequent | Rare |

Gemini appeared ~55% slower in this trial. However, later investigation (see [Latency](#latency)) revealed this was likely due to Instructor retrying on Pydantic validation failures rather than Gemini inference being slower — when retries were eliminated with tuned prompts, the gap narrowed to ~10%. Both surface largely the same source documents, but Anthropic cites more sources and explicitly flags what it couldn't find. Gemini reads more like an annotated bibliography; Anthropic reads more like an analyst's memo. Based on these findings, I worked with Claude to tune the prompt for Gemini to try and illicite more Anthropic like writing.

**Human A/B eval (5 prompts, 9 raters, 39 ratings):**

For the human a/b evals, I reused 5 research prompts from a colleague's recent Agentic Research explorations, as the team was already familiar with them and they are real production queries that people had actually run. I also used the tuned Gemini prompt for this trial. See the appendix for the prompts used and additional detail on the evals.

| Provider | Votes | % |
|----------|-------|---|
| Anthropic | 23 | 59% |
| Gemini | 11 | 28% |
| Same | 5 | 13% |

The design principles prompt was 8-0 Anthropic (Gemini over-expanded scope). Excluding that outlier, the remaining 4 prompts were 15-11-5 — still Anthropic but not as lopsided. The strongest signal was that writing quality mattered more than information quality, as both surfaced similar information except in some cases: raters preferred Anthropic's shorter sentences, answer-first structure, and tighter scoping even when Gemini surfaced useful information Anthropic missed (although these are limited feedback comments).

### Meeting Summaries

**Automated benchmark (2 meetings):**

For this trial, I used 2 recent all-hands meeting transcripts from our seed database (Jan 13, 2026 and Jan 20, 2026). Note that I would have used more, but these were the only two team meetings with transcripts (not just a summary). Similar to the Research trials above, I at-first used the production prompt for Gemini, and also saw similar patterns.

| Metric | Anthropic | Gemini |
|--------|-----------|--------|
| Latency | 47-49s | 94-172s |
| Key points per summary | 7-10 | 2-6 |
| Direct quotes | Throughout | Sparse |

Anthropic appeared 2-3.5x faster locally in this trial. As with research, this gap was largely caused by Instructor retrying on validation failures — a later run with tuned prompts and retry instrumentation showed zero retries and latencies within ~10% (47s vs 52s). Gemini produces substantially less detailed summaries, covering the main topics but missing secondary discussion threads and providing less context on why points matter. Based on these learnings, I tuned the prompt for Gemini to use in the human evaluations. More details on this and the following trial can also be found in the appendix.

**Human A/B eval (2 meetings, 6 raters, 12 ratings):**

These are limited results as we only had 6 raters at the time of writing, but are still indicative. For this trial, I used the tuned prompt for Gemini to try and illicit more Anthropic-esque writing styles.

| Provider | Votes | % |
|----------|-------|---|
| Anthropic | 6 | 50% |
| Gemini | 4 | 33% |
| Same | 2 | 17% |

Closer than research, though on only 12 ratings. Jan 13 went 5-1 Anthropic (completeness and metrics cited). Jan 20 went 3-1-2 Gemini — the only prompt across all evals where Gemini won, partly driven by a factual error in Anthropic's output (likely a transcription artifact, not a model issue).

### Custom Inbox Views

Side-by-side comparison across three views (screen recording not published — it shows a real inbox):

- **Daily driver inbox feed**: Very similar. Same emails, same categories, minor label differences.
- **By goals**: Anthropic noticeably better. Gemini lumped engineering work under one broad goal; Anthropic broke it into granular goal-to-email mappings.
- **Urgent/important matrix**: Roughly the same. Constrained categories level the playing field.

Pattern: Gemini matches Anthropic on structured/constrained tasks but fell behind on the goals view, which may be due to the number of goals to match to. This could indicate slightly weaker attention across the context window, or could just be random from the nature of LLMs.

### Combined Results

| Benchmark | Anthropic | Gemini | Same |
|-----------|-----------|--------|------|
| Research (human eval) | 59% | 28% | 13% |
| Meetings (human eval) | 50% | 33% | 17% |
| Custom views (qualitative) | Slight edge | Comparable | — |

## Engineering Considerations

### Prompt Tuning Required

Gemini does not respond to the same prompts the way Claude does. I created provider-specific prompt overrides for both research (`command.gemini.md.jinja`) and meeting summaries (`generate_summary.gemini.md.jinja`). The pattern is a conditional in the job code:

```python
template = (
    "extract_meeting_metadata/generate_summary.gemini.md.jinja"
    if settings.llm_provider == LLMProvider.GEMINI
    else "extract_meeting_metadata/generate_summary.md.jinja"
)
```

Key tuning patterns that helped:
- **Explicit writing style section** with BAD/GOOD examples — Gemini defaults to list-of-facts writing and needs concrete examples of the prose style you want.
- **Analytical framing** — instructions like "draw conclusions, identify patterns, synthesize across sources" pushed Gemini toward interpretation rather than just reporting.
- **Coverage expectations** — "a 60-minute meeting typically has 8-12 substantive topics" gave Gemini a target that prevented shallow 3-4 point summaries.
- **Anti-generalization rules** — instructions of the form "do not generalize to 'a client' when the transcript names the account" addressed Gemini's tendency to abstract away specifics.

Even with tuning, Gemini's instruction following was less stable than Claude's. The same prompt produced meaningfully different output structures across runs — one run might produce the expected `## Summary / ### Key Points` format while another flattened everything into a single narrative paragraph. Claude was more consistent.

### Instructor / Structured Output Compatibility

This was the biggest technical blocker. Instructor's JSON mode with Gemini has reliability issues:

- **Datetime coercion**: Gemini returns datetime values as raw strings. Pydantic's strict validation in Instructor's iterable wrapper rejects them. Fields like `starts_at: datetime | None` fail with `Input should be a valid datetime`.
- **Enum handling**: Similar coercion issues with enum types.

The validator solved the problems I saw, and I was able to run all the LLM jobs without noticeable errors. However, I also explored some alternatives:

1. **Google native structured output** (medium effort) — server-side schema enforcement, no Instructor dependency, but also no retry logic from what Claude saw in the docs, meaning we would need to roll our own
2. **PydanticAI** (high effort) — automatic validation retries, but a paradigm shift from our current `LLM` class. A colleague explored this separately in a prototype PR (internal, not public).


### Latency

Early benchmarks showed Gemini ~55% slower on research and 2-3.5x slower on meeting summaries. After instrumenting Instructor's retry hooks, we discovered the gap was largely driven by Instructor retrying on Pydantic validation failures (e.g. datetime coercion, enum handling) — not by slower inference. Instructor defaults to `max_retries=3`, so each validation failure could silently add 1-3 additional round-trips to the Gemini API.

A follow-up benchmark with tuned prompts showed zero retries and latencies within ~10%:

| Meeting | Anthropic | Gemini |
|---------|-----------|--------|
| All-hands (Jan 20) | 46.96s | 53.32s |
| All-hands (Jan 13) | 47.53s | 51.36s |

This means latency is likely a non-issue for a migration, provided structured output validation is clean. The Instructor compatibility issues (see above) are the real risk — if validation failures creep back, latency will degrade silently.

## Takeaways

**Gemini is a viable but not equivalent replacement.** It produces usable output across all three services. No feature is completely broken. But quality appears lower — particularly on tasks requiring synthesis, depth, and nuanced categorization. It is likely we could prompt tune around this, but would need to do further testing to see if there is a ceiling. The team mostly preferred Anthropic's output in blind evaluations.

**The gap is in writing quality, not information quality.** Gemini surfaces largely the same facts and sources. Where it falls behind is in how it presents them: shorter summaries, less connective prose, weaker scope discipline, less transparent about gaps. Raters chose Anthropic for "better business writing" and "more complete" even when Gemini had useful information Anthropic missed. This is good news, and points to this largely being a prompting problem - I ran limited prompt tuning as to not overload the team with evals, but would want to iterate further than the two versions I tried.

**Structured output needs engineering work.** Instructor compatibility issues are real so we would need to choose how to proceed there. The validation and coercion of string responses may be fine, but also feels fragile.

**Conciseness vs completeness is a personal preference.** Some raters consistently preferred Gemini's brevity; others consistently preferred Anthropic's thoroughness. A migration would trade completeness for conciseness — whether that's acceptable depends on which users' preferences you weight more heavily.


# Gemini Experiment Appendix

## Appendix A: Research Report Details

### Prompts Used

**Synthetic benchmark (8 prompts):**
1. "What are the current issues, blockers, and resolution strategies for our product documentation technical infrastructure that are preventing timely updates and feature alignment?"
2. "What are the current technical implementation patterns and user experience design principles we're applying to maintain clear human-AI boundaries in our AI-powered features?"
3. "What are the primary database performance issues we've encountered in production and what incident response patterns have emerged from our crisis management efforts?"
4. "What are the recommended approaches for balancing administrative efficiency with granular security controls when designing role-based permission inheritance models for our user group system?"
5. "What are the key milestones and outcomes from our recent B2B SaaS strategic pivot activities, including customer discovery findings, market repositioning decisions, and organizational changes required to execute the transition?"
6. "What are the specific technical documentation requirements and evidence standards that SR&ED reviewers expect for AI and decision intelligence projects?"
7. "What event format variations and attendee engagement tactics should I test to maximize meaningful professional connections and measure networking quality beyond simple attendance numbers?"
8. "What is our current employee equity compensation structure and what are the key considerations for optimizing our stock option and equity grant programs for talent retention?"

**Human A/B eval (5 prompts):** these were real production queries run against a real corpus, so
they are paraphrased here to remove names.
1. "Can you recommend updates I can make to my bio in the app based on things I actually work on, make decisions about, and the nature of my work based on the context you have?"
2. "What's the history of our design principles? Specifically looking for the original principles for 2024 relating to the product and in the time since. Everything that a new senior design engineer would need to get up to speed."
3. "How did I decide on the current ICP? Can you point me to the decisions and discussions I had surrounding the ICP? I'm especially curious on any tradeoffs I made, and the why behind our decision"
4. "What did [a named colleague] work on this summer?"
5. "I'm writing a software capitalization analysis for our financial audit. I need to determine our major phases of development in 2025. Can you propose 2-4 developmental phases we were in with our product last year?"

### Human Eval: Per-Prompt Results

| Prompt                    | Anthropic | Gemini | Same | Ratings |
|---------------------------|-----------|--------|------|---------|
| Bio recommendations       | 4         | 3      | 0    | 7       |
| Design principles         | 8         | 0      | 0    | 8       |
| ICP decision              | 3         | 3      | 2    | 8       |
| A colleague's summer work | 5         | 3      | 1    | 9       |
| Capitalization phases     | 3         | 2      | 2    | 7       |

### Human Eval: Selected Rater Comments

**Bio recommendations (4-3):** Close to a tie but comments favored Anthropic's structure. One rater noted Anthropic was "more actionable and rationale is provided after the real answer." Another would "combine both and edit the specifics out personally," valuing Anthropic's professional framing but Gemini's mentorship angle.

**Design principles (8-0):** The most decisive result. Gemini over-expanded into product vision, design ops methodology, and future concepts. One rater said the Gemini answer "stretches too far." Raters wanted the specific principles, not a survey of every design-adjacent decision.

**ICP decision (3-3-2):** The closest prompt and a genuine tie. Gemini surfaced specific rejected alternatives and who argued for them — exactly what the prompt asked for. Anthropic matched on writing quality. One rater: "I would probably combine both of these answers. they give different perspectives to the same question."

**Capitalization phases (3-2-2):** The clearest case of Gemini having a feature Anthropic lacks but losing on writing quality. Gemini included "Accounting Context" sub-sections mapping activities to standard capitalization terminology. One rater preferred Anthropic overall ("nearly identical to what I put in my accounting memo") but explicitly called out Gemini's accounting context as a strength.

### Themes from Research Evals

**Writing quality > information quality.** The strongest signal across all prompts. Raters preferred Anthropic's shorter sentences, answer-first structure, and tighter scoping even when Gemini surfaced useful information Anthropic missed. One rater's summary: "better business writing, shorter sentences, better formatting."

**Scope discipline separates the providers.** The 8-0 design principles result is the strongest data point: when users ask a specific question, staying focused wins decisively. Gemini's tendency to expand scope reads as unfocused rather than thorough.

**"Combine both" is a recurring wish.** Multiple raters across different prompts said they'd combine both answers. Anthropic writes better but Gemini sometimes finds different angles.

**Structure preferences are personal.** Some raters consistently valued Gemini's categorical structure (sections, time breakdowns). Others consistently valued Anthropic's sentence-level clarity. These appear to be stable individual preferences, not prompt-dependent.

---

## Appendix B: Meeting Summary Details

### Meetings Benchmarked

Two company all-hands meetings, January 13 and January 20, 2026 — the only two team meetings in
the seed database with full transcripts rather than just a summary. Roughly a dozen attendees
each. Topics spanned quarterly goal review and roadmap, account onboarding and renewal status,
user metrics, engineering progress, and go-to-market planning.

Attendee lists, account names, and the summaries themselves are not reproduced here. The
comparison below uses a **synthetic pair of summaries** written to preserve the property the real
pair demonstrated — a ~3x length difference and a large gap in retained specifics — over invented
content.

### Summary Comparison (synthetic illustration)

**Anthropic-style summary (opening):**
> The team reviewed the newly launched goals feature, now loaded with the quarter's OKRs that will
> guide all work this quarter. The CEO walked through the goals page, showing outcomes organized
> around customer adoption, revenue, and operations. The analytics lead shared metrics showing
> that the newest account had completed onboarding with three users exploring the product, though
> none had sent email yet; the team debated whether this signals an awareness problem or a desire
> problem. Engineering reported goals management moving into refinement, custom inbox views going
> live, and an MCP server now available for internal experimentation. Go-to-market outlined a new
> cadence of spinning up weekly campaigns, with this week aimed at two named vertical segments.

**Gemini-style summary (opening):**
> The team met to review the new goals feature and discuss sales plans. The CEO led the call and
> asked everyone to use the new goals tool themselves. The team shared data on a new client who is
> set up but not sending emails yet. Go-to-market laid out a plan to launch new sales campaigns
> weekly to find paying customers. The group also discussed technical updates for AI tools.

The Anthropic opening is roughly 3x the length and retains the specifics that make the summary
actionable — which users, the "awareness versus desire" framing, the exact campaign targets. Gemini
covers the same ground at a higher level but drops the context that explains why each topic
matters. That was the consistent pattern across both real meetings.

One incidental observation from the real pair: Gemini resolved speakers to full names where the
Anthropic output used first names only. For a summary of an internal meeting that is a minor
difference; for anything published or shared outside the company it is a meaningful one.

### Human Eval: Per-Meeting Results

| Meeting            | Anthropic | Gemini | Same | Ratings |
|--------------------|-----------|--------|------|---------|
| All-hands (Jan 13) | 5         | 1      | 0    | 6       |
| All-hands (Jan 20) | 1         | 3      | 2    | 6       |

### Human Eval: Selected Rater Comments

**Jan 13 (5-1, strong Anthropic):**
- (Anthropic): "Feels more complete, and as though it calls out more important issues (e.g. email sending as activation gap); I also like that answer 2 quoted actual metrics vs just the story."
- (Anthropic): "I lean towards answer one because it is a bit higher-level and not just quotes. But tbh I'd be fine with either answer."
- (Gemini): "Almost the same, but I prefer the Challenges & Risks in answer 1"

**Jan 20 (1-3-2, Gemini edge):**
- (Gemini): "I found answer two to be more concise."
- (Gemini): "Answer 2 says a particular person was a new team member, which is incorrect." (Note: likely a transcription artifact, not a model error — both providers work from the same raw transcript.)
- (Anthropic): "I prefer answer 1 if I had missed the meeting, that would have given a better holistic view of what was discussed."

### Themes from Meeting Evals

**Challenges & Risks is the differentiator.** The most cited reason for preferring one summary was the Challenges & Risks section — it requires the most synthesis and judgment, so it's where model differences show most.

**The gap is narrower than research.** Meeting summaries are more constrained (same transcript, same format template), which reduces the surface area for quality differences. Both providers produce usable meeting notes.

**Factual errors may be transcription, not model.** One vote against Anthropic was for apparently hallucinating that an existing colleague was a new team member. Both models are downstream of the same transcript quality — errors in the raw input propagate regardless of provider.
