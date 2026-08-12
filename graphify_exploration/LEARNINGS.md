# Experiment learnings

## The good

- **Confidence-tagged edges** (`EXTRACTED` / `INFERRED` / `AMBIGUOUS`) make the graph self-auditing — every relationship tells you whether graphify *found* it or *guessed* it.
- **The interactive HTML viz** is genuinely useful at small/medium scales — color-coded community nodes, dashed/solid edges by confidence, and shaded hyperedge hulls.
- **Query memory auto-enriches the graph.** Answers from `/graphify query`, `path`, and `explain` are saved to `graphify-out/memory/` and fold back into the graph on the next `--update`.
- **Skill-as-orchestrator design is elegant.** The heavy logic (subagent fan-out, cache merge, retry) lives in a Markdown skill file, not Python — the AI assistant *is* the runtime, which keeps the package small and extensible.

## The bad

- **Process to generate and query has real friction**: Since graphify is using a skills markdown, the markdown specifies what commands to run for a given executable action. So, to generate a graph, a series of python code blocks needs to be run, each with their own command request in Claude Code. Similar for querying the graph - bash commands are executed and requested in Claude Code. It works, but the step-by-step approval flow is a drag. Also, I'm hesitent to allow all permissions, so I was stuck babysitting the graphify executions.
- **Often less accurate than direct Claude questioning.** Graphify gave wrong info multiple times, over just a handful of queries - so the error rate is high. Asking these questions directly in Claude Code might take more time to process but would produce much better answers.
- **Answers leak graph jargon** — responses overuse "degrees of separation," "cohesion," and "community N" instead of plain domain-language explanations.
- **Token cost is high and opaque.** Daily usage easily passes 100M tokens between graph builds and queries, with little visibility into where it went.
- **Degrades fast at scale.** On the 13k-node `app/` build, Leiden over-fragmented into 1,878 mostly-singleton communities and the HTML viz auto-aggregated to a community-level view, losing node-level interactivity. At a certain point, communities are just labelled as "Community X" with no description, so it is not useful to analyze.
- **Overall**, for this codebase and this set of questions the graph-mediated route did not beat asking a coding agent directly. That is a finding about the approach at this corpus size, not a judgement on the project.

## Learnings related to knowledge graph generation

Comparing graphify's architecture against the prior knowledge graph work at Convictional, the following could be worth considering:

- **Hybrid AST + LLM extraction**
    - AST = Abstract syntax Tree - a tree-shaped representation of a source file's structure. A parser walks the code and produces a tree where each node is a syntactic construct (function definition, class, import statement, function call, etc.)
    - Our previous work have been mainly pure LLM; graphify pairs a free, deterministic AST pass (calls, imports, inheritance) with the LLM pass (semantic edges, rationale, cross-file links). Could cut costs and improve structural precision.
- **Categorical + continuous confidence with a discrete-value rubric**
    - Three-tier tag (`EXTRACTED` / `INFERRED` / `AMBIGUOUS`) plus a fixed set of allowed scores (0.95 / 0.85 / 0.75 / 0.65 / 0.55, never 0.5) avoids the bimodal-collapse failure mode LLMs often hit when given continuous confidence ranges.
    - Extracted: Literally explicit in the source. The LLM (or AST) saw the actual evidence: an import, a dbt ref(), a function call, a citation, a "see §3.2"
  reference. confidence_score = 1
    - Inferred: Not explicit, but a reasonable deduction. E.g., two functions probably share a data structure based on shape, or a class implements a documented protocol without saying so. Discrete confidence_score from {0.95, 0.85, 0.75, 0.65, 0.55} depending on how strong the inference is
    - Ambiguous: Uncertain but worth flagging rather than dropping. confidence_score 0.1–0.3. The skill instructs the LLM to mark these instead of inventing a high-confidence edge it isn't sure about. Surfaced explicitly in the report's "Suggested Questions" so a human can verify
- **Hyperedges as first-class group concepts**
    - Storing "all N functions in this auth flow" as one hyperedge preserves group meaning that pairwise-only graphs destroy.
- **Skill-as-orchestrator pattern**
    - Encoding the pipeline as a Markdown skill the AI assistant executes (rather than a Python notebook/script) makes orchestration auditable, makes swapping the semantic backend trivial, and integrates naturally with agentic workflows.
