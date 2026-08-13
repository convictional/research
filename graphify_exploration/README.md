# Exploring the Graphify tool

**Authors:** Matt Chequers, Adam McCabe

This "experiment" is meant to be an exploration of the [Graphify](https://github.com/safishamsi/graphify) tool.

The main goal is to explore the usage of the tool, and document any learnings about it that could be useful to us.
- If we might be able to use it in our day-to-day work, i.e. productivity
- If we can learn anything related to (knowledge) graph construction

## Tl;dr conclusions and learnings

> **Scope note.** These are one engineer's notes from a short evaluation, in mid-2026, of a
> fast-moving open-source tool against one particular codebase and one particular set of
> questions. They are a point-in-time assessment of **graph-RAG over a large codebase as an
> approach**, not a considered verdict on the project or its maintainers, and the tool has almost
> certainly moved on since. Read the technical findings; discount the impressions.

**Did graph-RAG beat asking a coding agent directly, for our questions?** No. The invocation
path had real friction (subagent dispatch needed supervision and repeated permission prompts),
answers were often less accurate than asking Claude Code directly against the same code,
responses tended to surface graph-internal jargon rather than answers, token cost was high and
hard to attribute, and quality degraded as the corpus grew. Those last three look like properties
of the approach at this corpus size, not implementation details.

**Worth considering for our own knowledge graph work**, though:

- **Hybrid AST + LLM extraction** — pair a free, deterministic AST pass (calls / imports / inheritance) with the LLM pass (semantic edges, rationale). Could cut costs, improves structural precision over our pure-LLM attempts.
- **Hyperedges as first-class group concepts** — keep "all N functions in this auth flow" as one record rather than flattening to pairwise edges.
- **Skill-as-orchestrator pattern** — encode the pipeline as a Markdown skill the AI executes (not a Python script). Auditable, swappable backend, agent-native.

Full notes in [LEARNINGS.md](LEARNINGS.md).

## Getting started

All commands are run from this directory (`experiments/graphify_exploration/`).

- `make install` — creates the local `.venv/`, installs `graphifyy` via `uv sync`, and copies graphify's `SKILL.md` to `.claude/skills/graphify/SKILL.md`. Run once.
- `make refresh` — pulls the latest `graphifyy` (graphify ships daily) and re-copies `SKILL.md`. Run when you want to update.
- `make clean` — removes `.venv/`, `graphify-out/`, and `.claude/skills/graphify/`. Tracked files (the empty `.claude/skills/.gitkeep` marker, Makefile, etc.) are preserved.
- `make help` — lists available targets.

After `make install`, activate the venv and launch Claude Code from this directory:

```bash
. .venv/bin/activate
claude
```

You don't necessarily have to start with a fresh Claude Code session. As long as you launch Claude Code from this directory, you can resume sessions from there and pick the skill up.

The `/graphify` skill will be available. The venv must be active so graphify's inline `python3 -c "..."` invocations resolve to the local interpreter (otherwise the skill installs `graphifyy` into system Python via `--break-system-packages`).

Output from any `/graphify` run lands in `graphify-out/` (gitignored).

### How this differs from graphify's docs

The official quickstart is `pip install graphifyy && graphify install`, which installs `graphifyy` into your system Python, copies `SKILL.md` into `~/.claude/skills/graphify/`, and appends a registration block to `~/.claude/CLAUDE.md`. There is no clean uninstall.

This setup keeps everything project-local instead:

- `graphifyy` lives in a local `.venv/` managed by `uv`, not system Python.
- `SKILL.md` is copied into `.claude/skills/graphify/` here, not `~/.claude/skills/`. Claude Code auto-discovers it from this directory.
- `~/.claude/CLAUDE.md` is never touched.
- `make clean` fully tears down the experiment with no global residue.

We never run `graphify install`. The trade-off: you must launch Claude Code from this directory with the venv active so graphify's inline `python3` invocations resolve to the local interpreter.

## Graphify usage

Once `/graphify` is loaded in your session, invoke it with a path to a folder.

The [graphify repo docs](https://github.com/safishamsi/graphify#common-commands) have a list of common commands.

### `/graphify [directory]`

Runs the full pipeline on the given directory:

1. **Detect** files (code, docs, papers, images, video) and report a corpus summary.
2. **Extract** structurally via tree-sitter AST (code) and semantically via dispatched subagents (docs, papers, images).
3. **Build** an in-memory NetworkX graph, **cluster** it with Leiden, and **analyze** for god nodes, surprising connections, and hyperedges.
4. **Export** to `graphify-out/`:
   - `graph.html` — interactive vis.js visualization
   - `GRAPH_REPORT.md` — narrative with god nodes, surprising connections, suggested questions
   - `graph.json` — canonical NetworkX graph (node-link form)

Path is relative to the directory Claude was launched from. Since you launch from `experiments/graphify_exploration/`, point at sibling experiments with `../<name>`:

```
/graphify ../common
```

Heads up: graphify's documented soft ceiling is ~5,000 nodes (the HTML viz fails past that on a single page) and Leiden fragments badly on monorepos. Start small.

### `/graphify query "<question>"`

Asks a question against the existing graph. Claude does a graph traversal (BFS by default), pulls the relevant subgraph, and answers using only what's in it.

```
/graphify query "what are the god nodes"
/graphify query "how does X reach Y" --dfs
/graphify query "..." --budget 1500
```

- `--dfs` traces a specific chain (good for "how does X get to Y" questions); BFS is the default and is better for broad context.
- `--budget N` caps the answer at N tokens.
- Each answer is saved to `graphify-out/memory/` so the next `--update` can fold it into the graph as a node.

**Caveat**: BFS seeds on semantic similarity to the question wording. Generic words ("overview", "metric") can pull irrelevant subgraphs. If you want a precise lineage trace between two known nodes, use `/graphify path "A" "B"` instead.

### `/graphify path "A" "B"`

Finds the shortest path between two named nodes. Claude prints the hop sequence (with each edge's relation + confidence) and explains in plain language what each hop means.

```
/graphify path "build_prompt" "aembed_query"
```

- Names must match existing node labels — fuzzy matching is limited.
- Returns "no path" if the two nodes are in disconnected components (common on small graphs with isolates).
- Edges are undirected by default; rerun the build with `--directed` to preserve source→target.
- More reliable than `query` when you know both endpoints — useful for lineage traces ("how does this source reach this mart?").

### `/graphify explain "<node_name>"`

Plain-language explanation of a single node and its immediate neighbors.

```
/graphify explain "fct_user_daily"
```

Prints the node's source location, type, community, degree, and connection list (relation + confidence per edge). Claude then writes a 3–5 sentence summary citing source locations.

- Best for "what is this thing and what touches it" questions.
- Like `path`, the name must match an existing node label.
- Saved to `graphify-out/memory/` for future `--update` enrichment.

## Exploration work

1. [What is graphify?](WHAT_IS_GRAPHIFY.md)
2. [First impressions running graphify](FIRST_IMPRESSIONS.md)
3. [Running graphify on `data/`](RUNNING_ON_DATA_DIRECTORY.md)
4. [Running graphify on `app/`](RUNNING_ON_APP_DIRECTORY.md)
5. [Learnings from this experiment](LEARNINGS.md)
