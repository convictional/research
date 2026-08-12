# What is Graphify?

A high-level account of [safishamsi/graphify](https://github.com/safishamsi/graphify) — what it is, and what it is not.

## TL;DR

Graphify is a **Claude-Code-style "skill" plus a Python CLI** that walks a folder of mixed-media files (source code, markdown, PDFs, images, video, URLs) and emits a **local, file-based knowledge graph** of the entities and relationships it found, along with an interactive HTML visualization, a markdown report, and an Obsidian vault. It runs entirely on your machine, stores its graph in NetworkX + JSON (no Neo4j or other database required), and is designed to give AI coding assistants a compact, persistent map they can query instead of re-reading raw files every session.

It is MIT-licensed and is written in Python (>=3.10).

## What it is

### Elevator pitch (verbatim from the graphify README)

> "A Claude Code skill. Type `/graphify` in Claude Code – it reads your files, builds a knowledge graph, and gives you back structure you didn't know was there."

### The problem it claims to solve

1. **Token-efficient context for AI assistants.** The README's headline benchmark claims **~71.5x fewer tokens per query** than re-reading the raw files (52-file Karpathy-style mixed corpus), with up to 100.8x on specific questions. The graph becomes a compact, persistent index agents can hit instead of stuffing raw source into context.
2. **The "raw folder" problem.** Karpathy-style `/raw` directories of papers, screenshots, code snippets, and notes. Graphify aims to surface the latent structure in such piles.
3. **Multi-agent code-evolution.** A `--watch` flag and a post-commit git hook keep the graph current as multiple agents write code in parallel.

### How it works (high level)

Pure-function, seven-stage pipeline (no shared state, no side effects outside `graphify-out/`):

```
detect() → extract() → build_graph() → cluster() → analyze() → report() → export()
```

- **Code** is parsed locally via tree-sitter ASTs (Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, C#, Kotlin, Scala, PHP).
- **Docs / PDFs / images / video** are read by the calling AI assistant (Claude, Codex, etc.) — the skill dispatches subagents to extract entities and relationships in a structured JSON schema. Graphify itself does **not** ship an LLM SDK; it leans on the host AI assistant.
- **Graph** is built in-memory with NetworkX, clustered with Leiden (via `graspologic`), analyzed for "god nodes" and "surprising connections," and persisted to JSON.
- **Cache** is keyed by SHA256 so re-runs only reprocess changed files; `--update`, `--watch`, and post-commit hooks all leverage it.

### Outputs (`graphify-out/`)

- `graph.json` — canonical NetworkX graph (node-link form).
- `graph.html` — interactive vis.js visualization with shaded community hulls.
- `GRAPH_REPORT.md` — narrative: god nodes, surprising connections, suggested questions.
- An Obsidian vault, optional `--wiki` for agent crawling, and optional `--svg`, `--graphml`, and Neo4j Cypher exports.
- A `cache/` of SHA256-keyed extractions for incremental rebuilds.

The expected workflow is to commit `graphify-out/` to git so a team starts with a shared map.

### Distinctive concepts

- **Confidence-tagged edges** — every edge is `EXTRACTED` (literal), `INFERRED` (deduced), or `AMBIGUOUS`. Surfaced verbatim in reports. The repo's pitch: *"honest about what it found vs guessed."*
- **God nodes** — high-degree concepts, with synthetic file-hub and stub-method nodes filtered out.
- **Surprising connections** — composite-scored cross-source/cross-community edges, with a plain-English "why" for each.
- **Hyperedges** — 3+ nodes participating in one concept, rendered as shaded hulls.
- **Skill-as-orchestrator** — much of the heavy logic (subagent fan-out, cache merge, retry) lives in a Markdown skill file (`graphify/skill.md`), not Python. The Python package is a toolbox; the AI assistant is the runtime.
- **Worked examples as evidence** — `worked/{httpx, mixed-corpus, karpathy-repos, ...}/` ship in-repo, each with its own `review.md` self-scoring the run candidly. This is the project's primary "evals."

### Tech stack

- **Python 3.10+**, single flat package (`graphify/`).
- **NetworkX** in-memory graph, JSON persistence.
- **graspologic** for Leiden community detection (lazy-imported).
- **tree-sitter** + 13 grammars for code AST extraction.
- **vis.js** for the interactive HTML viz (generated as a single self-contained file).
- Optional extras (`pip install graphifyy[all]`): `mcp`, `neo4j`, `pdf` (pypdf, html2text), `watch` (watchdog), `svg`, `office`, `video` (faster-whisper for local transcription).
- PyPI name is `graphifyy` (the `graphify` name is being reclaimed); CLI/skill command remain `graphify`.

### Install & run

```bash
pip install graphifyy
graphify install                     # writes the skill to ~/.claude/skills/graphify/
# inside Claude Code:
/graphify .                          # build a graph for the current folder
/graphify ./docs --update            # incremental rebuild
/graphify query "what connects auth to database?"
/graphify path "UserService" "DatabasePool"
graphify hook install                # post-commit auto-rebuild
```

### Maturity and community

- ~42.5k stars, ~4.6k forks, ~450k PyPI downloads, 78–210 open issues (active triage).
- Multiple releases per day in the v0.6.6 → v0.7.5 range, latest as of this writing **v0.7.5**.
- **Single-author project** (Safi Shamsi, AI Research Engineer based in London; medical-AI / KG background, MICAD 2025). No corporate backing visible.
- MIT-licensed. Contribution model emphasizes **adding worked examples** (`worked/{slug}/` with an honest `review.md`) as the most trust-building PR type.

## What it is NOT

This is the part that matters most for evaluating fit.

1. **Not a graph database, not Neo4j.** The runtime store is NetworkX-in-memory, persisted to JSON. Neo4j is an *export target* (`--neo4j` / `--neo4j-push`), not the engine. There is no Cypher query layer, no transactions, no remote graph store.
2. **Not a hosted service.** It is a local CLI plus a Markdown skill installed into your AI assistant. There is no SaaS, no managed offering, no auth layer. (The author runs a related product — Penpax / `graphifylabs.ai` — but graphify proper is purely OSS local tooling.)
3. **Not a runtime / dynamic-analysis tool.** ARCHITECTURE.md is explicit: *"processes codebases as static snapshots; real-time runtime behavior is not captured."* No tracing, no profiling, no execution-graph capture.
4. **Not true semantic understanding.** The honest `worked/httpx/review.md` admits structural extraction alone captures *"at most 25–30% of the interesting relationships in a Python codebase."* Issue #198 documents that the LLM-driven semantic layer runs **in parallel** to the AST layer rather than fusing with it — node IDs don't always match, so the semantic pass tends to behave as an exploratory side-graph.
5. **Not built for very large codebases out of the box.** Documented ceilings:
   - vis.js HTML viz fails past ~5,000 nodes (issue #447: 8,333 nodes — too large).
   - Leiden fragments into thousands of micro-communities on monorepos (issue #52: 410-file iOS project → 7,414 communities with near-zero cohesion).
   - Post-commit hooks broken on 50k-LOC Django (issue #541).
   - No built-in hierarchical aggregation yet (#265).
6. **Not a finished product.** v0.7.5, default branch `v7`, hundreds of open issues, multiple releases per day. Positioning is acknowledged-unresolved (issue #245: "coding-assistant skill or broader knowledge graph?"). Best characterized as a **mature research-grade prototype with an active OSS community**.
7. **Not a headless / CI-first tool.** Full pipeline expects an interactive AI-assistant slash command. A `graphify extract` headless mode exists but issue #698 confirms it's incomplete; semantic extraction in particular is not driven from CI.
8. **Not a replacement for grep / LSP / IDE navigation.** The bundled `AGENTS.md` instructs agents to *consult the graph first* and **fall back to source**. It is an orientation/overview aid, not a code-search engine.
9. **Not language-complete.** 13 languages with first-class tree-sitter grammars; recurring requests for Solidity, GDScript, Fortran. SvelteKit/NestJS/Astro projects hit AST-extractor bugs (#691, #692, #701, #700).
10. **Not a call-graph / program-analysis tool.** Inheritance-to-external-base edges are silently dropped on small corpora; call edges are `INFERRED` regex-ish heuristics in a second pass, not a sound static analysis.

## Known limits and failure modes

Concrete numbers, thresholds, and observed failures. Some come from upstream docs/issues; some from running graphify on Decide's own `experiments/common/` and `data/` directories.

### Hard size thresholds

- **HTML viz fails past ~5,000 nodes.** Issue #447 documents an 8,333-node project flagged "too large." Above this, graphify auto-aggregates to a community-level view, but you lose node-level navigation.
- **Corpus warning at >200 files OR >2,000,000 words.** The skill pauses and asks you to pick a subdir before proceeding.
- **Token-reduction benchmark only runs above 5,000 words.** Below that, graphify itself prints `"Corpus is ~X words — fits in a single context window. You may not need a graph."` (We hit this on `experiments/common/`, 1,507 words.)
- **Useful lower bound is ~6 files.** Per `worked/karpathy-repos/review.md`, `god_nodes()` and `suggest_questions()` silently return empty on smaller corpora.

### Semantic-capture ceiling

- From `worked/httpx/review.md`: *"the AST-only pipeline has a fundamental ceiling… captures at most 25–30% of the interesting relationships in a Python codebase."*
- The AST and LLM-driven semantic layers run **in parallel, not fused** (issue #198). Node IDs don't always match between the two passes — the semantic layer often behaves as an exploratory side-graph rather than enrichment.
- The httpx review notes: *"All 14 inheritance edges in `exceptions.py` are silently dropped"* and `Response.raise_for_status() calls HTTPStatusError()` is missing entirely. Inheritance to external bases (`Exception`, `ABC`) and many call relationships disappear.

### Observed failure modes on Decide's `data/` corpus

- **SQL and YAML aren't parsed as code.** Tree-sitter has 13 grammars; SQL is not one. Our 91 "code" files yielded only **8 AST nodes** — the rest came from the semantic LLM pass.
- **Same entity from one file can appear as two nodes.** `fct_user_daily.sql` produced both `fct_user_daily` and `int_user_daily` (the LLM treated a CTE as a peer of the model).
- **76 isolated nodes** (no edges) on a 158-file run — orphan doc concepts whose IDs didn't match the models they describe.
- **Leiden over-fragments small/heterogeneous corpora.** Top-4 community cohesion was ~0.08 on the `data/` run. Issue #52 reports 7,414 communities on a 410-file iOS project.
- **`/graphify query` BFS seeds on word-similarity to the question.** Generic terms like "overview" or "metric" can pull irrelevant subgraphs. `/graphify path A B` is more reliable when you know the endpoints.

### Operational gotchas

- **Repeated `/graphify <path>` overwrites `graphify-out/`.** No history, no parallel graphs from one output dir.
- **`--update` only scans the path you give it.** Saved Q&A memories under `graphify-out/memory/` are only picked up if that path is part of the next scan — awkward when graphify-out lives in a sibling tree to the corpus.
- **Token cost is real.** First-time build on `data/` (158 files / 37k words) was ~423k tokens (~$1.20 at Claude pricing). The SHA256 cache helps on re-runs but only for unchanged files.
- **The skill assumes `python3` resolves to a graphify-installed env.** Without an active venv, the skill pip-installs `graphifyy` into system Python with `--break-system-packages`. (Our Makefile-driven setup avoids this — see README.)
