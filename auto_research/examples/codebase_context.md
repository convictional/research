# Decide — Codebase Context

> **This is an example context file, and the goals section below is synthetic.** The real version
> was generated from the live goal board and carried actual owners, actual RAG statuses and actual
> completion percentages. Those have been replaced with a made-up board of the same shape, so the
> file still shows what `auto_research` feeds to the model. The product, infrastructure and
> past-experiments sections further down describe the real system.

## Current Goals & Priorities (Q1 2026)

These are the active organizational goals. Research recommendations should align with these priorities.

**Activation** (OFF TRACK, 20%) — Customers actively using goals to run their business. Owner: Strategy lead.
- Get ICP customers to load and review goals weekly

**Momentum** (ON TRACK, 85%) — Ship a Q1 roadmap we're proud of. Owner: Engineering lead.
- Ship 100% planned features on time, zero major scope cuts, success metrics for all features

**Integration** (OFF TRACK, 30%) — New features become part of how customers operate. Owner: Engineering lead.
- 1 customer articulates how we helped them do more with the same team
- 3 prospects ask about goals unprompted in demos

**Retention** (AT RISK, 40%) — Make sure early customers stick and get value. Owner: Strategy lead.
- 70% new customers reach activation milestone within 14 days, zero churn, referenceable customer

**AI Moat** (AT RISK, 0%) — Advance AI foundations to create durable product advantages. Owner: R&D lead.
- Support inbox algorithm development through PoCs and iterative improvements
- Identify production-viable pre-compression technique for high-entropy content
- Ship goal analytics and AI enablement
- Key framing: "premium on token spend" — tokens spent on Decide at a multiple of commodity prices should yield better outcomes than commodity token spend

**Validation** (AT RISK, 65%) — Validate solving a real problem for a defined customer. Owner: GTM lead.
**Conversion** (ON TRACK, 0%) — Acquire first meaningful set of paying customers. Owner: GTM lead.
**Discover** (ON TRACK, 0%) — Figure out how customers find and buy from us. Owner: GTM lead.
**Foundation** (ON TRACK, 0%) — Build operational basics for future scale. Owner: GTM lead.
**Capital Allocation** (ON TRACK, 70%) — Spend money where it matters most. Owner: Operations lead.
**Operations** (ON TRACK, 0%) — Keep the business running smoothly. Owner: Operations lead.
**Flourish** (ON TRACK, 0%) — Enable the team to do their best work. Owner: Operations lead.

## Product Overview

Decide is a collaborative decision-making and goal-tracking platform. Built with FastAPI, Tortoise ORM, PostgreSQL (pgvector), HTMX, and Alpine.js.

### Core Product Areas

- **Goals**: OKR-style hierarchical tracking with metrics, alignment scoring, AI-generated titles and suggestions. The flagship feature — activation is the top company priority.
- **Decision Processes**: Structured decision-making with options, criteria, evaluations (rating matrix), and AI-powered trade-off analysis. Generates options/criteria with deduplication via cosine similarity (0.9 threshold).
- **Meetings**: Transcript ingestion (via Recall AI), WebVTT parsing with speaker/timestamp extraction, automatic summarization, video-transcript sync in UI.
- **Research**: Multi-iteration intelligence gathering (configurable depth/breadth). Pipeline: ResearchJob → ResearchIterationJob → ResearchQueryJob chain with LLM-generated follow-up queries and learning extraction.
- **Email Integration**: Gmail API integration with watch/sync, AI-powered grouping by goals, smart collections. Mailgun for outbound.
- **Posts**: Team communication with threading, link previews, decision linking, and group scoping.
- **Content Search**: Unified content indexing across all workspace types with hybrid vector+text search.

### AI/NLP Features

- **LLM**: Anthropic Claude via `instructor` for structured outputs, 53+ Jinja2 prompt templates in `app/prompts/`
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions), stored in pgvector with HNSW indexes (m=16, ef_construction=64)
- **Hybrid Search**: 70% vector similarity (cosine) + 30% BM25 text rank, partitioned by content category
- **Goal Alignment**: Batch job scores all content against active goals — vector pre-filter then LLM scoring with few-shot examples (pinned = positive, deleted = negative). Semaphore-limited to 20 parallel LLM calls.
- **Decision Intelligence**: ExtractBasicsJob (title polish), GenerateCriteriaJob (5 criteria), GenerateOptionsJob (3 options), SuggestCollaboratorsJob. Few-shot learning from org's past good/bad examples.
- **Research Pipeline**: Multi-depth iterative search with LLM-generated queries, result review, learning extraction, and final synthesis (30K token budget).
- **Meeting Intelligence**: Transcript parsing, speaker attribution, title/summary generation.
- **Streaming**: SSE via `EventSourceResponse` + `CompletionStreamManager` for real-time LLM output.

## Infrastructure & Technical Capabilities

### Database & Search
- PostgreSQL with async asyncpg, Tortoise ORM. Cloud SQL support with auto-refreshing auth tokens.
- **pgvector**: HNSW index with `vector_cosine_ops`, ef_search=1000 at query time.
- **Full-text**: GIN indexes on TSVECTOR fields, `websearch_to_tsquery('english', ...)` for natural language queries.
- **Content model**: Unified `Content` table with embedding (1536D vector), text_search (TSVECTOR), normalized lookup fields, category/type enums, metadata JSON. All workspace content indexed here.
- **ContentSearch**: Hybrid scoring combining vector similarity and text rank. Max 500 vector + 300 text candidates before re-ranking.
- **ContentLookup**: Recency-boosted search with exponential decay (7-day half-life).

### LLM Infrastructure
- `AsyncAnthropic` client with instructor for structured output (Pydantic model extraction).
- Methods: `instructor_completion()`, `instructor_streaming_completion()`, `string_completion()`, `chat_completion()`, `instructor_multimodal_completion()`.
- Token usage tracking integrated with Sentry.
- Prompt engine: Jinja2 templates with base64 stripping, user syntax escaping, whitespace normalization.
- Organization and user context helpers for prompt building.

### Background Jobs
- **Runners**: ASYNCIO (dev), INLINE (test), CLOUD_TASKS (prod via Google Cloud Tasks HTTP push).
- **Job model**: Tracks type, details (JSON), status, error, timestamps. JobGroups for batch operations with completion callbacks.
- **Key job patterns**: Self-queuing pagination (GoalAlignment batches of 100 with offset), chained pipelines (Research), unique job deduplication, retry with dead job detection (104h).

### Real-Time
- PostgreSQL NOTIFY/LISTEN for WebSocket broadcasts. Multi-server safe.
- Client reconnection with exponential backoff (1s → 30s, 5 attempts).

### Other Infrastructure
- **Storage**: Local filesystem (dev) or GCS with signed URLs (prod). Magic bytes MIME validation.
- **Cache**: PostgreSQL-backed with TTL, atomic upsert, JSONB set operations.
- **Email**: Mailgun (outbound), Gmail API (inbound). Jinja2 templates with markdown rendering.

## Frontend & User Experience

Server-rendered HTML with HTMX progressive enhancement and Alpine.js for interactivity. No SPA.

### Key Patterns
- Full page renders always — layout auto-selected by `HX-Request` header. Use `hx-select` to extract fragments.
- `hx-swap="morph"` preserves Alpine state, focus, scroll position.
- Contenteditable inline editing with debounced saves (goals, descriptions).
- HTMX polling (`hx-trigger="every 1s"`) for AI generation progress.
- Complex Alpine logic extracted to TypeScript files registered with `Alpine.data()`.

### Where AI Surfaces in UI
- **Research dialog**: Command palette (Cmd+K) with preset prompts, results stream to inbox.
- **Meeting summaries**: Auto-generated after meeting ends, editable inline, AI icon badge.
- **Decision explanations**: AI-generated with polling until ready, `source.is_ai` flag.
- **Goal titles**: Auto-generated via background job with broadcast updates.
- **Options/criteria**: AI-generated with "Improve" feedback loop for regeneration.

### UX Improvement Opportunities
- Blocking waits for AI (spinners during research/summaries) — streaming could help.
- No progress visibility during multi-step research pipeline.
- Limited AI transparency — only decisions flag AI source consistently.
- Research results don't auto-link to relevant goals/decisions.

## Design & Architecture Constraints

- **Strict layering**: integrations → app → infra → config → lib. Enforced by import linter.
- **Models can't cross-depend** — use presenters to assemble across domains.
- **AI logic lives in jobs** (long-running) or routers (lightweight). Prompts in `app/prompts/`.
- **Async-first**: All I/O must be async with `transaction()` for multi-step DB operations.
- **Design philosophy**: Clarity and restraint over visual complexity. Layout/spacing over decoration. Progressive disclosure. Inline editing as default.
- **Testing**: Unit (no I/O) in `tests/unit/`, integration (with DB) in `tests/integration/`. Comprehensive tests over granular ones.

## Past Experiments & Research

40+ experiments in `experiments/` spanning LLM optimization, retrieval, and organizational analytics.

### Key Completed Work
- **Auto arXiv Researcher**: CLI tool for automated academic paper discovery and analysis. Searches arXiv for papers relevant to Decide's research priorities, generates reports using tiered LLM analysis (Sonnet for initial review, Opus for deep analysis). Configurable lookback window (minimum one week or since last report in output directory). Token truncation tuned for longer paper reviews. Outputs example reports to track findings over time. Includes codebase context document to ground relevance assessments. Enables systematic literature monitoring for AI/NLP techniques applicable to the product.
- **Multi-Vector Representation**: Production ColBERT comparison vs OpenAI embeddings vs hybrid 70/30. TREC-style evaluation with human annotations.
- **Train Research Report Judge**: 11-trial effort for automated quality scoring. Plateaued at 0.326 rank correlation. Recommends pivot to pairwise ranking.
- **LoRA Cassettes**: Episodic LoRA adapters for e5-base-v2 retrieval. Contrastive learning with replay buffers.
- **Decision DAGs**: Persistent decision trees with genetic algorithm evolution and Plotly visualization.
- **Agentic SQL**: NL-to-SQL with self-verification loops (3 refinement passes). Vector search for schema context.
- **Deep Research Ability**: Multi-iteration research tree builder with breadth/depth control and SERP theory.
- **Tune LLM Alignment**: OPRO (Gemini) and DSPy/MIPROv2 (Claude) for automated prompt optimization.
- **Writing Identification**: Siamese network for authorship verification. Best: Claude Opus at 72.86%.
- **Google Embeddings Comparison**: ~50% top-10 divergence vs OpenAI. Not drop-in replaceable.

### Techniques Already Explored
- Single-vector vs multi-vector retrieval (ColBERT)
- Hybrid search weighting (vector + keyword)
- Adapter-based fine-tuning (LoRA for domain retrieval)
- Pointwise quality scoring (with claim decomposition, ensemble voting, RAG verification — all plateaued)
- Agentic SQL with self-reflection
- Genetic algorithm optimization for decision paths
- Automated prompt optimization (OPRO, DSPy)
- Neural authorship verification (Siamese networks, BERT embeddings)
- Automated academic literature monitoring (arXiv search + tiered LLM analysis)

### Gaps Not Yet Explored
- Pairwise ranking for quality assessment
- Fine-tuning on domain-specific tasks (most work uses in-context learning)
- Multi-modal analysis (text + images, audio beyond transcripts)
- Continual/online learning in production
- Interactive user feedback loops for model refinement
- Formal verification for generated SQL/content
