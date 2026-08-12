# Auto Research Ideation

Scans ArXiv daily for SOTA techniques that could improve Decide. Builds up codebase context over time and produces detailed research plans for a human researcher to review.

## Usage

```bash
cd experiments
make run_experiment ARGS="auto_research"
```

Reports are written to `reports/YYYY-MM-DD.md`. Running twice on the same day overwrites the previous report.

## Pipeline

```
                                    codebase_context.md
                                           |
                                    +------v------+
                               +--->| 1. Context  |---> git log --since=1.day
                               |    | context.py  |---> LLM condense if > 4K tokens
                               |    +------+------+
                               |           |
                               |    +------v------+
                               |    | 2. Fetch    |
                               |    | arxiv.py    |---> ArXiv Atom API
                               |    +------+------+    (cs.IR, cs.AI, cs.HC, cs.GT, cs.CL)
                               |           |
                               |    +------v------+
                               |    | 3. Filter   |
                        context|    | main.py     |---> Sonnet 4.6 (structured output)
                               |    +------+------+    scores 1-10, keeps >= 6
                               |           |
                               |    +------v------+
                               |    | 4. Full Text|
                               |    | arxiv.py    |---> HTML (primary) / PDF (fallback)
                               |    +------+------+    truncate to 128K tokens
                               |           |
                               |    +------v------+
                               +--->| 5. Research |
                                    | main.py     |---> Opus 4.6 (free-form prose)
                                    +------+------+    1 call per paper, sequential
                                           |
                                    +------v------+
                                    | 6. Report   |
                                    | main.py     |---> reports/YYYY-MM-DD.md
                                    +------+------+
                                           |
                                    +------v------+
                                    | 7. Email    |
                                    | email.py    |---> Resend API (if enabled)
                                    +-------------+
```

## File Map

```
auto_research/
  __main__.py              # Entry point
  codebase_context.md      # Accumulated product context (updated daily by LLM)
  examples/                # Example codebase context and report for reference
  reports/                 # Daily output reports
  src/
    main.py                # Pipeline orchestration (stages 1-7)
    settings.py            # Config (API keys, model selection, email, paths)
    arxiv.py               # ArXiv API client + HTML/PDF full text extraction
    context.py             # Codebase context management (git log + LLM condensation)
    email.py               # Email delivery via Resend (stage 7)
    models.py              # Pydantic models (PaperRelevance, FilteredPapers)
    prompts/
      engine.py            # Jinja2 template engine
      filter_papers.md.jinja       # Paper relevance scoring prompt
      deep_research.md.jinja       # Deep analysis prompt
      codebase_summary.md.jinja    # Context update/condensation prompt
    utils/
      instruct_llm.py      # Anthropic instructor client (structured output)
      llm.py               # Raw Anthropic completion (prose output)
      tokens.py            # Token counting via tiktoken
```

## Setup

### 1. Configuration

Set in `.env.secrets`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | — | Required |
| `RESEND_API_KEY` | — | Required if email enabled |

Set in `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_MODEL` | `claude-opus-4-6` | Deep research model |
| `FILTER_MODEL` | `claude-sonnet-4-6` | Paper filtering model |
| `ARXIV_CATEGORIES` | `["cs.IR","cs.AI","cs.HC","cs.GT","cs.CL"]` | ArXiv categories to monitor |
| `EMAIL_ENABLED` | `false` | Send report via email after writing to disk |
| `EMAIL_TO` | — | Recipient address (required if email enabled) |
| `EMAIL_FROM` | `onboarding@resend.dev` | Sender address (Resend sandbox default) |

### 2. Bootstrap Codebase Context

The pipeline reads from `codebase_context.md` and incrementally updates it with recent commits. On first run this file won't exist, so you need to seed it. See `examples/` for reference output (a codebase context and a daily report).

The fastest way to build this is with Claude Code using a team of agents, each focused on one area of the codebase. From the repo root, ask Claude:

> Build a codebase context document at `experiments/auto_research/codebase_context.md`. This document is used by an automated ArXiv researcher to assess paper relevance, so focus on information that helps identify opportunities to apply new AI/NLP techniques to our product.
>
> **Step 1: Goals.** Use the Convictional MCP tool (`list_goals`) to fetch all active organizational goals. Place these at the top of the document under "Current Goals & Priorities" — they ground the researcher's relevance judgments.
>
> **Step 2: Codebase exploration.** Create a team of 7 agents, each exploring one area and writing a focused summary of what exists, how it works, key patterns, and what AI/NLP techniques are in use:
>
> 1. **Frontend** — `app/presenters/`, `app/routers/`, `app/templates/`, `app/static/` — UI patterns, HTMX/Alpine usage, where AI surfaces to users
> 2. **Jobs & Models** — `app/jobs/`, `app/models/` — domain model structure, background job patterns, AI-related jobs
> 3. **Infrastructure & DB** — `infra/`, `migrations/`, `config/` — database setup, pgvector, search infrastructure, caching
> 4. **Middleware & Integrations** — `app/middleware/`, `integrations/` — third-party services, OAuth, data connections
> 5. **Backend** — `app/helpers/`, `app/mailers/`, `app/prompts/`, `lib/` — LLM infrastructure, prompt patterns, utilities
> 6. **Design Principles** — `docs/`, `CLAUDE.md` — architectural constraints, design philosophy, coding conventions
> 7. **Experiments** — `experiments/` — past research, techniques explored, gaps not yet covered
>
> **Step 3: Consolidate** the goals and 7 summaries into a single markdown document structured like the example at `experiments/auto_research/examples/codebase_context.md`.

Once generated, the daily pipeline keeps it current via git log diffs and LLM condensation.

**Maintaining goals:** The pipeline has no automatic mechanism to fetch goals — it only updates the context via git commits. When organizational goals change, re-run the goals step above (or manually edit the goals section in `codebase_context.md`) to keep relevance assessments aligned.

### 3. Daily Cron (macOS launchd)

An example plist is included at `examples/com.convictional.auto-research.plist`. To set up:

1. Copy and fill in your paths (`$HOME` and `$REPO` are placeholders):
   ```bash
   cp experiments/auto_research/examples/com.convictional.auto-research.plist \
      ~/Library/LaunchAgents/com.convictional.auto-research.plist
   ```
2. Edit the copy — replace `$HOME` with your home directory and `$REPO` with the repo root (launchd doesn't expand variables).
3. Load and verify:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.convictional.auto-research.plist
   launchctl start com.convictional.auto-research   # test run
   cat ~/Library/Logs/auto-research.log              # check output
   ```

The schedule uses `StartCalendarInterval` which fires at local time and persists across reboots. No need to source `.env.secrets` — Pydantic Settings loads it automatically.
