# geo-analyzer

> Measure how Convictional shows up in the answers people are getting from
> ChatGPT, Claude, and Gemini — and whether the dropship-era version of us
> is fading from those answers.

A standalone Python subproject in the `convictional/decide` monorepo. Local-only
in v1: no GCP, no remote storage, no scheduled jobs. Clone the repo, set provider
API keys, run the CLI, get a markdown dashboard.

## Why this exists

Two shifts make this measurement load-bearing for Convictional right now:

**1. Discovery is moving from SEO to GEO.** When prospects ask "what is
organizational health software?" they are increasingly asking ChatGPT or
Gemini, not Google. Marketing teams have tracked SEO rankings for two
decades. We need the equivalent for generative engines: a periodic measurement
of how — and whether — we surface in the answers our buyers actually see.

**2. Convictional is repositioning.** We're moving from our legacy identity as
a B2B dropship integration platform to the new category we're creating:
**Organizational Health**. The repositioning only works if the generative
engines our prospects use start describing us the new way. The single most
important number this tool tracks is therefore: **what fraction of model
answers still describe Convictional as the dropship company?** We want that
trending toward zero.

## What this tool does

Every run does the same thing:

1. Take a hand-authored catalog of ~12 prompts organized into four tiers
   (broadest → brand-named) — `catalog/prompts/`.
2. Send each prompt to ten flagship LLM variants — five model families
   (`gpt-5.1`, `claude-opus-4-7`, `claude-sonnet-4-6`, `gemini-2.5-pro`,
   `gemini-2.5-flash`) × ungrounded (training priors only) and grounded
   (with web search enabled).
3. Score every response with deterministic extractors — pure-Python regex
   over the raw text. No LLM-as-judge in v1.
4. Aggregate per `(prompt × model)` cohort. Grounded mode runs N=3 samples;
   we emit both the majority-vote bool and the mean rate.
5. Persist to a local run directory: `tasks.jsonl` (full responses),
   `tasks.csv`, `scores.jsonl`, `scores.csv`, `manifest.json`, `summary.md`.

The catalog defines three measurement subjects:

| Subject                          | Kind         | Direction | What we want                        |
| -------------------------------- | ------------ | --------- | ----------------------------------- |
| `convictional_brand`             | brand        | grow      | Models surface Convictional more.   |
| `organizational_health_category` | category     | grow      | The category becomes legible.       |
| `convictional_legacy_dropship`   | anti-brand   | shrink    | Dropship association fades.         |

And five deterministic metrics per subject (see DESIGN §6.1 for algorithms):

- **Mention presence** — does the model name us, at a word boundary?
- **Ordinal rank** — when in a list, where do we land?
- **Share of voice** — `mentions(us) / (mentions(us) + mentions(competitors))`.
- **Brand-legacy conflation** (the headline) — does the response mention us *and*
  the dropship era in the same answer?
- **Cited URLs** — extracted from grounded responses; tagged when they hit our owned domains.

## What you get

Every `geo-analyzer run` produces a `data/runs/<run-id>/summary.md`. The excerpt
below is **synthetic** — a made-up brand and made-up numbers, standing in for the
real run this section originally showed — but the shape and every field are
exactly what the tool emits:

````markdown
# Run 2026-05-05-manual (completed • 11m48s)

## TL;DR

| subject | metric | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| `examplecorp_brand` | brand mention | 2/30 | 5/90 | 4/30 | 29/30 |
| `widget_ops_category` | category awareness | 1/30 | 6/90 | 27/30 | -- |
| `examplecorp_legacy_widgets` | conflation (want down) | 0/30 | 2/90 | 3/30 | 21/30 |

## examplecorp_legacy_widgets

**Conflation by tier**: `▁▁▂▇`  L1=0.0%(30)  L2=2.2%(90)  L3=10.0%(30)  L4=70.0%(30)

**Per-prompt conflation rate** (cohorts where brand and legacy term co-occurred, by tier and mode):

| tier | prompt | ungrounded | grounded | text |
|---|---|---|---|---|
| L1 | `companies-in-age-of-ai` | 0/5 | 0/5 | How have companies changed how they operate in the age of AI? |
| L2 | `best-team-chat` | 0/5 | 1/5 | What's the best team chat software for remote-first companies? |
| L3 | `what-is-widget-ops` | 0/5 | 1/5 | What is widget operations software? |
| L4 | `what-is-examplecorp` | 3/5 | 4/5 | What is ExampleCorp? |
| L4 | `examplecorp-product` | 3/5 | 4/5 | What does ExampleCorp's product do today? |
| L4 | `examplecorp-pricing` | 3/5 | 3/5 | How is ExampleCorp priced and who is it for? |

## Goal progress

- [RED] **legacy-widgets-l4-eoy** (examplecorp_legacy_widgets/brand_legacy_conflation/L4, target=0.05 by 2026-12-01): actual=0.700 expected=0.987

## Cost & runtime

- tasks: total=360 success=360 failed=0
- tokens: in=2,304,891 out=405,217
````

Each TL;DR cell is `cohorts-fired / total-cohorts` at that tier. A cohort is one
(prompt × model) pair — for grounded models the N=3 samples are aggregated to a
single cohort verdict. A row shaped like `low  low  low  high` says "we exist when
asked by name, but barely appear in broader category questions" — which is the
GEO problem in a single row. The same shape on the anti-brand row tells the
inverse story: the legacy-product association still fires at L4 (asked about the
company directly) but no longer leaks into broader questions.

Each subject also gets a per-prompt detail table so you can trace any tier
number to the exact prompts that drove it. The full `summary.md` also includes
per-model breakdowns, funnel sparklines, and the biggest grounded-vs-ungrounded
gaps. Raw responses are in `tasks.jsonl`.

## Quick start

```
cd experiments/geo-analyzer
uv sync
cp .env.secrets.example .env.secrets       # then fill in API keys (gitignored)

# Sanity-check the catalog
uv run geo-analyzer catalog validate

# Smallest possible end-to-end test (3 prompts × 1 model × 1 sample = 3 calls)
uv run geo-analyzer run --tier L4 --model openai:gpt-5.1:ungrounded --yes
uv run geo-analyzer report --open-latest

# The full matrix (~240 calls, ~$1-15 depending on tokens)
uv run geo-analyzer run --dry-run        # see matrix size + cost estimate first
uv run geo-analyzer run --yes
uv run geo-analyzer report --open-latest
uv run geo-analyzer status               # goal traffic-lights
```

## API keys

Three env vars, expected in `.env.secrets` (gitignored):

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

The CLI auto-loads both `.env` and `.env.secrets` (the latter overrides). Both
are gitignored. The committed template is `.env.secrets.example`.

## Commands

| Command                                              | What it does                                                                                                        |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `geo-analyzer catalog validate`                      | Cross-check `catalog/` for duplicate ids, unknown subjects, ungrounded-mode mistakes, etc.                          |
| `geo-analyzer probe "<prompt>" --model <id>`         | One-off: send a prompt to one model, print the response + tokens + cost. No persistence.                            |
| `geo-analyzer probe ... --sensitivity-samples N --temperature T` | Same but N samples; useful for "is this answer stable or coin-flip-dependent?" investigations (DESIGN §5.7). |
| `geo-analyzer run [filters]`                         | Execute the full matrix; persist to `data/runs/<run-id>/`. Default-resumes on the same day. Use `--no-resume` to start clean. |
| `geo-analyzer run --dry-run`                         | Print matrix size + estimated cost, exit. No API calls.                                                             |
| `geo-analyzer report [RUN_ID] [--open-latest]`       | Render `summary.md` for a run (default: latest). `--open-latest` also opens it in a browser.                        |
| `geo-analyzer report --since YYYY-MM-DD`             | Multi-run trend table from all local runs since that date.                                                          |
| `geo-analyzer status`                                | One-line traffic-light per goal against the latest run. Exits 3 if any goal is `[RED]`.                             |

`run` filters: `--tier L1`, `--subject convictional_brand`, `--model openai:gpt-5.1:grounded` — all repeatable.

## Layout

```
experiments/geo-analyzer/
├── DESIGN.md                       # full spec — read for the "why" of every choice
├── README.md                       # you are here
├── pyproject.toml / uv.lock        # uv-managed standalone subproject
│
├── catalog/                        # committed config (the measurement spec)
│   ├── subjects.yaml               # what we measure (brand, category, anti-brand)
│   ├── models.yaml                 # which LLMs, which modes, which tools
│   ├── goals.yaml                  # targets + traffic-light directions
│   └── prompts/
│       ├── l1_broad.yaml           # broadest user questions
│       ├── l2_adjacent.yaml        # category-adjacent ("what tools for engagement?")
│       ├── l3_category.yaml        # category-named ("what is org health software?")
│       └── l4_brand.yaml           # brand-named ("what is Convictional?")
│
├── data/                           # outputs only — entirely gitignored
│   └── runs/<run-id>/              # one dir per run
│       ├── manifest.json           # run metadata + catalog snapshot
│       ├── tasks.jsonl             # one row per task — includes full LLM response
│       ├── tasks.csv               # tasks minus `text` column (responses can be huge)
│       ├── scores.jsonl / .csv     # one row per (prompt, model, subject, metric)
│       └── summary.md              # the human-readable dashboard
│
├── src/geo_analyzer/
│   ├── catalog/                    # YAML loader + cross-ref validation
│   ├── providers/                  # one async adapter per provider (openai/anthropic/google)
│   ├── runner/                     # matrix expansion, retry, concurrency, scoring, orchestration
│   ├── reports/                    # summary.md generator + per-section computers
│   ├── scoring/                    # deterministic extractors + N=3 aggregation helpers
│   ├── storage/                    # run-dir layout, JSONL/CSV writers, manifest
│   ├── runtime.py                  # Run / Task / Score Pydantic types
│   ├── types.py                    # Catalog Pydantic types
│   └── cli.py                      # the typer entry-point
│
├── tests/                          # pytest unit tests (~230 tests, no API calls)
│   └── test_providers_live.py      # opt-in integration tests (`pytest -m live`, needs keys)
│
└── docs/plans/                     # phased implementation plans (P1 → P4)
```

## Status

Phase 4 complete. See [`DESIGN.md`](DESIGN.md) §14 for deferred v2/v3 scope.

Phase 5 (operational polish, no new user-facing features):

- `launchd` plist example for scheduled local runs.
- GitHub Actions CI workflow gating PRs that touch `experiments/geo-analyzer/**`.

## Where to go from here

- [`DESIGN.md`](DESIGN.md) — the full spec. Read this for the "why" behind every
  scoring decision, the v1/v2/v3 boundary, and what's deliberately out of scope.
- [`docs/plans/`](docs/plans/) — the phased implementation plans we executed
  to build this (Phase 1 = foundation, Phase 2 = providers, Phase 3 = runner,
  Phase 4 = reports). Useful as reference if you're extending the tool.
- `catalog/subjects.yaml`, `catalog/models.yaml`, `catalog/prompts/*.yaml`,
  `catalog/goals.yaml` — what to edit when the catalog content needs to change.
  Every committed config field is documented in DESIGN §4.

## Adding a goal

Edit `catalog/goals.yaml`:

```yaml
- id: brand-mention-l3-2027q1
  subject: convictional_brand
  metric: mention_presence
  tier: L3
  target: 0.40
  direction: above           # default; use 'below' for anti-signals
  created_at: 2026-04-30
  target_date: 2027-03-31
  notes: "L3 brand awareness milestone"
```

Then `uv run geo-analyzer status` to see the traffic-light against the latest
run. The traffic-light uses linear interpolation from `created_at` to
`target_date` — at 50% elapsed time you should be 50% of the way to `target`.
