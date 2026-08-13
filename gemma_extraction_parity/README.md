# Gemma vs Sonnet Extraction Parity

**Author:** Adam McCabe

## Question

Can Gemma 4 26B replace Claude Sonnet 4.6 for the learning extraction step in deep research without degrading report quality?

The hypothesis: if Gemma extracts >70% of the same learnings as Sonnet (measured as `Shared / (Shared + Sonnet-only)`), downstream reports won't be adversely affected. This would let us swap in a cheaper/self-hosted model for the most token-intensive step of the pipeline.

## How it works

The experiment mirrors the production deep research extraction pipeline:

1. **Search** -- queries the dev DB via full-text search to retrieve the same content chunks that production would see
2. **Extract** -- runs both Sonnet and Gemma over identical inputs, extracting structured learnings with citations. Gemma supports multi-pass extraction (`--passes N`) where follow-up passes find learnings missed by previous passes.
3. **Dedupe** -- uses Sonnet to de-duplicate each model's raw learnings (removes overlap from multiple search queries)
4. **Match** -- pairs de-duplicated Sonnet and Gemma learnings 1:1 by index, with tiebreaking for duplicates
5. **Report** -- produces a markdown parity report with per-topic stats, a pairing table, and derived Sonnet-only / Gemma-only lists

## CLI usage

All commands run from `experiments/`:

```bash
# Extract with Sonnet (results cached to output/sonnet_extraction.json)
PYTHONPATH=. uv run python gemma_extraction_parity --extract sonnet

# Extract with Gemma via Vertex AI (slow, ~5 QPM rate limit)
PYTHONPATH=. uv run python gemma_extraction_parity --extract gemma --prompt-version v5

# Extract with Gemma via local llama-server (much faster, no rate limits)
PYTHONPATH=. uv run python gemma_extraction_parity --extract gemma --prompt-version v5 --local-port 8080

# Multi-pass extraction (follow-up passes find missed learnings, early-exits on NONE)
PYTHONPATH=. uv run python gemma_extraction_parity --extract gemma --prompt-version v5 --local-port 8080 --passes 2

# Run dedupe + match on cached extractions
PYTHONPATH=. uv run python gemma_extraction_parity --diff --prompt-version v5

# Diff using local Gemma extraction cache (gemma_local_v5 instead of gemma_v5)
PYTHONPATH=. uv run python gemma_extraction_parity --diff --prompt-version v5 --local-port 8080

# Run a single prompt for quick iteration
PYTHONPATH=. uv run python gemma_extraction_parity --extract gemma --prompt-version v5 --prompt-id q3_software_capitalization --local-port 8080 --passes 2
```

For local Gemma, start llama-server first:
```bash
llama-server -m ~/models/gemma-4-26b-a4b-it/gemma-4-26B-A4B-it-Q8_0.gguf --port 8080
```

## Code layout

```
gemma_extraction_parity/
  __main__.py          CLI entry point (argparse)
  src/
    main.py            Orchestrator -- defines the 5 research prompts (30 queries total),
                       wires up search -> extract -> dedupe -> diff -> report
    extract.py         Sonnet and Gemma extraction via instructor (structured output)
    dedupe_diff.py     De-duplication and diff steps (both use Sonnet as judge)
    models.py          Pydantic models (ExtractionInput, MatchResult, ParityAnalysis, etc.)
    db.py              Raw asyncpg connection to dev DB + full-text search
    report.py          Markdown report generation
    settings.py        Pydantic settings (env vars, model names, paths)
    prompts/
      system.md.jinja              Shared system prompt for extraction
      sonnet/extract.md.jinja      Sonnet extraction user prompt
      gemma/v5.md.jinja            Gemma extraction prompt with few-shot quality examples
      gemma/v5_followup.md.jinja   Follow-up pass prompt (finds missed learnings)
      gemma/v4.md.jinja            Previous Gemma prompt version (baseline)
      engine.py                    Jinja2 template loader
  output/               Cached extractions and generated reports (gitignored)
```

## Key design decisions

- **Instructor JSON mode for Gemma**: Gemma doesn't support tool-use mode reliably; `instructor.Mode.JSON` works. Sonnet uses `instructor.from_anthropic` (tool-use mode).
- **Separate caches for local vs Vertex Gemma**: Local runs a quantized model (Q8_0) which may produce different results. Caches are tagged `gemma_v5` vs `gemma_local_v5`.
- **Serial requests for Vertex AI**: Rate-limited to ~5 QPM with 12s delays between requests and exponential backoff on 429s. Local Gemma uses concurrent requests.
- **Gemma max_tokens = 16384**: Gemma 4 is a thinking model that spends tokens on chain-of-thought reasoning before producing JSON. 4096 is not enough.
- **Index-based matching**: Shared learnings are identified by pairing Sonnet and Gemma de-duplicated lists by index (1:1), making counts deterministic. Duplicate indices trigger a tiebreak LLM call.
- **Multi-pass extraction**: Follow-up passes receive all previously extracted learnings and look for what was missed. Early-exits when the model returns `NONE`.
- **v5 few-shot examples**: Drawn from real v4 parity results — source excerpt + shallow Gemma extraction + rich Sonnet extraction — to teach specificity and quote inclusion.
