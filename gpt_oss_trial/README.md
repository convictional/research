# GPT-OSS Vertex AI Benchmark

**Author:** Adam McCabe

Comprehensive benchmark suite for testing OpenAI's open-weight GPT-OSS 120B model, served on Vertex AI Model Garden, with comparison capabilities against Claude Sonnet 4.5.

> Note: GPT-OSS is OpenAI's open-weight model family. Vertex AI is only the serving
> platform here — the model is not Google's. An earlier version of this README got that wrong.

## Features

- **Performance Benchmarking**: Latency (mean/p50/p95/p99), TTFT, throughput (tokens/sec)
- **Reliability Testing**: Error rates, error type classification
- **Database-Sampled Prompts**: Stratified sampling of recorded LLM requests by length
- **Model Comparison**: Parallel execution of same requests to Claude and GPT-OSS
- **Structured Output Testing**: Schema adherence testing using Instructor library
- **Evaluation Mode**: Cherry-picked prompts for quality review

## Setup

### Environment Variables

Add to `.env.secrets`:

```bash
GOOGLE_ACCESS_TOKEN=<your-gcloud-access-token>
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

### Install Dependencies

```bash
make install  # or: uv sync
```

## CLI Usage

### 1. Simple Single Request Test

Test the endpoint with a single request (no args):

```bash
uv run python -m src.main
```

### 2. GPT-OSS Benchmark

Run performance benchmark on GPT-OSS endpoint:

```bash
# Basic benchmark with simple prompt
uv run python -m src.main --requests 100 --concurrency 10

# With custom prompt
uv run python -m src.main --requests 50 --concurrency 5 --prompt "Your custom prompt"

# Adjust timeout
uv run python -m src.main --requests 10 --timeout 120
```

### 3. Database-Sampled Prompts

Sample recorded prompts from a local database (stratified by length). The original runs
sampled from a copy of the production request log; that database is not included in this
repository, so this path needs a local table of recorded requests to work against:

```bash
uv run python -m src.main --requests 100 --concurrency 10 --use-db-prompts
```

**Note**: Automatically filters out multi-turn conversations and uses complete request bodies including system prompts and tools.

### 4. Claude vs GPT-OSS Comparison

Run same requests to both models in parallel:

```bash
# With database prompts
uv run python -m src.main --requests 100 --concurrency 10 --use-db-prompts --compare-claude

# With simple prompt
uv run python -m src.main --requests 50 --concurrency 5 --compare-claude
```

**Output**: `comparison_claude_vs_gpt_YYYY-MM-DD_HH-MM-SS.csv` with per-request metrics from both models and delta columns.

### 5. Evaluation Mode

Run comparison on 6 cherry-picked prompts with full response text saved:

```bash
uv run python -m src.main --eval-mode --concurrency 2
```

**Output**: CSV includes `claude_response_text` and `gpt_response_text` columns for quality review.

**Cherry-picked request IDs** (hard-coded in `src/benchmark/eval_prompts.py`):
- Strategic planning questions
- ICP/market analysis
- Product vision exploration

### 6. Structured Output Testing

Test schema adherence using Instructor library with 8 synthetic prompts:

```bash
uv run python -m src.main --structured-test
```

**Tests 3 Pydantic models:**
- `SimpleResponse`: basic + confidence enum
- `ReasonedResponse`: adds reasoning + numeric confidence
- `ComplexResponse`: adds learnings list + caveats

**Output**: Success rates by schema complexity, validation errors, retry counts.

**Note**: GPT-OSS uses `Mode.JSON`, Claude uses `Mode.TOOLS` (forced tool calling).

## Output Files

All results saved to `./results/` directory:

### Standard Benchmark
- `benchmark_YYYY-MM-DD_HH-MM-SS.csv` - Per-request detailed metrics
- `benchmark_YYYY-MM-DD_HH-MM-SS_summary.csv` - Aggregate statistics

### Comparison Benchmark
- `comparison_claude_vs_gpt_YYYY-MM-DD_HH-MM-SS.csv` - Side-by-side metrics with deltas

### Structured Testing
- `structured_test_YYYY-MM-DD_HH-MM-SS.csv` - Schema adherence results with response JSON

## Metrics Collected

**Latency:**
- Mean, median (p50), p95, p99, min, max response times

**Time to First Token (TTFT):**
- Mean, median, p95, p99

**Throughput:**
- Tokens per second
- Requests per second

**Reliability:**
- Success rate (%)
- Error breakdown by type (401, 400, timeout, etc.)

**Prompt Statistics:**
- Length distribution (mean, median, min, max)
- Requests with tools (%)
- Requests with system prompts (%)

**Structured Output:**
- Validation success rate by schema complexity
- Retry counts
- Validation error types

## Architecture

```
src/
├── main.py                     # Entry point with CLI routing
├── cli.py                      # Command-line interface
├── llm/
│   ├── gpt_oss.py             # Vertex GPT-OSS streaming client
│   ├── claude.py              # Anthropic Claude streaming client
│   └── instructor_client.py   # Instructor wrapper (Mode.JSON for GPT-OSS)
├── benchmark/
│   ├── models.py              # Pydantic data models
│   ├── metrics.py             # Statistics calculations
│   ├── runner.py              # Benchmark orchestration
│   ├── comparison.py          # Parallel Claude vs GPT-OSS execution
│   ├── output.py              # Console formatting and CSV export
│   ├── prompts.py             # Database sampling with stratification
│   ├── eval_prompts.py        # Cherry-picked request IDs
│   ├── format_converter.py    # Anthropic → OpenAI format conversion
│   ├── structured_test.py     # Structured output benchmark
│   └── structured_prompts.py  # Synthetic prompts for structured testing
└── models/
    └── structured_responses.py # Pydantic schemas for structured testing

```

## Notes

- **Token Refresh**: Google Cloud access tokens expire - refresh before long benchmarks
- **Forced Tool Calling**: GPT-OSS doesn't support it - we use `tool_choice: "auto"`
- **Model Override**: Always uses `claude-sonnet-4-5-20250929` for Claude regardless of DB model
- **Multi-turn Filtering**: Database sampling excludes conversations with assistant messages or tool use/results
- **Data**: The sampled prompt corpus and the cherry-picked request IDs referenced a production database that is not published with this repository
- **Concurrency**: Controls max parallel requests (recommended: 10 for 100+ requests)
