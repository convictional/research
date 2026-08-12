# Automatic LLM Prompt Tuning

Experiment exploring techniques to automatically tune LLM prompts without manual prompt engineering.

## Why This Matters

Manual prompt engineering requires reading user content and iterating based on intuition. Under privacy constraints where we can't see user content, we need automated optimization that works on metrics alone. This experiment validates techniques that could be applied to production jobs.

## Task

Predict daily work priorities given historical context (emails, meetings, tasks, discussions). Ground truth comes from historical standup entries.

## Techniques

Two optimization techniques are implemented:

| Technique | Description | Models |
|-----------|-------------|--------|
| **OPRO** | Custom three-LLM loop (Generator, Judge, Optimizer) with trajectory-based meta-prompting | Gemini |
| **DSPy** | Declarative framework with MIPROv2 optimizer that handles instruction rewriting + few-shot selection | Claude |

See detailed documentation:
- [`docs/opro_implementation.md`](docs/opro_implementation.md) - Custom OPRO architecture and implementation
- [`docs/dspy_implementation.md`](docs/dspy_implementation.md) - DSPy migration and MIPROv2 comparison

## Setup

```bash
cd experiments/tune_llm_alignment
uv sync
```

Configure API keys in `.env.secrets`:
- `GEMINI_API_KEY` - For OPRO (Google AI Studio)
- `ANTHROPIC_API_KEY` - For DSPy
- `OPENAI_API_KEY` - For embeddings in context retrieval

## Usage

### Run OPRO Optimization

```bash
uv run python scripts/01_run_opro.py
```

### Run DSPy Optimization

```bash
uv run python scripts/02_run_dspy.py [--team] [--recall-metric]
```

Options:
- `--team` - Use team-level data (all team members' priorities)
- `--recall-metric` - Use simpler recall-based metric instead of full LLM judge

### Regenerate Team Dataset

```bash
uv run python scripts/03_regenerate_team_data.py
```

## Data

Dataset splits in `data/` and `data_team/`:
- `train.jsonl` - Training examples for optimization
- `dev.jsonl` - Validation during optimization
- `test.jsonl` - Held-out final evaluation

Results saved in `results/`.

## References

- [OPRO: Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)
- [DSPy: Compiling Declarative Language Model Calls](https://github.com/stanfordnlp/dspy)
- [Optimization by PROmpting Guide](https://cameronrwolfe.substack.com/p/automatic-prompt-optimization)
