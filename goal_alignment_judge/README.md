# Goal Alignment Judge

**Author:** Adam McCabe

Per-user LLM-as-a-judge scorer that predicts whether a user would pin, delete, or leave content neutral for a given organizational goal. Uses DSPy's GEPA optimizer to discover personalized scoring prompts from individual rater signal.

## Status

Proof of concept — experimental pipeline validated on 2 raters, production code TBD.

## Key results

- **Single-rater test macro F1: 0.76-0.79** (vs production baseline of 0.38-0.49)
- Per-user optimization required — cross-rater transfer fails (kappa 0.19-0.38)
- Best config: Opus optimizer + Sonnet scorer, GEPA medium, ~84 min/user
- Haiku viable for inference (retains 87-99% of Sonnet quality)

See [EXPERIMENT.md](EXPERIMENT.md) for the full research arc and findings.

## Contents

- `EXPERIMENT.md` — Full experiment report (WIP)
- `research-log.md` — Detailed research log covering all phases
- `notes/` — Ablation study plans and key observations
