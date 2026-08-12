"""Per-model token pricing.

Values are best-effort as of 2026-04-27. Verify against the provider's pricing
page when API keys are configured — these are estimates, not billing.

Format: model_name -> (USD_per_1k_input_tokens, USD_per_1k_output_tokens)
"""

from __future__ import annotations

PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5.1": (0.0025, 0.0100),
    # Anthropic
    "claude-opus-4-7": (0.0150, 0.0750),
    "claude-sonnet-4-6": (0.0030, 0.0150),
    # Google
    "gemini-2.5-pro": (0.00125, 0.0050),
    "gemini-2.5-flash": (0.000075, 0.0003),
}


def estimate_cost(model_name: str, *, tokens_in: int, tokens_out: int) -> float:
    """Return USD cost estimate for a given (model, token usage). Unknown
    models return 0.0 (Phase 2 doesn't fail closed on missing pricing — the
    field is informational; surfacing $0 highlights the gap when reviewing
    a run rather than blocking the run)."""
    rates = PRICING.get(model_name)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (tokens_in / 1000.0) * in_rate + (tokens_out / 1000.0) * out_rate
