from __future__ import annotations

from geo_analyzer.providers.pricing import PRICING, estimate_cost


class TestEstimateCost:
    def test_known_model_computes_cost(self) -> None:
        # gpt-5.1: PRICING entry will be (input_per_1k, output_per_1k).
        # 1000 input + 500 output → 1*input_rate + 0.5*output_rate.
        cost = estimate_cost("gpt-5.1", tokens_in=1000, tokens_out=500)
        in_rate, out_rate = PRICING["gpt-5.1"]
        expected = in_rate + 0.5 * out_rate
        assert abs(cost - expected) < 1e-9

    def test_unknown_model_returns_zero(self) -> None:
        assert estimate_cost("not-a-real-model", tokens_in=1000, tokens_out=500) == 0.0

    def test_zero_tokens_zero_cost(self) -> None:
        assert estimate_cost("gpt-5.1", tokens_in=0, tokens_out=0) == 0.0

    def test_pricing_covers_seed_catalog_models(self) -> None:
        # Every model in the seed catalog should have pricing.
        # If pricing is missing the adapter still works, but the cost will be 0
        # which we want to flag at the catalog level rather than silently.
        seed_models = [
            "gpt-5.1",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ]
        for m in seed_models:
            assert m in PRICING, f"missing pricing for seed model {m!r}"
