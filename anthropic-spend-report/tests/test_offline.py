"""Offline unit tests for anthropic_spend — no API calls, no admin key, no secrets.

    cd experiments/anthropic-spend-report && python3 -m unittest discover -s tests

Covers the logic that breaks under our own edits (roster, allocation, parsing).
API-drift is covered separately by the live `--self-test` (see SKILL.md).
"""
import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import anthropic_spend  # noqa: E402  (import after the sys.path tweak above)


class AllocateTests(unittest.TestCase):
    K = ("2025-01-01", "claude-x", "0-200k", "standard", "output_tokens")

    def test_reconciles_and_nonzero_remainder_goes_to_largest(self):
        # 100 over A=1,B=1,C=5 (total 7): A,B floor to 14.285714 each and the
        # accumulated floor remainder lands on the largest holder C. C's share
        # (71.428572) EXCEEDS its bare proportional (71.428571), proving the
        # remainder was routed to C -- this fails if it goes to a smaller holder.
        alloc, unattr = anthropic_spend.allocate({self.K: Decimal("100")}, {self.K: {"A": 1, "B": 1, "C": 5}})
        self.assertEqual(sum(alloc.values()), Decimal("100"))            # reconciles exactly
        self.assertEqual(sum(unattr.values()), Decimal("0"))
        self.assertEqual(alloc[("2025-01", "A")], Decimal("14.285714"))
        self.assertEqual(alloc[("2025-01", "B")], Decimal("14.285714"))
        self.assertEqual(alloc[("2025-01", "C")], Decimal("71.428572"))
        self.assertGreater(alloc[("2025-01", "C")], alloc[("2025-01", "A")])

    def test_remainder_target_independent_of_insertion_order(self):
        a1, _ = anthropic_spend.allocate({self.K: Decimal("100")}, {self.K: {"A": 1, "B": 1, "C": 5}})
        a2, _ = anthropic_spend.allocate({self.K: Decimal("100")}, {self.K: {"C": 5, "B": 1, "A": 1}})
        self.assertEqual(dict(a1), dict(a2))

    def test_all_shares_non_negative_on_pathological_split(self):
        # regression: eight tiny near-equal shares + one larger holder used to make
        # the largest holder's (amt - used) go negative when each share rounded UP.
        shares = {f"h{i}": 1 for i in range(8)}
        shares["BIG"] = 2
        alloc, _ = anthropic_spend.allocate({self.K: Decimal("0.0000075")}, {self.K: shares})
        self.assertTrue(all(v >= 0 for v in alloc.values()), dict(alloc))
        self.assertEqual(sum(alloc.values()), Decimal("0.0000075"))     # still reconciles

    def test_tie_break_is_deterministic(self):
        # equal token counts: the remainder must land on the same key regardless of
        # the order the usage API happened to return the rows in.
        a1, _ = anthropic_spend.allocate({self.K: Decimal("1")}, {self.K: {"A": 1, "B": 1, "C": 1}})
        a2, _ = anthropic_spend.allocate({self.K: Decimal("1")}, {self.K: {"C": 1, "A": 1, "B": 1}})
        self.assertEqual(dict(a1), dict(a2))
        self.assertEqual(a1[("2025-01", "C")], Decimal("0.333334"))     # last by tie-break

    def test_indivisible_split_still_reconciles(self):
        alloc, _ = anthropic_spend.allocate({self.K: Decimal("100")}, {self.K: {"A": 1, "B": 1, "C": 1}})
        self.assertEqual(sum(alloc.values()), Decimal("100"))

    def test_unattributed_when_no_matching_usage(self):
        alloc, unattr = anthropic_spend.allocate({self.K: Decimal("55")}, {})   # cost line, no usage
        self.assertEqual(dict(alloc), {})
        self.assertEqual(unattr["2025-01"], Decimal("55"))

    def test_allocation_conserves_total(self):
        # the headline invariant, offline: allocated + unattributed == input cost,
        # across a line WITH usage and one WITHOUT (which falls to unattributed).
        k2 = ("2025-01-02", "claude-y", "0-200k", "batch", "uncached_input_tokens")
        cost = {self.K: Decimal("123.45"), k2: Decimal("67.89")}
        alloc, unattr = anthropic_spend.allocate(cost, {self.K: {"A": 3, "B": 5}})
        self.assertEqual(sum(alloc.values()) + sum(unattr.values()), Decimal("191.34"))


class ReconcileTests(unittest.TestCase):
    def test_zero_residual_when_balanced(self):
        alloc = {("2025-01", "A"): Decimal("60"), ("2025-01", "B"): Decimal("30")}
        unattr = {"2025-01": Decimal("5")}
        nontok = {("2025-01", "other"): Decimal("5")}
        self.assertEqual(anthropic_spend.reconcile(Decimal("100"), alloc, unattr, nontok), Decimal("0"))

    def test_residual_surfaces_a_shortfall(self):
        self.assertEqual(
            anthropic_spend.reconcile(Decimal("100"), {("2025-01", "A"): Decimal("90")}, {}, {}), Decimal("10"))


class ValidationTests(unittest.TestCase):
    def test_month_regex_rejects_calendar_invalid_and_malformed(self):
        for bad in ("2026-13", "2026-00", "2026-1", "2026-99", "abc", "202601", "2026-1a"):
            self.assertIsNone(anthropic_spend.MONTH_RE.fullmatch(bad), bad)

    def test_month_regex_accepts_valid(self):
        for good in ("2026-01", "2026-12", "2024-06"):
            self.assertIsNotNone(anthropic_spend.MONTH_RE.fullmatch(good), good)


class MonthWindowTests(unittest.TestCase):
    def test_labels_and_exclusive_end_boundary(self):
        w = list(anthropic_spend.month_windows("2024-01", "2024-03"))
        self.assertEqual([x[0] for x in w], ["2024-01", "2024-02", "2024-03"])
        self.assertEqual(w[0][1], "2024-01-01T00:00:00Z")
        self.assertEqual(w[0][2], "2024-02-02T00:00:00Z")        # +1 day past next-month-01

    def test_year_rollover(self):
        w = list(anthropic_spend.month_windows("2024-12", "2025-01"))
        self.assertEqual([x[0] for x in w], ["2024-12", "2025-01"])


class TokenCountsTests(unittest.TestCase):
    def test_nested_cache_creation_and_missing_fields(self):
        c = anthropic_spend.token_counts({
            "uncached_input_tokens": 5,
            "output_tokens": 3,
            "cache_creation": {"ephemeral_1h_input_tokens": 2, "ephemeral_5m_input_tokens": 1},
        })
        self.assertEqual(c["cache_creation.ephemeral_1h_input_tokens"], 2)
        self.assertEqual(c["cache_creation.ephemeral_5m_input_tokens"], 1)
        self.assertEqual(c["cache_read_input_tokens"], 0)   # absent -> 0, never KeyError


if __name__ == "__main__":
    unittest.main()
