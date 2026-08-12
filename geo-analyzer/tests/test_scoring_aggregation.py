from __future__ import annotations

import pytest

from geo_analyzer.scoring.aggregation import (
    majority_vote,
    mean_of_floats,
    mean_rate,
    median_or_none,
)


class TestMajorityVote:
    @pytest.mark.parametrize(
        "samples,expected",
        [
            ([True, True, True], True),
            ([True, True, False], True),
            ([True, False, False], False),
            ([False, False, False], False),
            ([True, False], True),  # tie → True (favor signal presence)
            ([True], True),
            ([False], False),
        ],
    )
    def test_simple(self, samples: list[bool], expected: bool) -> None:
        assert majority_vote(samples) is expected

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            majority_vote([])


class TestMedianOrNone:
    def test_all_ints(self) -> None:
        assert median_or_none([1, 2, 3]) == 2
        assert median_or_none([1, 3]) == 2

    def test_drops_none_before_median(self) -> None:
        assert median_or_none([None, 2, 4]) == 3

    def test_all_none_returns_none(self) -> None:
        assert median_or_none([None, None]) is None

    def test_empty_returns_none(self) -> None:
        assert median_or_none([]) is None


class TestMeanOfFloats:
    def test_basic(self) -> None:
        assert mean_of_floats([0.0, 1.0]) == 0.5

    def test_skips_none(self) -> None:
        assert mean_of_floats([None, 0.5, 1.0]) == 0.75

    def test_all_none_returns_none(self) -> None:
        assert mean_of_floats([None, None]) is None

    def test_empty_returns_none(self) -> None:
        assert mean_of_floats([]) is None


class TestMeanRate:
    def test_basic(self) -> None:
        result = mean_rate([True, False, False])
        assert abs(result - 1 / 3) < 1e-9

    def test_all_true(self) -> None:
        assert mean_rate([True, True]) == 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            mean_rate([])
