"""Tests for alignsim.src.engine.ops_logic — process project lifecycle and bonuses."""

import math
import random

import pytest

from alignsim.src.engine.ops_logic import (
    compute_degradation_pct,
    compute_effective_bonus,
    compute_maintenance_cost,
    compute_process_bonus,
    compute_project_progress,
    get_active_bonus,
    tick_bonus_durations,
)
from alignsim.src.models.entities import ProcessProjectStatus

from .factories import make_active_bonus, make_process_project


# --- compute_project_progress ---

def test_project_progress_completes():
    """Last needed turn returns True (this turn would complete it)."""
    project = make_process_project(
        ops_capacity_cost=2, duration_turns=3, progress_turns=2,
        status=ProcessProjectStatus.in_progress,
    )
    assert compute_project_progress(project, ops_capacity=2) is True


def test_project_progress_not_yet():
    """Earlier turn returns False (still in-progress, not complete)."""
    project = make_process_project(
        ops_capacity_cost=2, duration_turns=3, progress_turns=0,
        status=ProcessProjectStatus.in_progress,
    )
    assert compute_project_progress(project, ops_capacity=2) is False


def test_project_progress_already_completed():
    """Completed project never advances further."""
    project = make_process_project(
        ops_capacity_cost=2, duration_turns=3, progress_turns=3,
        status=ProcessProjectStatus.completed,
    )
    assert compute_project_progress(project, ops_capacity=2) is False


def test_project_progress_insufficient_capacity():
    """Capacity below the project's cost cannot advance."""
    project = make_process_project(
        ops_capacity_cost=4, duration_turns=3, progress_turns=2,
        status=ProcessProjectStatus.in_progress,
    )
    assert compute_project_progress(project, ops_capacity=3) is False


# --- compute_process_bonus ---

def test_process_bonus_deterministic():
    """Same seed yields identical bonus."""
    project = make_process_project(
        bonus_base=0.05, bonus_scale_factor=0.02, bonus_max=0.20,
        target_team_capacity_invested=10,
    )
    a = compute_process_bonus(project, random.Random(42))
    b = compute_process_bonus(project, random.Random(42))
    assert a == b


def test_process_bonus_scales_with_investment():
    """Higher target-team investment raises the mean of the sampled bonus."""
    proj_low = make_process_project(target_team_capacity_invested=0,
                                    bonus_base=0.05, bonus_scale_factor=0.05,
                                    bonus_max=1.0)
    proj_high = make_process_project(target_team_capacity_invested=100,
                                     bonus_base=0.05, bonus_scale_factor=0.05,
                                     bonus_max=1.0)
    low_avg = sum(compute_process_bonus(proj_low, random.Random(s))
                  for s in range(50)) / 50
    high_avg = sum(compute_process_bonus(proj_high, random.Random(s))
                   for s in range(50)) / 50
    assert high_avg > low_avg


def test_process_bonus_capped_at_max():
    """Result never exceeds bonus_max even with extreme investment."""
    project = make_process_project(
        bonus_base=0.5, bonus_scale_factor=10.0, bonus_max=0.20,
        target_team_capacity_invested=1000,
    )
    for s in range(50):
        b = compute_process_bonus(project, random.Random(s))
        assert 0 <= b <= 0.20


# --- compute_degradation_pct ---

def test_degradation_pct_fresh_and_expired():
    """Fresh bonus (turns_remaining = duration) → 0; near-expired → ~1."""
    fresh = make_active_bonus(turns_remaining=12, bonus_duration_turns=12)
    half = make_active_bonus(turns_remaining=6, bonus_duration_turns=12)
    near_end = make_active_bonus(turns_remaining=1, bonus_duration_turns=12)
    assert compute_degradation_pct(fresh) == 0.0
    assert compute_degradation_pct(half) == pytest.approx(0.5)
    assert compute_degradation_pct(near_end) == pytest.approx(11 / 12)


def test_degradation_pct_zero_duration():
    """Zero duration → fully degraded (defensive)."""
    zero_dur = make_active_bonus(turns_remaining=0, bonus_duration_turns=0)
    assert compute_degradation_pct(zero_dur) == 1.0


# --- compute_effective_bonus ---

def test_effective_bonus_linear():
    """Effective = bonus_value * (turns_remaining / duration)."""
    fresh = make_active_bonus(bonus_value=0.20, turns_remaining=12,
                              bonus_duration_turns=12)
    half = make_active_bonus(bonus_value=0.20, turns_remaining=6,
                             bonus_duration_turns=12)
    quarter = make_active_bonus(bonus_value=0.20, turns_remaining=3,
                                bonus_duration_turns=12)
    assert compute_effective_bonus(fresh) == pytest.approx(0.20)
    assert compute_effective_bonus(half) == pytest.approx(0.10)
    assert compute_effective_bonus(quarter) == pytest.approx(0.05)


def test_effective_bonus_with_floor():
    """Floored bonus: effective = floor + (peak - floor) * spike_fraction.

    peak 0.20, frac 0.25 → floor 0.05. Full duration = 0.20; fully expired = the floor 0.05.
    """
    full = make_active_bonus(bonus_value=0.20, permanent_floor_fraction=0.25,
                             turns_remaining=12, bonus_duration_turns=12)
    half = make_active_bonus(bonus_value=0.20, permanent_floor_fraction=0.25,
                             turns_remaining=6, bonus_duration_turns=12)
    expired = make_active_bonus(bonus_value=0.20, permanent_floor_fraction=0.25,
                                turns_remaining=0, bonus_duration_turns=12)
    assert compute_effective_bonus(full) == pytest.approx(0.20)
    # half: 0.05 + (0.20 - 0.05) * 0.5 = 0.125
    assert compute_effective_bonus(half) == pytest.approx(0.125)
    assert compute_effective_bonus(expired) == pytest.approx(0.05)


# --- compute_maintenance_cost ---

def test_maintenance_cost():
    """Maintenance = max(1, round(degradation * original_cost))."""
    # Half degraded, original cost 4 → round(0.5*4)=2
    half_deg = make_active_bonus(turns_remaining=6, bonus_duration_turns=12,
                                 original_ops_capacity_cost=4)
    assert compute_maintenance_cost(half_deg) == 2

    # Fresh → degradation 0 → max(1, 0) = 1 (floor)
    fresh = make_active_bonus(turns_remaining=12, bonus_duration_turns=12,
                              original_ops_capacity_cost=4)
    assert compute_maintenance_cost(fresh) == 1

    # Near full degradation
    near_end = make_active_bonus(turns_remaining=1, bonus_duration_turns=12,
                                 original_ops_capacity_cost=6)
    # degradation = 11/12, round(11/12*6) = round(5.5) = 6 (banker's rounding rounds 5.5 to 6)
    assert compute_maintenance_cost(near_end) in {5, 6}


# --- get_active_bonus ---

def test_get_active_bonus_takes_strongest_not_sum():
    """Same-type bonuses REPLACE rather than stack: the strongest effective bonus wins.

    A tier-0 (0.20) and a tier-1 (0.30) of the same type yield 0.30 (a net +0.10 win), not
    0.50 — process improvement is incremental, not additive.
    """
    bonuses = [
        make_active_bonus(project_id="A", target_function="sales",
                          bonus_type="conversion_rate", bonus_value=0.20,
                          turns_remaining=12, bonus_duration_turns=12),  # full = 0.20 (tier-0)
        make_active_bonus(project_id="B", target_function="sales",
                          bonus_type="conversion_rate", bonus_value=0.30,
                          turns_remaining=12, bonus_duration_turns=12),  # full = 0.30 (tier-1)
        make_active_bonus(project_id="C", target_function="engineering",
                          bonus_type="bug_rate_reduction", bonus_value=0.30,
                          turns_remaining=12, bonus_duration_turns=12),
    ]
    sales_conv = get_active_bonus(bonuses, "sales", "conversion_rate")
    eng_bug = get_active_bonus(bonuses, "engineering", "bug_rate_reduction")
    sales_other = get_active_bonus(bonuses, "sales", "other_type")
    assert sales_conv == pytest.approx(0.30)  # max(0.20, 0.30), NOT 0.50
    assert eng_bug == pytest.approx(0.30)
    assert sales_other == 0.0


def test_get_active_bonus_floored_at_zero_still_contributes():
    """A floored bonus pinned at turns_remaining=0 contributes its permanent floor to the max."""
    bonuses = [
        # Spike fully decayed; floor = 0.20 * 0.25 = 0.05 persists.
        make_active_bonus(project_id="T0", target_function="sales",
                          bonus_type="conversion_rate", bonus_value=0.20,
                          permanent_floor_fraction=0.25,
                          turns_remaining=0, bonus_duration_turns=12),
    ]
    assert get_active_bonus(bonuses, "sales", "conversion_rate") == pytest.approx(0.05)


# --- tick_bonus_durations ---

def test_tick_bonus_durations():
    """Each tick decrements turns_remaining; bonuses with new_turns <= 0 are removed."""
    bonuses = [
        make_active_bonus(project_id="A", turns_remaining=3),
        make_active_bonus(project_id="B", turns_remaining=1),  # will expire
        make_active_bonus(project_id="C", turns_remaining=12),
    ]
    ticked = tick_bonus_durations(bonuses)
    ids = {b.project_id for b in ticked}
    assert ids == {"A", "C"}
    a = next(b for b in ticked if b.project_id == "A")
    c = next(b for b in ticked if b.project_id == "C")
    assert a.turns_remaining == 2
    assert c.turns_remaining == 11


def test_tick_preserves_floored_bonus_at_zero():
    """A floored bonus ticks down to 0 and STAYS (pinned); a non-floored sibling is removed."""
    bonuses = [
        make_active_bonus(project_id="FLOOR", turns_remaining=1,
                          bonus_value=0.20, permanent_floor_fraction=0.25),
        make_active_bonus(project_id="DECAY", turns_remaining=1,
                          permanent_floor_fraction=0.0),
    ]
    # First tick: FLOOR 1 → pinned at 0 (kept); DECAY 1 → removed.
    ticked = tick_bonus_durations(bonuses)
    ids = {b.project_id for b in ticked}
    assert ids == {"FLOOR"}
    floored = next(b for b in ticked if b.project_id == "FLOOR")
    assert floored.turns_remaining == 0
    # Its effective value is exactly the floor — still delivering, not worthless.
    assert compute_effective_bonus(floored) == pytest.approx(0.05)

    # Second tick: floored bonus stays pinned at 0 indefinitely.
    ticked_again = tick_bonus_durations(ticked)
    assert {b.project_id for b in ticked_again} == {"FLOOR"}
    assert ticked_again[0].turns_remaining == 0
