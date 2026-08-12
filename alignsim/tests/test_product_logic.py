"""Tests for alignsim.src.engine.product_logic — pure product/engineering mechanics."""

import math
import random

import pytest

from alignsim.src.engine.product_logic import (
    _bug_fix_cost,
    _sample_severity,
    apply_build_progress,
    compute_bug_fix_progress,
    compute_tech_debt_delta,
    inject_bugs,
    inject_emergent_needs,
    poisson_sample,
)
from alignsim.src.models.entities import (
    BugSeverity,
    CustomerStage,
    FeatureStatus,
    QualityLevel,
)
from alignsim.src.models.scenario import CalibrationParams

from .factories import make_bug, make_customer, make_emergent_need, make_feature


# --- apply_build_progress ---

def test_build_progress_basic():
    """No calibration: progress increment = capacity / cost * 100."""
    feature = make_feature(cost={"mvp": 10}, status=FeatureStatus.not_started)
    progress, status = apply_build_progress(feature, 5, QualityLevel.mvp, calibration=None)
    assert progress == pytest.approx(50.0)
    assert status == FeatureStatus.in_progress


def test_build_progress_completes():
    """Capacity equal to cost completes the feature with no calibration."""
    feature = make_feature(cost={"mvp": 8}, status=FeatureStatus.not_started)
    progress, status = apply_build_progress(feature, 8, QualityLevel.mvp, calibration=None)
    assert progress == 100.0
    assert status == FeatureStatus.shipped_mvp


def test_build_progress_upgrade_delta_cost():
    """MVP→Solid pays only the delta (solid_cost - mvp_cost), not full solid cost."""
    feature = make_feature(
        cost={"mvp": 10, "solid": 25, "polished": 40},
        status=FeatureStatus.shipped_mvp,
    )
    # Delta is 25 - 10 = 15. Provide 15 capacity.
    progress, status = apply_build_progress(feature, 15, QualityLevel.solid, calibration=None)
    assert progress == 100.0
    assert status == FeatureStatus.shipped_solid


def test_build_progress_diminishing_returns():
    """Capacity above optimal yields sublinear progress vs. capacity at optimal."""
    cal = CalibrationParams(build_min_turns_factor=0.0)  # disable min_turns floor
    # Cost large enough that the 65% per-turn cap doesn't bind
    feature_at = make_feature(cost={"mvp": 200}, status=FeatureStatus.not_started, turns_worked=20)
    feature_over = feature_at.model_copy(deep=True)

    p_at, _ = apply_build_progress(feature_at, cal.build_optimal_capacity, QualityLevel.mvp, cal)
    p_over, _ = apply_build_progress(feature_over, cal.build_optimal_capacity * 2, QualityLevel.mvp, cal)

    # Doubling capacity should yield less than 2x progress due to diminishing returns
    assert p_over > p_at
    assert p_over < 2 * p_at


def test_build_progress_max_progress_cap():
    """Per-turn progress is capped at build_max_progress_pct of remaining cost."""
    cal = CalibrationParams(build_min_turns_factor=0.0)
    # Use capacity not far above optimal so diminishing returns don't shrink it
    # below the 65% cap. Cost 10, capacity 20 → effective ≈ 18.2 > cap of 6.5.
    feature = make_feature(cost={"mvp": 10}, status=FeatureStatus.not_started, turns_worked=20)
    progress, status = apply_build_progress(feature, 20, QualityLevel.mvp, cal)
    assert progress == pytest.approx(cal.build_max_progress_pct, rel=1e-6)
    assert status == FeatureStatus.in_progress


def test_build_progress_minimum_turns():
    """Large features must take ceil(remaining_cost * factor) turns, floor of 2."""
    cal = CalibrationParams()  # default factor 0.15
    # cost 100, factor 0.15 → min_turns = max(2, ceil(15)) = 15.
    # progress=90 already, capacity=20 → effective ≈ 18.2 (under 65 cap),
    # progress_increment ≈ 18.2%, new_progress capped at 100.
    # turns_worked_after = 2 < 15, so held at 99.9.
    feature = make_feature(
        cost={"mvp": 100}, status=FeatureStatus.not_started, turns_worked=1, progress=90.0,
    )
    progress, status = apply_build_progress(feature, 20, QualityLevel.mvp, cal)
    assert progress == 99.9
    assert status == FeatureStatus.in_progress


def test_build_progress_minimum_turns_floor_two():
    """Even a tiny feature takes at least 2 turns when calibration is applied."""
    cal = CalibrationParams()
    # cost 4, factor 0.15 → ceil(0.6) = 1, but floor is 2.
    # progress already 50, capacity 4 (= cost) so we'd reach 100 in this turn,
    # but turns_worked_after = 1 < 2, so held at 99.9.
    feature = make_feature(
        cost={"mvp": 4}, status=FeatureStatus.not_started, turns_worked=0, progress=50.0,
    )
    progress, status = apply_build_progress(feature, 4, QualityLevel.mvp, cal)
    assert progress == 99.9
    assert status == FeatureStatus.in_progress


def test_build_progress_zero_cost():
    """Feature with zero cost for the target quality returns unchanged."""
    feature = make_feature(
        cost={"mvp": 0, "solid": 15},
        status=FeatureStatus.in_progress,
        progress=42.0,
    )
    progress, status = apply_build_progress(feature, 5, QualityLevel.mvp, calibration=None)
    assert progress == 42.0
    assert status == FeatureStatus.in_progress


# --- compute_tech_debt_delta ---

def test_tech_debt_mvp_vs_solid_vs_polished():
    """MVP work adds the most debt, polished adds the least."""
    cal = CalibrationParams()
    mvp_delta = compute_tech_debt_delta({"mvp": 10}, 0, cal)
    solid_delta = compute_tech_debt_delta({"solid": 10}, 0, cal)
    polished_delta = compute_tech_debt_delta({"polished": 10}, 0, cal)
    assert mvp_delta == pytest.approx(1.0)
    assert solid_delta == pytest.approx(0.5)
    assert polished_delta == pytest.approx(0.2)
    assert mvp_delta > solid_delta > polished_delta


def test_tech_debt_infra_reduces():
    """Infrastructure capacity reduces debt by 1.0 per 5 units (default)."""
    cal = CalibrationParams()
    delta = compute_tech_debt_delta({}, 5, cal)
    assert delta == pytest.approx(-1.0)


def test_tech_debt_mixed():
    """Net effect is debt_increase - debt_decrease."""
    cal = CalibrationParams()
    # 10 mvp (+1.0) + 10 polished (+0.2) - 5 infra (-1.0) = 0.2
    delta = compute_tech_debt_delta({"mvp": 10, "polished": 10}, 5, cal)
    assert delta == pytest.approx(0.2)


# --- inject_bugs ---

def _shipped_feature():
    return make_feature(id="F01", status=FeatureStatus.shipped_mvp)


def test_inject_bugs_deterministic():
    """Same seed yields the same number and types of bugs."""
    cal = CalibrationParams()
    features = [_shipped_feature()]
    customers = {"C01": make_customer(id="C01", stage=CustomerStage.customer,
                                       feature_needs={"F01": {"mvp": 0.5}})}
    bugs_a = inject_bugs(10.0, features, cal, customers, 1, 5, random.Random(123))
    bugs_b = inject_bugs(10.0, features, cal, customers, 1, 5, random.Random(123))
    assert [(b.severity, b.feature_id) for b in bugs_a] == [
        (b.severity, b.feature_id) for b in bugs_b
    ]


def test_inject_bugs_scales_with_debt():
    """Higher debt level produces more bugs on average."""
    cal = CalibrationParams()
    features = [_shipped_feature()]
    low_total = sum(
        len(inject_bugs(2.0, features, cal, {}, 1, 5, random.Random(s)))
        for s in range(50)
    )
    high_total = sum(
        len(inject_bugs(20.0, features, cal, {}, 1, 5, random.Random(s)))
        for s in range(50)
    )
    assert high_total > low_total


def test_inject_bugs_severity_distribution():
    """Severity sampling matches the calibrated 20/40/40 distribution."""
    cal = CalibrationParams()
    rng = random.Random(123)
    counts = {BugSeverity.critical: 0, BugSeverity.major: 0, BugSeverity.minor: 0}
    for _ in range(10_000):
        s = _sample_severity(cal, rng)
        counts[s] += 1
    # Tolerances: ~3 sigma for binomial
    assert 1700 < counts[BugSeverity.critical] < 2300
    assert 3700 < counts[BugSeverity.major] < 4300
    assert 3700 < counts[BugSeverity.minor] < 4300


def test_inject_bugs_empty_without_shipped_features():
    """Nothing shipped → no bugs, even at high debt."""
    cal = CalibrationParams()
    bugs = inject_bugs(50.0, [], cal, {}, 1, 5, random.Random(0))
    assert bugs == []


def test_inject_bugs_rate_reduction():
    """A non-zero process bug-rate reduction strictly reduces bugs over many trials."""
    cal = CalibrationParams()
    features = [_shipped_feature()]
    base_total = sum(
        len(inject_bugs(15.0, features, cal, {}, 1, 5, random.Random(s),
                        bug_rate_reduction=0.0))
        for s in range(100)
    )
    reduced_total = sum(
        len(inject_bugs(15.0, features, cal, {}, 1, 5, random.Random(s),
                        bug_rate_reduction=0.5))
        for s in range(100)
    )
    assert reduced_total < base_total


def test_inject_bugs_affected_customers():
    """Only customers with feature in feature_needs and stage=customer are affected."""
    cal = CalibrationParams()
    features = [_shipped_feature()]
    customers = {
        "active": make_customer(id="active", stage=CustomerStage.customer,
                                feature_needs={"F01": {"mvp": 0.5}}),
        "lead": make_customer(id="lead", stage=CustomerStage.lead,
                              feature_needs={"F01": {"mvp": 0.5}}),
        "irrelevant": make_customer(id="irrelevant", stage=CustomerStage.customer,
                                    feature_needs={"F02": {"mvp": 0.5}}),
    }
    # Use a large debt to almost guarantee at least one bug
    bugs = inject_bugs(50.0, features, cal, customers, 1, 5, random.Random(0))
    assert bugs, "expected at least one bug at debt=50"
    for bug in bugs:
        assert bug.affected_customers == ["active"]


# --- bug fix mechanics ---

def test_bug_fix_costs():
    """Critical=4, major=2, minor=1 capacity required."""
    assert _bug_fix_cost(BugSeverity.critical) == 4
    assert _bug_fix_cost(BugSeverity.major) == 2
    assert _bug_fix_cost(BugSeverity.minor) == 1


def test_compute_bug_fix_progress_threshold():
    """compute_bug_fix_progress is True iff capacity >= severity cost."""
    crit = make_bug(severity=BugSeverity.critical)
    minor = make_bug(severity=BugSeverity.minor)
    assert compute_bug_fix_progress(crit, 4) is True
    assert compute_bug_fix_progress(crit, 3) is False
    assert compute_bug_fix_progress(minor, 1) is True
    assert compute_bug_fix_progress(minor, 0) is False


# --- poisson_sample ---

def test_poisson_sample_zero_lambda():
    assert poisson_sample(0.0, random.Random(0)) == 0
    assert poisson_sample(-1.0, random.Random(0)) == 0


def test_poisson_sample_deterministic():
    a = poisson_sample(3.0, random.Random(42))
    b = poisson_sample(3.0, random.Random(42))
    assert a == b


def test_poisson_sample_mean_approximates_lambda():
    """Empirical mean of many samples ≈ lambda."""
    rng = random.Random(0)
    samples = [poisson_sample(5.0, rng) for _ in range(5_000)]
    mean = sum(samples) / len(samples)
    assert 4.7 < mean < 5.3


# --- inject_emergent_needs ---

def _active(cid="C1", **kw):
    kw.setdefault("stage", CustomerStage.customer)
    return make_customer(id=cid, **kw)


def _feat(fid, status=FeatureStatus.not_started, depends_on=None):
    return make_feature(id=fid, status=status, depends_on=depends_on or [])


def test_inject_emergent_determinism_per_seed():
    cal = CalibrationParams(emergent_need_injection_floor=4.0)
    customers = {"C1": _active(known_needs=["F01"])}
    features = {f.id: f for f in [_feat("F01", FeatureStatus.shipped_mvp), _feat("F02"), _feat("F03")]}

    def run():
        needs = inject_emergent_needs(cal, customers, features, [], 1, 5, random.Random(99))
        return [(n.id, n.customer_id, n.feature_id) for n in needs]

    assert run() == run()


def test_inject_emergent_never_targets_known_needs():
    cal = CalibrationParams(emergent_need_injection_floor=8.0)
    customers = {"C1": _active(known_needs=["F02"])}
    features = {f.id: f for f in [_feat("F02"), _feat("F03")]}
    needs = inject_emergent_needs(cal, customers, features, [], 1, 5, random.Random(3))
    assert needs  # something injected
    assert all(n.feature_id != "F02" for n in needs)
    assert all(n.feature_id == "F03" for n in needs)


def test_inject_emergent_no_duplicate_open_need_per_customer():
    cal = CalibrationParams(emergent_need_injection_floor=8.0)
    customers = {"C1": _active(known_needs=["F01"])}
    features = {f.id: f for f in [_feat("F01", FeatureStatus.shipped_mvp), _feat("F02"), _feat("F03")]}
    existing = [make_emergent_need(id="EN_001", customer_id="C1", feature_id="F02")]
    needs = inject_emergent_needs(cal, customers, features, existing, 2, 5, random.Random(3))
    # No new need re-uses F02 (already open), and no two new needs share a feature.
    new_feats = [n.feature_id for n in needs]
    assert "F02" not in new_feats
    assert len(new_feats) == len(set(new_feats))
    # Combined with the existing open need, every (customer, feature) is unique.
    all_pairs = [(n.customer_id, n.feature_id) for n in existing + needs]
    assert len(all_pairs) == len(set(all_pairs))


def test_inject_emergent_active_customers_only():
    cal = CalibrationParams(emergent_need_injection_floor=8.0)
    customers = {
        "C1": _active(cid="C1", known_needs=["F01"]),
        "L1": make_customer(id="L1", stage=CustomerStage.lead, known_needs=["F01"]),
    }
    features = {f.id: f for f in [_feat("F01", FeatureStatus.shipped_mvp), _feat("F02"), _feat("F03")]}
    needs = inject_emergent_needs(cal, customers, features, [], 1, 5, random.Random(3))
    assert needs
    assert all(n.customer_id == "C1" for n in needs)


def test_inject_emergent_no_active_customers_returns_empty():
    cal = CalibrationParams(emergent_need_injection_floor=8.0)
    customers = {"L1": make_customer(id="L1", stage=CustomerStage.lead)}
    features = {f.id: f for f in [_feat("F02"), _feat("F03")]}
    assert inject_emergent_needs(cal, customers, features, [], 1, 5, random.Random(3)) == []


def test_inject_emergent_excludes_shipped_features():
    cal = CalibrationParams(emergent_need_injection_floor=8.0)
    customers = {"C1": _active(known_needs=["F01"])}
    features = {f.id: f for f in [
        _feat("F01", FeatureStatus.shipped_mvp),
        _feat("F02", FeatureStatus.shipped_solid),
        _feat("F03", FeatureStatus.not_started),
    ]}
    needs = inject_emergent_needs(cal, customers, features, [], 1, 5, random.Random(3))
    assert needs
    assert all(n.feature_id == "F03" for n in needs)  # only non-shipped, non-known option


def test_inject_emergent_skips_when_no_eligible_feature():
    cal = CalibrationParams(emergent_need_injection_floor=8.0)
    customers = {"C1": _active(known_needs=["F02"])}
    # Only feature is the customer's known need → nothing eligible.
    features = {f.id: f for f in [_feat("F02")]}
    assert inject_emergent_needs(cal, customers, features, [], 1, 5, random.Random(3)) == []


def test_inject_emergent_lambda_scales_with_active_count():
    """Expected injections scale with the number of active customers (rate term)."""
    cal = CalibrationParams(emergent_need_injection_rate=1.0, emergent_need_injection_floor=0.0)
    features = {f.id: f for f in [_feat("F01", FeatureStatus.shipped_mvp)] + [_feat(f"F{i:02d}") for i in range(2, 12)]}

    def mean_count(num_active):
        customers = {f"C{i}": _active(cid=f"C{i}", known_needs=["F01"]) for i in range(num_active)}
        total = 0
        trials = 200
        for s in range(trials):
            total += len(inject_emergent_needs(cal, customers, features, [], 1, 5, random.Random(s)))
        return total / trials

    assert mean_count(5) > mean_count(1)


def test_inject_emergent_ids_are_sequential_from_next_id():
    cal = CalibrationParams(emergent_need_injection_floor=4.0)
    customers = {"C1": _active(known_needs=["F01"])}
    features = {f.id: f for f in [_feat("F01", FeatureStatus.shipped_mvp), _feat("F02"), _feat("F03")]}
    needs = inject_emergent_needs(cal, customers, features, [], 7, 5, random.Random(99))
    assert [n.id for n in needs] == [f"EN_{7 + i:03d}" for i in range(len(needs))]
