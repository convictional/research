"""Tests for alignsim.src.engine.customer_generator — procedural customer generation."""

import random

import pytest

from alignsim.src.engine.customer_generator import (
    filter_hidden_by_features,
    generate_customer,
    generate_discovery_candidates,
    generate_inbound_candidates,
    get_adjacent_features,
    get_feature_tier,
)
from alignsim.src.engine.customer_logic import compute_rubric_satisfaction
from alignsim.src.models.entities import CustomerStage, FeatureStatus, Segment
from alignsim.src.models.scenario import CustomerGeneratorConfig

from .factories import make_customer, make_feature, make_generator_config


def _make_test_config() -> CustomerGeneratorConfig:
    return CustomerGeneratorConfig(
        feature_segment_affinity={
            "F01": {"startup": 0.25, "growth": 0.25, "mid_market": 0.25, "enterprise": 0.25},
            "F02": {"growth": 0.45, "startup": 0.40, "mid_market": 0.10, "enterprise": 0.05},
            "F14": {"enterprise": 0.80, "growth": 0.15, "mid_market": 0.05, "startup": 0.00},
        },
        rubric_archetypes={
            "startup": {"feature_coverage": 0.30, "price": 0.35, "maturity": 0.15, "support": 0.20},
            "growth": {"feature_coverage": 0.40, "price": 0.20, "maturity": 0.20, "support": 0.20},
            "mid_market": {"feature_coverage": 0.40, "price": 0.15, "maturity": 0.25, "support": 0.20},
            "enterprise": {"feature_coverage": 0.45, "price": 0.10, "maturity": 0.30, "support": 0.15},
        },
        segment_weights={"startup": 0.30, "growth": 0.30, "mid_market": 0.25, "enterprise": 0.15},
        size_distributions={
            "startup": {1: 0.40, 2: 0.40, 3: 0.20},
            "growth": {2: 0.30, 3: 0.40, 4: 0.30},
            "mid_market": {2: 0.20, 3: 0.40, 4: 0.30, 5: 0.10},
            "enterprise": {3: 0.15, 4: 0.40, 5: 0.45},
        },
        deal_value_per_size={"startup": 900, "growth": 1350, "mid_market": 1125, "enterprise": 1800},
        discovery_difficulty_range={"tier2": (1.5, 2.5), "tier3": (3.0, 4.5), "tier4": (4.5, 5.5)},
        timeline_range={
            "startup": (24, 35), "growth": (22, 30),
            "mid_market": (20, 28), "enterprise": (16, 22),
        },
    )


def _make_test_features_dict() -> dict:
    features = [
        make_feature(id="F01", name="Core", depends_on=[], status=FeatureStatus.shipped_mvp),
        make_feature(id="F02", name="Reports", depends_on=["F01"]),
        make_feature(id="F03", name="Alerts", depends_on=["F01"]),
        make_feature(id="F04", name="Dashboard", depends_on=["F01"]),
        make_feature(id="F06", name="Analytics", depends_on=["F02"]),
        make_feature(id="F14", name="Compliance", depends_on=["F06"]),
    ]
    return {f.id: f for f in features}


# --- get_feature_tier ---

def test_get_feature_tier():
    features_dict = _make_test_features_dict()
    assert get_feature_tier("F01", features_dict) == 1
    assert get_feature_tier("F02", features_dict) == 2
    assert get_feature_tier("F06", features_dict) == 3
    assert get_feature_tier("F14", features_dict) == 4
    assert get_feature_tier("MISSING", features_dict) == 1


# --- get_adjacent_features ---

def test_get_adjacent_features():
    features_dict = _make_test_features_dict()
    adj_f02 = get_adjacent_features("F02", features_dict)
    assert "F01" in adj_f02
    assert "F03" in adj_f02 or "F04" in adj_f02
    assert "F06" in adj_f02

    adj_f01 = get_adjacent_features("F01", features_dict)
    assert "F02" in adj_f01
    assert "F03" in adj_f01
    assert "F04" in adj_f01


# --- generate_customer ---

def test_generate_customer_valid_pydantic():
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    rng = random.Random(42)
    customer = generate_customer(["F02"], ["F01"], config, features_dict, rng, 1)
    assert customer.id == "G0001"
    assert customer.size >= 1
    assert customer.deal_value > 0


def test_generate_customer_rubric_sums_to_one():
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    rng = random.Random(42)
    for seed in range(10):
        rng = random.Random(seed)
        customer = generate_customer(["F02"], ["F01"], config, features_dict, rng, seed)
        total = (
            customer.rubric.feature_coverage
            + customer.rubric.price
            + customer.rubric.maturity
            + customer.rubric.support
        )
        assert abs(total - 1.0) < 0.01, f"Rubric sum {total} != 1.0 for seed {seed}"


def test_generate_customer_id_format():
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    rng = random.Random(42)
    c1 = generate_customer(["F02"], ["F01"], config, features_dict, rng, 1)
    c2 = generate_customer(["F02"], ["F01"], config, features_dict, rng, 42)
    assert c1.id == "G0001"
    assert c2.id == "G0042"


def test_generate_customer_feature_needs_from_targets():
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    rng = random.Random(42)
    customer = generate_customer(["F02"], ["F01"], config, features_dict, rng, 1)
    assert "F02" in customer.feature_needs


def test_generate_customer_segment_correlates_with_affinity():
    """Targeting F14 (enterprise=0.80) should produce mostly enterprise customers."""
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    segments: list[str] = []
    for seed in range(100):
        rng = random.Random(seed)
        c = generate_customer(["F14"], ["F01"], config, features_dict, rng, seed)
        segments.append(c.segment.value)
    enterprise_pct = segments.count("enterprise") / len(segments)
    assert enterprise_pct > 0.30, f"Expected >30% enterprise, got {enterprise_pct:.0%}"


def test_generate_customer_deal_value_scales_with_size():
    """Larger sizes should produce higher deal values on average."""
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    values_by_size: dict[int, list[int]] = {}
    for seed in range(200):
        rng = random.Random(seed)
        c = generate_customer(["F02"], ["F01"], config, features_dict, rng, seed)
        values_by_size.setdefault(c.size, []).append(c.deal_value)
    sizes = sorted(values_by_size.keys())
    if len(sizes) >= 2:
        avg_small = sum(values_by_size[sizes[0]]) / len(values_by_size[sizes[0]])
        avg_large = sum(values_by_size[sizes[-1]]) / len(values_by_size[sizes[-1]])
        assert avg_large > avg_small


def test_generate_customer_deterministic_with_seed():
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    c1 = generate_customer(["F02"], ["F01"], config, features_dict, random.Random(99), 1)
    c2 = generate_customer(["F02"], ["F01"], config, features_dict, random.Random(99), 1)
    assert c1.id == c2.id
    assert c1.segment == c2.segment
    assert c1.size == c2.size
    assert c1.deal_value == c2.deal_value
    assert c1.feature_needs == c2.feature_needs
    assert c1.rubric == c2.rubric


# --- generate_discovery_candidates ---

def test_generate_discovery_candidates_count():
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    rng = random.Random(42)
    candidates = generate_discovery_candidates(
        ["F02"], ["F01"], config, features_dict, rng, start_id=1, count=5,
    )
    assert len(candidates) == 5
    ids = [c.id for c in candidates]
    assert len(set(ids)) == 5


# --- generate_inbound_candidates ---

def test_generate_inbound_candidates_overlap_shipped():
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    rng = random.Random(42)
    candidates = generate_inbound_candidates(
        ["F01", "F02"], config, features_dict, rng, start_id=1, count=10,
    )
    assert len(candidates) == 10
    overlap_count = sum(
        1 for c in candidates
        if set(c.feature_needs.keys()) & {"F01", "F02"}
    )
    assert overlap_count > 0


# --- filter_hidden_by_features ---

def test_filter_hidden_by_features():
    c1 = make_customer(id="C1", is_visible=False, feature_needs={"F02": {"mvp": 0.5}})
    c2 = make_customer(id="C2", is_visible=False, feature_needs={"F03": {"mvp": 0.5}})
    c3 = make_customer(id="C3", is_visible=False, feature_needs={"F05": {"mvp": 0.5}})
    result = filter_hidden_by_features([c1, c2, c3], ["F02", "F03"])
    assert len(result) == 2
    ids = {c.id for c in result}
    assert ids == {"C1", "C2"}


# --- Integration tests ---

def test_generated_customer_integrates_with_satisfaction():
    """compute_rubric_satisfaction returns valid 0.0-1.0 on a generated customer."""
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    rng = random.Random(42)
    customer = generate_customer(["F01"], ["F01"], config, features_dict, rng, 1)
    satisfaction = compute_rubric_satisfaction(customer, features_dict)
    assert 0.0 <= satisfaction <= 1.0


def test_generated_customer_pipeline_advancement():
    """Generated customer starts as invisible lead — can be set visible and advanced."""
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    rng = random.Random(42)
    customer = generate_customer(["F02"], ["F01"], config, features_dict, rng, 1)
    assert customer.stage == CustomerStage.lead
    assert customer.is_visible is False
    customer.is_visible = True
    customer.stage = CustomerStage.prospect
    assert customer.stage == CustomerStage.prospect


def test_determinism_across_full_generation():
    """Same seed + same actions = identical generated customers."""
    config = _make_test_config()
    features_dict = _make_test_features_dict()

    def run(seed: int) -> list[str]:
        rng = random.Random(seed)
        candidates = generate_discovery_candidates(
            ["F02"], ["F01"], config, features_dict, rng, start_id=1, count=5,
        )
        return [
            f"{c.id}:{c.segment.value}:{c.size}:{c.deal_value}"
            for c in candidates
        ]

    assert run(99) == run(99)


def test_discovery_prefers_handwritten_before_generated():
    """filter_hidden_by_features returns handwritten customers that match target features."""
    handwritten = [
        make_customer(id="C1", is_visible=False, feature_needs={"F01": {"mvp": 0.6}}),
        make_customer(id="C2", is_visible=False, feature_needs={"F02": {"mvp": 0.5}}),
    ]
    matching = filter_hidden_by_features(handwritten, ["F01"])
    assert len(matching) == 1
    assert matching[0].id == "C1"


def test_generated_customer_churn_drivers_valid():
    """Churn drivers reference valid feature IDs from feature_needs."""
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    for seed in range(20):
        rng = random.Random(seed)
        customer = generate_customer(["F02"], ["F01"], config, features_dict, rng, seed)
        for feat_id in customer.churn_drivers:
            assert feat_id in customer.feature_needs


def test_generated_customer_desired_price_when_config_has_range():
    """When desired_price_discount_range is set, generated customers get a desired_price_point."""
    config = _make_test_config()
    config.desired_price_discount_range = {
        "startup": (0.05, 0.20),
        "growth": (0.05, 0.15),
        "mid_market": (0.03, 0.12),
        "enterprise": (0.02, 0.10),
    }
    features_dict = _make_test_features_dict()
    customers_with_price = 0
    for seed in range(50):
        rng = random.Random(seed)
        c = generate_customer(["F02"], ["F01"], config, features_dict, rng, seed)
        if c.desired_price_point > 0:
            customers_with_price += 1
            assert c.desired_price_point <= c.deal_value
            assert c.desired_price_point >= 50
    assert customers_with_price == 50


def test_generated_customer_no_desired_price_without_config():
    """Without desired_price_discount_range, desired_price_point stays 0."""
    config = _make_test_config()
    features_dict = _make_test_features_dict()
    rng = random.Random(42)
    c = generate_customer(["F02"], ["F01"], config, features_dict, rng, 1)
    assert c.desired_price_point == 0


def test_seed_stage_customers_have_desired_prices():
    """Handwritten seed_stage customers all get desired_price_point assigned."""
    from alignsim.src.scenarios.seed_stage import create_seed_stage_scenario
    scenario = create_seed_stage_scenario(seed=42)
    with_price = [c for c in scenario.customers if c.desired_price_point > 0]
    assert len(with_price) == len(scenario.customers)
    for c in with_price:
        assert c.desired_price_point <= c.deal_value
        assert c.desired_price_point >= 50


def test_make_generator_config_factory():
    """Factory produces a valid config."""
    config = make_generator_config()
    assert "F01" in config.feature_segment_affinity
    assert "startup" in config.segment_weights
