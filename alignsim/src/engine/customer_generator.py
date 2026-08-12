"""Procedural customer generation engine.

All scenario-specific data comes from CustomerGeneratorConfig — no hardcoded
feature IDs, segment names, or deal values. The engine reads config fields and
works for any scenario that provides a valid config.
"""

import math
import random

from alignsim.src.models.entities import Customer, CustomerRubric, CustomerStage, Engagement, Segment
from alignsim.src.models.scenario import CustomerGeneratorConfig


def get_feature_tier(feature_id: str, features_dict: dict) -> int:
    """Determine tier by counting dependency depth. Tier 1 = 0 hops."""
    feature = features_dict.get(feature_id)
    if feature is None:
        return 1
    if not feature.depends_on:
        return 1
    return 1 + max(get_feature_tier(dep, features_dict) for dep in feature.depends_on)


def get_adjacent_features(feature_id: str, features_dict: dict) -> list[str]:
    """Return features one DAG hop away (shared parent or direct dependency)."""
    feature = features_dict.get(feature_id)
    if feature is None:
        return []

    adjacent: set[str] = set()

    for dep in feature.depends_on:
        adjacent.add(dep)

    for fid, f in features_dict.items():
        if fid == feature_id:
            continue
        if feature_id in f.depends_on:
            adjacent.add(fid)
        if feature.depends_on and any(dep in f.depends_on for dep in feature.depends_on):
            adjacent.add(fid)

    adjacent.discard(feature_id)
    return sorted(adjacent)


def _weighted_choice(weights: dict[str, float], rng: random.Random) -> str:
    """Sample a key from a {key: weight} dict."""
    items = list(weights.items())
    keys = [k for k, _ in items]
    vals = [v for _, v in items]
    total = sum(vals)
    if total <= 0:
        return rng.choice(keys)
    r = rng.random() * total
    cumulative = 0.0
    for key, val in zip(keys, vals):
        cumulative += val
        if r <= cumulative:
            return key
    return keys[-1]


def _weighted_choice_int(weights: dict[int, float], rng: random.Random) -> int:
    """Sample an int key from a {int: weight} dict."""
    items = list(weights.items())
    keys = [k for k, _ in items]
    vals = [v for _, v in items]
    total = sum(vals)
    if total <= 0:
        return rng.choice(keys)
    r = rng.random() * total
    cumulative = 0.0
    for key, val in zip(keys, vals):
        cumulative += val
        if r <= cumulative:
            return key
    return keys[-1]


def generate_customer(
    target_features: list[str],
    shipped_features: list[str],
    config: CustomerGeneratorConfig,
    features_dict: dict,
    rng: random.Random,
    customer_id: int,
) -> Customer:
    """Generate a single customer based on target features and config."""
    # 1. Segment selection: blend affinities from target features with base weights
    blended_weights: dict[str, float] = {}
    for seg, base_w in config.segment_weights.items():
        blended_weights[seg] = base_w

    if target_features:
        for feat_id in target_features:
            affinity = config.feature_segment_affinity.get(feat_id, {})
            for seg, w in affinity.items():
                blended_weights[seg] = blended_weights.get(seg, 0) + w

    segment_str = _weighted_choice(blended_weights, rng)
    segment = Segment(segment_str)

    # 2. Size
    size_dist = config.size_distributions.get(segment_str, {1: 1.0})
    size = _weighted_choice_int(size_dist, rng)

    # 3. Deal value
    base_deal = config.deal_value_per_size.get(segment_str, 1000)
    jitter = config.deal_value_jitter
    raw_deal = size * base_deal * rng.uniform(1 - jitter, 1 + jitter)
    deal_value = max(50, round(raw_deal / 50) * 50)

    # 4. Feature needs (1-3 features)
    min_needs, max_needs = config.feature_needs_count
    num_needs = rng.randint(min_needs, max_needs)

    feature_needs: dict[str, dict[str, float]] = {}
    for feat_id in target_features[:1]:
        base = round(rng.uniform(0.3, 0.5), 2)
        feature_needs[feat_id] = {
            "mvp": round(base, 2),
            "solid": round(base + 0.2, 2),
            "polished": round(base + 0.4, 2),
        }

    remaining_slots = num_needs - len(feature_needs)
    if remaining_slots > 0:
        adjacent_pool: list[str] = []
        for feat_id in target_features:
            adjacent_pool.extend(get_adjacent_features(feat_id, features_dict))
        adjacent_pool = [f for f in set(adjacent_pool) if f not in feature_needs]
        rng.shuffle(adjacent_pool)
        for feat_id in adjacent_pool[:remaining_slots]:
            base = round(rng.uniform(0.3, 0.5), 2)
            feature_needs[feat_id] = {
                "mvp": round(base, 2),
                "solid": round(base + 0.2, 2),
                "polished": round(base + 0.4, 2),
            }

    if not feature_needs and target_features:
        feat_id = target_features[0]
        base = round(rng.uniform(0.3, 0.5), 2)
        feature_needs[feat_id] = {
            "mvp": round(base, 2),
            "solid": round(base + 0.2, 2),
            "polished": round(base + 0.4, 2),
        }

    # 5. Rubric weights: archetype + noise, normalize to 1.0
    archetype = config.rubric_archetypes.get(segment_str, {
        "feature_coverage": 0.25, "price": 0.25, "maturity": 0.25, "support": 0.25,
    })
    rubric_raw: dict[str, float] = {}
    for comp, base_w in archetype.items():
        rubric_raw[comp] = max(0.01, base_w + rng.uniform(-0.08, 0.08))
    rubric_total = sum(rubric_raw.values())
    rubric_norm = {k: round(v / rubric_total, 3) for k, v in rubric_raw.items()}

    residual = round(1.0 - sum(rubric_norm.values()), 3)
    if residual != 0 and rubric_norm:
        first_key = next(iter(rubric_norm))
        rubric_norm[first_key] = round(rubric_norm[first_key] + residual, 3)

    rubric = CustomerRubric(
        feature_coverage=rubric_norm.get("feature_coverage", 0.25),
        price=rubric_norm.get("price", 0.25),
        maturity=rubric_norm.get("maturity", 0.25),
        support=rubric_norm.get("support", 0.25),
    )

    # 6. Dealbreakers: tier-3+ features in needs
    dealbreakers: list[str] = []
    for feat_id in feature_needs:
        tier = get_feature_tier(feat_id, features_dict)
        if tier >= 4:
            if rng.random() < 0.60:
                dealbreakers.append(feat_id)
        elif tier >= 3:
            if rng.random() < 0.30:
                dealbreakers.append(feat_id)

    # 7. Known needs: 50-80% of feature_needs keys
    all_need_keys = list(feature_needs.keys())
    known_fraction = rng.uniform(0.5, 0.8)
    known_count = max(1, round(len(all_need_keys) * known_fraction))
    known_needs = sorted(rng.sample(all_need_keys, min(known_count, len(all_need_keys))))

    # 8. Timeline
    timeline_range = config.timeline_range.get(segment_str, (20, 30))
    timeline = rng.randint(timeline_range[0], timeline_range[1])

    # 9. Discovery difficulty (tier-based)
    max_tier = max((get_feature_tier(f, features_dict) for f in target_features), default=2)
    tier_key = f"tier{max_tier}" if max_tier >= 2 else "tier2"
    diff_range = config.discovery_difficulty_range.get(tier_key, (1.5, 3.0))
    discovery_difficulty = round(rng.uniform(diff_range[0], diff_range[1]), 1)

    # 10. Churn drivers: features with highest satisfaction weights become churn drivers
    churn_drivers: dict[str, float] = {}
    if feature_needs:
        total_sat = sum(max(scores.values()) for scores in feature_needs.values())
        if total_sat > 0:
            for feat_id, scores in feature_needs.items():
                weight = round(max(scores.values()) / total_sat, 2)
                if weight > 0:
                    churn_drivers[feat_id] = weight

    # 11. Close threshold (per-customer jitter around mean)
    close_threshold = max(0.50, min(0.95, rng.gauss(
        config.close_threshold_mean, config.close_threshold_std,
    )))

    # 12. Desired price point (pricing negotiation)
    desired_price_point = 0
    discount_range = config.desired_price_discount_range.get(segment_str)
    if discount_range is not None:
        discount = rng.uniform(discount_range[0], discount_range[1])
        desired_price_point = max(50, round(deal_value * (1 - discount) / 50) * 50)

    return Customer(
        id=f"G{customer_id:04d}",
        size=size,
        segment=segment,
        stage=CustomerStage.lead,
        engagement=Engagement.cold,
        known_needs=known_needs,
        deal_value=deal_value,
        rubric=rubric,
        feature_needs=feature_needs,
        dealbreakers=dealbreakers,
        timeline=timeline,
        timeline_original=timeline,
        churn_drivers=churn_drivers,
        discovery_difficulty=discovery_difficulty,
        health=8.0,
        is_visible=False,
        close_threshold=round(close_threshold, 3),
        desired_price_point=desired_price_point,
    )


def generate_discovery_candidates(
    target_features: list[str],
    shipped_features: list[str],
    config: CustomerGeneratorConfig,
    features_dict: dict,
    rng: random.Random,
    start_id: int,
    count: int,
) -> list[Customer]:
    """Generate discovery candidates. 60% use requested targets, 40% use adjacent features."""
    candidates: list[Customer] = []
    for i in range(count):
        cid = start_id + i
        if rng.random() < 0.4 and target_features:
            adj_pool: list[str] = []
            for feat_id in target_features:
                adj_pool.extend(get_adjacent_features(feat_id, features_dict))
            adj_pool = list(set(adj_pool))
            if adj_pool:
                effective_targets = [rng.choice(adj_pool)]
            else:
                effective_targets = target_features
        else:
            effective_targets = target_features

        customer = generate_customer(
            effective_targets, shipped_features, config, features_dict, rng, cid,
        )
        candidates.append(customer)
    return candidates


def _weighted_sample_no_replace(
    items: list[str],
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list[str]:
    """Sample k items without replacement, proportional to weights (deterministic)."""
    pool = list(items)
    w = list(weights)
    chosen: list[str] = []
    for _ in range(min(k, len(pool))):
        total = sum(w)
        if total <= 0:
            idx = rng.randrange(len(pool))
        else:
            r = rng.random() * total
            cumulative = 0.0
            idx = len(pool) - 1
            for i, wi in enumerate(w):
                cumulative += wi
                if r <= cumulative:
                    idx = i
                    break
        chosen.append(pool.pop(idx))
        w.pop(idx)
    return chosen


def generate_inbound_candidates(
    shipped_features: list[str],
    config: CustomerGeneratorConfig,
    features_dict: dict,
    rng: random.Random,
    start_id: int,
    count: int,
    awareness: dict[str, float] | None = None,
    awareness_bias: float = 0.0,
) -> list[Customer]:
    """Generate inbound candidates. 60-80% of needs overlap shipped, 20-40% adjacent.

    When awareness is supplied (with a positive bias), the shipped-feature seed is sampled
    with a weight toward high-awareness features — so marketing spend pulls inbound leads
    toward the features it hyped. This shapes WHICH features inbound favours, not the count.
    """
    candidates: list[Customer] = []
    for i in range(count):
        cid = start_id + i
        if shipped_features:
            sample_size = min(rng.randint(1, 3), len(shipped_features))
            if awareness and awareness_bias > 0:
                weights = [1.0 + awareness_bias * awareness.get(f, 0.0) for f in shipped_features]
                shipped_sample = _weighted_sample_no_replace(shipped_features, weights, sample_size, rng)
            else:
                shipped_sample = rng.sample(shipped_features, sample_size)
        else:
            shipped_sample = []

        customer = generate_customer(
            shipped_sample, shipped_features, config, features_dict, rng, cid,
        )
        candidates.append(customer)
    return candidates


def filter_hidden_by_features(
    hidden_customers: list[Customer],
    target_features: list[str],
) -> list[Customer]:
    """Return handwritten hidden customers whose feature_needs overlap with target_features."""
    target_set = set(target_features)
    return [
        c for c in hidden_customers
        if target_set & set(c.feature_needs.keys())
    ]
