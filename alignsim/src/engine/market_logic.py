"""Pure functions for marketing, discovery, and competitive event mechanics.

All functions compute and return values — no state mutation.
"""

import math
import random

from alignsim.src.models.entities import (
    Competitor,
    CompetitorEvent,
    Customer,
    CustomerStage,
    Engagement,
    Feature,
    FeatureStatus,
    PendingAwareness,
    Segment,
)
from alignsim.src.models.scenario import CalibrationParams, ChannelProfile


def compute_inbound_leads(
    marketing_history: list[int],
    calibration: CalibrationParams,
    process_bonus: float = 0.0,
) -> int:
    """Compute number of inbound leads for this turn based on lagged marketing investment.

    Marketing investment takes `lag_turns` to produce leads.
    process_bonus (from Ops Marketing Analytics project) multiplies effectiveness.
    """
    lag = calibration.marketing_lag_turns
    base = calibration.base_inbound_rate

    # Look at marketing investment from `lag` turns ago
    lagged_investment = 0
    if len(marketing_history) >= lag:
        lagged_investment = marketing_history[-lag]

    effectiveness = calibration.marketing_effectiveness * (1.0 + process_bonus)
    leads = base + (lagged_investment * effectiveness)
    return max(0, int(leads))


# ---------------------------------------------------------------------------
# Marketing awareness (keystone mechanic)
# ---------------------------------------------------------------------------


def effective_awareness_targets(
    target_features: list[str],
    features: dict[str, Feature],
) -> list[str]:
    """Resolve a market action's targets to concrete feature IDs.

    Empty target_features = broad across all shipped + in-progress features (marketing may
    build awareness for not-yet-shipped features, unlike discovery which is shipped-only).
    Explicit targets are filtered to features that actually exist.
    """
    if target_features:
        return [fid for fid in target_features if fid in features]

    eligible_statuses = {
        FeatureStatus.in_progress,
        FeatureStatus.shipped_mvp,
        FeatureStatus.shipped_solid,
        FeatureStatus.shipped_polished,
    }
    return [fid for fid, f in features.items() if f.status in eligible_statuses]


def schedule_awareness(
    profile: ChannelProfile,
    target_features: list[str],
    capacity: int,
    current_turn: int,
) -> list[PendingAwareness]:
    """Schedule the lagged, spread awareness increments for one market action.

    Total contribution = capacity * efficiency, split evenly across the targeted features,
    then spread evenly over `spread` turns starting at current_turn + lag. spread=1 = a
    single burst (events); large spread = durable trickle (content).
    """
    if not target_features:
        return []

    total = capacity * profile.efficiency
    per_feature = total / len(target_features)
    per_turn = per_feature / profile.spread

    pending: list[PendingAwareness] = []
    for feature_id in target_features:
        for step in range(profile.spread):
            pending.append(PendingAwareness(
                land_turn=current_turn + profile.lag + step,
                feature_id=feature_id,
                amount=per_turn,
            ))
    return pending


def mature_pending_awareness(
    pending: list[PendingAwareness],
    current_turn: int,
) -> tuple[list[PendingAwareness], list[PendingAwareness]]:
    """Partition pending awareness into (matured this turn, still waiting).

    An increment matures when the game turn reaches its land_turn (<= for robustness).
    """
    matured = [p for p in pending if p.land_turn <= current_turn]
    remaining = [p for p in pending if p.land_turn > current_turn]
    return matured, remaining


def decay_awareness(
    awareness: dict[str, float],
    decay: float,
    epsilon: float,
) -> dict[str, float]:
    """Apply per-turn decay to every feature's awareness stock, dropping tiny stocks."""
    decayed: dict[str, float] = {}
    for feature_id, value in awareness.items():
        new_value = value * (1.0 - decay)
        if new_value >= epsilon:
            decayed[feature_id] = new_value
    return decayed


def compute_awareness_score(customer: Customer, awareness: dict[str, float]) -> float:
    """A customer's awareness score = the awareness of the most-hyped feature they need."""
    if not customer.feature_needs:
        return 0.0
    return max((awareness.get(f, 0.0) for f in customer.feature_needs), default=0.0)


def compute_awareness_reveal(
    awareness_score: float,
    calibration: CalibrationParams,
    rng: random.Random,
) -> tuple[Engagement, int]:
    """Compute a freshly-revealed customer's engagement + timeline bonus from the awareness score.

    Pure — returns (engagement, timeline_bonus); the caller applies them. High awareness → warm
    (rarely hot); low → cold. timeline_bonus only ever extends the timeline (more chances for Sales
    to close), never shortens it. Used uniformly at every reveal site (inbound + discovery),
    covering both handwritten and generated leads.
    """
    if awareness_score >= calibration.awareness_warm_threshold:
        engagement = Engagement.warm
        if (
            awareness_score >= calibration.awareness_hot_threshold
            and rng.random() < calibration.awareness_hot_prob
        ):
            engagement = Engagement.hot
    else:
        engagement = Engagement.cold

    timeline_bonus = 0
    if awareness_score > 0 and calibration.awareness_hot_threshold > 0:
        frac = min(1.0, awareness_score / calibration.awareness_hot_threshold)
        timeline_bonus = round(calibration.awareness_timeline_bonus_max * frac)

    return engagement, timeline_bonus


def awareness_lead_weight(
    customer: Customer,
    awareness: dict[str, float],
    bias: float,
) -> float:
    """Selection weight pulling inbound toward customers needing high-awareness features."""
    return 1.0 + bias * compute_awareness_score(customer, awareness)


def scan_competitor_radar(
    competitors: dict[str, Competitor],
    customers: dict[str, Customer],
    awareness: dict[str, float],
    current_turn: int,
    calibration: CalibrationParams,
    rng: random.Random,
) -> list[str]:
    """Passively sense upcoming competitor events touching features marketing is active in.

    Returns fuzzy signal payloads of the form "<feature_area>:<soon|upcoming>" — never an
    exact turn, customer, or event ID. Only events within radar_lookahead_turns and only
    feature areas marketing has awareness on can be sensed; detection probability scales
    with that awareness (plus uncertainty jitter). Deterministic per seed.
    """
    lookahead = calibration.radar_lookahead_turns
    imminent_cutoff = max(1, lookahead // 2)
    hot_threshold = calibration.awareness_hot_threshold or 1.0

    signals: list[str] = []
    for competitor in competitors.values():
        for event in competitor.events:
            delta = event.turn - current_turn
            if delta <= 0 or delta > lookahead:
                continue

            # Map affected customers -> the feature areas they need, keep only ones we track.
            feature_areas: set[str] = set()
            for cid in event.affected_customers:
                cust = customers.get(cid)
                if cust is not None:
                    feature_areas.update(cust.feature_needs.keys())

            relevant = [(f, awareness.get(f, 0.0)) for f in sorted(feature_areas) if awareness.get(f, 0.0) > 0]
            if not relevant:
                continue

            max_aware = max(a for _, a in relevant)
            prob = calibration.radar_base_prob * min(1.0, max_aware / hot_threshold)
            prob += rng.uniform(-calibration.radar_uncertainty_jitter, calibration.radar_uncertainty_jitter)
            if rng.random() < prob:
                area = max(relevant, key=lambda x: x[1])[0]
                timing = "soon" if delta <= imminent_cutoff else "upcoming"
                signals.append(f"{area}:{timing}")

    return signals


# ---------------------------------------------------------------------------
# Sales-gated pipeline progression (Marketing<->Sales co-investment)
# ---------------------------------------------------------------------------

# One-stage progression ladder. Hard-capped at in_deal: closing (in_deal -> customer) must run
# the full dealbreaker + rubric gate via a normal proposal/negotiate, never via progression.
_PROGRESSION_LADDER: dict[CustomerStage, CustomerStage] = {
    CustomerStage.lead: CustomerStage.prospect,
    CustomerStage.prospect: CustomerStage.qualified,
    CustomerStage.qualified: CustomerStage.in_deal,
}


def next_pipeline_stage_capped(stage: CustomerStage) -> CustomerStage | None:
    """The next pipeline stage one step up, or None at/above in_deal (or terminal)."""
    return _PROGRESSION_LADDER.get(stage)


def roll_pipeline_progression(
    channel: str,
    marketing_cap: int,
    collab_cap: int,
    calibration: CalibrationParams,
    rng: random.Random,
) -> bool:
    """Roll whether a one-stage pipeline progression succeeds for a budget-channel campaign.

    p = clamp(base[ch] * (1 + collab_scale*log1p(min(m, s)) + budget_scale*log1p(budget_$K)), 0, max)
    where m = marketing capacity on the channel, s = sales collab capacity (min(m,s) is the joint
    Liebig commitment — both teams must commit), and budget_$K = m * budget_cost_per_capacity / 1000.
    Returns False (no draw) if the channel is unfunded or has no joint commitment.
    """
    base = calibration.progression_base_prob.get(channel, 0.0)
    if base <= 0 or marketing_cap <= 0 or collab_cap <= 0:
        return False

    profile = calibration.channel_profiles.get(channel)
    budget_per_cap = profile.budget_cost_per_capacity if profile is not None else 0
    budget_k = marketing_cap * budget_per_cap / 1000.0
    joint = min(marketing_cap, collab_cap)

    p = base * (
        1.0
        + calibration.progression_collab_scale * math.log1p(joint)
        + calibration.progression_budget_scale * math.log1p(budget_k)
    )
    p = max(0.0, min(p, calibration.progression_max_prob))
    return rng.random() < p


def discover_customers(
    capacity: int,
    hidden_customers: list[Customer],
    segment_filter: str | None,
    rng: random.Random,
    process_bonus: float = 0.0,
) -> list[str]:
    """Attempt to discover customers from the hidden pool.

    Returns list of customer IDs that were discovered.
    process_bonus (from Ops Discovery Playbook) increases discovery probability.
    """
    if not hidden_customers:
        return []

    # Filter by segment if specified
    candidates = hidden_customers
    if segment_filter:
        try:
            seg = Segment(segment_filter)
            candidates = [c for c in hidden_customers if c.segment == seg]
        except ValueError:
            candidates = hidden_customers

    if not candidates:
        return []

    discovered = []
    remaining_capacity = capacity
    for customer in candidates:
        if remaining_capacity <= 0:
            break
        # Probability = capacity_allocated / discovery_difficulty, boosted by process bonus
        probability = min(remaining_capacity / customer.discovery_difficulty * (1.0 + process_bonus), 0.95)
        if rng.random() < probability:
            discovered.append(customer.id)
        remaining_capacity -= 1  # each attempt costs 1 capacity unit

    return discovered


def fire_competitive_events(competitors: dict[str, Competitor], turn: int) -> list[CompetitorEvent]:
    """Get all competitive events scheduled for this turn."""
    events = []
    for competitor in competitors.values():
        for event in competitor.events:
            if event.turn == turn:
                events.append(event)
    return events


def apply_competitive_pressure(
    customer: Customer,
    events: list[CompetitorEvent],
) -> float:
    """Compute the competitive pressure on a customer from this turn's events.

    Returns the new competitive_pressure value.
    """
    pressure = customer.competitive_pressure

    for event in events:
        if customer.id in event.affected_customers:
            # Pressure increases based on rubric impact
            impact = sum(event.rubric_impact.values()) / max(len(event.rubric_impact), 1)
            pressure += impact * 0.3

    # Decay pressure over time (but not below 0)
    pressure = max(0.0, pressure - 0.05)

    return min(pressure, 1.0)


def check_competitor_deal_win(
    customer: Customer,
    competitor_satisfaction: float,
    player_satisfaction: float,
) -> bool:
    """Check if a competitor wins a deal over the player.

    A competitor wins if their rubric satisfaction exceeds the player's
    for a customer whose timeline has expired.
    """
    return competitor_satisfaction > player_satisfaction
