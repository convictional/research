"""Pure functions for Ops cross-functional analyses.

`compute_analysis(analysis_type, state, calibration) -> dict` converts the OBSERVABLE past
(event log, per-pool capacity used/available, churn/marketing history, visible customer
stage/deal_value) into shared foresight. Each analysis is cross-functional — uncomputable by
any single partitioned (C3) role-agent because it never observed the other streams.

Hard constraints:
  * NO RNG — a pure, deterministic function of observable history (analysis adds no entropy).
  * NO HIDDEN FIELDS — never reads true rubric, desired_price_point, close_threshold,
    unrevealed churn_drivers, or undiscovered emergent needs. Only public/observable state.
  * DESCRIPTIVE / PREDICTIVE, never prescriptive-optimal (no "best move" — that would game the
    decision-quality metric).

This module is intentionally self-contained: the analysis suite can be extended or re-balanced
without touching the action/resolution plumbing. The seams to extend are the AnalysisType
literal (models/actions.py) and the dispatch in compute_analysis below.
"""

import statistics
from collections import defaultdict

from alignsim.src.models.entities import CustomerStage
from alignsim.src.models.game_state import GameState
from alignsim.src.models.scenario import CalibrationParams

# Trailing window (turns) for "recent" vs "lifetime" comparisons.
TRAILING_WINDOW = 8

# awareness_attribution ESTIMATES the marketing->lead lag from observed data rather than reading
# the engine's true marketing_lag_turns (deliberately withheld from agents). These are properties
# of the ANALYSIS — the widest lag it will try to detect, and the minimum aligned (spend, lead)
# pairs before a candidate lag is credible — NOT engine constants.
MAX_LAG_SWEEP = 12
MIN_LAG_PAIRS = 3

# Engine pool names and where each turn's usage / current availability live.
_POOLS = ("engineering", "sales", "support", "marketing", "ops")
_POOL_USED_ATTR = {
    "engineering": "eng_capacity_used",
    "sales": "sales_capacity_used",
    "support": "support_capacity_used",
    "marketing": "marketing_capacity_used",
    "ops": "ops_capacity_used",
}
_POOL_AVAIL_ATTR = {
    "engineering": "eng_capacity",
    "sales": "sales_capacity",
    "support": "support_capacity",
    "marketing": "marketing_capacity",
    "ops": "ops_capacity",
}

# Ordered pipeline stages used by the conversion funnel.
_PIPELINE_STAGES = ("lead", "prospect", "qualified", "in_deal")


# --- small shared helpers -------------------------------------------------


def _trailing(records: list, window: int) -> list:
    return records[-window:] if window > 0 else list(records)


def _count_prefix(records: list, prefix: str) -> int:
    return sum(1 for r in records for e in r.events if e.startswith(prefix))


def _safe_ratio(numerator: float, denominator: float, ndigits: int = 3) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, ndigits)


def _confidence_for_n(n: int) -> str:
    """Honest confidence band by sample size — no point estimate dressed as certainty."""
    if n < 3:
        return "none"
    if n < 8:
        return "low"
    if n < 16:
        return "medium"
    return "high"


# --- conversion_funnel (sales) -------------------------------------------


def _conversion_funnel(state: "GameState", calibration: "CalibrationParams") -> dict:
    records = state.turn_history
    window = _trailing(records, TRAILING_WINDOW)

    def tally_transitions(recs: list) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in recs:
            for e in r.events:
                if e.startswith("stage_advanced:"):
                    parts = e.split(":")
                    if len(parts) >= 3 and "->" in parts[2]:
                        counts[parts[2]] = counts.get(parts[2], 0) + 1
        return counts

    transitions_lifetime = tally_transitions(records)
    transitions_window = tally_transitions(window)

    won_window = _count_prefix(window, "deal_won:")
    won_lifetime = _count_prefix(records, "deal_won:")
    lost_window = _count_prefix(window, "deal_lost:")
    lost_lifetime = _count_prefix(records, "deal_lost:")
    resets_window = _count_prefix(window, "timeline_expired_reset:")
    resets_lifetime = _count_prefix(records, "timeline_expired_reset:")

    # Chained funnel conversion (each ratio relative to the upstream transition count). The
    # in_deal->customer step is the close (emitted as deal_won, not stage_advanced).
    lead_to_prospect = transitions_lifetime.get("lead->prospect", 0)
    prospect_to_qualified = transitions_lifetime.get("prospect->qualified", 0)
    qualified_to_in_deal = transitions_lifetime.get("qualified->in_deal", 0)
    funnel_rates = {
        "prospect_to_qualified": _safe_ratio(prospect_to_qualified, lead_to_prospect),
        "qualified_to_in_deal": _safe_ratio(qualified_to_in_deal, prospect_to_qualified),
        "in_deal_to_close": _safe_ratio(won_lifetime, qualified_to_in_deal),
    }

    # Median turns-in-stage: reconstruct each customer's stage timeline from the event log
    # (stage_advanced + deal_won as ->customer + timeline_expired_reset as ->lead), then measure
    # the gap between entering a stage and leaving it.
    cust_timeline: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in records:
        for e in r.events:
            if e.startswith("stage_advanced:"):
                parts = e.split(":")
                if len(parts) >= 3 and "->" in parts[2]:
                    cust_timeline[parts[1]].append((r.turn, parts[2].split("->")[1]))
            elif e.startswith("deal_won:"):
                cust_timeline[e.split(":")[1]].append((r.turn, "customer"))
            elif e.startswith("timeline_expired_reset:"):
                cust_timeline[e.split(":")[1]].append((r.turn, "lead"))

    durations: dict[str, list[int]] = defaultdict(list)
    for events in cust_timeline.values():
        events.sort(key=lambda x: x[0])
        for (t1, stage_in), (t2, _stage_next) in zip(events, events[1:]):
            durations[stage_in].append(t2 - t1)
    median_turns_in_stage = {
        stage: round(statistics.median(ds), 1) for stage, ds in durations.items() if ds
    }

    # Trend: deals won in the trailing window vs the window before it.
    prior_window = (
        records[-2 * TRAILING_WINDOW : -TRAILING_WINDOW] if len(records) >= 2 * TRAILING_WINDOW else []
    )
    prior_won = _count_prefix(prior_window, "deal_won:")
    if not prior_window:
        trend = "insufficient_data"
    elif won_window > prior_won:
        trend = "improving"
    elif won_window < prior_won:
        trend = "declining"
    else:
        trend = "stable"

    # Current visible pipeline snapshot (counts + sticker value by stage).
    snapshot = {s: 0 for s in _PIPELINE_STAGES}
    value_by_stage = {s: 0 for s in _PIPELINE_STAGES}
    for c in state.customers.values():
        if c.is_visible and c.stage.value in snapshot:
            snapshot[c.stage.value] += 1
            value_by_stage[c.stage.value] += c.deal_value

    return {
        "analysis_type": "conversion_funnel",
        "window_turns": TRAILING_WINDOW,
        "stage_transitions_window": transitions_window,
        "stage_transitions_lifetime": transitions_lifetime,
        "funnel_conversion_rates": funnel_rates,
        "median_turns_in_stage": median_turns_in_stage,
        "deals_won": {"window": won_window, "lifetime": won_lifetime},
        "deals_lost": {"window": lost_window, "lifetime": lost_lifetime},
        "timeline_resets": {"window": resets_window, "lifetime": resets_lifetime},
        "trend": trend,
        "pipeline_snapshot": snapshot,
        "pipeline_value_by_stage": value_by_stage,
        "sample_size": won_lifetime + lost_lifetime + sum(transitions_lifetime.values()),
    }


# --- retention_efficiency (cs) -------------------------------------------


def _retention_efficiency(state: "GameState", calibration: "CalibrationParams") -> dict:
    records = state.turn_history
    window = _trailing(records, TRAILING_WINDOW)
    churn_hist = state.churn_history
    window_churn = churn_hist[-TRAILING_WINDOW:]

    churn_window_total = sum(window_churn)
    churn_lifetime_total = sum(churn_hist)
    active_now = sum(1 for c in state.customers.values() if c.stage == CustomerStage.customer)

    # Intervention success ratio.
    succ = sum(
        1 for r in records for e in r.events
        if e.startswith("churn_intervention:") and e.endswith(":success")
    )
    fail = sum(
        1 for r in records for e in r.events
        if e.startswith("churn_intervention:") and e.endswith(":failed")
    )

    # Expansions per support capacity (trailing window).
    exp_window = _count_prefix(window, "expansion:")
    exp_lifetime = _count_prefix(records, "expansion:")
    support_used_window = sum(r.support_capacity_used for r in window)

    # Churn trend: trailing window total vs the window before it (lower = improving).
    prior_churn = (
        churn_hist[-2 * TRAILING_WINDOW : -TRAILING_WINDOW] if len(churn_hist) >= 2 * TRAILING_WINDOW else []
    )
    if not prior_churn:
        churn_trend = "insufficient_data"
    elif churn_window_total < sum(prior_churn):
        churn_trend = "improving"
    elif churn_window_total > sum(prior_churn):
        churn_trend = "worsening"
    else:
        churn_trend = "stable"

    return {
        "analysis_type": "retention_efficiency",
        "window_turns": TRAILING_WINDOW,
        "churn": {
            "window_total": churn_window_total,
            "lifetime_total": churn_lifetime_total,
            "window_avg_per_turn": _safe_ratio(churn_window_total, len(window_churn)) or 0.0,
            "lifetime_avg_per_turn": _safe_ratio(churn_lifetime_total, len(churn_hist)) or 0.0,
        },
        "active_customers_now": active_now,
        "intervention": {
            "successes": succ,
            "failures": fail,
            "success_ratio": _safe_ratio(succ, succ + fail),
        },
        "expansions": {"window": exp_window, "lifetime": exp_lifetime},
        "support_capacity_used_window": support_used_window,
        "expansions_per_support_capacity": _safe_ratio(exp_window, support_used_window, 4),
        "churn_trend": churn_trend,
        "sample_size": churn_lifetime_total + succ + fail + exp_lifetime,
    }


# --- awareness_attribution (marketing) -----------------------------------


def _awareness_attribution(state: "GameState", calibration: "CalibrationParams") -> dict:
    records = state.turn_history
    n_turns = len(records)

    # Align marketing capacity to completed turns (marketing_history may carry the current,
    # not-yet-recorded turn). Leads/closes derived from the same completed records.
    mkt = list(state.marketing_history[:n_turns])
    leads_by_turn = [sum(1 for e in r.events if e.startswith("inbound_lead:")) for r in records]
    closes_by_turn = [sum(1 for e in r.events if e.startswith("deal_won:")) for r in records]

    # ESTIMATE the marketing->lead lag from observed data: for each candidate lag, correlate
    # marketing spend at turn t with inbound leads at t+lag, and keep the lag with the strongest
    # (most positive) relationship. The sweep ceiling MAX_LAG_SWEEP is the analysis's OWN constant
    # — the engine's true marketing_lag_turns is never read (no estimation back-door; the estimate
    # is an honest function of observable history).
    best_lag: int | None = None
    best_corr: float | None = None
    best_n = 0
    lags_evaluated = 0
    for cand in range(0, MAX_LAG_SWEEP + 1):
        pairs = [
            (mkt[t], leads_by_turn[t + cand])
            for t in range(len(mkt))
            if t + cand < len(leads_by_turn)
        ]
        if len(pairs) < MIN_LAG_PAIRS:
            continue
        lags_evaluated += 1
        try:
            c = round(statistics.correlation([p[0] for p in pairs], [p[1] for p in pairs]), 3)
        except statistics.StatisticsError:
            continue  # zero variance at this lag — correlation undefined, skip
        if best_corr is None or c > best_corr:
            best_corr, best_lag, best_n = c, cand, len(pairs)

    total_mkt = sum(mkt)
    total_leads = sum(leads_by_turn)
    total_closes = sum(closes_by_turn)

    confidence = _confidence_for_n(best_n)
    result = {
        "analysis_type": "awareness_attribution",
        "estimated_lag_turns": best_lag,  # data-derived best-fit; None until estimable
        "marketing_to_leads_correlation": best_corr,
        "lags_evaluated": lags_evaluated,
        "confidence": confidence,
        "leads_per_marketing_capacity": _safe_ratio(total_leads, total_mkt, 4),
        "closes_per_lead": _safe_ratio(total_closes, total_leads),
        "totals": {
            "marketing_capacity": total_mkt,
            "inbound_leads": total_leads,
            "deals_won": total_closes,
        },
        "sample_size": best_n,
    }

    # Honest, actionable insufficiency signal — keyed off the analysis's OWN sweep ceiling, not any
    # engine constant: until there's enough history to span the full lag range, say how many more
    # turns until a complete sweep becomes possible.
    turns_until_full_sweep = max(0, (MAX_LAG_SWEEP + MIN_LAG_PAIRS) - n_turns)
    if turns_until_full_sweep > 0:
        result["note"] = (
            f"lag estimate limited by available history — re-run in ~{turns_until_full_sweep} "
            f"turn(s) to evaluate marketing->lead lags up to {MAX_LAG_SWEEP}"
        )

    return result


# --- capacity_bottleneck (any) -------------------------------------------


def _capacity_bottleneck(state: "GameState", calibration: "CalibrationParams") -> dict:
    records = state.turn_history
    window = _trailing(records, TRAILING_WINDOW)
    n = len(window)

    pool_stats: dict[str, dict] = {}
    for pool in _POOLS:
        total_used = sum(getattr(r, _POOL_USED_ATTR[pool]) for r in window)
        avg_used = total_used / n if n else 0.0
        available = getattr(state.resources, _POOL_AVAIL_ATTR[pool])
        pool_stats[pool] = {
            "avg_used": round(avg_used, 2),
            "available": available,
            "utilization": _safe_ratio(avg_used, available),
        }

    saturated = [
        p for p, s in pool_stats.items()
        if s["utilization"] is not None and s["utilization"] >= 0.85
    ]
    idle = [
        p for p, s in pool_stats.items()
        if s["utilization"] is not None and s["utilization"] <= 0.15
    ]

    # Capacity-bound rejections per pool (trailing window), parsed from validator reasons of the
    # form "Insufficient {pool} capacity: ...". Budget rejections ("Insufficient budget ...") are
    # ignored because their second token is not a pool name.
    rejections = {p: 0 for p in _POOLS}
    for r in window:
        for rej in r.actions_rejected:
            reason = getattr(rej, "reason", "")
            if reason.startswith("Insufficient "):
                parts = reason.split()
                if len(parts) >= 2 and parts[1] in rejections:
                    rejections[parts[1]] += 1

    return {
        "analysis_type": "capacity_bottleneck",
        "window_turns": TRAILING_WINDOW,
        "pools": pool_stats,
        "saturated_pools": saturated,
        "idle_pools": idle,
        "capacity_rejections_window": rejections,
        "total_capacity_rejections_window": sum(rejections.values()),
        "sample_size": n,
    }


# --- dispatch ------------------------------------------------------------

_DISPATCH = {
    "conversion_funnel": _conversion_funnel,
    "retention_efficiency": _retention_efficiency,
    "awareness_attribution": _awareness_attribution,
    "capacity_bottleneck": _capacity_bottleneck,
}


def compute_analysis(
    analysis_type: str,
    state: "GameState",
    calibration: "CalibrationParams",
) -> dict:
    """Compute one cross-functional analysis from observable history (pure, no RNG).

    The result dict always carries an ``analysis_type`` key so the requester can route it.
    """
    fn = _DISPATCH.get(analysis_type)
    if fn is None:
        return {"analysis_type": analysis_type, "error": "unknown_analysis_type"}
    return fn(state, calibration)
