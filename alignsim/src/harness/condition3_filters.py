"""Condition 3: observation filtering, action validation, and permission configs.

In Condition 3, each agent controls a single business function and sees only
function-specific slices of the game state. This module enforces information
asymmetry boundaries between agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alignsim.src.engine.game import GameEngine
from alignsim.src.game_cli import _observation_to_dict
from alignsim.src.models.observations import TurnObservation

# ---------------------------------------------------------------------------
# Function naming: agent function names vs engine function names
# ---------------------------------------------------------------------------
# Agent functions use "support" (matching the capacity pool name).
# The engine uses "cs" in HireAction.hiring_function, FireAction.function,
# and PendingHire.hiring_function/target_function.

AGENT_TO_ENGINE_FUNCTION: dict[str, str] = {
    "engineering": "engineering",
    "sales": "sales",
    "support": "cs",
    "marketing": "marketing",
    "ops": "ops",
}

ENGINE_TO_AGENT_FUNCTION: dict[str, str] = {v: k for k, v in AGENT_TO_ENGINE_FUNCTION.items()}

ALL_FUNCTIONS: frozenset[str] = frozenset(AGENT_TO_ENGINE_FUNCTION.keys())

STARTING_FUNCTIONS: frozenset[str] = frozenset({"engineering", "sales", "marketing"})


# ---------------------------------------------------------------------------
# Permission configuration per function
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionPermissions:
    core_actions: frozenset[str]
    obs_sections: frozenset[str]
    event_prefixes: frozenset[str]
    allowed_queries: frozenset[str]
    allowed_computes: frozenset[str]


FUNCTION_PERMISSIONS: dict[str, FunctionPermissions] = {
    "engineering": FunctionPermissions(
        core_actions=frozenset({"build", "fix_bugs", "infrastructure"}),
        obs_sections=frozenset({"global", "product_eng"}),
        event_prefixes=frozenset({
            "feature_shipped", "feature_upgraded", "bug_fixed",
            "infrastructure_work", "bug_injected",
        }),
        allowed_queries=frozenset({"feature", "bugs", "rejections"}),
        allowed_computes=frozenset({"maturity", "maturity-if", "capacity-cost"}),
    ),
    "sales": FunctionPermissions(
        core_actions=frozenset({"sell", "discover", "market_support"}),
        obs_sections=frozenset({"global", "sales"}),
        # pipeline_progression + market_support[_unmatched] are the Marketing<->Sales co-investment
        # handshake — shared between sales + marketing only (Sales sees its pipeline move; it does
        # NOT get marketing's channel/awareness internals). Kept out of SHARED_EVENT_PREFIXES.
        event_prefixes=frozenset({
            "timeline_started", "deal_won", "stage_advanced",
            "timeline_expired_reset", "deal_lost", "discovered",
            "pipeline_progression", "market_support", "market_support_unmatched",
        }),
        allowed_queries=frozenset({"customer", "rejections"}),
        allowed_computes=frozenset({"satisfaction", "capacity-cost"}),
    ),
    "support": FunctionPermissions(
        core_actions=frozenset({"support"}),
        obs_sections=frozenset({"global", "cs"}),
        # NOTE: emergent_need_injected is deliberately ABSENT — admitting it would bypass the
        # health_check discovery gate. Only reveal/met/expired (post-discovery lifecycle) and
        # the CS-owned churn_intervention outcome are surfaced. Keep emergent_need_injected out
        # of every role's prefixes and out of SHARED_EVENT_PREFIXES.
        event_prefixes=frozenset({
            "churn", "expansion", "churn_intervention",
            "emergent_need_revealed", "emergent_need_met", "emergent_need_expired",
        }),
        allowed_queries=frozenset({"customer", "rejections"}),
        allowed_computes=frozenset({"satisfaction"}),
    ),
    "marketing": FunctionPermissions(
        core_actions=frozenset({"market"}),
        obs_sections=frozenset({"global", "marketing_history"}),
        # competitor_radar + awareness_built are marketing-ONLY (a unique information source).
        # They must stay out of every other role's prefixes and out of SHARED_EVENT_PREFIXES —
        # Sales/Product see only the EFFECT (warm leads), never the awareness values or radar.
        # pipeline_progression + market_support[_unmatched] are shared with sales (co-investment).
        event_prefixes=frozenset({
            "inbound_lead", "competitor_radar", "awareness_built",
            "pipeline_progression", "market_support", "market_support_unmatched",
        }),
        allowed_queries=frozenset({"rejections"}),
        allowed_computes=frozenset(),
    ),
    "ops": FunctionPermissions(
        core_actions=frozenset({"ops_project", "ops_analysis"}),
        obs_sections=frozenset({"global", "ops"}),
        event_prefixes=frozenset({
            "ops_project_started", "ops_project_completed",
            "ops_project_refresh", "ops_project_restarted",
            "ops_project_support",
        }),
        allowed_queries=frozenset({"rejections"}),
        allowed_computes=frozenset(),
    ),
}

SHARED_EVENT_PREFIXES: frozenset[str] = frozenset({
    "hire_started", "hire_arrived", "hire_cancelled", "hire_sustained",
    "fire", "game_over", "competitive",
    # The analysis handshake echo/waste — routed tf-aware in _is_shared_event_visible so only the
    # Ops provider and the specific requesting function (parts[1]) ever see it.
    "ops_analysis", "analysis_unmatched",
})


# ---------------------------------------------------------------------------
# Observation filtering
# ---------------------------------------------------------------------------

def filter_observation(
    obs: TurnObservation,
    function: str,
    engine: GameEngine,
) -> dict[str, Any]:
    """Return a dict containing only the observation sections visible to *function*."""
    perms = FUNCTION_PERMISSIONS[function]
    full = _observation_to_dict(obs)
    filtered: dict[str, Any] = {}

    for section in perms.obs_sections:
        if section == "marketing_history":
            filtered["marketing_history"] = _build_marketing_obs(engine, obs)
        elif section in full:
            filtered[section] = full[section]

    # Cross-functional analysis results: delivered to the REQUESTER only. Keyed by this agent's
    # function, so it is structurally impossible for another role to surface another's result.
    filtered["analyses_received_this_turn"] = engine.state.pending_analyses.get(function, [])

    return filtered


def _build_marketing_obs(engine: GameEngine, obs: TurnObservation) -> dict[str, Any]:
    """Construct marketing-specific observation from game state."""
    state = engine.state
    lag = engine.scenario.calibration.marketing_lag_turns

    lagged = 0
    if len(state.marketing_history) >= lag:
        lagged = state.marketing_history[-lag]

    leads_this_turn: list[str] = []
    total_leads = 0
    radar_signals: list[str] = []
    collab_received_this_turn: list[dict[str, Any]] = []
    pipeline_progressions_this_turn: list[str] = []
    for record in state.turn_history:
        is_last_turn = record.turn == state.turn - 1
        for event in record.events:
            if event.startswith("inbound_lead:"):
                total_leads += 1
                if is_last_turn:
                    leads_this_turn.append(event.split(":", 1)[1])
            elif event.startswith("competitor_radar:") and is_last_turn:
                # Fuzzy "<feature_area>:<soon|upcoming>" — no exact turn/customer/ID.
                radar_signals.append(event.split(":", 1)[1])
            elif event.startswith("market_support:") and is_last_turn:
                # market_support:<channel>:capacity=<collab>:matched=<m>
                parts = event.split(":")
                channel = parts[1] if len(parts) > 1 else ""
                collab_cap = parts[2].split("=", 1)[1] if len(parts) > 2 and "=" in parts[2] else ""
                matched = parts[3].split("=", 1)[1] if len(parts) > 3 and "=" in parts[3] else ""
                collab_received_this_turn.append({
                    "channel": channel,
                    "sales_capacity": int(collab_cap) if collab_cap.isdigit() else collab_cap,
                    "marketing_capacity": int(matched) if matched.isdigit() else matched,
                })
            elif event.startswith("pipeline_progression:") and is_last_turn:
                # pipeline_progression:<id>:<from>-><to>
                pipeline_progressions_this_turn.append(event.split(":", 1)[1])

    marketing_bonus_active = any(
        b.bonus_type == "marketing_effectiveness"
        for b in state.active_process_bonuses
    )

    # Per-feature awareness stock (rounded), and what's still maturing and when so the agent
    # can plan around the channel lag. Aggregate pending increments by (feature, land_turn).
    awareness_by_feature = {fid: round(v, 2) for fid, v in sorted(state.awareness.items())}

    pending_totals: dict[tuple[str, int], float] = {}
    for p in state.pending_awareness:
        key = (p.feature_id, p.land_turn)
        pending_totals[key] = pending_totals.get(key, 0.0) + p.amount
    pending_awareness_summary = [
        {"feature": fid, "matures_turn": land_turn, "amount": round(amount, 2)}
        for (fid, land_turn), amount in sorted(pending_totals.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    ]

    return {
        "capacity_invested_per_turn": list(state.marketing_history),
        "lag_turns": lag,
        "lagged_investment_now_converting": lagged,
        "leads_generated_this_turn": leads_this_turn,
        "total_leads_generated": total_leads,
        "marketing_bonus_active": marketing_bonus_active,
        "sales_momentum": obs.global_dashboard.sales_momentum,
        "awareness_by_feature": awareness_by_feature,
        "pending_awareness_summary": pending_awareness_summary,
        "competitor_radar": radar_signals,
        "collab_received_this_turn": collab_received_this_turn,
        "pipeline_progressions_this_turn": pipeline_progressions_this_turn,
    }


# ---------------------------------------------------------------------------
# Event filtering
# ---------------------------------------------------------------------------

def filter_events(
    events: list[str],
    function: str,
    hire_owners: dict[str, str] | None = None,
) -> list[str]:
    """Filter turn events to only those visible to *function*.

    *hire_owners* maps hire_id → engine function name of the initiator.
    Required for correct filtering of hire_sustained/cancelled/arrived events
    (whose event strings only contain target_function, not hiring_function).
    """
    perms = FUNCTION_PERMISSIONS[function]
    engine_fn = AGENT_TO_ENGINE_FUNCTION[function]
    owners = hire_owners or {}
    filtered: list[str] = []

    for event in events:
        prefix = event.split(":")[0]

        if prefix in perms.event_prefixes:
            filtered.append(event)
        elif prefix in SHARED_EVENT_PREFIXES:
            if _is_shared_event_visible(event, prefix, engine_fn, owners):
                filtered.append(event)

    return filtered


def _is_shared_event_visible(
    event: str,
    prefix: str,
    engine_fn: str,
    hire_owners: dict[str, str],
) -> bool:
    if prefix in ("game_over", "competitive"):
        return True

    if prefix == "fire":
        parts = event.split(":")
        return len(parts) >= 2 and parts[1] == engine_fn

    if prefix in ("ops_analysis", "analysis_unmatched"):
        # ops_analysis:{tf}:{at} / analysis_unmatched:{tf}:{at}. Visible to the Ops provider and
        # to the requesting function only (parts[1] is the engine target_function, e.g. "cs").
        parts = event.split(":")
        return engine_fn == "ops" or (len(parts) >= 2 and parts[1] == engine_fn)

    if prefix.startswith("hire_"):
        parts = event.split(":")
        if len(parts) < 2:
            return False
        hire_id = parts[1]
        if prefix == "hire_started":
            return _parse_hire_started_owner(event) == engine_fn
        return hire_owners.get(hire_id) == engine_fn

    return False


def _parse_hire_started_owner(event: str) -> str | None:
    """Extract hiring function from a hire_started event.

    Format: hire_started:{id}:{target_fn}[:cross_hire_from_{hiring_fn}]:arrives_in_...
    For native hires, hiring_fn == target_fn (2nd colon-delimited field).
    """
    if ":cross_hire_from_" in event:
        for part in event.split(":"):
            if part.startswith("cross_hire_from_"):
                return part[len("cross_hire_from_"):]
    parts = event.split(":")
    return parts[2] if len(parts) >= 3 else None


# ---------------------------------------------------------------------------
# Action validation (runs before the engine validator)
# ---------------------------------------------------------------------------

@dataclass
class FunctionActionValidation:
    valid_actions: list[dict] = field(default_factory=list)
    rejected_actions: list[dict] = field(default_factory=list)


def validate_function_actions(
    actions: list[dict],
    function: str,
    engine: GameEngine,
) -> FunctionActionValidation:
    """Validate that actions belong to the function's allowed set.

    Returns valid actions to forward to the engine and rejected actions
    with reasons to return to the agent.
    """
    perms = FUNCTION_PERMISSIONS[function]
    engine_fn = AGENT_TO_ENGINE_FUNCTION[function]
    result = FunctionActionValidation()

    for action in actions:
        if not isinstance(action, dict):
            result.rejected_actions.append({
                "action": action,
                "reason": (
                    f"action must be a JSON object with an 'action_type' field, got "
                    f"{type(action).__name__}: {action!r}"
                ),
            })
            continue
        action_type = action.get("action_type", "")

        if action_type in perms.core_actions:
            result.valid_actions.append(action)
            continue

        if action_type == "hire":
            if action.get("hiring_function") == engine_fn:
                result.valid_actions.append(action)
            else:
                result.rejected_actions.append({
                    "action": action,
                    "reason": f"{function} agent can only hire from the {engine_fn} pool",
                })
            continue

        if action_type == "sustain_hire":
            hire_id = action.get("hire_id")
            hire = next(
                (h for h in engine.state.pending_hires if h.id == hire_id),
                None,
            )
            if hire is None:
                result.valid_actions.append(action)
            elif hire.hiring_function == engine_fn:
                result.valid_actions.append(action)
            else:
                owner = ENGINE_TO_AGENT_FUNCTION.get(
                    hire.hiring_function, hire.hiring_function,
                )
                result.rejected_actions.append({
                    "action": action,
                    "reason": f"{function} agent cannot sustain hire {hire_id} (owned by {owner})",
                })
            continue

        if action_type == "fire":
            if action.get("function") == engine_fn:
                result.valid_actions.append(action)
            else:
                result.rejected_actions.append({
                    "action": action,
                    "reason": f"{function} agent can only fire from {engine_fn}",
                })
            continue

        if action_type == "ops_project_support":
            project_id = action.get("project_id")
            project = engine.state.process_projects.get(project_id)
            if project is None:
                result.valid_actions.append(action)
            elif project.target_function == function:
                result.valid_actions.append(action)
            else:
                result.rejected_actions.append({
                    "action": action,
                    "reason": (
                        f"{function} agent cannot support project {project_id} "
                        f"(targets {project.target_function})"
                    ),
                })
            continue

        if action_type == "market_support":
            # Only Sales co-invests in Marketing's campaign. (Sales has it in core_actions and is
            # accepted above; this catches every other role with a clear reason.)
            if function == "sales":
                result.valid_actions.append(action)
            else:
                result.rejected_actions.append({
                    "action": action,
                    "reason": f"only the sales agent may submit market_support, not {function}",
                })
            continue

        if action_type == "ops_analysis":
            # Only Ops runs cross-functional analyses. (Ops has it in core_actions and is accepted
            # above; this catches every other role with a clear reason.)
            if function == "ops":
                result.valid_actions.append(action)
            else:
                result.rejected_actions.append({
                    "action": action,
                    "reason": f"only the ops agent may submit ops_analysis, not {function}",
                })
            continue

        if action_type == "analysis_scope":
            # A team may only scope an analysis FOR ITSELF (target_function == own function). Not in
            # any role's core_actions — this tf-check is the gate (mirrors ops_project_support).
            tf = action.get("target_function")
            if tf == engine_fn:
                result.valid_actions.append(action)
            else:
                owner = ENGINE_TO_AGENT_FUNCTION.get(tf, tf)
                result.rejected_actions.append({
                    "action": action,
                    "reason": (
                        f"{function} agent may only scope analyses for its own function; "
                        f"target_function {tf} belongs to {owner}"
                    ),
                })
            continue

        result.rejected_actions.append({
            "action": action,
            "reason": f"Action type '{action_type}' is not allowed for {function}",
        })

    return result


# ---------------------------------------------------------------------------
# Query / compute permission checks
# ---------------------------------------------------------------------------

def is_query_allowed(function: str, query_type: str) -> bool:
    return query_type in FUNCTION_PERMISSIONS[function].allowed_queries


def is_compute_allowed(function: str, compute_type: str) -> bool:
    return compute_type in FUNCTION_PERMISSIONS[function].allowed_computes
