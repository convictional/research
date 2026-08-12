"""Game CLI: the interface between agentic harnesses and the game engine.

State is persisted via pickle between invocations. All output is JSON
to stdout; human-readable messages go to stderr.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

from alignsim.src.engine.game import GameEngine
from alignsim.src.engine.observer import ObservationGenerator
from alignsim.src.engine.scoring import compute_goal_attainment
from alignsim.src.harness.inspector import GameInspector
from alignsim.src.models.actions import (
    ACTION_CLASSES,
    GameAction,
    TurnActions,
)
from alignsim.src.models.entities import QualityLevel
from alignsim.src.models.goals import score_to_player_dict
from alignsim.src.scenarios.playtest import create_playtest_scenario
from alignsim.src.scenarios.seed_stage import create_seed_stage_scenario

SCENARIO_FACTORIES = {
    "playtest": create_playtest_scenario,
    "seed_stage": create_seed_stage_scenario,
}


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

STATE_FILE = "engine.pkl"
REJECTIONS_FILE = "rejections.json"
LOG_FILE = "game_log.jsonl"


def _state_dir(args) -> Path:
    return Path(args.state_dir)


def _save_engine(engine: GameEngine, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / STATE_FILE, "wb") as f:
        pickle.dump(engine, f)


def _load_engine(state_dir: Path) -> GameEngine:
    path = state_dir / STATE_FILE
    if not path.exists():
        _err(f"No game state found at {path}. Run 'init' first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_rejections(state_dir: Path) -> list[dict]:
    path = state_dir / REJECTIONS_FILE
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_rejections(rejections: list[dict], state_dir: Path) -> None:
    with open(state_dir / REJECTIONS_FILE, "w") as f:
        json.dump(rejections, f, indent=2)


INTERNAL_SCORES_FILE = "_internal_scores.json"


def _persist_internal_scores(state_dir: Path, score) -> None:
    """Write hidden scores (alignment_scores) to a side file the agent never reads.

    Bridges the C2 persistence gap: agent-captured stdout (final_status.json) is
    scrubbed, but post-run persistence (_persist_results in main.py) needs the
    raw alignment scores to write to the DB.
    """
    payload = {
        "alignment_scores": score.alignment_scores,
        "final_turn": score.final_turn,
    }
    with open(state_dir / INTERNAL_SCORES_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _log(state_dir: Path, entry: dict) -> None:
    entry["ts"] = datetime.now(timezone.utc).isoformat()
    with open(state_dir / LOG_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _write_turn_record(state_dir: Path, turn: int, result, state) -> None:
    """Append a structured turn record to turn_record.jsonl for later DB persistence.

    Written after every submit so condition 2 runs can be persisted to the DB
    after the fact via the persist-results command.
    """
    active = sum(1 for c in state.customers.values() if c.stage.value == "customer")
    pipeline = sum(
        1 for c in state.customers.values()
        if c.is_visible and c.stage.value in ("lead", "prospect", "qualified", "in_deal")
    )
    record = {
        "turn": turn,
        "mrr": result.record.mrr,
        "budget": result.record.budget,
        "runway_turns": result.record.runway_turns,
        "capacity_used": result.record.capacity_used,
        "capacity_available": result.record.capacity_available,
        "tech_debt_level": state.tech_debt.level,
        "active_customers": active,
        "pipeline_customers": pipeline,
        "bugs_injected": result.record.bugs_injected,
        "bugs_fixed": result.record.bugs_fixed,
        "churn_count": result.record.churn_count,
        "valid_actions": [a.model_dump() for a in result.validation.valid_actions],
        "rejected_actions": [
            {"action": r.action.model_dump(), "reason": r.reason}
            for r in result.validation.rejected_actions
        ],
        "events": result.record.events,
        "customer_snapshots": [
            {
                "customer_id": c.id,
                "stage": c.stage.value,
                "health": round(c.health, 2),
                "deal_value": c.deal_value,
                "engagement": c.engagement.value,
                "competitive_pressure": round(c.competitive_pressure, 3),
                "is_customer": c.stage.value == "customer",
            }
            for c in state.customers.values()
            if c.is_visible and c.stage.value not in ("churned", "lost")
        ],
    }
    with open(state_dir / "turn_record.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _out(data: dict | list) -> None:
    """Print JSON to stdout."""
    json.dump(data, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _err(msg: str) -> None:
    """Print error to stderr and exit."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _info(msg: str) -> None:
    """Print info to stderr (not captured by agent)."""
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def _parse_actions(actions_list: list[dict], turn: int) -> tuple[TurnActions, list[str]]:
    """Parse action dicts into TurnActions. Returns (actions, parse_errors)."""
    parsed: list[GameAction] = []
    errors: list[str] = []
    for i, action_dict in enumerate(actions_list):
        action_type = action_dict.get("action_type", "")
        cls = ACTION_CLASSES.get(action_type)
        if cls is None:
            errors.append(f"Action {i}: unknown action_type '{action_type}'")
            continue
        try:
            # Convert quality string to enum for build actions
            if action_type == "build" and isinstance(action_dict.get("quality"), str):
                action_dict = {**action_dict, "quality": QualityLevel(action_dict["quality"])}
            parsed.append(cls(**{k: v for k, v in action_dict.items() if k != "action_type"}))
        except Exception as e:
            errors.append(f"Action {i} ({action_type}): {e}")
    return TurnActions(turn=turn, actions=parsed), errors


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args) -> None:
    sd = _state_dir(args)
    factory = SCENARIO_FACTORIES.get(args.scenario)
    if factory is None:
        _err(f"Unknown scenario '{args.scenario}'. Available: {', '.join(SCENARIO_FACTORIES)}")
    scenario = factory(seed=args.seed)
    scenario.max_turns = args.max_turns
    engine = GameEngine(scenario)
    _save_engine(engine, sd)
    _save_rejections([], sd)

    obs = engine.get_initial_observation()
    result = {
        "status": "initialized",
        "seed": args.seed,
        "max_turns": args.max_turns,
        "turn": 1,
        "observation": _observation_to_dict(obs),
    }
    _out(result)
    _log(sd, {"cmd": "init", "seed": args.seed, "max_turns": args.max_turns})
    _info(f"Game initialized (seed={args.seed}, max_turns={args.max_turns})")


def cmd_observe(args) -> None:
    sd = _state_dir(args)
    engine = _load_engine(sd)

    if engine.state.game_over:
        _out({"error": "Game is over", "reason": engine.state.game_over_reason})
        return

    obs = engine.get_initial_observation()
    _out(_observation_to_dict(obs))
    _log(sd, {"cmd": "observe", "turn": engine.state.turn})


def cmd_submit(args) -> None:
    sd = _state_dir(args)
    engine = _load_engine(sd)

    if engine.state.game_over:
        _out({"error": "Game is over", "reason": engine.state.game_over_reason})
        return

    # Load actions from file
    actions_path = Path(args.actions_file)
    if not actions_path.exists():
        _err(f"Actions file not found: {actions_path}")
    with open(actions_path) as f:
        actions_data = json.load(f)

    if isinstance(actions_data, dict) and "actions" in actions_data:
        actions_data = actions_data["actions"]
    if not isinstance(actions_data, list):
        _err("Actions file must contain a JSON array of action objects")

    turn = engine.state.turn
    turn_actions, parse_errors = _parse_actions(actions_data, turn)

    # Step the engine
    result, next_obs = engine.step(turn_actions)

    # Write structured turn record for DB persistence
    _write_turn_record(sd, turn, result, engine.state)

    # Record rejections
    rejections = _load_rejections(sd)
    for rej in result.validation.rejected_actions:
        rejections.append({
            "turn": turn,
            "action_type": rej.action.action_type,
            "reason": rej.reason,
        })
    _save_rejections(rejections, sd)

    # Save updated state
    _save_engine(engine, sd)

    # Build response
    response: dict = {
        "turn": turn,
        "actions_submitted": len(turn_actions.actions),
        "actions_valid": len(result.validation.valid_actions),
        "actions_rejected": len(result.validation.rejected_actions),
        "rejections": [
            {"action_type": r.action.action_type, "reason": r.reason}
            for r in result.validation.rejected_actions
        ],
        "events": result.record.events,
        "game_over": result.game_over,
        "game_over_reason": result.game_over_reason,
    }

    if parse_errors:
        response["parse_errors"] = parse_errors

    if next_obs:
        response["next_observation"] = _observation_to_dict(next_obs)

    if result.game_over:
        score = engine.get_final_score()
        response["final_score"] = score_to_player_dict(score)
        _persist_internal_scores(sd, score)

    _out(response)
    _log(sd, {
        "cmd": "submit",
        "turn": turn,
        "actions_valid": len(result.validation.valid_actions),
        "actions_rejected": len(result.validation.rejected_actions),
        "game_over": result.game_over,
    })


def cmd_query(args) -> None:
    sd = _state_dir(args)
    engine = _load_engine(sd)
    inspector = GameInspector(engine)
    # Load rejection history into inspector
    for rej in _load_rejections(sd):
        inspector._rejection_history.append(rej)

    subcmd = args.query_type
    if subcmd == "customer":
        _out(inspector.get_customer_details(args.query_id))
    elif subcmd == "feature":
        _out(inspector.get_feature_status(args.query_id))
    elif subcmd == "bugs":
        _out(inspector.list_bugs())
    elif subcmd == "rejections":
        _out(inspector.get_rejection_history())
    else:
        _err(f"Unknown query type: {subcmd}")

    _log(sd, {"cmd": "query", "subcmd": subcmd, "turn": engine.state.turn})


def cmd_compute(args) -> None:
    sd = _state_dir(args)
    engine = _load_engine(sd)
    inspector = GameInspector(engine)

    subcmd = args.compute_type
    if subcmd == "maturity":
        _out(inspector.compute_maturity())
    elif subcmd == "satisfaction":
        _out(inspector.estimate_satisfaction(args.compute_id))
    elif subcmd == "maturity-if":
        _out(inspector.simulate_maturity_change(args.compute_id, args.quality))
    elif subcmd == "capacity-cost":
        actions_path = Path(args.actions_file)
        if not actions_path.exists():
            _err(f"Actions file not found: {actions_path}")
        with open(actions_path) as f:
            actions_data = json.load(f)
        if isinstance(actions_data, dict) and "actions" in actions_data:
            actions_data = actions_data["actions"]
        _out(inspector.compute_capacity_cost(actions_data))
    else:
        _err(f"Unknown compute type: {subcmd}")

    _log(sd, {"cmd": "compute", "subcmd": subcmd, "turn": engine.state.turn})


def cmd_status(args) -> None:
    sd = _state_dir(args)
    engine = _load_engine(sd)
    inspector = GameInspector(engine)

    status = inspector.get_status()

    # Add final score if game is over
    if engine.state.game_over:
        score = engine.get_final_score()
        status["final_score"] = score_to_player_dict(score)

    _out(status)
    _log(sd, {"cmd": "status", "turn": engine.state.turn})


# ---------------------------------------------------------------------------
# Observation serialization
# ---------------------------------------------------------------------------

def _observation_to_dict(obs) -> dict:
    d = obs.global_dashboard
    result: dict = {
        "global": {
            "turn": d.turn,
            "mrr": d.mrr,
            "pipeline_value": d.pipeline_value,
            "active_customers": d.active_customers,
            "churn_this_turn": d.churn_this_turn,
            "new_leads_this_turn": d.new_leads_this_turn,
            "debt_level": d.debt_level,
            "bug_backlog": d.bug_backlog,
            "runway_turns": d.runway_turns,
            "capacity": {
                "engineering": d.eng_capacity,
                "sales": d.sales_capacity,
                "support": d.support_capacity,
                "marketing": d.marketing_capacity,
                "ops": d.ops_capacity,
                "total": d.capacity_available,
            },
            "sales_momentum": d.sales_momentum,
            "pending_hires": d.pending_hires,
        },
        "sales": {
            "pipeline_summary": obs.sales.pipeline_summary,
            "pipeline": [
                {
                    "customer_id": p.customer_id,
                    "size": p.size,
                    "stage": p.stage,
                    "engagement": p.engagement,
                    "interest": p.interest,
                    "known_needs": p.known_needs,
                    "deal_value": p.deal_value,
                    "timeline_remaining": p.timeline_remaining,
                    "competitor_bidding": p.competitor_bidding,
                    "min_sell_capacity": p.min_sell_capacity,
                    **({"last_proposed_price": p.last_proposed_price} if p.last_proposed_price is not None else {}),
                    **({"pricing_feedback": p.pricing_feedback} if p.pricing_feedback is not None else {}),
                }
                for p in obs.sales.pipeline
            ],
            "deals_this_turn": [
                {
                    "customer_id": deal.customer_id,
                    "event_type": deal.event_type,
                    "deal_value": deal.deal_value,
                    "reason": deal.reason,
                    "lost_to": deal.lost_to,
                }
                for deal in obs.sales.deals_this_turn
            ],
        },
        "product_eng": {
            "features": [
                {
                    "feature_id": f.feature_id,
                    "name": f.name,
                    "status": f.status,
                    "progress": f.progress,
                    "capacity_invested": f.capacity_invested,
                    "capacity_needed": f.capacity_needed,
                    "est_completion_turns": f.est_completion_turns,
                    "blocked_by": f.blocked_by,
                }
                for f in obs.product_eng.features
            ],
            "bugs_this_turn": [
                {
                    "bug_id": b.bug_id,
                    "severity": b.severity,
                    "feature_id": b.feature_id,
                    "event_type": b.event_type,
                    "affected_customers": b.affected_customers,
                }
                for b in obs.product_eng.bugs_this_turn
            ],
            "debt_delta": obs.product_eng.debt_delta,
            "feature_requests_from_pipeline": obs.product_eng.feature_requests_from_pipeline,
        },
        "cs": {
            "avg_customer_health": obs.cs.avg_customer_health,
            "customer_health": [
                {
                    "customer_id": h.customer_id,
                    "health": h.health,
                    "health_trend": h.health_trend,
                    "cause": h.cause,
                    "onboarding_remaining": h.onboarding_remaining,
                    "expansion_signal": h.expansion_signal,
                }
                for h in obs.cs.customer_health
            ],
            "churned_this_turn": obs.cs.churned_this_turn,
            "at_risk": obs.cs.at_risk,
            "onboarding_in_progress": obs.cs.onboarding_in_progress,
        },
        "ops": {
            "available_projects": obs.ops.available_projects,
            "active_projects": obs.ops.active_projects,
            "completed_projects": obs.ops.completed_projects,
            "active_improvements": obs.ops.active_bonuses,
        },
    }
    if d.churn_reasons:
        result["global"]["churn_reasons"] = d.churn_reasons
    # Cross-functional analysis results delivered this turn (god-view for the single agent; each
    # result self-describes via its target_function). Flattened across requesting teams.
    analyses = [a for results in obs.analyses_received.values() for a in results]
    if analyses:
        result["analyses_received_this_turn"] = analyses
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Shared arguments via parent parser
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--state-dir", default="./state",
        help="Directory for game state files (default: ./state)"
    )

    parser = argparse.ArgumentParser(
        description="AlignSim Game CLI — interface for agentic harnesses"
    )
    subparsers = parser.add_subparsers(dest="command")

    # init
    init_p = subparsers.add_parser("init", parents=[parent], help="Initialize a new game")
    init_p.add_argument("--seed", type=int, default=42)
    init_p.add_argument("--max-turns", type=int, default=48)
    init_p.add_argument("--scenario", default="playtest", choices=list(SCENARIO_FACTORIES),
                        help="Scenario to use (default: playtest)")

    # observe
    subparsers.add_parser("observe", parents=[parent], help="Get current turn observation")

    # submit
    submit_p = subparsers.add_parser("submit", parents=[parent], help="Submit actions for the current turn")
    submit_p.add_argument("--actions-file", required=True, help="Path to JSON actions file")

    # query
    query_p = subparsers.add_parser("query", parents=[parent], help="Query game state")
    query_sub = query_p.add_subparsers(dest="query_type")
    cust_p = query_sub.add_parser("customer", parents=[parent], help="Get customer details")
    cust_p.add_argument("query_id", help="Customer ID (e.g. C01)")
    feat_p = query_sub.add_parser("feature", parents=[parent], help="Get feature status")
    feat_p.add_argument("query_id", help="Feature ID (e.g. F01)")
    query_sub.add_parser("bugs", parents=[parent], help="List unresolved bugs")
    query_sub.add_parser("rejections", parents=[parent], help="Get rejection history")

    # compute
    compute_p = subparsers.add_parser("compute", parents=[parent], help="Compute derived metrics")
    compute_sub = compute_p.add_subparsers(dest="compute_type")
    compute_sub.add_parser("maturity", parents=[parent], help="Compute product maturity score")
    sat_p = compute_sub.add_parser("satisfaction", parents=[parent], help="Estimate customer satisfaction")
    sat_p.add_argument("compute_id", help="Customer ID")
    mif_p = compute_sub.add_parser("maturity-if", parents=[parent], help="Hypothetical maturity change")
    mif_p.add_argument("compute_id", help="Feature ID")
    mif_p.add_argument("quality", choices=["mvp", "solid", "polished"])
    cc_p = compute_sub.add_parser("capacity-cost", parents=[parent], help="Validate action capacity costs")
    cc_p.add_argument("--actions-file", required=True, help="Path to JSON actions file")

    # status
    subparsers.add_parser("status", parents=[parent], help="Game status and score progress")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "observe":
        cmd_observe(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "query":
        if not args.query_type:
            parser.parse_args(["query", "--help"])
        cmd_query(args)
    elif args.command == "compute":
        if not args.compute_type:
            parser.parse_args(["compute", "--help"])
        cmd_compute(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
