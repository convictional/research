"""CLI entry point for AlignSim experiment."""

import argparse
import asyncio
import json
import random
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn

from alignsim.src.analysis.metrics import compute_game_metrics
from alignsim.src.engine.game import GameEngine
from alignsim.src.harness.condition1 import SingleLLMHarness
from alignsim.src.harness.condition3_filters import STARTING_FUNCTIONS
from alignsim.src.harness.condition3_orchestrator import Orchestrator, create_app
from alignsim.src.harness.condition4_orchestrator import (
    ChannelOrchestrator,
    ConvictionalOrchestrator,
    create_c4a_app,
    create_c4b_app,
)
from alignsim.src.models.actions import (
    BuildAction,
    DiscoverAction,
    FixBugsAction,
    GameAction,
    HireAction,
    InfrastructureAction,
    MarketAction,
    SellAction,
    SupportAction,
    TurnActions,
)
from alignsim.src.models.entities import CustomerStage, FeatureStatus, QualityLevel
from alignsim.src.persistence.database import close_db, init_db, try_init_db
from alignsim.src.persistence.run_logger import RunLogger, collect_run_token_usage
from alignsim.src.scenarios.playtest import create_playtest_scenario
from alignsim.src.scenarios.seed_stage import create_seed_stage_scenario
from alignsim.src.settings import settings
from alignsim.src.web.app import app as web_app

SCENARIO_FACTORIES = {
    "playtest": create_playtest_scenario,
    "seed_stage": create_seed_stage_scenario,
}


def _print_alignment_scores(alignment_scores: dict | None) -> None:
    """Print Layer 2 alignment scores to researcher stdout.

    Hidden from agents (they get score_to_player_dict serialization), surfaced
    here in terminal output for researchers and post-run inspection.
    """
    if not alignment_scores:
        return
    print(f"  Alignment Goals (hidden from agent, 1.0 = par):")
    for key, val in alignment_scores.items():
        if not isinstance(val, dict):
            continue
        score = val.get("score")
        if score is None:
            continue
        print(f"    {key:<22}{score:.4f}")


def run():
    parser = argparse.ArgumentParser(description="AlignSim: Goal Alignment Benchmark Game")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # test-engine command
    test_parser = subparsers.add_parser("test-engine", help="Run engine smoke test with random actions")
    test_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    test_parser.add_argument("--max-turns", type=int, default=48, help="Maximum turns to play")
    test_parser.add_argument("--verbose", action="store_true", help="Print detailed turn info")
    test_parser.add_argument("--scenario", default="playtest", choices=list(SCENARIO_FACTORIES),
                             help="Scenario to use (default: playtest)")

    # run command (LLM-powered)
    run_parser = subparsers.add_parser("run", help="Run a game with LLM player (Condition 1)")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--max-turns", type=int, default=48, help="Maximum turns to play")
    run_parser.add_argument("--model", type=str, default=None, help="Override LLM model")
    run_parser.add_argument("--scenario", default="playtest", choices=list(SCENARIO_FACTORIES),
                            help="Scenario to use (default: playtest)")

    # run-sandbox command (Condition 2: Claude Code in sandbox)
    sandbox_parser = subparsers.add_parser("run-sandbox", help="Run a game with Claude Code in sandbox (Condition 2)")
    sandbox_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    sandbox_parser.add_argument("--max-turns", type=int, default=48, help="Maximum turns to play")
    sandbox_parser.add_argument("--model", type=str, default="", help="Claude model for sandbox agent")
    sandbox_parser.add_argument("--interactive", action="store_true", help="Interactive mode (Condition 2.5)")
    sandbox_parser.add_argument("--scenario", default="playtest", choices=list(SCENARIO_FACTORIES),
                                help="Scenario to use (default: playtest)")
    sandbox_parser.add_argument("--skip-db", action="store_true", help="Skip DB persistence after run")
    sandbox_parser.add_argument("--harness", type=str, default="", help="Agent harness: claude-code or pi (default: auto-detect from model)")

    # persist-results command (post-hoc DB persistence for condition 2 runs)
    persist_parser = subparsers.add_parser("persist-results", help="Persist a condition 2 results directory to the DB")
    persist_parser.add_argument("--results-dir", required=True, help="Path to a results/<run_id>/ directory")
    persist_parser.add_argument("--player-type", default=None,
                                help="Override the decision-maker tag (e.g. human_guided for human-in-the-loop runs)")
    persist_parser.add_argument("--run-mode", default=None, help="Record run mode in config (e.g. interactive)")

    # persist-interactive command: recover an interactive/human-guided run from the sandbox
    persist_int_parser = subparsers.add_parser(
        "persist-interactive",
        help="Pull an interactive (e.g. human-guided) Condition 2 run out of the sandbox, score it, and persist it")
    persist_int_parser.add_argument("run_id", help="Run ID = the ~/game/<run_id> directory name in the sandbox")
    persist_int_parser.add_argument(
        "--player-type", default="human_guided",
        help="Decision-maker for this run (default: human_guided; use llm_agent if the agent played and you only watched)")

    # migrate-db command (reconcile schema drift on an existing DB, without a full persist)
    subparsers.add_parser("migrate-db", help="Ensure tables exist and add any missing columns to alignsim_runs")

    # run-orchestrator command (Condition 3: multi-agent orchestrator server)
    orch_parser = subparsers.add_parser("run-orchestrator", help="Start the Condition 3 orchestrator server")
    orch_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    orch_parser.add_argument("--max-turns", type=int, default=48, help="Maximum turns to play")
    orch_parser.add_argument("--scenario", default="seed_stage", choices=list(SCENARIO_FACTORIES),
                             help="Scenario to use (default: seed_stage)")
    orch_parser.add_argument("--port", type=int, default=9000, help="Port for orchestrator API")
    orch_parser.add_argument("--timeout", type=float, default=600.0, help="Per-turn submit timeout in seconds")
    orch_parser.add_argument("--output-dir", type=str, default=None, help="Directory for turn records and logs")

    # run-multi-agent command (Condition 3: full multi-agent game in sandbox)
    multi_parser = subparsers.add_parser("run-multi-agent", help="Run multi-agent game in sandbox (Condition 3)")
    multi_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    multi_parser.add_argument("--max-turns", type=int, default=48, help="Maximum turns to play")
    multi_parser.add_argument("--model", type=str, default="", help="Claude model for agents")
    multi_parser.add_argument("--scenario", default="seed_stage", choices=list(SCENARIO_FACTORIES),
                              help="Scenario to use (default: seed_stage)")
    multi_parser.add_argument("--interactive", action="store_true", help="Interactive mode (drop into shell)")
    multi_parser.add_argument("--harness", type=str, default="", choices=["claude-code", "pi", ""],
                              help="Agent harness (default: auto-detect from model)")
    multi_parser.add_argument("--skip-db", action="store_true", help="Skip DB persistence after run")

    # run-orchestrator-c4 command (Condition 4: substrate-varied orchestrator server)
    orch4_parser = subparsers.add_parser(
        "run-orchestrator-c4", help="Start the Condition 4 orchestrator server (channels or convictional)")
    orch4_parser.add_argument("--substrate", choices=["channels", "convictional"], default="channels",
                              help="Collaboration substrate: channels (C4a) or convictional (C4b)")
    orch4_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    orch4_parser.add_argument("--max-turns", type=int, default=48, help="Maximum turns to play")
    orch4_parser.add_argument("--scenario", default="seed_stage", choices=list(SCENARIO_FACTORIES),
                              help="Scenario to use (default: seed_stage)")
    orch4_parser.add_argument("--port", type=int, default=9100, help="Port for orchestrator API")
    orch4_parser.add_argument("--timeout", type=float, default=600.0, help="Per-turn submit timeout in seconds")
    orch4_parser.add_argument("--output-dir", type=str, default=None, help="Directory for turn records and logs")

    # run-multi-agent-c4 command (Condition 4: full multi-agent game in sandbox)
    multi4_parser = subparsers.add_parser(
        "run-multi-agent-c4", help="Run multi-agent Condition 4 game in sandbox (channels or convictional)")
    multi4_parser.add_argument("--substrate", choices=["channels", "convictional"], default="channels",
                               help="Collaboration substrate: channels (C4a) or convictional (C4b)")
    multi4_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    multi4_parser.add_argument("--max-turns", type=int, default=48, help="Maximum turns to play")
    multi4_parser.add_argument("--model", type=str, default="", help="Claude model for agents")
    multi4_parser.add_argument("--scenario", default="seed_stage", choices=list(SCENARIO_FACTORIES),
                               help="Scenario to use (default: seed_stage)")
    multi4_parser.add_argument("--interactive", action="store_true", help="Interactive mode (drop into shell)")
    multi4_parser.add_argument("--harness", type=str, default="", choices=["claude-code", "pi", ""],
                               help="Agent harness (default: auto-detect from model)")
    multi4_parser.add_argument("--skip-db", action="store_true", help="Skip DB persistence after run")

    # play command (interactive web UI)
    play_parser = subparsers.add_parser("play", help="Play interactively in browser")
    play_parser.add_argument("--port", type=int, default=8420, help="Port for web server")

    args = parser.parse_args()

    if args.command == "test-engine":
        _run_engine_test(args)
    elif args.command == "run":
        asyncio.run(_run_llm_game(args))
    elif args.command == "run-sandbox":
        _run_sandbox_game(args)
    elif args.command == "run-orchestrator":
        _run_orchestrator(args)
    elif args.command == "run-multi-agent":
        _run_multi_agent(args)
    elif args.command == "run-orchestrator-c4":
        _run_orchestrator_c4(args)
    elif args.command == "run-multi-agent-c4":
        _run_multi_agent_c4(args)
    elif args.command == "persist-results":
        asyncio.run(_persist_results(Path(args.results_dir),
                                     player_type=args.player_type, run_mode=args.run_mode))
    elif args.command == "persist-interactive":
        _persist_interactive(args)
    elif args.command == "migrate-db":
        asyncio.run(_migrate_db())
    elif args.command == "play":
        _run_web(args)
    else:
        parser.print_help()


def _run_engine_test(args):
    """Smoke test: run the engine with random valid actions."""
    print(f"=== AlignSim Engine Smoke Test (seed={args.seed}, max_turns={args.max_turns}, scenario={args.scenario}) ===\n")

    scenario = SCENARIO_FACTORIES[args.scenario](seed=args.seed)
    scenario.max_turns = args.max_turns
    engine = GameEngine(scenario)
    action_rng = random.Random(args.seed + 1000)  # separate RNG for action generation

    obs = engine.get_initial_observation()
    print(f"Turn 1 | MRR: {obs.global_dashboard.mrr:,} | Runway: {obs.global_dashboard.runway_turns:.1f} turns | "
          f"Active: {obs.global_dashboard.active_customers} | Capacity: {obs.global_dashboard.capacity_available}")

    for turn in range(1, args.max_turns + 1):
        actions = _generate_random_actions(engine, action_rng, turn)
        result, next_obs = engine.step(actions)

        if args.verbose:
            print(f"\n  Actions: {len(result.validation.valid_actions)} valid, "
                  f"{len(result.validation.rejected_actions)} rejected")
            for event in result.record.events:
                print(f"  Event: {event}")

        if next_obs:
            d = next_obs.global_dashboard
            bugs_str = ", ".join(f"{k}: {v}" for k, v in d.bug_backlog.items()) if d.bug_backlog else "none"
            print(f"Turn {d.turn:2d} | MRR: {d.mrr:>8,} | Runway: {d.runway_turns:>6.1f} turns | "
                  f"Active: {d.active_customers:2d} | Debt: {d.debt_level:<8s} | Bugs: {bugs_str} | "
                  f"Churn: {len(d.churn_this_turn)}")

        if result.game_over:
            print(f"\n  Game Over: {result.game_over_reason}")
            break

    score = engine.get_final_score()
    print(f"\n=== Final Score ===")
    print(f"  Primary Goals (1.0 = par):")
    print(f"    MRR:        {score.mrr_score:.4f} (${score.final_mrr:,} / ${scenario.primary_goal.mrr_target:,})")
    print(f"    Churn:      {score.churn_score:.4f} (avg rate: {score.avg_churn_rate:.4f})")
    print(f"    Runway:     {score.runway_score:.4f} ({score.final_runway_turns:.1f} turns)")
    print(f"    Composite:  {score.composite:.4f} (geo mean)")
    print(f"    Pareto:     {score.pareto_score:.4f} (min)")
    if score.function_scores:
        print(f"  Function Goals (1.0 = par):")
        for fn, s in score.function_scores.items():
            print(f"    {fn:<14}{s:.4f}")
        print(f"    Composite:  {score.function_composite:.4f} (geo mean)")
        print(f"    Pareto:     {score.function_pareto:.4f} (min)")

    _print_alignment_scores(score.alignment_scores)

    # Determinism check
    print(f"\n=== Determinism Check ===")
    scenario2 = SCENARIO_FACTORIES[args.scenario](seed=args.seed)
    scenario2.max_turns = args.max_turns
    engine2 = GameEngine(scenario2)
    action_rng2 = random.Random(args.seed + 1000)

    obs2 = engine2.get_initial_observation()
    for turn in range(1, args.max_turns + 1):
        actions2 = _generate_random_actions(engine2, action_rng2, turn)
        result2, next_obs2 = engine2.step(actions2)
        if result2.game_over:
            break

    score2 = engine2.get_final_score()
    if score.composite == score2.composite and score.final_mrr == score2.final_mrr:
        print("  PASS: Deterministic (same seed + same actions = identical results)")
    else:
        print("  FAIL: Non-deterministic!")
        print(f"  Run 1: composite={score.composite}, mrr={score.final_mrr}")
        print(f"  Run 2: composite={score2.composite}, mrr={score2.final_mrr}")


def _generate_random_actions(engine: GameEngine, rng: random.Random, turn: int) -> TurnActions:
    """Generate random but valid-ish actions for smoke testing."""
    state = engine.state
    actions: list[GameAction] = []
    capacity_remaining = state.resources.capacity_per_turn

    # Allocate roughly: 40% engineering, 25% sales, 15% CS, 10% discovery, 10% marketing
    eng_budget = int(capacity_remaining * 0.4)
    sales_budget = int(capacity_remaining * 0.25)
    cs_budget = int(capacity_remaining * 0.15)
    discover_budget = int(capacity_remaining * 0.1)
    market_budget = capacity_remaining - eng_budget - sales_budget - cs_budget - discover_budget

    # Engineering: build unshipped features or fix bugs
    unresolved_bugs = [b for b in state.bugs if not b.is_resolved]
    if unresolved_bugs and rng.random() < 0.4:
        fix_capacity = min(eng_budget, 4)
        if fix_capacity > 0:
            actions.append(FixBugsAction(capacity=fix_capacity))
            eng_budget -= fix_capacity

    # Build random unfinished features
    buildable = [
        f for f in state.features.values()
        if f.status in (FeatureStatus.not_started, FeatureStatus.in_progress, FeatureStatus.shipped_mvp, FeatureStatus.shipped_solid)
        and all(
            state.features[dep].status.value.startswith("shipped")
            for dep in f.depends_on if dep in state.features
        )
    ]
    if buildable and eng_budget > 0:
        feature = rng.choice(buildable)
        target = rng.choice([QualityLevel.mvp, QualityLevel.solid])
        # Ensure target is higher than current
        quality_order = {"not_started": 0, "in_progress": 0, "shipped_mvp": 1, "shipped_solid": 2, "shipped_polished": 3}
        target_order = {"mvp": 1, "solid": 2, "polished": 3}
        if target_order[target.value] > quality_order.get(feature.status.value, 0):
            build_cap = min(eng_budget, rng.randint(4, 10))
            if build_cap > 0:
                actions.append(BuildAction(feature_id=feature.id, quality=target, capacity=build_cap))
                eng_budget -= build_cap

    # Infrastructure with remaining eng budget
    if eng_budget > 2:
        actions.append(InfrastructureAction(capacity=eng_budget))

    # Sales: advance pipeline customers
    pipeline = [
        c for c in state.customers.values()
        if c.is_visible and c.stage in (CustomerStage.lead, CustomerStage.prospect, CustomerStage.qualified, CustomerStage.in_deal)
    ]
    if pipeline and sales_budget > 0:
        for customer in rng.sample(pipeline, min(3, len(pipeline))):
            if sales_budget <= 0:
                break
            sell_map = {
                CustomerStage.lead: "outbound",
                CustomerStage.prospect: "demo",
                CustomerStage.qualified: "demo",
                CustomerStage.in_deal: "proposal",
            }
            sell_action = sell_map.get(customer.stage, "outbound")
            cap = min(sales_budget, rng.randint(2, 4))
            actions.append(SellAction(customer_id=customer.id, sell_action=sell_action, capacity=cap))
            sales_budget -= cap

    # CS: support active customers
    active = [c for c in state.customers.values() if c.stage == CustomerStage.customer]
    at_risk = [c for c in active if c.health < 6]
    targets = at_risk if at_risk else active
    if targets and cs_budget > 0:
        for customer in rng.sample(targets, min(2, len(targets))):
            if cs_budget <= 0:
                break
            cap = min(cs_budget, rng.randint(2, 3))
            support_type = "churn_intervention" if customer.health < 5 else "health_check"
            actions.append(SupportAction(customer_id=customer.id, support_action=support_type, capacity=cap))
            cs_budget -= cap

    # Discovery
    hidden = [c for c in state.customers.values() if not c.is_visible]
    if hidden and discover_budget > 0:
        actions.append(DiscoverAction(capacity=discover_budget))

    # Marketing
    if market_budget > 0:
        actions.append(MarketAction(channel="content", capacity=market_budget))

    return TurnActions(turn=turn, actions=actions)


async def _run_llm_game(args):
    """Run a game with the LLM harness (Condition 1)."""
    print(f"=== AlignSim LLM Game (seed={args.seed}, max_turns={args.max_turns}, scenario={args.scenario}) ===\n")

    # Initialize persistence (graceful degradation)
    db_available = await try_init_db()
    run_logger: RunLogger | None = None

    scenario = SCENARIO_FACTORIES[args.scenario](seed=args.seed)
    scenario.max_turns = args.max_turns
    engine = GameEngine(scenario)

    model = args.model or settings.llm_model
    harness = SingleLLMHarness(model=model)

    # Start game
    scenario_info = engine.get_scenario_info()
    await harness.on_game_start(scenario_info)

    if db_available:
        run_logger = await RunLogger.create(
            scenario_name=scenario.name,
            condition="condition1",
            player_type="llm",
            model=model,
            seed=args.seed,
            max_turns=args.max_turns,
            config={"temperature": settings.temperature, "max_tokens": settings.max_tokens},
        )

    obs = engine.get_initial_observation()
    state_summary = engine.get_state_summary()

    print(f"Turn 1 | MRR: {obs.global_dashboard.mrr:,} | Model: {model}")

    for turn in range(1, args.max_turns + 1):
        print(f"\n--- Turn {turn} ---")

        # Get LLM decision
        actions = await harness.decide(obs, state_summary)
        print(f"  LLM submitted {len(actions.actions)} actions")

        # Log LLM trace
        if run_logger:
            await run_logger.log_llm_trace(
                turn=turn,
                system_prompt=harness.last_system_prompt or "",
                user_prompt=harness.last_user_prompt or "",
                response=actions,
                model=model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                latency_ms=harness.last_latency_ms,
                error=harness.last_error,
            )

        # Step engine
        result, next_obs = engine.step(actions)

        # Log turn
        if run_logger:
            await run_logger.log_turn(turn, result.record, engine.state)

        valid_count = len(result.validation.valid_actions)
        rejected_count = len(result.validation.rejected_actions)
        if rejected_count > 0:
            print(f"  Validation: {valid_count} valid, {rejected_count} rejected")
            for rej in result.validation.rejected_actions:
                print(f"    Rejected: {rej.reason}")

        # Feed rejection info back to harness for next turn's prompt
        harness.on_turn_result([rej.reason for rej in result.validation.rejected_actions])

        for event in result.record.events:
            if any(kw in event for kw in ("deal_won", "churn", "game_over", "feature_shipped", "expansion")):
                print(f"  Event: {event}")

        if next_obs:
            d = next_obs.global_dashboard
            print(f"  MRR: {d.mrr:,} | Runway: {d.runway_turns:.1f} turns | Active: {d.active_customers} | "
                  f"Debt: {d.debt_level}")
            obs = next_obs
            state_summary = engine.get_state_summary()

        if result.game_over:
            print(f"\n  Game Over: {result.game_over_reason}")
            break

    # Final score
    score = engine.get_final_score()
    await harness.on_game_end(score, engine.state)

    # Finalize persistence
    if run_logger:
        await run_logger.finalize(
            score, engine.state.game_over_reason, engine.state.turn - 1,
            token_usage=dict(harness.token_usage) if harness.token_usage else None,
        )

    if db_available:
        await close_db()

    print(f"\n=== Final Score ===")
    print(f"  Primary Goals (1.0 = par):")
    print(f"    MRR:        {score.mrr_score:.4f} (${score.final_mrr:,})")
    print(f"    Churn:      {score.churn_score:.4f} (avg rate: {score.avg_churn_rate:.4f})")
    print(f"    Runway:     {score.runway_score:.4f} ({score.final_runway_turns:.1f} turns)")
    print(f"    Composite:  {score.composite:.4f} (geo mean)")
    print(f"    Pareto:     {score.pareto_score:.4f} (min)")
    if score.function_scores:
        print(f"  Function Goals (1.0 = par):")
        for fn, s in score.function_scores.items():
            print(f"    {fn:<14}{s:.4f}")
        print(f"    Composite:  {score.function_composite:.4f} (geo mean)")
        print(f"    Pareto:     {score.function_pareto:.4f} (min)")

    _print_alignment_scores(score.alignment_scores)

    # Save results with full metrics (keep existing JSON output)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = settings.output_path / f"{args.scenario}_condition1_{args.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_game_metrics(engine.state, score)
    results = {
        "scenario": scenario.name,
        "condition": "condition1",
        "model": model,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "turn_count": engine.state.turn - 1,
        "game_over_reason": engine.state.game_over_reason,
        **metrics,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_dir}")


def _run_sandbox_game(args):
    """Run a game with Claude Code in the Lima sandbox (Condition 2)."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "sandbox_run_condition2.sh"
    if not script.exists():
        print(f"Error: sandbox_run_condition2.sh not found at {script}", file=sys.stderr)
        sys.exit(1)

    cmd = [str(script), "--seed", str(args.seed), "--max-turns", str(args.max_turns),
           "--scenario", args.scenario]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.interactive:
        cmd.append("--interactive")
    if getattr(args, "harness", ""):
        cmd.extend(["--harness", args.harness])
    if getattr(args, "skip_db", False):
        cmd.append("--skip-db")

    sys.exit(subprocess.call(cmd))


def _persist_interactive(args):
    """Recover an interactive/human-guided sandbox run and persist it.

    Interactive runs never execute the autonomous wrap-up (score → collect → persist), so their
    game state stays in the sandbox and nothing lands in the DB. This shells to the helper that
    pulls the state out of the sandbox, scores it on the host (current scoring code), assembles a
    results dir, and persists it tagged with the given player_type (default human_guided).
    """
    script = Path(__file__).resolve().parent.parent / "scripts" / "persist_interactive.sh"
    if not script.exists():
        print(f"Error: persist_interactive.sh not found at {script}", file=sys.stderr)
        sys.exit(1)
    cmd = [str(script), args.run_id, "--player-type", args.player_type]
    sys.exit(subprocess.call(cmd))


async def _migrate_db() -> None:
    """Reconcile the DB schema on demand: create any missing tables and add missing columns.

    init_db() runs generate_schemas() + the ADD COLUMN IF NOT EXISTS reconciler, so this is the
    explicit way to bring a pre-existing analysis DB up to date (plot_runs/compare_conditions
    connect read-only and never trigger the reconciler themselves).
    """
    await init_db()
    await close_db()
    print("Schema reconciled: tables and alignsim_runs columns ensured.")


async def _persist_results(
    results_dir: Path, player_type: str | None = None, run_mode: str | None = None
) -> None:
    """Persist a condition 2 results directory to the DB.

    Reads run_metadata.json, turn_record.jsonl, and final_status.json from the
    results directory and writes the full run (snapshots, actions, events, customer
    snapshots) to PostgreSQL via RunLogger.

    player_type overrides the metadata's decision-maker tag (e.g. "human_guided" for an
    interactive human-in-the-loop run, so it is not pooled with autonomous "llm_agent"
    cells). run_mode is recorded in the run config (e.g. "interactive").
    """
    metadata_path = results_dir / "run_metadata.json"
    final_path = results_dir / "final_status.json"

    if not metadata_path.exists():
        print(f"Error: run_metadata.json not found in {results_dir}", file=sys.stderr)
        sys.exit(1)
    if not final_path.exists():
        print(f"Error: final_status.json not found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    metadata = json.loads(metadata_path.read_text())
    final = json.loads(final_path.read_text())

    # Bridge the C2 persistence gap: agent-captured stdout is player-scrubbed,
    # so hidden Layer 2 alignment scores live in a side file written by the
    # engine. Merge into the score dict before persisting.
    internal_path = results_dir / "_internal_scores.json"
    if internal_path.exists():
        internal = json.loads(internal_path.read_text())
        final.setdefault("final_score", {})["alignment_scores"] = internal.get("alignment_scores", {})

    db_available = await try_init_db()
    if not db_available:
        print("DB not available — skipping persistence", file=sys.stderr)
        sys.exit(2)

    # Overrides for non-autonomous runs (e.g. interactive human-guided sessions persisted via
    # `persist-interactive`): player_type marks who actually made the decisions, and run_mode
    # records how the run was launched. Both are stored so a human-in-the-loop run is never
    # silently pooled with the autonomous llm_agent grid.
    if player_type:
        metadata["player_type"] = player_type
    config = dict(metadata.get("config") or {})
    effective_run_mode = run_mode or metadata.get("run_mode")
    if effective_run_mode:
        config["run_mode"] = effective_run_mode
    if metadata["player_type"] == "human_guided":
        config["human_intervened"] = True

    run_logger = await RunLogger.create(
        scenario_name=metadata["scenario"],
        condition=metadata["condition"],
        player_type=metadata["player_type"],
        model=metadata.get("model") or None,
        seed=metadata["seed"],
        max_turns=metadata["max_turns"],
        harness=metadata.get("harness") or metadata.get("agent_cli"),
        thinking=metadata.get("thinking"),
        config=config,
    )

    turn_record_path = results_dir / "turn_record.jsonl"
    if turn_record_path.exists():
        lines = [l for l in turn_record_path.read_text().splitlines() if l.strip()]
        for line in lines:
            tr = json.loads(line)
            await run_logger.log_turn_from_dict(tr)
        print(f"  Logged {len(lines)} turns")
    else:
        print("  Warning: turn_record.jsonl not found — no per-turn data persisted")

    # collect_run_token_usage discovers both layouts: C2's single top-level
    # transcript and C3's per-agent transcripts in <function>/ subdirs.
    token_usage = collect_run_token_usage(results_dir)
    if token_usage:
        for model, usage in token_usage.items():
            total_in = usage["input_tokens"] + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"]
            print(f"  Tokens ({model}): {total_in:,} in, {usage['output_tokens']:,} out")

    score = final.get("final_score", {})
    await run_logger.finalize_from_dict(
        score=score,
        game_over_reason=final.get("game_over_reason"),
        turns_played=final.get("turn", 1) - 1,
        token_usage=token_usage,
    )

    await close_db()
    print(f"Persisted run {run_logger.run_id} to DB (composite={score.get('composite', 0):.4f})")


def _orchestrator_lifespan(orchestrator: Orchestrator):
    """Shared uvicorn lifespan: run the watchdog, and flush all logs on shutdown.

    Used by both the C3 and C4 servers so the watchdog lifecycle and log-flush can't
    diverge between conditions. C4b (ConvictionalOrchestrator) additionally flushes its
    durable Posts + Goals artifacts.
    """

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(orchestrator.run_watchdog())
        yield
        task.cancel()
        orchestrator.write_chat_log()
        if isinstance(orchestrator, ConvictionalOrchestrator):
            orchestrator.write_posts_log()
            orchestrator.write_goals_log()

    return lifespan


def _run_orchestrator(args):
    """Start the Condition 3 orchestrator server."""
    scenario = SCENARIO_FACTORIES[args.scenario](seed=args.seed)
    scenario.max_turns = args.max_turns
    engine = GameEngine(scenario)

    orchestrator = Orchestrator(engine, submit_timeout_s=args.timeout, output_dir=args.output_dir)
    for fn in sorted(STARTING_FUNCTIONS):
        orchestrator._register_agent_unlocked(fn)

    app = create_app(orchestrator)
    app.router.lifespan_context = _orchestrator_lifespan(orchestrator)

    print(f"Starting Condition 3 orchestrator on port {args.port}")
    print(f"  Scenario: {args.scenario}, Seed: {args.seed}, Max turns: {args.max_turns}")
    print(f"  Starting agents: {sorted(STARTING_FUNCTIONS)}")
    print(f"  Submit timeout: {args.timeout}s")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


def _run_multi_agent(args):
    """Run a multi-agent game with Claude Code agents in the Lima sandbox (Condition 3)."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "sandbox_run_condition3.sh"
    if not script.exists():
        print(f"Error: sandbox_run_condition3.sh not found at {script}", file=sys.stderr)
        sys.exit(1)

    cmd = [str(script), "--seed", str(args.seed), "--max-turns", str(args.max_turns),
           "--scenario", args.scenario]
    if args.model:
        cmd.extend(["--model", args.model])
    if getattr(args, "harness", ""):
        cmd.extend(["--harness", args.harness])
    if args.interactive:
        cmd.append("--interactive")
    if getattr(args, "skip_db", False):
        cmd.append("--skip-db")

    sys.exit(subprocess.call(cmd))


def _run_orchestrator_c4(args):
    """Start the Condition 4 orchestrator server (channels = C4a, convictional = C4b)."""
    scenario = SCENARIO_FACTORIES[args.scenario](seed=args.seed)
    scenario.max_turns = args.max_turns
    engine = GameEngine(scenario)

    if args.substrate == "channels":
        orchestrator = ChannelOrchestrator(engine, submit_timeout_s=args.timeout, output_dir=args.output_dir)
        app = create_c4a_app(orchestrator)
        label = "4a (channels)"
    else:
        orchestrator = ConvictionalOrchestrator(engine, submit_timeout_s=args.timeout, output_dir=args.output_dir)
        app = create_c4b_app(orchestrator)
        label = "4b (convictional)"

    for fn in sorted(STARTING_FUNCTIONS):
        orchestrator._register_agent_unlocked(fn)

    app.router.lifespan_context = _orchestrator_lifespan(orchestrator)

    print(f"Starting Condition {label} orchestrator on port {args.port}")
    print(f"  Scenario: {args.scenario}, Seed: {args.seed}, Max turns: {args.max_turns}")
    print(f"  Starting agents: {sorted(STARTING_FUNCTIONS)}")
    print(f"  Substrate: {args.substrate}")
    print(f"  Submit timeout: {args.timeout}s")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


def _run_multi_agent_c4(args):
    """Run a multi-agent Condition 4 game with Claude Code agents in the Lima sandbox."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "sandbox_run_condition4.sh"
    if not script.exists():
        print(f"Error: sandbox_run_condition4.sh not found at {script}", file=sys.stderr)
        sys.exit(1)

    cmd = [str(script), "--seed", str(args.seed), "--max-turns", str(args.max_turns),
           "--scenario", args.scenario, "--substrate", args.substrate]
    if args.model:
        cmd.extend(["--model", args.model])
    if getattr(args, "harness", ""):
        cmd.extend(["--harness", args.harness])
    if args.interactive:
        cmd.append("--interactive")
    if getattr(args, "skip_db", False):
        cmd.append("--skip-db")

    sys.exit(subprocess.call(cmd))


def _run_web(args):
    """Launch the interactive web UI."""
    print(f"Starting AlignSim at http://localhost:{args.port}")
    uvicorn.run(web_app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    run()
