import argparse
import sys

from tqdm import tqdm

from common.io import load_pickle_file
from common.prompt_template_engine import initialize_and_register_prompt_templates

from .data.load_organization_data import load_organization_data
from .decision_mapping.dm_mapper import run_dm_mapper
from .decision_mapping.dsm_mapper import run_dsm_mapper
from .decision_mapping.gm_mapper import run_gm_mapper
from .decision_mapping.render import render_and_save
from .evaluation.aggregator import aggregate
from .evaluation.calibration_pilot import (
    check_pilot_passed,
    run_calibration_pilot,
)
from .evaluation.moe_judge import run_cell_judges
from .evaluation.research_summary import summarize_artifact
from .evaluation.results_matrix import build_results_matrix
from .evaluation.rubric import cell_id
from .goal_mining.pipeline import run_pipeline
from .models import Decision, FinalizedGoalSet
from .settings import logger, settings

_CONDITIONS = ["unstated", "stated", "mixed"]
_SCHEMAS = ["dm", "dsm", "gm"]


async def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Ensure the shared corpus is cached. The decision corpus is identical across conditions."""
    load_from_cache = args.load_from_cache
    activity_events, decisions, stated_goals = await load_organization_data(load_from_cache=load_from_cache)
    print(f"Shared corpus ready: {len(activity_events)} events, {len(decisions)} decisions, "
          f"{len(stated_goals)} stated goals (Convictional seed)")
    print("Conditions will use this shared corpus with varying goal-set provenance:")
    print("  unstated — LLM-mined unstated goals only (step 1, no stated goals)")
    print("  stated   — human-written stated goals only (step 2, skip step 1)")
    print("  mixed    — LLM-mined unstated + human-written stated, merged (steps 1+2)")


async def _run_mine_goals(condition: str, load_from_cache: bool) -> None:
    activity_events, decisions, convictional_stated_goals = await load_organization_data(load_from_cache=True)

    # Determine goal-set provenance and whether to run unstated mining
    if condition == "unstated":
        # Fresh-onboarded company: 0 written goals; mine unstated from activity
        stated_goals = []
        mine_unstated = True
    elif condition == "stated":
        # Company with written goals but no approved unstated corpus
        stated_goals = convictional_stated_goals
        mine_unstated = False
    else:  # mixed
        # Future platform state: both sources merged
        stated_goals = convictional_stated_goals
        mine_unstated = True

    output_path = settings.condition_output_path(condition)
    finalized = await run_pipeline(
        condition=condition,
        activity_events=activity_events,
        decisions=decisions,
        stated_goals=stated_goals,
        output_path=output_path,
        mine_unstated=mine_unstated,
        load_from_cache=load_from_cache,
    )

    pkl_path = output_path / "step5_final_goal_set.pkl"
    md_path = output_path / "final_goal_set.md"
    print(f"\nDone — condition: {condition}")
    print(f"  pkl: {pkl_path}")
    print(f"  md:  {md_path}")
    print(f"  Goals: {len(finalized.goals)}, Steps run: {finalized.run_metadata['steps_run']}")


async def cmd_mine_goals(args: argparse.Namespace) -> None:
    await _run_mine_goals(args.condition, args.load_from_cache)


async def cmd_mine_all(args: argparse.Namespace) -> None:
    load_from_cache = args.load_from_cache

    for condition in _CONDITIONS:
        print(f"\n{'='*60}")
        print(f"  Condition: {condition}")
        print(f"{'='*60}")
        await _run_mine_goals(condition, load_from_cache)


def _load_goal_set(condition: str) -> FinalizedGoalSet:
    """Load the FinalizedGoalSet pkl for a condition."""
    pkl = settings.condition_output_path(condition) / "step5_final_goal_set.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"Goal set not found: {pkl}\n"
            f"Run 'mine_goals --condition {condition}' first."
        )
    size_kb = pkl.stat().st_size / 1024
    print(f"  [goal set] {pkl.name} ({size_kb:.1f} KB)")
    return load_pickle_file(pkl)


def _load_shared_decisions() -> list[Decision]:
    pkl = settings.shared_output_path / "decisions.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"Shared decisions not found: {pkl}\n"
            "Run 'build_dataset' first."
        )
    size_kb = pkl.stat().st_size / 1024
    print(f"  [decisions] {pkl.name} ({size_kb:.1f} KB)")
    return load_pickle_file(pkl)


async def _run_map(
    condition: str,
    schema: str,
    load_from_cache: bool,
) -> None:
    goal_set = _load_goal_set(condition)
    decisions = _load_shared_decisions()
    output_path = settings.condition_output_path(condition)

    goals = goal_set.goals
    print(f"\n  {condition} / {schema}: {len(decisions)} decisions, {len(goals)} goals")

    if schema == "dm":
        mapping = await run_dm_mapper(decisions, goals, condition, output_path, load_from_cache)
    elif schema == "dsm":
        mapping = await run_dsm_mapper(decisions, goals, condition, output_path, load_from_cache)
    elif schema == "gm":
        mapping = await run_gm_mapper(decisions, goals, condition, output_path, load_from_cache)
    else:
        raise ValueError(f"Unknown schema: {schema}")

    render_and_save(mapping, decisions, goals, output_path)


async def cmd_map_decisions(args: argparse.Namespace) -> None:
    await _run_map(args.condition, args.schema, args.load_from_cache)


async def cmd_map_all(args: argparse.Namespace) -> None:
    load_from_cache = args.load_from_cache

    for condition in _CONDITIONS:
        for schema in _SCHEMAS:
            print(f"\n{'='*60}")
            print(f"  map_all: condition={condition}, schema={schema}")
            print(f"{'='*60}")
            await _run_map(condition, schema, load_from_cache)


# ── Phase 2.5: Summarization (obfuscation layer) ──────────────────────────────

def _load_mapping_md(condition: str, schema: str) -> tuple[str, int]:
    """Load schema-masked mapping md. Returns (content, word_count)."""
    md_path = settings.condition_output_path(condition) / f"mapping_{schema}.md"
    if not md_path.exists():
        raise FileNotFoundError(
            f"Mapping md not found: {md_path}\n"
            f"Run 'map_decisions --condition {condition} --schema {schema}' first."
        )
    content = md_path.read_text()
    wc = len(content.split())
    return content, wc


def _load_summary_md(condition: str, schema: str) -> tuple[str, int]:
    """Load the fixed-length research summary. Returns (content, word_count)."""
    md_path = settings.condition_output_path(condition) / f"summary_{schema}.md"
    if not md_path.exists():
        raise FileNotFoundError(
            f"Summary md not found: {md_path}\n"
            f"Run 'summarize --condition {condition} --schema {schema}' first. "
            "The summarize step must run before judge."
        )
    content = md_path.read_text()
    wc = len(content.split())
    return content, wc


async def _run_summarize(
    condition: str,
    schema: str,
    load_from_cache: bool = True,
) -> None:
    rendered_md, _ = _load_mapping_md(condition, schema)
    output_path = settings.condition_output_path(condition)
    summary_md, wc = await summarize_artifact(
        rendered_md=rendered_md,
        output_path=output_path,
        schema=schema,
        load_from_cache=load_from_cache,
    )
    print(f"  {condition}/{schema}: summary {wc} words")


async def cmd_summarize(args: argparse.Namespace) -> None:
    await _run_summarize(args.condition, args.schema, args.load_from_cache)


async def cmd_summarize_all(args: argparse.Namespace) -> None:
    load_from_cache = args.load_from_cache

    for condition in _CONDITIONS:
        for schema in _SCHEMAS:
            print(f"\n{'='*60}")
            print(f"  summarize_all: condition={condition}, schema={schema}")
            print(f"{'='*60}")
            await _run_summarize(condition, schema, load_from_cache)


# ── Phase 3: Evaluation ───────────────────────────────────────────────────────

async def _run_judge(
    condition: str,
    schema: str,
    temperature: float,
    load_from_cache: bool,
) -> None:
    cid = cell_id(condition, schema)
    # Judges score the fixed-length research summary, not the raw mapping artifact
    summary_md, wc = _load_summary_md(condition, schema)
    output_path = settings.condition_output_path(condition)

    runs = await run_cell_judges(
        cell_id=cid,
        rendered_md=summary_md,
        rendered_word_count=wc,
        schema=schema,
        output_path=output_path,
        temperature=temperature,
        load_from_cache=load_from_cache,
    )
    agg = aggregate(runs, condition, schema)
    print(
        f"  {cid} T={temperature}: trimmed_mean={agg.trimmed_mean_overall:.2f}, "
        f"variance={agg.inter_judge_variance:.1f}"
    )


async def cmd_calibration_pilot(args: argparse.Namespace) -> None:
    result = await run_calibration_pilot(load_from_cache=args.load_from_cache)
    if not result.passed:
        print("\n  HALTING: calibration pilot failed. Fix the summarizer/rubric before running judge_all.")
        sys.exit(1)


async def cmd_judge(args: argparse.Namespace) -> None:
    if not check_pilot_passed():
        print("⚠  Calibration pilot has not passed. Run 'calibration_pilot' first.")
        print("   Proceeding anyway — this is a warning, not a block.")

    temperature = float(args.temperature)
    await _run_judge(args.condition, args.schema, temperature, args.load_from_cache)


async def cmd_judge_all(args: argparse.Namespace) -> None:
    if not check_pilot_passed():
        print("⚠  Calibration pilot has not passed. Run 'calibration_pilot' first.")
        print("   Proceeding anyway — this is a warning, not a block.")

    temperature = float(args.temperature)
    for condition in _CONDITIONS:
        for schema in _SCHEMAS:
            print(f"\n{'='*60}")
            print(f"  judge_all: {condition} / {schema} T={temperature}")
            print(f"{'='*60}")
            await _run_judge(condition, schema, temperature, args.load_from_cache)


async def cmd_run_all(args: argparse.Namespace) -> None:
    """Run the whole experiment end to end, with a phase-level progress bar.

    Phases run in order; each reuses the same args namespace (it carries
    --temperature and --load-from-cache). The calibration pilot is a hard gate —
    if it fails it exits before judging, exactly as when run standalone.
    """
    phases: list[tuple[str, object]] = [
        ("Build dataset", cmd_build_dataset),
        ("Mine goals (3 conditions)", cmd_mine_all),
        ("Map decisions (9 cells)", cmd_map_all),
        ("Summarize / obfuscation (9 cells)", cmd_summarize_all),
        ("Calibration pilot (gate)", cmd_calibration_pilot),
        ("Judge (9 cells)", cmd_judge_all),
        ("Build results matrix", cmd_build_matrix),
    ]

    pbar = tqdm(phases, desc="Experiment", unit="phase")
    for label, fn in pbar:
        pbar.set_postfix_str(label)
        pbar.write(f"\n{'#'*60}\n#  Phase: {label}\n{'#'*60}")
        await fn(args)
    pbar.close()
    print(f"\n✓ Experiment complete (T={float(args.temperature)}). See RESULTS.md and output/results_matrix.{{csv,json}}.")


async def cmd_build_matrix(args: argparse.Namespace) -> None:
    build_results_matrix(temperature=float(args.temperature), load_from_cache=args.load_from_cache)


async def main() -> None:
    settings.init_output_dirs()
    initialize_and_register_prompt_templates(settings.prompts_path)

    parser = argparse.ArgumentParser(description="Decisions-to-Goals Experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build_dataset", help="Fetch and cache the shared corpus (decisions + events + stated goals)")
    build_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    mine_p = subparsers.add_parser("mine_goals", help="Run the goal mining pipeline for one condition")
    mine_p.add_argument("--condition", required=True, choices=_CONDITIONS)
    mine_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    all_p = subparsers.add_parser("mine_all", help="Run the pipeline for all three conditions")
    all_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    map_p = subparsers.add_parser("map_decisions", help="Run one mapping schema for one condition")
    map_p.add_argument("--condition", required=True, choices=_CONDITIONS)
    map_p.add_argument("--schema", required=True, choices=_SCHEMAS)
    map_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    map_all_p = subparsers.add_parser("map_all", help="Run all 9 mapping cells (3 conditions × 3 schemas)")
    map_all_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    # Phase 2.5 — summarization (obfuscation layer)
    sum_p = subparsers.add_parser("summarize", help="Compress one mapping artifact to a fixed-length research summary")
    sum_p.add_argument("--condition", required=True, choices=_CONDITIONS)
    sum_p.add_argument("--schema", required=True, choices=_SCHEMAS)
    sum_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    sum_all_p = subparsers.add_parser("summarize_all", help="Compress all 9 mapping artifacts to fixed-length summaries")
    sum_all_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    # Phase 3 — evaluation commands
    subparsers.add_parser("calibration_pilot", help="MUST run before judge_all: length normalization + bias checks").add_argument(
        "--load-from-cache", action=argparse.BooleanOptionalAction, default=True
    )

    judge_p = subparsers.add_parser("judge", help="Judge one cell (condition × schema) using its research summary")
    judge_p.add_argument("--condition", required=True, choices=_CONDITIONS)
    judge_p.add_argument("--schema", required=True, choices=_SCHEMAS)
    judge_p.add_argument("--temperature", type=float, default=0.0)
    judge_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    judge_all_p = subparsers.add_parser("judge_all", help="Judge all 9 cells at the given temperature")
    judge_all_p.add_argument("--temperature", type=float, default=0.0)
    judge_all_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    matrix_p = subparsers.add_parser("build_matrix", help="Assemble RESULTS.md + output/results_matrix.{csv,json}")
    matrix_p.add_argument("--temperature", type=float, default=0.0, help="Which temperature's judge caches to assemble")
    matrix_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    run_all_p = subparsers.add_parser("run_all", help="Run the whole experiment end to end (all phases, with progress bar)")
    run_all_p.add_argument("--temperature", type=float, default=0.0, help="Judging temperature (default 0.0)")
    run_all_p.add_argument("--load-from-cache", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(sys.argv[1:])

    command_map = {
        "build_dataset": cmd_build_dataset,
        "mine_goals": cmd_mine_goals,
        "mine_all": cmd_mine_all,
        "map_decisions": cmd_map_decisions,
        "map_all": cmd_map_all,
        "summarize": cmd_summarize,
        "summarize_all": cmd_summarize_all,
        "calibration_pilot": cmd_calibration_pilot,
        "judge": cmd_judge,
        "judge_all": cmd_judge_all,
        "build_matrix": cmd_build_matrix,
        "run_all": cmd_run_all,
    }

    await command_map[args.command](args)
