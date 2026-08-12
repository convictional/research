from pathlib import Path
from typing import Literal

from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit

from ..models import ActivityEvent, Decision, FinalizedGoalSet, StatedGoal
from ..settings import logger, settings
from .step1_unstated_extraction import run_step1
from .step2_stated_validation import run_step2
from .step3_consolidation import run_step3
from .step4_alignment_report import run_step4
from .step5_summary import run_step5


def _write_markdown(goal_set: FinalizedGoalSet, md_path: Path) -> None:
    lines = [
        f"# Goal Set — Condition: {goal_set.condition_name}",
        "",
        f"**Steps run:** {goal_set.run_metadata['steps_run']}",
        f"**Model IDs:** {goal_set.run_metadata['model_ids']}",
        "",
        "---",
        "",
        goal_set.summary_markdown,
        "",
        "---",
        "",
        f"## Canonical Goals ({len(goal_set.goals)})",
        "",
    ]
    for g in goal_set.goals:
        stated_tag = "stated" if g.is_stated else "unstated"
        lines.append(f"### [{g.id}] {g.title} `{stated_tag}` (support={g.activity_support_score:.2f})")
        lines.append("")
        lines.append(g.description)
        lines.append("")

    if goal_set.alignment_report:
        lines += [
            "---",
            "",
            "## Goal Alignment Report",
            "",
            goal_set.alignment_report.summary,
            "",
        ]
        for r in goal_set.alignment_report.relations:
            lines.append(f"- **{r.relation}** (conf={r.confidence:.2f}): {r.label}")
            lines.append(f"  - A: {r.goal_a_id}")
            lines.append(f"  - B: {r.goal_b_id}")
            lines.append("")

    md_path.write_text("\n".join(lines))


async def run_pipeline(
    condition: Literal["unstated", "stated", "mixed"],
    activity_events: list[ActivityEvent],
    decisions: list[Decision],
    stated_goals: list[StatedGoal],
    output_path: Path,
    mine_unstated: bool,
    load_from_cache: bool = True,
) -> FinalizedGoalSet:
    """Run the 5-step goal mining pipeline for a single condition.

    mine_unstated: if True, run step 1 (unstated extraction). Set False for the
      'stated' condition where no unstated mining is desired.
    """
    pkl_path = output_path / "step5_final_goal_set.pkl"

    if load_from_cache and pkl_path.exists():
        log_cache_hit(pkl_path)
        return load_pickle_file(pkl_path)

    print(f"\nRunning pipeline: condition={condition}, mine_unstated={mine_unstated}")

    # Step 1 — unstated extraction (skipped for 'stated' condition)
    if mine_unstated:
        candidates = await run_step1(
            activity_events=activity_events,
            decisions=decisions,
            output_path=output_path,
            load_from_cache=load_from_cache,
        )
    else:
        candidates = []
        logger.info(f"Step 1 SKIPPED — stated-only condition (mine_unstated=False)")
        print(f"  Step 1: SKIPPED (stated-only condition — no unstated mining)")

    # Step 2 — stated goal validation (no-op when stated_goals == [])
    evidence, step2_notes = await run_step2(
        stated_goals=stated_goals,
        activity_events=activity_events,
        decisions=decisions,
        output_path=output_path,
        load_from_cache=load_from_cache,
    )

    # Step 3 — consolidation (embeddings for deduplication only)
    canonical_goals = await run_step3(
        candidates=candidates,
        stated_goals=stated_goals,
        evidence=evidence,
        output_path=output_path,
        load_from_cache=load_from_cache,
    )

    # Step 4 — alignment report
    alignment_report = await run_step4(
        canonical_goals=canonical_goals,
        output_path=output_path,
        load_from_cache=load_from_cache,
    )

    # Step 5 — summary
    summary_markdown = await run_step5(
        canonical_goals=canonical_goals,
        alignment_report=alignment_report,
        output_path=output_path,
        load_from_cache=load_from_cache,
    )

    # Build steps_run based on what actually ran (step 1 skipped for stated-only)
    steps_run = ([1] if mine_unstated else []) + [2, 3, 4, 5]

    run_metadata: dict = {
        "steps_run": steps_run,
        "model_ids": settings.model_ids,
        "mine_unstated": mine_unstated,
    }
    run_metadata.update(step2_notes)

    goal_set = FinalizedGoalSet(
        condition_name=condition,
        goals=canonical_goals,
        alignment_report=alignment_report,
        summary_markdown=summary_markdown,
        run_metadata=run_metadata,
    )

    dump_to_pickle_file(goal_set, pkl_path)
    md_path = output_path / "final_goal_set.md"
    _write_markdown(goal_set, md_path)

    print(f"\n  Finalized goal set → {pkl_path}")
    print(f"  Human-readable    → {md_path}")
    return goal_set
