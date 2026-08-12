import json
import random
from pathlib import Path

from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt
from src.models import (
    BatchAnalysis,
    DiscoveredRubric,
    RatedReport,
    RefinedRubric,
    SynthesizedRubric,
)
from src.settings import settings, logger


def _sample_reports_across_scores(
    reports: list[RatedReport], n: int = settings.rubric_sample_size
) -> list[RatedReport]:
    by_score: dict[int, list[RatedReport]] = {}
    for r in reports:
        by_score.setdefault(r.quality_score, []).append(r)

    per_score = max(1, n // len(by_score))
    sampled = []
    for score in sorted(by_score.keys()):
        pool = by_score[score]
        random.shuffle(pool)
        sampled.extend(pool[: min(per_score, len(pool))])

    random.shuffle(sampled)
    return sampled[:n]


def _batch(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _analyze_batch(batch: list[RatedReport]) -> BatchAnalysis:
    system_prompt = build_prompt("rubric_discovery_system.txt.jinja")
    user_prompt = build_prompt("rubric_discovery_user.txt.jinja", reports=batch)

    return await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=BatchAnalysis,
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


async def _synthesize_rubric(batch_analyses: list[BatchAnalysis], version: int = 1) -> DiscoveredRubric:
    system_prompt = build_prompt("rubric_discovery_system.txt.jinja")
    analyses_text = "\n\n".join(
        f"Batch {i+1}:\nPatterns: {json.dumps([p.model_dump() for p in a.patterns], indent=2)}\n"
        f"Observations: {a.observations}"
        for i, a in enumerate(batch_analyses)
    )
    user_prompt = build_prompt("rubric_synthesis_user.txt.jinja", analyses_text=analyses_text)

    synthesized = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=SynthesizedRubric,
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    return DiscoveredRubric(
        dimensions=synthesized.dimensions,
        version=version,
        general_notes=synthesized.general_notes,
    )


async def discover_rubric(train_reports: list[RatedReport], version: int = 1) -> DiscoveredRubric:
    logger.info(f"Discovering rubric v{version} from {len(train_reports)} training reports")

    sampled = _sample_reports_across_scores(train_reports)
    logger.info(f"Sampled {len(sampled)} reports across score levels")

    batches = _batch(sampled, settings.rubric_batch_size)
    logger.info(f"Analyzing {len(batches)} batches")

    batch_analyses = []
    for i, batch in enumerate(batches):
        logger.info(f"Analyzing batch {i+1}/{len(batches)}")
        analysis = await _analyze_batch(batch)
        batch_analyses.append(analysis)

    logger.info("Synthesizing rubric from batch analyses")
    rubric = await _synthesize_rubric(batch_analyses, version=version)

    save_path = settings.rubric_path / f"rubric_v{version}.json"
    save_path.write_text(rubric.model_dump_json(indent=2))
    logger.info(f"Saved rubric v{version} to {save_path}")

    _print_rubric(rubric)
    return rubric


async def refine_rubric(
    current_rubric: DiscoveredRubric,
    rubric_changes: list[str],
    prompt_changes: list[str],
) -> DiscoveredRubric:
    new_version = current_rubric.version + 1
    logger.info(f"Refining rubric from v{current_rubric.version} to v{new_version}")

    system_prompt = build_prompt("rubric_discovery_system.txt.jinja")
    user_prompt = build_prompt(
        "rubric_refinement_user.txt.jinja",
        current_rubric=current_rubric,
        rubric_changes=rubric_changes,
        prompt_changes=prompt_changes,
    )

    refined = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=RefinedRubric,
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    rubric = DiscoveredRubric(
        dimensions=refined.dimensions,
        version=new_version,
        general_notes=f"Refined from v{current_rubric.version}. {refined.change_summary}",
    )

    save_path = settings.rubric_path / f"rubric_v{new_version}.json"
    save_path.write_text(rubric.model_dump_json(indent=2))
    logger.info(f"Saved rubric v{new_version} to {save_path}")

    _print_rubric(rubric)
    return rubric


def load_rubric(version: int | None = None) -> DiscoveredRubric:
    if version is not None:
        path = settings.rubric_path / f"rubric_v{version}.json"
        if not path.exists():
            raise FileNotFoundError(f"Rubric v{version} not found at {path}")
        return DiscoveredRubric.model_validate_json(path.read_text())

    rubric_files = sorted(settings.rubric_path.glob("rubric_v*.json"))
    if not rubric_files:
        raise FileNotFoundError(f"No rubric files found in {settings.rubric_path}")
    return DiscoveredRubric.model_validate_json(rubric_files[-1].read_text())


def _print_rubric(rubric: DiscoveredRubric) -> None:
    print(f"\n{'='*60}")
    print(f"  Discovered Rubric v{rubric.version}")
    print(f"{'='*60}")
    for dim in rubric.dimensions:
        print(f"\n  {dim.name} (weight: {dim.weight:.2f})")
        print(f"  {dim.description}")
        for anchor in sorted(dim.anchors, key=lambda a: a.score):
            print(f"    {anchor.score}: {anchor.description}")
    if rubric.general_notes:
        print(f"\n  Notes: {rubric.general_notes}")
