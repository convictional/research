"""Length-bias calibration pilot — MUST pass before running the full judge sweep.

Two checks:

Check A (new core): Cross-schema length normalization.
  Summarizes all three schemas for the 'mixed' condition and verifies that the
  summaries are within the configured word band and that the max/min ratio is ≤ 1.25.
  This directly proves the obfuscation layer is working: GM artifacts are ~70% larger
  than DM raw but their summaries must be indistinguishably similar in length.

Check B (retained): Padding-bias guard.
  Tests whether the judge ensemble scores a padded (+50% words, zero new information)
  version of the GM summary higher than the real summary. If delta > 0.2 points, the
  information_density rubric dimension needs re-tightening.

passed = Check A OK AND Check B OK.
"""
from datetime import datetime, timezone
from pathlib import Path

from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit

from ..settings import logger, settings
from .aggregator import aggregate
from .moe_judge import run_cell_judges
from .research_summary import summarize_artifact
from .rubric import CalibrationResult, cell_id

PILOT_CONDITION = "mixed"
PILOT_SCHEMA = "gm"
PILOT_CELL_ID = cell_id(PILOT_CONDITION, PILOT_SCHEMA)
PILOT_CACHE = settings.output_path / "calibration_pilot.pkl"
CALIBRATION_PASSED_FLAG = settings.output_path / "calibration_pilot_passed.flag"

LENGTH_BIAS_THRESHOLD = 0.2
MAX_PAIRWISE_RATIO_THRESHOLD = 1.25


def _pad_artifact(md: str) -> str:
    """Create a padded version with ~50% more words; zero new analytical information.

    Uses verbatim repetition of content lines so judges cannot gain information
    from the filler — any score increase must be pure length bias.
    """
    current_wc = len(md.split())
    target_extra = current_wc // 2

    # Extract non-header, non-separator content lines
    content_lines = [
        line for line in md.split("\n")
        if line.strip() and not line.startswith("#") and not line.startswith("|") and not line.startswith("---")
    ]

    header = (
        "\n\n---\n\n"
        "## Extended Documentation\n\n"
        "This section provides a verbose restatement of the candidate briefing above. "
        "No new analytical conclusions, connections, or reasoning are introduced here. "
        "The entries documented above remain the complete and authoritative record.\n\n"
    )

    # Cycle through content lines until we have enough words
    filler_words: list[str] = []
    for _ in range(20):  # upper bound on cycling
        for line in content_lines:
            filler_words.extend(line.split())
            if len(filler_words) >= target_extra:
                break
        if len(filler_words) >= target_extra:
            break

    return md + header + " ".join(filler_words[:target_extra])


def _load_mapping_md(schema: str) -> tuple[str, int]:
    """Load the schema-masked mapping markdown for the mixed condition."""
    md_path = settings.condition_output_path(PILOT_CONDITION) / f"mapping_{schema}.md"
    if not md_path.exists():
        raise FileNotFoundError(
            f"Mapping md not found: {md_path}\n"
            f"Run 'map_decisions --condition {PILOT_CONDITION} --schema {schema}' first."
        )
    content = md_path.read_text()
    return content, len(content.split())


def _load_summary_md(schema: str) -> tuple[str, int]:
    """Load the pre-built summary for the mixed condition."""
    md_path = settings.condition_output_path(PILOT_CONDITION) / f"summary_{schema}.md"
    if not md_path.exists():
        raise FileNotFoundError(
            f"Summary md not found: {md_path}\n"
            f"Run 'summarize --condition {PILOT_CONDITION} --schema {schema}' first."
        )
    content = md_path.read_text()
    return content, len(content.split())


async def run_calibration_pilot(load_from_cache: bool = True) -> CalibrationResult:
    """Run the two-check calibration pilot.

    Returns CalibrationResult with pass/fail verdict. Writes a flag file if passed.
    """
    if load_from_cache and PILOT_CACHE.exists():
        log_cache_hit(PILOT_CACHE)
        result: CalibrationResult = load_pickle_file(PILOT_CACHE)
        _print_result(result)
        return result

    print("\n" + "=" * 60)
    print("  CALIBRATION PILOT: obfuscation-layer guards")
    print(f"  Condition: {PILOT_CONDITION} (richest goal set)")
    print("=" * 60)

    # ── Check A: cross-schema length normalization ─────────────────────────────
    print("\n  Check A: verifying summaries are length-normalized across schemas...")
    summary_word_counts: dict[str, int] = {}
    for schema in ["dm", "dsm", "gm"]:
        # Summarize (or load from cache)
        mapping_md, _ = _load_mapping_md(schema)
        output_path = settings.condition_output_path(PILOT_CONDITION)
        _, wc = await summarize_artifact(
            rendered_md=mapping_md,
            output_path=output_path,
            schema=schema,
            load_from_cache=load_from_cache,
        )
        summary_word_counts[schema] = wc
        print(f"    {schema}: {wc} words")

    counts = list(summary_word_counts.values())
    min_wc = min(counts)
    max_wc = max(counts)
    max_pairwise_ratio = max_wc / min_wc if min_wc > 0 else float("inf")

    band_ok_each = all(
        settings.summary_word_min <= wc <= settings.summary_word_max * 1.15
        for wc in counts
    )
    ratio_ok = max_pairwise_ratio <= MAX_PAIRWISE_RATIO_THRESHOLD
    length_band_ok = band_ok_each and ratio_ok

    print(f"    Band [{settings.summary_word_min}–{int(settings.summary_word_max * 1.15)}]: {'✓' if band_ok_each else '✗'}")
    print(f"    Max/min ratio: {max_pairwise_ratio:.2f} (threshold {MAX_PAIRWISE_RATIO_THRESHOLD}): {'✓' if ratio_ok else '✗'}")

    # ── Check B: padding-bias guard on the GM summary ──────────────────────────
    print("\n  Check B: padding-bias guard on GM summary...")
    gm_summary_md, gm_summary_wc = _load_summary_md(PILOT_SCHEMA)
    padded_md = _pad_artifact(gm_summary_md)
    padded_wc = len(padded_md.split())
    print(f"    Real summary: {gm_summary_wc} words")
    print(f"    Padded summary: {padded_wc} words (+{padded_wc - gm_summary_wc} = {(padded_wc/gm_summary_wc - 1)*100:.0f}%)")

    print("\n  Running 9 judges on REAL summary...")
    real_runs = await run_cell_judges(
        cell_id=PILOT_CELL_ID + "__real",
        rendered_md=gm_summary_md,
        rendered_word_count=gm_summary_wc,
        schema=f"{PILOT_SCHEMA}_pilot_real",
        output_path=settings.condition_output_path(PILOT_CONDITION),
        temperature=0.0,
        load_from_cache=load_from_cache,
    )

    print("\n  Running 9 judges on PADDED summary...")
    padded_runs = await run_cell_judges(
        cell_id=PILOT_CELL_ID + "__padded",
        rendered_md=padded_md,
        rendered_word_count=padded_wc,
        schema=f"{PILOT_SCHEMA}_pilot_padded",
        output_path=settings.condition_output_path(PILOT_CONDITION),
        temperature=0.0,
        load_from_cache=load_from_cache,
    )

    real_agg = aggregate(real_runs, PILOT_CONDITION, PILOT_SCHEMA)
    padded_agg = aggregate(padded_runs, PILOT_CONDITION, PILOT_SCHEMA)

    real_score = real_agg.trimmed_mean_overall
    padded_score = padded_agg.trimmed_mean_overall
    delta = padded_score - real_score
    length_bias_ok = delta <= LENGTH_BIAS_THRESHOLD

    # ── Assemble result ────────────────────────────────────────────────────────
    passed = length_band_ok and length_bias_ok

    warn_parts = []
    if not length_band_ok:
        warn_parts.append(
            f"LENGTH NORMALIZATION FAILED: summaries are not within the required band "
            f"or ratio (max/min={max_pairwise_ratio:.2f}, threshold={MAX_PAIRWISE_RATIO_THRESHOLD}). "
            "The obfuscation layer is not normalizing volume. Tighten research_summary_user.jinja."
        )
    if not length_bias_ok:
        warn_parts.append(
            f"LENGTH BIAS DETECTED: padded summary scored {delta:.1f} points HIGHER than real "
            f"(threshold: {LENGTH_BIAS_THRESHOLD}). "
            "The information_density rubric dimension needs re-tightening."
        )
    warn = " | ".join(warn_parts) if warn_parts else None

    result = CalibrationResult(
        ran_at=datetime.now(timezone.utc),
        summary_word_counts=summary_word_counts,
        length_band_ok=length_band_ok,
        max_pairwise_word_ratio=round(max_pairwise_ratio, 3),
        real_trimmed_mean=real_score,
        padded_trimmed_mean=padded_score,
        delta=delta,
        length_bias_ok=length_bias_ok,
        passed=passed,
        threshold=LENGTH_BIAS_THRESHOLD,
        warning_message=warn,
    )

    dump_to_pickle_file(result, PILOT_CACHE)
    _print_result(result)

    if passed:
        CALIBRATION_PASSED_FLAG.write_text("passed")
        print(f"\n  Flag written: {CALIBRATION_PASSED_FLAG}")

    return result


def _print_result(result: CalibrationResult) -> None:
    print("\n" + "=" * 60)
    print(f"  CALIBRATION RESULT: {'✓ PASSED' if result.passed else '✗ FAILED'}")
    print(f"\n  Check A — Length normalization:")
    if result.summary_word_counts:
        for schema, wc in result.summary_word_counts.items():
            print(f"    {schema}: {wc} words")
    print(f"    Band OK: {result.length_band_ok}  |  Max/min ratio: {result.max_pairwise_word_ratio:.2f}")
    print(f"\n  Check B — Padding-bias guard:")
    print(f"    Real summary trimmed mean:   {result.real_trimmed_mean:.2f}")
    print(f"    Padded summary trimmed mean: {result.padded_trimmed_mean:.2f}")
    print(f"    Delta (padded-real):         {result.delta:.2f} (threshold: {result.threshold})")
    if result.warning_message:
        print(f"\n  ⚠  {result.warning_message}")
    print("=" * 60)


def check_pilot_passed() -> bool:
    """Return True if the calibration pilot has been run and passed."""
    return CALIBRATION_PASSED_FLAG.exists()
