"""Fixed-length research summary — the obfuscation layer between Phase 2 mapping and Phase 3 judging.

Each Phase-2 artifact (rendered mapping markdown, 16k–30k words) is compressed into a
fixed-length neutral prose summary (~450–600 words) before the MoE judge ensemble scores it.
This eliminates the volume gap between schemas: GM artifacts are naturally ~70% larger than
DM/DSM, which gives GM an irreducible LLM-judging advantage when passed raw.

The summary deliberately hides how the source artifact was structured (single mapping,
scored mapping, or relationship graph) so judges compare information density on a level
playing field, not structural depth.

NOTE: The summarizer uses Claude Sonnet, which is also one of the three judge models.
This overlap is known and unavoidable without a non-Claude summarizer; it is documented
here rather than hidden. The calibration pilot's cross-schema length check is the
empirical guard that confirms the obfuscation is working.
"""
import re
import warnings
from pathlib import Path

from pydantic import BaseModel

from ..instruct_helper import ainstruct_llm, set_async_instructor_client
from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit
from common.prompt_template_engine import build_prompt

from ..settings import CLAUDE_OPUS, logger, settings

# Forbidden tokens in the generated summary — structural tells that would leak the
# underlying schema to the judge. Two tiers:
#   Hard-fail: unambiguous schema tells → raise ValueError immediately
#   Warn-and-retry: ambiguous ordinary English words → one retry with explicit instruction

_HARD_FAIL_TOKENS = [
    "graph", "node", "nodes", "edge", "edges", "vertex", "vertices",
    "network", "diagram", "threshold", "relation type", "relationship type",
    "relation kind", "closed vocabulary", "labeled edge",
]
_HARD_FAIL_PATTERNS = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in _HARD_FAIL_TOKENS]

# Decimal scores in [0,1] like "0.82" or ".82" — DSM's score tell
_DECIMAL_SCORE_PATTERN = re.compile(r"\b0?\.\d{2}\b")

# Bracketed confidence tiers — DM/GM render [high] [medium] [low]
_BRACKET_TIER_PATTERN = re.compile(r"\[(high|medium|low)\]", re.IGNORECASE)

_WARN_RETRY_TOKENS = ["scored", "scoring", "score range", "single-goal", "one-to-one", "at most one goal"]
_WARN_RETRY_PATTERNS = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in _WARN_RETRY_TOKENS]

# A leaked structural tell or an over-budget summary is regenerated with an explicit
# correction up to this many times before we give up. The summarizer runs at temp 0,
# so a leak is deterministic for a given prompt — appending the exact offending words
# to the prompt changes the output and reliably removes them. Only a tell that
# survives every retry aborts the run.
_MAX_FIXUP_RETRIES = 3


def _check_hard_fails(text: str) -> list[str]:
    found = []
    for pattern, token in zip(_HARD_FAIL_PATTERNS, _HARD_FAIL_TOKENS):
        if pattern.search(text):
            found.append(token)
    if _DECIMAL_SCORE_PATTERN.search(text):
        found.append("decimal score (0.NN)")
    if _BRACKET_TIER_PATTERN.search(text):
        found.append("[high/medium/low] bracket tier")
    return found


def _check_warn_retry(text: str) -> list[str]:
    return [t for pattern, t in zip(_WARN_RETRY_PATTERNS, _WARN_RETRY_TOKENS) if pattern.search(text)]


class ResearchSummary(BaseModel):
    summary_md: str


async def _call_summarizer(
    system_prompt: str,
    user_prompt: str,
) -> str:
    set_async_instructor_client(CLAUDE_OPUS, settings.anthropic_api_key)
    result: ResearchSummary = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ResearchSummary,
        llm_model=settings.summarizer_model,
        temperature=0.0,
        max_tokens=4096,
    )
    return result.summary_md


async def summarize_artifact(
    rendered_md: str,
    output_path: Path,
    schema: str,
    load_from_cache: bool = True,
) -> tuple[str, int]:
    """Compress a schema-masked mapping artifact into a fixed-length neutral summary.

    Returns (summary_md, word_count). Caches to summary_{schema}.md (human-
    inspectable) and summary_{schema}.pkl (provenance + word count).
    """
    cache_pkl = output_path / f"summary_{schema}.pkl"

    if load_from_cache and cache_pkl.exists():
        log_cache_hit(cache_pkl)
        cached = load_pickle_file(cache_pkl)
        return cached["summary_md"], cached["word_count"]

    print(f"  Summarizing {schema} → fixed-length research summary [{settings.summarizer_model}]...")

    system_prompt = build_prompt("research_summary_system.jinja")
    user_prompt = build_prompt(
        "research_summary_user.jinja",
        artifact_markdown=rendered_md,
        word_target=settings.summary_word_target,
        word_min=settings.summary_word_min,
        word_max=settings.summary_word_max,
    )

    summary_md = await _call_summarizer(system_prompt, user_prompt)

    # Post-generation fixup loop. A summary can leak structural tells (hard-fail
    # tokens like "graph", soft warn-retry tokens) or exceed the word budget. Both
    # are regenerated with an explicit correction naming exactly what was wrong, so
    # the model can target it. We loop until the summary is clean or retries are
    # exhausted, then apply the final hard guards below.
    word_ceiling = settings.summary_word_max * 1.15
    for attempt in range(_MAX_FIXUP_RETRIES):
        hard_fails = _check_hard_fails(summary_md)
        warn_tokens = _check_warn_retry(summary_md)
        wc = len(summary_md.split())
        over_by = wc - settings.summary_word_max if wc > word_ceiling else 0

        leaked = hard_fails + warn_tokens
        if not leaked and not over_by:
            break

        instructions = []
        if leaked:
            instructions.append(
                f"Your previous response used these forbidden words: {', '.join(leaked)}. "
                "They reveal how the source was structured and must NEVER appear. Remove every "
                "one of them, describing the decision-to-goal relationships in plain business prose."
            )
        if over_by:
            instructions.append(
                f"Your previous response was {wc} words, {over_by} over the "
                f"{settings.summary_word_max}-word maximum. Rewrite to fit strictly within "
                f"{settings.summary_word_min}–{settings.summary_word_max} words; cut ruthlessly."
            )
        logger.warning(
            f"summarize_artifact({schema}): fixup attempt {attempt + 1}/{_MAX_FIXUP_RETRIES} "
            f"(leaked={leaked or 'none'}, words={wc}). Regenerating."
        )
        retry_user = user_prompt + (
            "\n\nIMPORTANT — your previous attempt violated these rules; fix all of them while "
            "keeping the same five sections and content:\n- " + "\n- ".join(instructions)
        )
        summary_md = await _call_summarizer(system_prompt, retry_user)

    # Final hard guards. A surviving hard-fail tell aborts (the obfuscation guarantee
    # cannot be met). An over-budget summary only warns — never truncate, never crash
    # the sweep; the calibration pilot's length check is the empirical backstop.
    hard_fails = _check_hard_fails(summary_md)
    if hard_fails:
        raise ValueError(
            f"summarize_artifact({schema}): hard-fail schema tells still present after "
            f"{_MAX_FIXUP_RETRIES} retries: {hard_fails}. The summary leaks structural "
            "information to the judge. Fix the prompt."
        )

    wc = len(summary_md.split())
    if wc > word_ceiling:
        warnings.warn(
            f"summarize_artifact({schema}): {wc} words still exceeds budget after "
            f"{_MAX_FIXUP_RETRIES} retries. Proceeding with oversized summary — "
            "re-run calibration_pilot to detect length leakage.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Save human-inspectable markdown
    md_path = output_path / f"summary_{schema}.md"
    md_path.write_text(summary_md)

    # Save pkl with provenance
    cache_data = {
        "summary_md": summary_md,
        "word_count": wc,
        "model_id": settings.summarizer_model,
        "schema": schema,
    }
    dump_to_pickle_file(cache_data, cache_pkl)

    print(f"  Summary done: {wc} words → {md_path}")
    return summary_md, wc
