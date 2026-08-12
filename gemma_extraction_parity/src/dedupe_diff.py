import logging
import re
from collections import Counter

import instructor
from anthropic import AsyncAnthropic

from src.models import MatchResult, ParityAnalysis, SharedLearningPair
from src.settings import settings

logger = logging.getLogger(__name__)

DEDUP_SYSTEM_PROMPT = """\
You are analyzing research learnings extracted from internal company documents.

Your task: de-duplicate a list of learnings. Many learnings say the same thing because they were extracted from overlapping document sets across multiple search queries.

Instructions:
1. Group learnings that describe the same event, decision, person, or fact — even if they include slightly different supporting details or quotes. Merge them into one.
2. For each group, produce a single canonical learning that combines the best details from all versions. Prefer specific dates, names, dollar amounts, and direct quotes.
3. Only keep two learnings as separate if they describe genuinely independent facts (different events, different people, different decisions).
4. Return ONLY a numbered list of unique learnings. No preamble, no explanation.
5. Preserve the [^content:...] citation markers exactly as they appear. When merging, keep all unique citations.

Examples of learnings that SHOULD be merged into one:
- "The Chief of Staff was identified as a critical ICP champion. The CEO stated 'we need to make them look good.'" AND "The CEO noted the Chief of Staff is the primary champion, stating 'we need it to be good for the chief of staff's career.'" → Same fact (Chief of Staff = ICP champion, same quote). Merge.
- "The Goal Alignment Graph was identified as a 'killer feature' showing how work bubbles up to goals." AND "The chief of staff argued the Goal Alignment Graph answers questions 'impossible to answer from Slack.'" → Same feature being described. Merge into one learning with both details.
- "The company sold its legacy marketplace business in January 2025." AND "The legacy marketplace divestiture in January 2025 enabled the pivot to AI tools." → Same event. Merge."""

MATCH_SYSTEM_PROMPT = """\
You are comparing two numbered lists of de-duplicated research learnings — list A and list B — extracted from the same underlying documents by two different models.

Your task: identify pairs of learnings that convey the same core fact or insight.

Rules:
1. For each pair, provide the 1-based index from list A and the 1-based index from list B.
2. Only pair learnings that genuinely describe the same fact. Different details about the same broad topic are NOT a match.
3. Each index should appear in at most one pair. If a learning has no counterpart, leave it unpaired.
4. When in doubt, leave a learning unpaired rather than forcing a weak match."""

TIEBREAK_SYSTEM_PROMPT = """\
You are resolving a duplicate detection issue in a research learning comparison.

A single learning from one model was paired with multiple learnings from the other model. This implies either:
(a) The multiply-paired learnings are actually duplicates that de-duplication missed, OR
(b) One or more of the pairings is a false match.

For each group below, decide: are the multiply-paired learnings duplicates of each other? If yes, pick the best one and the correct pair. If no, pick the single strongest pair and mark the rest as unpaired."""


def count_deduped(text: str) -> int:
    return len([line for line in text.strip().splitlines() if re.match(r"^\d+\.", line.strip())])


def _parse_numbered_list(text: str) -> list[str]:
    items = []
    for line in text.strip().splitlines():
        match = re.match(r"^\d+\.\s*(.+)", line.strip())
        if match:
            items.append(match.group(1))
    return items


async def deduplicate_learnings(learnings: list[str], variant: str) -> tuple[str, list[str]]:
    if not learnings:
        return "(no learnings)", []

    numbered = "\n".join(f"{i + 1}. {l}" for i, l in enumerate(learnings))
    user_prompt = f"De-duplicate these {len(learnings)} {variant} learnings:\n\n{numbered}"

    client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    message = await client.messages.create(
        model=settings.sonnet_model,
        max_tokens=8192,
        temperature=0.0,
        system=DEDUP_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    deduped_text = message.content[0].text
    deduped_list = _parse_numbered_list(deduped_text)
    unique_count = count_deduped(deduped_text)
    logger.info(f"{variant}: {len(learnings)} raw → {unique_count} unique")
    return deduped_text, deduped_list


async def match_learnings(
    a_deduped: list[str],
    b_deduped: list[str],
    variant_a: str,
    variant_b: str,
) -> MatchResult:
    a_numbered = "\n".join(f"{i + 1}. {l}" for i, l in enumerate(a_deduped))
    b_numbered = "\n".join(f"{i + 1}. {l}" for i, l in enumerate(b_deduped))
    user_prompt = (
        f"## {variant_a.capitalize()} Learnings (List A, {len(a_deduped)} items)\n\n{a_numbered}\n\n"
        f"## {variant_b.capitalize()} Learnings (List B, {len(b_deduped)} items)\n\n{b_numbered}"
    )

    client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    instructor_client = instructor.from_anthropic(client)

    result = await instructor_client.chat.completions.create(
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=16384,
        model=settings.sonnet_model,
        temperature=0.0,
        response_model=MatchResult,
        system=MATCH_SYSTEM_PROMPT,
    )

    # The LLM occasionally hallucinates an out-of-range index; drop those pairs so
    # downstream 1-based indexing (resolve_duplicates, derive_parity, report) can't crash.
    valid_pairs = []
    for pair in result.pairs:
        if 1 <= pair.a_index <= len(a_deduped) and 1 <= pair.b_index <= len(b_deduped):
            valid_pairs.append(pair)
        else:
            logger.warning(
                f"Dropping out-of-range match pair (a={pair.a_index}/{len(a_deduped)}, "
                f"b={pair.b_index}/{len(b_deduped)}) for {variant_a} vs {variant_b}"
            )
    result.pairs = valid_pairs
    return result


def _find_duplicate_indices(pairs: list[SharedLearningPair]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Find indices that appear in more than one pair.

    Returns (a_dupes, b_dupes) where each maps a duplicated index
    to the list of pair positions it appears in.
    """
    a_counts: dict[int, list[int]] = {}
    b_counts: dict[int, list[int]] = {}
    for i, pair in enumerate(pairs):
        a_counts.setdefault(pair.a_index, []).append(i)
        b_counts.setdefault(pair.b_index, []).append(i)
    a_dupes = {idx: positions for idx, positions in a_counts.items() if len(positions) > 1}
    b_dupes = {idx: positions for idx, positions in b_counts.items() if len(positions) > 1}
    return a_dupes, b_dupes


async def _tiebreak_group(
    anchor_learning: str,
    anchor_side: str,
    candidate_learnings: list[tuple[int, str]],
    candidate_side: str,
) -> int:
    """Ask the LLM which candidate is the true match for the anchor. Returns the winning index."""
    candidates_text = "\n".join(f"  {idx}. {text}" for idx, text in candidate_learnings)
    user_prompt = (
        f"Anchor ({anchor_side} learning):\n  {anchor_learning}\n\n"
        f"Candidates ({candidate_side} learnings) — all were paired with the anchor:\n{candidates_text}\n\n"
        f"Which single candidate index is the strongest match? If none are strong matches, pick the best one. "
        f"Respond with just the index number."
    )

    client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    message = await client.messages.create(
        model=settings.sonnet_model,
        max_tokens=64,
        temperature=0.0,
        system=TIEBREAK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = message.content[0].text.strip()
    match = re.search(r"\d+", text)
    if match:
        return int(match.group())
    return candidate_learnings[0][0]


async def resolve_duplicates(
    pairs: list[SharedLearningPair],
    a_deduped: list[str],
    b_deduped: list[str],
    variant_a: str,
    variant_b: str,
) -> tuple[list[SharedLearningPair], list[str]]:
    """Resolve cases where an index appears in multiple pairs.

    Returns (clean_pairs, warnings).
    """
    a_dupes, b_dupes = _find_duplicate_indices(pairs)
    if not a_dupes and not b_dupes:
        return pairs, []

    warnings: list[str] = []
    drop_positions: set[int] = set()

    for a_idx, positions in a_dupes.items():
        a_text = a_deduped[a_idx - 1]
        candidates = [(pairs[p].b_index, b_deduped[pairs[p].b_index - 1]) for p in positions]
        warnings.append(
            f"{variant_a} #{a_idx} paired with {len(positions)} {variant_b} learnings "
            f"(#{', #'.join(str(pairs[p].b_index) for p in positions)}) — "
            f"implies {variant_b} dedupe missed a duplicate"
        )
        winner = await _tiebreak_group(a_text, variant_a, candidates, variant_b)
        for p in positions:
            if pairs[p].b_index != winner:
                drop_positions.add(p)

    for b_idx, positions in b_dupes.items():
        b_text = b_deduped[b_idx - 1]
        candidates = [(pairs[p].a_index, a_deduped[pairs[p].a_index - 1]) for p in positions]
        warnings.append(
            f"{variant_b} #{b_idx} paired with {len(positions)} {variant_a} learnings "
            f"(#{', #'.join(str(pairs[p].a_index) for p in positions)}) — "
            f"implies {variant_a} dedupe missed a duplicate"
        )
        winner = await _tiebreak_group(b_text, variant_b, candidates, variant_a)
        for p in positions:
            if pairs[p].a_index != winner:
                drop_positions.add(p)

    clean_pairs = [p for i, p in enumerate(pairs) if i not in drop_positions]

    # After dropping, re-check for remaining duplicates (from overlapping resolutions)
    still_duped_a, still_duped_b = _find_duplicate_indices(clean_pairs)
    if still_duped_a or still_duped_b:
        seen_a: set[int] = set()
        seen_b: set[int] = set()
        final_pairs = []
        for p in clean_pairs:
            if p.a_index in seen_a or p.b_index in seen_b:
                continue
            seen_a.add(p.a_index)
            seen_b.add(p.b_index)
            final_pairs.append(p)
        clean_pairs = final_pairs
        warnings.append("Had to drop additional pairs after tiebreaking to ensure 1:1 mapping")

    logger.info(f"Duplicate resolution: {len(pairs)} pairs → {len(clean_pairs)} clean ({len(warnings)} warnings)")
    return clean_pairs, warnings


def derive_parity(
    pairs: list[SharedLearningPair],
    a_deduped: list[str],
    b_deduped: list[str],
    warnings: list[str],
) -> ParityAnalysis:
    paired_a = {p.a_index for p in pairs}
    paired_b = {p.b_index for p in pairs}

    a_only = [l for i, l in enumerate(a_deduped, 1) if i not in paired_a]
    b_only = [l for i, l in enumerate(b_deduped, 1) if i not in paired_b]

    return ParityAnalysis(
        shared=pairs,
        a_only=a_only,
        b_only=b_only,
        a_deduped=a_deduped,
        b_deduped=b_deduped,
        duplicate_warnings=warnings,
    )
