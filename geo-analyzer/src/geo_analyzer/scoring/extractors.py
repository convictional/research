"""Deterministic extractors. All operate on raw response text — no network.

Conventions:
- Aliases match case-insensitively at word boundaries.
- Multi-word aliases are matched as whole phrases (with internal whitespace
  collapsed). Word boundary on each side of the phrase.
- When two alias matches overlap, the longer match wins (no double counting).
- Offsets are (start, end) byte positions into the original (un-normalized) text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from geo_analyzer.types import Subject, SubjectKind

type _Span = tuple[int, int]


def _empty_spans() -> list[_Span]:
    return []


@dataclass(frozen=True)
class MentionResult:
    present: bool
    offsets: list[_Span] = field(default_factory=_empty_spans)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Compile a case-insensitive word-bounded pattern for one alias.

    Internal whitespace in multi-word aliases is collapsed to \\s+ so
    'organizational health' matches 'organizational  health' too.
    """
    parts = [re.escape(p) for p in alias.split()]
    body = r"\s+".join(parts) if len(parts) > 1 else parts[0]
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _find_all_alias_spans(text: str, aliases: list[str]) -> list[_Span]:
    """Return non-overlapping spans for any alias match. Longer aliases win on overlap."""
    spans: list[_Span] = []
    # Match longest aliases first so 'organizational health' beats 'health'.
    for alias in sorted(aliases, key=len, reverse=True):
        for m in _alias_pattern(alias).finditer(text):
            s, e = m.start(), m.end()
            if not any(_overlaps((s, e), existing) for existing in spans):
                spans.append((s, e))
    return sorted(spans)


def _overlaps(a: _Span, b: _Span) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def mention_presence(text: str, subject: Subject) -> MentionResult:
    """True if any alias for `subject` appears in `text` at a word boundary."""
    if not text:
        return MentionResult(present=False, offsets=[])
    spans = _find_all_alias_spans(text, list(subject.aliases))
    return MentionResult(present=bool(spans), offsets=spans)


def mention_count(text: str, subject: Subject) -> int:
    """Number of distinct (non-overlapping) alias matches in `text`."""
    if not text:
        return 0
    return len(_find_all_alias_spans(text, list(subject.aliases)))


@dataclass(frozen=True)
class OrdinalResult:
    rank: int | None
    """1-based position of the subject in the first detected list, or None."""


_NUMBERED_LINE = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")
_BULLET_LINE = re.compile(r"^\s*[-*+]\s+(.+)$")
_ORDINAL_PHRASES = [
    ("first", 1),
    ("second", 2),
    ("third", 3),
    ("fourth", 4),
    ("fifth", 5),
    ("sixth", 6),
    ("seventh", 7),
    ("eighth", 8),
    ("ninth", 9),
    ("tenth", 10),
]


def ordinal_rank(text: str, subject: Subject) -> OrdinalResult:
    """Return the subject's 1-based position in the first detected enumerated list,
    or None if no list is found or the subject does not appear in any list.

    Detection order:
      1. Contiguous markdown-numbered lines (1. ... 2. ...).
      2. Contiguous bullet lines (- ... or * ...).
      3. Inline ordinal-phrase enumeration ('first... second... third...').
    The first list with a hit wins.
    """
    aliases = list(subject.aliases)

    rank = _rank_in_numbered_list(text, aliases)
    if rank is not None:
        return OrdinalResult(rank=rank)

    rank = _rank_in_bullet_list(text, aliases)
    if rank is not None:
        return OrdinalResult(rank=rank)

    rank = _rank_in_ordinal_phrases(text, aliases)
    return OrdinalResult(rank=rank)


def _alias_in(text: str, aliases: list[str]) -> bool:
    return any(_alias_pattern(a).search(text) for a in aliases)


def _rank_in_numbered_list(text: str, aliases: list[str]) -> int | None:
    """Walk lines; the first contiguous block of `\\d+. ` lines becomes the list."""
    lines = text.splitlines()
    in_list = False
    items: list[str] = []
    for line in lines:
        m = _NUMBERED_LINE.match(line)
        if m:
            in_list = True
            items.append(m.group(2))
        elif in_list and line.strip() == "":
            # Blank line mid-list: common in markdown LLM output.
            # TODO(phase-3): two numbered lists separated only by blanks merge into one here.
            # Validate against real responses before trusting ranks beyond a single list.
            continue
        elif in_list:
            break  # end of the first numbered block
    if not items:
        return None
    for i, item in enumerate(items, start=1):
        if _alias_in(item, aliases):
            return i
    return None


def _rank_in_bullet_list(text: str, aliases: list[str]) -> int | None:
    lines = text.splitlines()
    in_list = False
    items: list[str] = []
    for line in lines:
        m = _BULLET_LINE.match(line)
        if m:
            in_list = True
            items.append(m.group(1))
        elif in_list and line.strip() == "":
            continue
        elif in_list:
            break
    if not items:
        return None
    for i, item in enumerate(items, start=1):
        if _alias_in(item, aliases):
            return i
    return None


def _rank_in_ordinal_phrases(text: str, aliases: list[str]) -> int | None:
    """Find the first ordinal phrase whose immediately following sentence
    contains an alias. The 'sentence' is the substring from the phrase to
    the next sentence terminator or the next ordinal phrase, whichever comes first.
    """
    lower = text.lower()
    starts: list[tuple[int, int]] = []  # (offset, rank)
    for word, n in _ORDINAL_PHRASES:
        for m in re.finditer(rf"(?<!\w){re.escape(word)}(?!\w)", lower):
            starts.append((m.start(), n))
    if not starts:
        return None
    starts.sort()
    for i, (start, rank) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        terminator = re.search(r"[.!?\n]", text[start:end])
        scope_end = start + terminator.end() if terminator else end
        if _alias_in(text[start:scope_end], aliases):
            return rank
    return None


@dataclass(frozen=True)
class ShareOfVoiceResult:
    value: float | None
    """mentions(subject) / (mentions(subject) + sum(mentions(competitor)))
    or None when the denominator is 0."""
    subject_mentions: int = 0
    competitor_mentions: int = 0


def share_of_voice(
    text: str,
    subject: Subject,
    subjects_by_id: dict[str, Subject],
) -> ShareOfVoiceResult:
    """Compute share of voice for `subject` against its declared competitors.

    `subjects_by_id` is the catalog's full subject map. Competitor ids that
    aren't in the map contribute 0 mentions (silently — the catalog
    loader may declare a competitor that's a string-only label, not a Subject).
    """
    sub_n = mention_count(text, subject)
    comp_n = 0
    for cid in subject.competitors:
        comp = subjects_by_id.get(cid)
        if comp is None:
            continue
        comp_n += mention_count(text, comp)
    denom = sub_n + comp_n
    value: float | None = sub_n / denom if denom > 0 else None
    return ShareOfVoiceResult(value=value, subject_mentions=sub_n, competitor_mentions=comp_n)


@dataclass(frozen=True)
class BrandLegacyResult:
    fired: bool
    brand_offsets: list[_Span] = field(default_factory=_empty_spans)
    legacy_offsets: list[_Span] = field(default_factory=_empty_spans)


def brand_legacy_conflation(
    text: str,
    brand: Subject,
    legacy: Subject,
) -> BrandLegacyResult:
    """True when both `brand` and `legacy` aliases co-occur in `text`.

    `legacy.kind` must be ANTI_BRAND — guards against accidental misuse.
    """
    if legacy.kind != SubjectKind.ANTI_BRAND:
        raise ValueError(f"brand_legacy_conflation: legacy subject must have kind=anti_brand; " f"got {legacy.kind!r}")
    brand_spans = _find_all_alias_spans(text, list(brand.aliases))
    legacy_spans = _find_all_alias_spans(text, list(legacy.aliases))
    fired = bool(brand_spans) and bool(legacy_spans)
    return BrandLegacyResult(
        fired=fired,
        brand_offsets=brand_spans,
        legacy_offsets=legacy_spans,
    )


@dataclass(frozen=True)
class CitationResult:
    urls: list[str]
    owned_urls: list[str]


_URL_RE = re.compile(r"https?://[^\s)>\]\"]+")


def extract_citations(text: str, *, owned_domains: list[str]) -> CitationResult:
    """Extract URLs and tag any whose host matches an owned domain (or subdomain)."""
    seen: list[str] = []
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?")
        if url not in seen:
            seen.append(url)
    owned_set = {d.lower() for d in owned_domains}
    owned: list[str] = []
    for url in seen:
        host = _host_of(url)
        if host is None:
            continue
        if any(host == d or host.endswith("." + d) for d in owned_set):
            owned.append(url)
    return CitationResult(urls=seen, owned_urls=owned)


def _host_of(url: str) -> str | None:
    m = re.match(r"https?://([^/?#]+)", url)
    if not m:
        return None
    return m.group(1).lower()
