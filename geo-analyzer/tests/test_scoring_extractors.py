from __future__ import annotations

from geo_analyzer.scoring.extractors import (
    BrandLegacyResult,
    CitationResult,
    MentionResult,
    OrdinalResult,
    ShareOfVoiceResult,
    brand_legacy_conflation,
    extract_citations,
    mention_count,
    mention_presence,
    ordinal_rank,
    share_of_voice,
)
from geo_analyzer.types import Subject, SubjectKind


def _brand() -> Subject:
    return Subject(
        id="convictional_brand",
        kind=SubjectKind.BRAND,
        aliases=["convictional", "Convictional"],
        definition="x",
    )


class TestMentionPresence:
    def test_simple_word_match(self) -> None:
        out = mention_presence("Convictional is a platform.", _brand())
        assert out.present is True
        assert len(out.offsets) == 1
        start, end = out.offsets[0]
        assert "Convictional" in "Convictional is a platform."[start:end]

    def test_case_insensitive(self) -> None:
        out = mention_presence("CONVICTIONAL helps leaders.", _brand())
        assert out.present is True

    def test_word_boundary_required(self) -> None:
        # 'unconvictional' should NOT match alias 'convictional'.
        out = mention_presence("This is unconvictional behavior.", _brand())
        assert out.present is False
        assert out.offsets == []

    def test_substring_in_other_word_does_not_match(self) -> None:
        # 'preconvictionalism' must not match 'convictional'.
        out = mention_presence("preconvictionalism is bad.", _brand())
        assert out.present is False

    def test_returns_offsets_for_each_match(self) -> None:
        text = "Convictional and convictional both."
        out = mention_presence(text, _brand())
        assert out.present is True
        assert len(out.offsets) == 2

    def test_returns_false_for_empty_text(self) -> None:
        out = mention_presence("", _brand())
        assert out.present is False
        assert out.offsets == []

    def test_multi_word_alias_match(self) -> None:
        sub = Subject(
            id="organizational_health_category",
            kind=SubjectKind.CATEGORY,
            aliases=["organizational health", "org health"],
            definition="x",
        )
        out = mention_presence(
            "Organizational health is an emerging category. Org Health overlaps culture.",
            sub,
        )
        assert out.present is True
        assert len(out.offsets) == 2

    def test_result_is_pure_dataclass(self) -> None:
        out = mention_presence("Convictional", _brand())
        assert isinstance(out, MentionResult)


class TestMentionCount:
    def test_distinct_alias_matches_counted(self) -> None:
        text = "Convictional and convictional and CONVICTIONAL."
        assert mention_count(text, _brand()) == 3

    def test_zero_when_absent(self) -> None:
        assert mention_count("nothing here", _brand()) == 0

    def test_overlapping_aliases_dedup_by_offset(self) -> None:
        sub = Subject(
            id="organizational_health_category",
            kind=SubjectKind.CATEGORY,
            aliases=["organizational health", "health"],
            definition="x",
        )
        # 'organizational health' and 'health' both match — 'health' is contained.
        # Count should be 1 (the longer alias wins; we don't double-count overlapping spans).
        assert mention_count("organizational health is good.", sub) == 1


class TestOrdinalRank:
    def test_markdown_numbered_list(self) -> None:
        text = """Top platforms:

1. Lattice
2. Convictional
3. Culture Amp
"""
        out = ordinal_rank(text, _brand())
        assert out.rank == 2

    def test_bullet_list_returns_position(self) -> None:
        text = """Some options:

- Lattice
- Workday
- Convictional
"""
        out = ordinal_rank(text, _brand())
        assert out.rank == 3

    def test_ordinal_phrases(self) -> None:
        text = "First, Lattice is mature. Second, Workday is broad. " "Third, Convictional is the newcomer."
        out = ordinal_rank(text, _brand())
        assert out.rank == 3

    def test_no_list_returns_none(self) -> None:
        text = "Convictional is a platform for organizational health."
        out = ordinal_rank(text, _brand()).rank
        assert out is None

    def test_subject_not_in_list_returns_none(self) -> None:
        text = """1. Lattice
2. Workday
3. Culture Amp
"""
        out = ordinal_rank(text, _brand())
        assert out.rank is None

    def test_returns_first_match_when_multiple_lists(self) -> None:
        text = """1. Lattice
2. Convictional

Later: 1. Workday 2. Convictional
"""
        out = ordinal_rank(text, _brand())
        assert out.rank == 2  # picks earliest list's rank

    def test_result_dataclass(self) -> None:
        out = ordinal_rank("1. Convictional", _brand())
        assert isinstance(out, OrdinalResult)
        assert out.rank == 1


class TestShareOfVoice:
    def _subjects_with_competitors(self) -> tuple[Subject, dict[str, Subject]]:
        brand = Subject(
            id="convictional_brand",
            kind=SubjectKind.BRAND,
            aliases=["convictional"],
            definition="x",
            competitors=["lattice", "culture_amp"],
        )
        lattice = Subject(
            id="lattice",
            kind=SubjectKind.BRAND,
            aliases=["lattice"],
            definition="x",
        )
        ca = Subject(
            id="culture_amp",
            kind=SubjectKind.BRAND,
            aliases=["culture amp", "cultureamp"],
            definition="x",
        )
        return brand, {brand.id: brand, lattice.id: lattice, ca.id: ca}

    def test_simple_share(self) -> None:
        brand, sbi = self._subjects_with_competitors()
        out = share_of_voice("Convictional and Lattice are tools.", brand, sbi)
        assert out.value == 0.5

    def test_subject_alone(self) -> None:
        brand, sbi = self._subjects_with_competitors()
        out = share_of_voice("Convictional is great.", brand, sbi)
        assert out.value == 1.0

    def test_zero_denominator_returns_none(self) -> None:
        brand, sbi = self._subjects_with_competitors()
        out = share_of_voice("nothing on topic here", brand, sbi)
        assert out.value is None

    def test_subject_with_no_competitors_returns_one_when_present(self) -> None:
        sub = Subject(
            id="organizational_health_category",
            kind=SubjectKind.CATEGORY,
            aliases=["organizational health"],
            definition="x",
        )
        out = share_of_voice("organizational health is critical.", sub, {sub.id: sub})
        assert out.value == 1.0

    def test_unknown_competitor_id_is_ignored(self) -> None:
        brand = Subject(
            id="x",
            kind=SubjectKind.BRAND,
            aliases=["xtool"],
            definition="x",
            competitors=["does_not_exist", "lattice"],
        )
        lattice = Subject(id="lattice", kind=SubjectKind.BRAND, aliases=["lattice"], definition="x")
        out = share_of_voice("xtool and Lattice", brand, {brand.id: brand, lattice.id: lattice})
        # 1 of (1 + 1) since unknown competitor contributes 0.
        assert out.value == 0.5

    def test_result_dataclass(self) -> None:
        brand, sbi = self._subjects_with_competitors()
        out = share_of_voice("Convictional", brand, sbi)
        assert isinstance(out, ShareOfVoiceResult)


class TestBrandLegacyConflation:
    def _pair(self) -> tuple[Subject, Subject]:
        brand = Subject(
            id="convictional_brand",
            kind=SubjectKind.BRAND,
            aliases=["Convictional"],
            definition="x",
        )
        legacy = Subject(
            id="convictional_legacy_dropship",
            kind=SubjectKind.ANTI_BRAND,
            aliases=["dropship", "drop shipping platform"],
            definition="x",
        )
        return brand, legacy

    def test_co_occurrence_fires(self) -> None:
        brand, legacy = self._pair()
        out = brand_legacy_conflation("Convictional was a dropship platform that pivoted.", brand, legacy)
        assert out.fired is True
        assert len(out.brand_offsets) == 1
        assert len(out.legacy_offsets) == 1

    def test_brand_alone_does_not_fire(self) -> None:
        brand, legacy = self._pair()
        assert brand_legacy_conflation("Convictional helps leaders.", brand, legacy).fired is False

    def test_legacy_alone_does_not_fire(self) -> None:
        brand, legacy = self._pair()
        assert brand_legacy_conflation("Some dropship platforms exist.", brand, legacy).fired is False

    def test_neither_does_not_fire(self) -> None:
        brand, legacy = self._pair()
        out = brand_legacy_conflation("nothing relevant", brand, legacy)
        assert out.fired is False
        assert out.brand_offsets == []
        assert out.legacy_offsets == []

    def test_legacy_must_be_anti_brand(self) -> None:
        # Defensive: pass a non-anti_brand legacy by mistake. Function should reject.
        brand, _ = self._pair()
        bad_legacy = Subject(id="x", kind=SubjectKind.BRAND, aliases=["dropship"], definition="x")
        import pytest as _pytest

        with _pytest.raises(ValueError, match="anti_brand"):
            brand_legacy_conflation("Convictional dropship", brand, bad_legacy)

    def test_result_dataclass(self) -> None:
        brand, legacy = self._pair()
        out = brand_legacy_conflation("Convictional dropship pivot.", brand, legacy)
        assert isinstance(out, BrandLegacyResult)


class TestCitations:
    def test_extracts_http_and_https(self) -> None:
        text = "See https://example.com/x and http://other.example/path."
        out = extract_citations(text, owned_domains=[])
        assert sorted(out.urls) == [
            "http://other.example/path",
            "https://example.com/x",
        ]
        assert out.owned_urls == []

    def test_owned_domain_tagged(self) -> None:
        text = "Read https://convictional.com/blog and https://example.com."
        out = extract_citations(text, owned_domains=["convictional.com"])
        assert out.owned_urls == ["https://convictional.com/blog"]

    def test_subdomain_of_owned_domain_tagged(self) -> None:
        text = "Read https://blog.convictional.com/post"
        out = extract_citations(text, owned_domains=["convictional.com"])
        assert out.owned_urls == ["https://blog.convictional.com/post"]

    def test_owned_domain_partial_match_does_not_tag(self) -> None:
        # 'convictional.com' must not match 'notconvictional.com'.
        text = "https://notconvictional.com/x"
        out = extract_citations(text, owned_domains=["convictional.com"])
        assert out.owned_urls == []

    def test_no_urls_returns_empty(self) -> None:
        out = extract_citations("no urls here", owned_domains=["convictional.com"])
        assert out.urls == []
        assert out.owned_urls == []

    def test_dedup_repeated_urls(self) -> None:
        text = "See https://x.com/a and https://x.com/a again."
        out = extract_citations(text, owned_domains=[])
        assert out.urls == ["https://x.com/a"]

    def test_result_dataclass(self) -> None:
        out = extract_citations("https://example.com", owned_domains=[])
        assert isinstance(out, CitationResult)
        assert out.urls == ["https://example.com"]
        assert out.owned_urls == []
