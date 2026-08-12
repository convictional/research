from __future__ import annotations

from pathlib import Path

import pytest

from geo_analyzer.catalog import load_catalog
from geo_analyzer.reports.funnel import (
    FunnelTier,
    compute_funnel,
    render_sparkline,
)
from geo_analyzer.runtime import Score
from geo_analyzer.types import Catalog


@pytest.fixture(scope="module")
def real_catalog() -> Catalog:
    project_root = Path(__file__).resolve().parents[1]
    return load_catalog(project_root / "catalog")


def _binary_score(prompt_id: str, value: bool, subject: str = "convictional_brand") -> Score:
    return Score(
        run_id="r",
        prompt_id=prompt_id,
        model_id="openai:gpt-5.1:ungrounded",
        subject_id=subject,
        metric="mention_presence",
        value=value,
        scoring_method="deterministic",
        sample_aggregation="single",
    )


class TestComputeFunnel:
    def test_tiers_use_catalog_lookup(self, real_catalog: Catalog) -> None:
        # Seed catalog: prompt.broad.l1.companies-in-age-of-ai is L1;
        # prompt.brand.l4.what-is-convictional is L4.
        scores = [
            _binary_score("prompt.broad.l1.companies-in-age-of-ai", False),
            _binary_score("prompt.brand.l4.what-is-convictional", True),
        ]
        funnel = compute_funnel(scores, real_catalog, subject_id="convictional_brand")
        l1 = next(t for t in funnel if t.tier == "L1")
        l4 = next(t for t in funnel if t.tier == "L4")
        assert l1.rate == 0.0
        assert l4.rate == 1.0

    def test_skips_subjects_with_no_scores(self, real_catalog: Catalog) -> None:
        funnel = compute_funnel([], real_catalog, subject_id="convictional_brand")
        # All tiers present but rate=None for empty cohorts.
        tiers = {t.tier: t.rate for t in funnel}
        assert tiers == {"L1": None, "L2": None, "L3": None, "L4": None}

    def test_unknown_prompt_id_skipped(self, real_catalog: Catalog) -> None:
        scores = [_binary_score("prompt.does.not.exist", True)]
        funnel = compute_funnel(scores, real_catalog, subject_id="convictional_brand")
        assert all(t.rate is None for t in funnel)

    def test_result_dataclass(self, real_catalog: Catalog) -> None:
        funnel = compute_funnel([], real_catalog, subject_id="convictional_brand")
        assert isinstance(funnel[0], FunnelTier)


class TestRenderSparkline:
    def test_basic_progression(self) -> None:
        bars = render_sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
        assert len(bars) == 5

    def test_none_renders_as_blank(self) -> None:
        bars = render_sparkline([None, 0.5, None])
        assert len(bars) == 3
