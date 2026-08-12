from __future__ import annotations

from pathlib import Path

import pytest

from geo_analyzer.catalog import load_catalog
from geo_analyzer.runner.matrix import (
    PendingTask,
    expand_matrix,
    filter_catalog,
)
from geo_analyzer.types import Catalog


@pytest.fixture(scope="module")
def real_catalog() -> Catalog:
    """Load the actual seed catalog so tests reflect real shape."""
    project_root = Path(__file__).resolve().parents[1]
    return load_catalog(project_root / "catalog")


class TestFilterCatalog:
    def test_no_filters_returns_full_catalog(self, real_catalog: Catalog) -> None:
        result = filter_catalog(real_catalog, tiers=None, subjects=None, model_ids=None)
        # Asserts the catalog has at least the seed shape — exact counts grow
        # over time as prompts/competitors get added; checking >= keeps this
        # test from breaking on every catalog change.
        assert len(result.prompts) >= 12
        assert len(result.models) >= 10

    def test_tier_filter(self, real_catalog: Catalog) -> None:
        result = filter_catalog(real_catalog, tiers=["L1"], subjects=None, model_ids=None)
        assert len(result.prompts) == 3
        assert all(p.tier == "L1" for p in result.prompts)

    def test_subject_filter_keeps_prompts_targeting_subject(self, real_catalog: Catalog) -> None:
        result = filter_catalog(
            real_catalog,
            tiers=None,
            subjects=["convictional_legacy_dropship"],
            model_ids=None,
        )
        # Only L4 brand prompts target the legacy subject in the seed catalog.
        assert all("convictional_legacy_dropship" in p.targets for p in result.prompts)

    def test_model_filter_inactive_models_dropped(self, real_catalog: Catalog) -> None:
        result = filter_catalog(
            real_catalog,
            tiers=None,
            subjects=None,
            model_ids=["openai:gpt-5.1:grounded"],
        )
        assert len(result.models) == 1
        assert result.models[0].id == "openai:gpt-5.1:grounded"


class TestExpandMatrix:
    def test_ungrounded_emits_one_per_prompt(self, real_catalog: Catalog) -> None:
        result = filter_catalog(
            real_catalog,
            tiers=["L1"],
            subjects=None,
            model_ids=["openai:gpt-5.1:ungrounded"],
        )
        tasks = expand_matrix(result, run_id="2026-04-29-manual")
        # 3 L1 prompts x 1 model x 1 sample
        assert len(tasks) == 3
        assert all(t.sample_n == 0 for t in tasks)

    def test_grounded_emits_n_samples_per_prompt(self, real_catalog: Catalog) -> None:
        result = filter_catalog(
            real_catalog,
            tiers=["L1"],
            subjects=None,
            model_ids=["openai:gpt-5.1:grounded"],
        )
        tasks = expand_matrix(result, run_id="2026-04-29-manual")
        # 3 L1 prompts x 1 model x 3 samples
        assert len(tasks) == 9
        sample_ns = sorted({t.sample_n for t in tasks})
        assert sample_ns == [0, 1, 2]

    def test_pending_task_has_required_fields(self, real_catalog: Catalog) -> None:
        result = filter_catalog(
            real_catalog,
            tiers=["L1"],
            subjects=None,
            model_ids=["openai:gpt-5.1:ungrounded"],
        )
        tasks = expand_matrix(result, run_id="r1")
        t = tasks[0]
        assert isinstance(t, PendingTask)
        assert t.run_id == "r1"
        assert t.prompt_id.startswith("prompt.")
        assert t.model_id == "openai:gpt-5.1:ungrounded"
        assert isinstance(t.sample_n, int)

    def test_inactive_model_excluded(self, real_catalog: Catalog) -> None:
        # Build a Catalog where one model is inactive
        models = [
            m if m.id != "openai:gpt-5.1:ungrounded" else m.model_copy(update={"active": False})
            for m in real_catalog.models
        ]
        new_cat = Catalog(
            subjects=real_catalog.subjects,
            prompts=real_catalog.prompts,
            providers=real_catalog.providers,
            models=models,
        )
        filtered = filter_catalog(new_cat, tiers=None, subjects=None, model_ids=None)
        tasks = expand_matrix(filtered, run_id="r1")
        assert all(t.model_id != "openai:gpt-5.1:ungrounded" for t in tasks)
