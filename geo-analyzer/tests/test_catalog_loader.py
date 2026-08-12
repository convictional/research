from __future__ import annotations

from pathlib import Path

import pytest

from geo_analyzer.catalog.loader import CatalogError, load_catalog


class TestLoadCatalog:
    def test_happy_path(self, fake_catalog_dir: Path) -> None:
        cat = load_catalog(fake_catalog_dir)
        assert {s.id for s in cat.subjects} == {"convictional_brand"}
        assert {p.id for p in cat.prompts} == {"prompt.brand.l1.intro"}
        assert {m.id for m in cat.models} == {"openai:gpt-5.1:ungrounded"}
        assert "openai" in cat.providers
        assert cat.providers["openai"].concurrency == 4

    def test_missing_subjects_yaml_errors(self, tmp_path: Path) -> None:
        (tmp_path / "catalog" / "prompts").mkdir(parents=True)
        with pytest.raises(CatalogError, match="subjects.yaml"):
            load_catalog(tmp_path / "catalog")

    def test_duplicate_subject_id_errors(self, fake_catalog_dir: Path) -> None:
        (fake_catalog_dir / "subjects.yaml").write_text(
            """
- id: convictional_brand
  kind: brand
  aliases: [a]
  definition: x
- id: convictional_brand
  kind: category
  aliases: [b]
  definition: y
""".strip()
        )
        with pytest.raises(CatalogError, match="duplicate subject id"):
            load_catalog(fake_catalog_dir)

    def test_prompt_target_must_exist(self, fake_catalog_dir: Path) -> None:
        (fake_catalog_dir / "prompts" / "l1_broad.yaml").write_text(
            """
- id: prompt.brand.l1.intro
  tier: L1
  text: x
  targets: [does_not_exist]
  version: 1
  authored_at: 2026-04-23
""".strip()
        )
        with pytest.raises(CatalogError, match="unknown subject"):
            load_catalog(fake_catalog_dir)

    def test_model_provider_must_be_declared(self, fake_catalog_dir: Path) -> None:
        (fake_catalog_dir / "models.yaml").write_text(
            """
providers:
  openai: {concurrency: 4}
models:
  - id: anthropic:claude-opus-4-7:ungrounded
    provider: anthropic
    model_name: claude-opus-4-7
    mode: ungrounded
    active: true
    config: {}
    sampling: {n: 1, temperature: 0}
""".strip()
        )
        with pytest.raises(CatalogError, match="undeclared provider"):
            load_catalog(fake_catalog_dir)

    def test_ungrounded_must_have_n_eq_1_temp_eq_0(self, fake_catalog_dir: Path) -> None:
        # DESIGN §5.3 enforces this combination semantically; the loader rejects.
        (fake_catalog_dir / "models.yaml").write_text(
            """
providers:
  openai: {concurrency: 4}
models:
  - id: openai:gpt-5.1:ungrounded
    provider: openai
    model_name: gpt-5.1
    mode: ungrounded
    active: true
    config: {}
    sampling: {n: 3, temperature: null, seed: 42}
""".strip()
        )
        with pytest.raises(CatalogError, match="ungrounded.*n=1.*temperature=0"):
            load_catalog(fake_catalog_dir)

    def test_grounded_temperature_null_allowed(self, fake_catalog_dir: Path) -> None:
        (fake_catalog_dir / "models.yaml").write_text(
            """
providers:
  openai: {concurrency: 4}
models:
  - id: openai:gpt-5.1:grounded
    provider: openai
    model_name: gpt-5.1
    mode: grounded
    active: true
    config: {tools: [{type: web_search}]}
    sampling: {n: 3, temperature: null, seed: 42}
""".strip()
        )
        cat = load_catalog(fake_catalog_dir)
        assert cat.models[0].mode == "grounded"
        assert cat.models[0].sampling.temperature is None

    def test_duplicate_prompt_id_across_files_errors(self, fake_catalog_dir: Path) -> None:
        (fake_catalog_dir / "prompts" / "l2_adjacent.yaml").write_text(
            """
- id: prompt.brand.l1.intro
  tier: L2
  text: collision
  targets: [convictional_brand]
  version: 1
  authored_at: 2026-04-23
""".strip()
        )
        with pytest.raises(CatalogError, match="duplicate prompt id"):
            load_catalog(fake_catalog_dir)

    def test_prompt_tier_must_match_filename(self, fake_catalog_dir: Path) -> None:
        # Defensive: file is l1_broad.yaml so prompts inside should be tier L1.
        (fake_catalog_dir / "prompts" / "l1_broad.yaml").write_text(
            """
- id: prompt.brand.l3.bad
  tier: L3
  text: x
  targets: [convictional_brand]
  version: 1
  authored_at: 2026-04-23
""".strip()
        )
        with pytest.raises(CatalogError, match="tier mismatch"):
            load_catalog(fake_catalog_dir)

    def test_legacy_of_only_on_anti_brand(self, fake_catalog_dir: Path) -> None:
        (fake_catalog_dir / "subjects.yaml").write_text(
            """
- id: convictional_brand
  kind: brand
  aliases: [Convictional]
  definition: x
- id: bad_legacy
  kind: brand
  aliases: [b]
  definition: x
  legacy_of: convictional_brand
""".strip()
        )
        with pytest.raises(CatalogError, match="legacy_of can only be set on anti_brand"):
            load_catalog(fake_catalog_dir)

    def test_legacy_of_must_reference_known_subject(self, fake_catalog_dir: Path) -> None:
        (fake_catalog_dir / "subjects.yaml").write_text(
            """
- id: convictional_brand
  kind: brand
  aliases: [Convictional]
  definition: x
- id: legacy
  kind: anti_brand
  aliases: [dropship]
  definition: x
  legacy_of: does_not_exist
""".strip()
        )
        with pytest.raises(CatalogError, match="legacy_of references unknown subject"):
            load_catalog(fake_catalog_dir)

    def test_legacy_of_target_must_be_brand(self, fake_catalog_dir: Path) -> None:
        (fake_catalog_dir / "subjects.yaml").write_text(
            """
- id: convictional_brand
  kind: brand
  aliases: [Convictional]
  definition: x
- id: org_health
  kind: category
  aliases: [organizational health]
  definition: x
- id: legacy
  kind: anti_brand
  aliases: [dropship]
  definition: x
  legacy_of: org_health
""".strip()
        )
        with pytest.raises(CatalogError, match="legacy_of must point at a brand"):
            load_catalog(fake_catalog_dir)
