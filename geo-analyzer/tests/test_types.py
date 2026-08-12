from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from geo_analyzer.types import (
    ModelSpec,
    Prompt,
    ProviderConfig,
    SamplingConfig,
    Subject,
    SubjectKind,
)


class TestSubject:
    def test_brand_subject_parses(self) -> None:
        s = Subject(
            id="convictional_brand",
            kind=SubjectKind.BRAND,
            aliases=["convictional", "Convictional"],
            definition="Convictional positioning.",
            competitors=["lattice"],
            owned_domains=["convictional.com"],
        )
        assert s.id == "convictional_brand"
        assert s.kind == SubjectKind.BRAND

    def test_anti_brand_kind_allowed(self) -> None:
        s = Subject(
            id="convictional_legacy_dropship",
            kind=SubjectKind.ANTI_BRAND,
            aliases=["dropship"],
            definition="Legacy.",
        )
        assert s.kind == SubjectKind.ANTI_BRAND

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Subject(id="x", kind="weirdkind", aliases=["x"], definition="x")  # type: ignore[arg-type]

    def test_empty_aliases_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Subject(id="x", kind=SubjectKind.BRAND, aliases=[], definition="x")

    def test_id_must_be_snake_case(self) -> None:
        with pytest.raises(ValidationError):
            Subject(id="Convictional Brand", kind=SubjectKind.BRAND, aliases=["x"], definition="x")


class TestPrompt:
    def test_valid_prompt(self) -> None:
        p = Prompt(
            id="prompt.category.l3.what-is-org-health",
            tier="L3",
            text="What is organizational health software?",
            targets=["organizational_health_category"],
            version=1,
            authored_at=date(2026, 4, 23),
        )
        assert p.tier == "L3"
        assert p.version == 1

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Prompt(
                id="x",
                tier="L9",  # type: ignore[arg-type]
                text="x",
                targets=["x"],
                version=1,
                authored_at=date(2026, 4, 23),
            )

    def test_targets_must_not_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            Prompt(
                id="x",
                tier="L1",
                text="x",
                targets=[],
                version=1,
                authored_at=date(2026, 4, 23),
            )


class TestModelSpec:
    def test_grounded_model(self) -> None:
        m = ModelSpec(
            id="openai:gpt-5.1:grounded",
            provider="openai",
            model_name="gpt-5.1",
            mode="grounded",
            active=True,
            config={"tools": [{"type": "web_search"}]},
            sampling=SamplingConfig(n=3, temperature=None, seed=42),
        )
        assert m.sampling.n == 3
        assert m.sampling.temperature is None  # null → provider default

    def test_ungrounded_must_have_temperature_zero_n_one(self) -> None:
        # DESIGN §5.3: ungrounded mode is N=1 at temp=0.
        # The type permits other configs (catalog validator enforces semantics) —
        # but n<1 is structurally invalid.
        with pytest.raises(ValidationError):
            SamplingConfig(n=0, temperature=0)

    def test_negative_n_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SamplingConfig(n=-1, temperature=None)

    def test_id_format_validated(self) -> None:
        # provider:model_name:mode shape
        with pytest.raises(ValidationError):
            ModelSpec(
                id="openai-gpt5",  # missing colons
                provider="openai",
                model_name="gpt-5.1",
                mode="grounded",
                active=True,
                config={},
                sampling=SamplingConfig(n=3, temperature=None),
            )


class TestProviderConfig:
    def test_default_retry(self) -> None:
        pc = ProviderConfig(concurrency=8)
        assert pc.retry.max_attempts == 3
        assert pc.retry.backoff_base_s == 2

    def test_concurrency_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(concurrency=0)
