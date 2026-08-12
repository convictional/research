"""Read catalog YAML from disk, validate cross-references, return a Catalog.

Layout it expects (relative to the catalog root passed in):

    catalog/
      subjects.yaml          # list of Subject dicts
      models.yaml            # {providers: {<id>: ProviderConfig}, models: [ModelSpec, ...]}
      prompts/
        l1_broad.yaml        # list of Prompt dicts; tier must be 'L1'
        l2_adjacent.yaml     # tier must be 'L2'
        l3_category.yaml     # tier must be 'L3'
        l4_brand.yaml        # tier must be 'L4'

Validation runs in two phases:
  1. Pydantic parse — every entity individually well-formed.
  2. Cross-ref checks — duplicate ids, unknown subjects, undeclared providers,
     ungrounded N/temperature semantics, prompt tier vs. filename consistency.

A single CatalogError is raised on the first cross-ref violation; Pydantic
errors propagate as ValidationError (the CLI wraps both — see Task 7).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, cast

import yaml

from geo_analyzer.types import (
    Catalog,
    ModelSpec,
    Prompt,
    PromptTier,
    ProviderConfig,
    Subject,
)


class CatalogError(ValueError):
    """Raised when catalog YAML parses cleanly but fails cross-reference checks."""


_TIER_BY_FILENAME: dict[str, PromptTier] = {
    "l1_broad.yaml": "L1",
    "l2_adjacent.yaml": "L2",
    "l3_category.yaml": "L3",
    "l4_brand.yaml": "L4",
}


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise CatalogError(f"missing required file: {path.name}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_catalog(catalog_dir: Path) -> Catalog:
    subjects = _load_subjects(catalog_dir / "subjects.yaml")
    providers, models = _load_models(catalog_dir / "models.yaml")
    prompts = _load_prompts(catalog_dir / "prompts", subjects)

    # Cross-ref: prompt targets must reference known subjects.
    subject_ids = {s.id for s in subjects}
    for p in prompts:
        for t in p.targets:
            if t not in subject_ids:
                raise CatalogError(f"prompt {p.id!r} targets unknown subject {t!r}")

    # Cross-ref: model.provider must be declared in providers map.
    for m in models:
        if m.provider not in providers:
            raise CatalogError(f"model {m.id!r} references undeclared provider {m.provider!r}")

    # Cross-ref: legacy_of must point at a brand subject and only be set on anti_brand.
    from geo_analyzer.types import SubjectKind  # local to avoid circular import

    for s in subjects:
        if s.legacy_of is None:
            continue
        if s.kind != SubjectKind.ANTI_BRAND:
            raise CatalogError(
                f"subject {s.id!r}: legacy_of can only be set on anti_brand subjects (got kind={s.kind.value!r})"
            )
        target = next((t for t in subjects if t.id == s.legacy_of), None)
        if target is None:
            raise CatalogError(f"subject {s.id!r}: legacy_of references unknown subject {s.legacy_of!r}")
        if target.kind != SubjectKind.BRAND:
            raise CatalogError(
                f"subject {s.id!r}: legacy_of must point at a brand subject (got kind={target.kind.value!r})"
            )

    return Catalog(
        subjects=subjects,
        prompts=prompts,
        providers=providers,
        models=models,
    )


def _load_subjects(path: Path) -> list[Subject]:
    raw = _read_yaml(path)
    if not isinstance(raw, list):
        raise CatalogError("subjects.yaml must be a top-level list")
    items = cast(list[Any], raw)
    subjects = [Subject.model_validate(item) for item in items]
    _check_unique([s.id for s in subjects], kind="subject id")
    return subjects


def _load_models(path: Path) -> tuple[dict[str, ProviderConfig], list[ModelSpec]]:
    raw = _read_yaml(path)
    if not isinstance(raw, dict) or "providers" not in raw or "models" not in raw:
        raise CatalogError("models.yaml must have top-level 'providers' and 'models' keys")
    raw_dict = cast(dict[str, Any], raw)
    providers: dict[str, ProviderConfig] = {
        name: ProviderConfig.model_validate(cfg) for name, cfg in cast(dict[str, Any], raw_dict["providers"]).items()
    }
    models = [ModelSpec.model_validate(item) for item in cast(list[Any], raw_dict["models"])]
    _check_unique([m.id for m in models], kind="model id")
    for m in models:
        _validate_sampling_for_mode(m)
    return providers, models


def _load_prompts(prompts_dir: Path, subjects: list[Subject]) -> list[Prompt]:
    if not prompts_dir.is_dir():
        raise CatalogError(f"missing prompts directory: {prompts_dir}")
    all_prompts: list[Prompt] = []
    for filename, expected_tier in _TIER_BY_FILENAME.items():
        path = prompts_dir / filename
        if not path.exists():
            raise CatalogError(f"missing required prompts file: {filename}")
        raw = _read_yaml(path)
        if raw is None:
            # Empty file — allow.
            continue
        if not isinstance(raw, list):
            raise CatalogError(f"{filename}: must be a top-level list")
        for item in cast(list[Any], raw):
            prompt = Prompt.model_validate(item)
            if prompt.tier != expected_tier:
                raise CatalogError(
                    f"tier mismatch in {filename}: prompt {prompt.id!r} declares "
                    f"tier {prompt.tier!r} but filename implies {expected_tier!r}"
                )
            all_prompts.append(prompt)
    _check_unique([p.id for p in all_prompts], kind="prompt id")
    return all_prompts


def _validate_sampling_for_mode(m: ModelSpec) -> None:
    """Encode DESIGN §5.3 semantics: ungrounded → N=1 at temp=0; grounded → N≥1."""
    if m.mode == "ungrounded" and (m.sampling.n != 1 or m.sampling.temperature != 0):
        raise CatalogError(
            f"model {m.id!r}: ungrounded mode requires n=1 and temperature=0; "
            f"got n={m.sampling.n}, temperature={m.sampling.temperature!r}"
        )
    # grounded: any n≥1, any temperature (typically null = provider default).


def _check_unique(values: list[str], *, kind: str) -> None:
    dupes = [v for v, c in Counter(values).items() if c > 1]
    if dupes:
        raise CatalogError(f"duplicate {kind}: {sorted(dupes)!r}")
