"""Catalog x matrix -> list of PendingTask.

A PendingTask is what the orchestrator dispatches. After the call resolves
it gets converted into a runtime Task (with status, tokens, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from geo_analyzer.types import Catalog, ModelSpec, Prompt, PromptTier


@dataclass(frozen=True)
class PendingTask:
    run_id: str
    prompt_id: str
    model_id: str
    sample_n: int


def filter_catalog(
    catalog: Catalog,
    *,
    tiers: list[PromptTier] | None,
    subjects: list[str] | None,
    model_ids: list[str] | None,
) -> Catalog:
    """Return a new Catalog with prompts/models narrowed by the given filters.

    - tiers=None means all tiers; otherwise keep only prompts whose tier matches.
    - subjects=None means all subjects; otherwise keep only prompts whose
      `targets` contains any of the named subject ids.
    - model_ids=None means all models; otherwise keep only those ids.
    Inactive models are always excluded regardless of `model_ids`.
    """
    prompts: list[Prompt] = list(catalog.prompts)
    if tiers is not None:
        tier_set = set(tiers)
        prompts = [p for p in prompts if p.tier in tier_set]
    if subjects is not None:
        subj_set = set(subjects)
        prompts = [p for p in prompts if any(t in subj_set for t in p.targets)]

    models: list[ModelSpec] = [m for m in catalog.models if m.active]
    if model_ids is not None:
        id_set = set(model_ids)
        models = [m for m in models if m.id in id_set]

    return Catalog(
        subjects=catalog.subjects,
        prompts=prompts,
        providers=catalog.providers,
        models=models,
    )


def expand_matrix(catalog: Catalog, *, run_id: str) -> list[PendingTask]:
    """Cartesian-product prompts x models, with N samples per (prompt, model).

    N comes from model.sampling.n — typically 1 for ungrounded, 3 for grounded.
    """
    out: list[PendingTask] = []
    for prompt in catalog.prompts:
        for model in catalog.models:
            for n in range(model.sampling.n):
                out.append(
                    PendingTask(
                        run_id=run_id,
                        prompt_id=prompt.id,
                        model_id=model.id,
                        sample_n=n,
                    )
                )
    return out
