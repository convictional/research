"""The run loop. Composes matrix + retry + concurrency + scoring + storage."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from geo_analyzer.providers import ProbeRequest, ProviderError, get_provider
from geo_analyzer.runner.concurrency import ConcurrencyManager
from geo_analyzer.runner.matrix import PendingTask, expand_matrix
from geo_analyzer.runner.retry import retry_with_backoff
from geo_analyzer.runner.scoring import score_run
from geo_analyzer.runtime import (
    Run,
    RunStatus,
    RunTrigger,
    Task,
    TaskStatus,
)
from geo_analyzer.storage import (
    Manifest,
    append_jsonl,
    build_run_id,
    read_manifest,
    read_tasks_jsonl,
    run_paths_for,
    write_jsonl,
    write_manifest,
    write_scores_csv,
    write_tasks_csv,
)
from geo_analyzer.types import Catalog, ModelSpec, Prompt


@dataclass(frozen=True)
class RunSummary:
    run: Run
    n_success: int
    n_failed: int


async def run(
    *,
    catalog: Catalog,
    data_dir: Path,
    run_date: date,
    trigger: RunTrigger,
    api_keys: dict[str, str],
    resume: bool,
    on_run_start: Callable[[int], None] | None = None,
    on_task_complete: Callable[[], None] | None = None,
) -> RunSummary:
    """Execute a run end-to-end.

    Side effects: creates `data_dir/runs/<run-id>/` and writes manifest.json,
    tasks.jsonl, scores.jsonl, tasks.csv, scores.csv inside it.

    Optional progress callbacks (called from the asyncio loop):
      - `on_run_start(n_pending)` fires once after the resume filter, with the
        actual number of tasks that will be dispatched (0 if fully resumed).
      - `on_task_complete()` fires once per task finished (success or failure).
    """
    run_id = build_run_id(run_date, trigger=trigger)
    rp = run_paths_for(data_dir, run_id)
    rp.ensure()

    # Preserve the original started_at across resumes so wall-time reflects
    # "how long since the first kickoff of this run id" rather than the no-op
    # resume's instant. Only set started_at to now() on a fresh run.
    started = datetime.now(UTC)
    if rp.manifest.exists():
        try:
            existing = read_manifest(rp.manifest)
            started = existing.run.started_at
        except Exception:
            pass  # corrupt or absent — fall back to now()
    run_obj = Run(id=run_id, trigger=trigger, started_at=started, status=RunStatus.IN_PROGRESS)

    # Persist initial manifest so partial runs are interpretable.
    manifest = Manifest(
        run=run_obj,
        subject_ids=[s.id for s in catalog.subjects],
        prompt_ids=[p.id for p in catalog.prompts],
        model_ids=[m.id for m in catalog.models],
        catalog_hash=_hash_catalog(catalog),
    )
    write_manifest(rp.manifest, manifest)

    # Resume: load existing completed task keys.
    completed_keys: set[tuple[str, str, str, int]] = set()
    if resume and rp.tasks_jsonl.exists():
        for t in read_tasks_jsonl(rp.tasks_jsonl):
            if t.status == TaskStatus.SUCCESS:
                completed_keys.add(t.key())

    pending_all = expand_matrix(catalog, run_id=run_id)
    pending = [pt for pt in pending_all if (run_id, pt.prompt_id, pt.model_id, pt.sample_n) not in completed_keys]

    if on_run_start is not None:
        on_run_start(len(pending))

    # Only build adapters/semaphores for providers actually used by active
    # models in the filtered catalog — a `--model openai:...` filter shouldn't
    # require ANTHROPIC_API_KEY to be set.
    used_providers = {m.provider for m in catalog.models}
    cm = ConcurrencyManager(caps={name: catalog.providers[name].concurrency for name in used_providers})
    providers_by_name = {name: get_provider(name, api_key=api_keys[name]) for name in used_providers}

    prompts_by_id: dict[str, Prompt] = {p.id: p for p in catalog.prompts}
    models_by_id: dict[str, ModelSpec] = {m.id: m for m in catalog.models}

    async def _execute(pt: PendingTask) -> None:
        prompt = prompts_by_id[pt.prompt_id]
        model = models_by_id[pt.model_id]
        provider = providers_by_name[model.provider]
        retry_cfg = catalog.providers[model.provider].retry

        async def _do_call() -> None:
            async with cm.semaphore_for(model.provider):
                req = ProbeRequest(model=model, prompt=prompt.text)
                resp = await provider.call(req)
                _append_task(
                    rp.tasks_jsonl,
                    pt,
                    status=TaskStatus.SUCCESS,
                    text=resp.text,
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    cost_usd_estimate=resp.cost_usd_estimate,
                    latency_ms=resp.latency_ms,
                    error=None,
                )

        try:
            await retry_with_backoff(
                _do_call,
                max_attempts=retry_cfg.max_attempts,
                backoff_base_s=retry_cfg.backoff_base_s,
            )
        except ProviderError as e:
            _append_task(
                rp.tasks_jsonl,
                pt,
                status=TaskStatus.FAILED,
                text="",
                tokens_in=0,
                tokens_out=0,
                cost_usd_estimate=0.0,
                latency_ms=0,
                error=str(e),
            )
        finally:
            if on_task_complete is not None:
                on_task_complete()

    await asyncio.gather(*(_execute(pt) for pt in pending))

    # Re-read all tasks (including ones from prior partial runs) for scoring.
    all_tasks = read_tasks_jsonl(rp.tasks_jsonl)

    subjects = {s.id: s for s in catalog.subjects}
    prompt_targets = {p.id: list(p.targets) for p in catalog.prompts}
    # Scores are derived from tasks — overwrite scores.jsonl on every run so
    # resuming the same run id doesn't accumulate duplicate score lines.
    scores = score_run(all_tasks, run_id=run_id, prompt_targets=prompt_targets, subjects=subjects)
    write_jsonl(rp.scores_jsonl, (s.model_dump(mode="json") for s in scores))

    write_tasks_csv(rp.tasks_csv, all_tasks)
    write_scores_csv(rp.scores_csv, scores)

    n_success = sum(1 for t in all_tasks if t.status == TaskStatus.SUCCESS)
    n_failed = sum(1 for t in all_tasks if t.status == TaskStatus.FAILED)

    finished_run = run_obj.model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "finished_at": datetime.now(UTC),
        }
    )
    final_manifest = manifest.model_copy(update={"run": finished_run})
    write_manifest(rp.manifest, final_manifest)

    return RunSummary(run=finished_run, n_success=n_success, n_failed=n_failed)


def _append_task(
    path: Path,
    pt: PendingTask,
    *,
    status: TaskStatus,
    text: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd_estimate: float,
    latency_ms: int,
    error: str | None,
) -> None:
    t = Task(
        run_id=pt.run_id,
        prompt_id=pt.prompt_id,
        model_id=pt.model_id,
        sample_n=pt.sample_n,
        status=status,
        text=text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd_estimate=cost_usd_estimate,
        latency_ms=latency_ms,
        error=error,
    )
    append_jsonl(path, t.model_dump(mode="json"))


def _hash_catalog(catalog: Catalog) -> str:
    """SHA-256 of the catalog's deterministic JSON representation."""
    payload = catalog.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
