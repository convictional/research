from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from geo_analyzer.catalog import load_catalog
from geo_analyzer.providers.base import ProbeRequest, ProviderError, ProviderResponse
from geo_analyzer.runner.matrix import filter_catalog
from geo_analyzer.runner.orchestrator import RunSummary
from geo_analyzer.runner.orchestrator import run as orchestrator_run
from geo_analyzer.runtime import RunStatus, TaskStatus
from geo_analyzer.storage import read_jsonl_dicts, read_manifest, run_paths_for
from geo_analyzer.types import Catalog


class _FakeProvider:
    """Records every call. Returns a deterministic response per request."""

    def __init__(self, name: str = "openai") -> None:
        self.name = name
        self.calls: list[ProbeRequest] = []

    async def call(self, request: ProbeRequest) -> ProviderResponse:
        self.calls.append(request)
        text = f"Convictional response for {request.model.id} prompt={request.prompt!r}"
        return ProviderResponse(
            text=text,
            tokens_in=10,
            tokens_out=20,
            cost_usd_estimate=0.001,
            latency_ms=50,
            raw={},
        )


@pytest.fixture(scope="module")
def real_catalog() -> Catalog:
    project_root = Path(__file__).resolve().parents[1]
    return load_catalog(project_root / "catalog")


@pytest.mark.asyncio
async def test_orchestrator_writes_tasks_and_scores(real_catalog: Catalog, tmp_path: Path) -> None:
    cat = filter_catalog(
        real_catalog,
        tiers=["L1"],
        subjects=None,
        model_ids=["openai:gpt-5.1:ungrounded"],
    )
    fake = _FakeProvider()

    with patch("geo_analyzer.runner.orchestrator.get_provider", return_value=fake):
        summary: RunSummary = await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
        )

    assert summary.run.status == RunStatus.COMPLETED
    assert summary.n_success == 3  # 3 L1 prompts x 1 model x 1 sample
    assert summary.n_failed == 0

    rp = run_paths_for(tmp_path, "2026-04-29-manual")
    tasks_lines = read_jsonl_dicts(rp.tasks_jsonl)
    scores_lines = read_jsonl_dicts(rp.scores_jsonl)
    assert len(tasks_lines) == 3
    assert len(scores_lines) > 0
    assert rp.tasks_csv.exists()
    assert rp.scores_csv.exists()

    manifest = read_manifest(rp.manifest)
    assert manifest.run.status == RunStatus.COMPLETED
    assert len(manifest.prompt_ids) == 3


@pytest.mark.asyncio
async def test_orchestrator_resumes_partial_run(real_catalog: Catalog, tmp_path: Path) -> None:
    cat = filter_catalog(
        real_catalog,
        tiers=["L1"],
        subjects=None,
        model_ids=["openai:gpt-5.1:ungrounded"],
    )
    fake = _FakeProvider()

    with patch("geo_analyzer.runner.orchestrator.get_provider", return_value=fake):
        # First run: complete normally.
        await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
        )
        first_call_count = len(fake.calls)
        assert first_call_count == 3

        # Capture score-line count after first run for the regression assertion below.
        rp = run_paths_for(tmp_path, "2026-04-29-manual")
        first_score_count = len(read_jsonl_dicts(rp.scores_jsonl))

        # Second run with resume=True: tasks already complete, no new calls.
        await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
        )
        assert len(fake.calls) == first_call_count  # no additional calls

        # Regression: scores.jsonl should NOT accumulate duplicates across resumes.
        # Scores are derived from tasks; the same tasks should produce the same
        # number of score lines, not 2x.
        second_score_count = len(read_jsonl_dicts(rp.scores_jsonl))
        assert second_score_count == first_score_count


@pytest.mark.asyncio
async def test_orchestrator_records_failed_tasks(real_catalog: Catalog, tmp_path: Path) -> None:
    cat = filter_catalog(
        real_catalog,
        tiers=["L1"],
        subjects=None,
        model_ids=["openai:gpt-5.1:ungrounded"],
    )

    class _AlwaysFails:
        name = "openai"

        async def call(self, request: ProbeRequest) -> ProviderResponse:
            raise ProviderError("rate limit")

    with patch("geo_analyzer.runner.orchestrator.get_provider", return_value=_AlwaysFails()):
        summary = await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
        )

    assert summary.n_failed == 3
    assert summary.n_success == 0
    rp = run_paths_for(tmp_path, "2026-04-29-manual")
    tasks_lines = read_jsonl_dicts(rp.tasks_jsonl)
    assert all(t["status"] == TaskStatus.FAILED.value for t in tasks_lines)


@pytest.mark.asyncio
async def test_orchestrator_progress_callbacks(real_catalog: Catalog, tmp_path: Path) -> None:
    cat = filter_catalog(
        real_catalog,
        tiers=["L1"],
        subjects=None,
        model_ids=["openai:gpt-5.1:ungrounded"],
    )
    fake = _FakeProvider()

    starts: list[int] = []
    completes: list[None] = []

    with patch("geo_analyzer.runner.orchestrator.get_provider", return_value=fake):
        # First run: 3 tasks pending → on_run_start(3), on_task_complete x3
        await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
            on_run_start=starts.append,
            on_task_complete=lambda: completes.append(None),
        )
        assert starts == [3]
        assert len(completes) == 3

        # Second run (full resume): 0 pending → on_run_start(0), no completes
        starts.clear()
        completes.clear()
        await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
            on_run_start=starts.append,
            on_task_complete=lambda: completes.append(None),
        )
        assert starts == [0]
        assert completes == []


@pytest.mark.asyncio
async def test_orchestrator_preserves_started_at_across_resumes(real_catalog: Catalog, tmp_path: Path) -> None:
    """A no-op resume must NOT bump started_at — wall-time should reflect
    the original kickoff, not the resume's instant."""
    cat = filter_catalog(
        real_catalog,
        tiers=["L1"],
        subjects=None,
        model_ids=["openai:gpt-5.1:ungrounded"],
    )
    fake = _FakeProvider()

    with patch("geo_analyzer.runner.orchestrator.get_provider", return_value=fake):
        await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
        )
        rp = run_paths_for(tmp_path, "2026-04-29-manual")
        first_started_at = read_manifest(rp.manifest).run.started_at

        await orchestrator_run(
            catalog=cat,
            data_dir=tmp_path,
            run_date=date(2026, 4, 29),
            trigger="manual",
            api_keys={"openai": "sk-test"},
            resume=True,
        )
        second_started_at = read_manifest(rp.manifest).run.started_at

    assert first_started_at == second_started_at
