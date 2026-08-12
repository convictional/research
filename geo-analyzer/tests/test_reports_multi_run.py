from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from geo_analyzer.reports.multi_run import (
    MultiRunRow,
    compute_multi_run_trends,
)
from geo_analyzer.runtime import Run, RunStatus
from geo_analyzer.storage import (
    Manifest,
    append_jsonl,
    run_paths_for,
    write_manifest,
)


def _seed(tmp_path: Path, run_id: str, date_str: str, presence_value: bool) -> None:
    rp = run_paths_for(tmp_path, run_id)
    rp.ensure()
    started = datetime.fromisoformat(date_str + "T09:00:00+00:00")
    finished = started.replace(minute=5)
    run = Run(id=run_id, trigger="manual", started_at=started, finished_at=finished, status=RunStatus.COMPLETED)
    write_manifest(
        rp.manifest,
        Manifest(
            run=run,
            subject_ids=[],
            prompt_ids=[],
            model_ids=[],
            catalog_hash="x",
        ),
    )
    append_jsonl(
        rp.scores_jsonl,
        {
            "run_id": run_id,
            "prompt_id": "prompt.broad.l1.companies-in-age-of-ai",
            "model_id": "openai:gpt-5.1:ungrounded",
            "subject_id": "convictional_brand",
            "metric": "mention_presence",
            "value": presence_value,
            "scoring_method": "deterministic",
            "sample_aggregation": "single",
        },
    )


class TestComputeMultiRunTrends:
    def test_returns_one_row_per_run(self, tmp_path: Path) -> None:
        _seed(tmp_path, "2026-04-15-manual", "2026-04-15", presence_value=False)
        _seed(tmp_path, "2026-04-22-manual", "2026-04-22", presence_value=True)
        _seed(tmp_path, "2026-04-29-manual", "2026-04-29", presence_value=True)
        rows = compute_multi_run_trends(tmp_path, since=None)
        # Three runs x one (subject, metric) = three rows
        assert len(rows) == 3
        assert [r.run_id for r in rows] == [
            "2026-04-15-manual",
            "2026-04-22-manual",
            "2026-04-29-manual",
        ]

    def test_since_filters_older(self, tmp_path: Path) -> None:
        _seed(tmp_path, "2026-04-15-manual", "2026-04-15", presence_value=False)
        _seed(tmp_path, "2026-04-29-manual", "2026-04-29", presence_value=True)
        rows = compute_multi_run_trends(tmp_path, since=date(2026, 4, 20))
        assert [r.run_id for r in rows] == ["2026-04-29-manual"]

    def test_no_runs_returns_empty(self, tmp_path: Path) -> None:
        assert compute_multi_run_trends(tmp_path, since=None) == []

    def test_row_carries_topline_fields(self, tmp_path: Path) -> None:
        _seed(tmp_path, "2026-04-29-manual", "2026-04-29", presence_value=True)
        rows = compute_multi_run_trends(tmp_path, since=None)
        row = rows[0]
        assert isinstance(row, MultiRunRow)
        assert row.subject_id == "convictional_brand"
        assert row.metric == "mention_presence"
        assert row.prompt_level_rate == 1.0  # one True score → rate=1.0
