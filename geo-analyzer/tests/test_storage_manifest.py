from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from geo_analyzer.runtime import Run, RunStatus
from geo_analyzer.storage.manifest import (
    Manifest,
    read_manifest,
    write_manifest,
)


def _run() -> Run:
    return Run(
        id="2026-04-29-manual",
        trigger="manual",
        started_at=datetime(2026, 4, 29, 9, 0, tzinfo=UTC),
    )


class TestManifest:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        m = Manifest(
            run=_run(),
            subject_ids=["convictional_brand", "convictional_legacy_dropship"],
            prompt_ids=["prompt.broad.l1.companies-in-age-of-ai"],
            model_ids=["openai:gpt-5.1:ungrounded"],
            catalog_hash="abc123",
        )
        write_manifest(path, m)
        loaded = read_manifest(path)
        assert loaded.run.id == m.run.id
        assert loaded.subject_ids == m.subject_ids
        assert loaded.catalog_hash == "abc123"

    def test_status_round_trips(self, tmp_path: Path) -> None:
        m = Manifest(
            run=_run().model_copy(update={"status": RunStatus.COMPLETED}),
            subject_ids=[],
            prompt_ids=[],
            model_ids=[],
            catalog_hash="x",
        )
        path = tmp_path / "manifest.json"
        write_manifest(path, m)
        assert read_manifest(path).run.status == RunStatus.COMPLETED
