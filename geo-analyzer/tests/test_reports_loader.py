from __future__ import annotations

from pathlib import Path

from geo_analyzer.reports.loader import (
    latest_run_id,
    list_run_ids,
    read_scores_jsonl,
)
from geo_analyzer.runtime import Score
from geo_analyzer.storage import append_jsonl, run_paths_for


def _score(prompt_id: str = "p", metric: str = "mention_presence") -> Score:
    return Score(
        run_id="r",
        prompt_id=prompt_id,
        model_id="openai:gpt-5.1:ungrounded",
        subject_id="convictional_brand",
        metric=metric,
        value=True,
        scoring_method="deterministic",
        sample_aggregation="single",
    )


class TestReadScoresJsonl:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.jsonl"
        append_jsonl(path, _score("p1").model_dump(mode="json"))
        append_jsonl(path, _score("p2", "share_of_voice").model_dump(mode="json"))
        loaded = read_scores_jsonl(path)
        assert len(loaded) == 2
        assert {s.prompt_id for s in loaded} == {"p1", "p2"}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_scores_jsonl(tmp_path / "nope.jsonl") == []


class TestListRunIds:
    def test_returns_sorted_ids(self, tmp_path: Path) -> None:
        for run_id in ("2026-04-29-manual", "2026-04-22-launchd-weekly", "2026-04-15-manual"):
            run_paths_for(tmp_path, run_id).ensure()
        ids = list_run_ids(tmp_path)
        # Sorted ascending so "latest" is last.
        assert ids == ["2026-04-15-manual", "2026-04-22-launchd-weekly", "2026-04-29-manual"]

    def test_empty_data_dir_returns_empty(self, tmp_path: Path) -> None:
        assert list_run_ids(tmp_path) == []

    def test_skips_non_directories(self, tmp_path: Path) -> None:
        run_paths_for(tmp_path, "2026-04-29-manual").ensure()
        # A stray file under runs/ should be ignored.
        (tmp_path / "runs" / "stray.txt").write_text("noise")
        ids = list_run_ids(tmp_path)
        assert ids == ["2026-04-29-manual"]


class TestLatestRunId:
    def test_returns_most_recent(self, tmp_path: Path) -> None:
        for run_id in ("2026-04-15-manual", "2026-04-29-manual"):
            run_paths_for(tmp_path, run_id).ensure()
        assert latest_run_id(tmp_path) == "2026-04-29-manual"

    def test_no_runs_returns_none(self, tmp_path: Path) -> None:
        assert latest_run_id(tmp_path) is None
