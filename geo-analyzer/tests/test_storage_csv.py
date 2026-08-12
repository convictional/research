from __future__ import annotations

import csv
from pathlib import Path

from geo_analyzer.runtime import Score, Task, TaskStatus
from geo_analyzer.storage.csv_export import write_scores_csv, write_tasks_csv


def _task(prompt_id: str = "p1") -> Task:
    return Task(
        run_id="r1",
        prompt_id=prompt_id,
        model_id="m:n:grounded",
        sample_n=0,
        status=TaskStatus.SUCCESS,
        text="long response here",
        tokens_in=10,
        tokens_out=5,
        cost_usd_estimate=0.001,
        latency_ms=100,
    )


def _score(metric: str = "mention_presence") -> Score:
    return Score(
        run_id="r1",
        prompt_id="p1",
        model_id="m:n:grounded",
        subject_id="s",
        metric=metric,
        value=True,
        scoring_method="deterministic",
        sample_aggregation="single",
    )


class TestWriteTasksCsv:
    def test_excludes_text_column(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.csv"
        write_tasks_csv(path, [_task(), _task("p2")])
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2
        assert "text" not in rows[0]
        for col in (
            "run_id",
            "prompt_id",
            "model_id",
            "sample_n",
            "status",
            "tokens_in",
            "tokens_out",
            "cost_usd_estimate",
            "latency_ms",
        ):
            assert col in rows[0]

    def test_empty_list_writes_header_only(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.csv"
        write_tasks_csv(path, [])
        text = path.read_text()
        assert text.count("\n") == 1  # just header


class TestWriteScoresCsv:
    def test_writes_all_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.csv"
        write_scores_csv(path, [_score(), _score("ordinal_rank")])
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2
        for col in (
            "run_id",
            "prompt_id",
            "model_id",
            "subject_id",
            "metric",
            "value",
            "scoring_method",
            "sample_aggregation",
        ):
            assert col in rows[0]

    def test_value_serialization_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.csv"
        s_int = _score("ordinal_rank").model_copy(update={"value": 3})
        s_float = _score("share_of_voice").model_copy(update={"value": 0.5})
        s_none = _score("share_of_voice").model_copy(update={"value": None})
        write_scores_csv(path, [s_int, s_float, s_none])
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["value"] == "3"
        assert rows[1]["value"] == "0.5"
        assert rows[2]["value"] == ""  # None → empty cell
