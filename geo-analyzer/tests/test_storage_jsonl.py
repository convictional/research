from __future__ import annotations

from pathlib import Path

import pytest

from geo_analyzer.runtime import Task, TaskStatus
from geo_analyzer.storage.jsonl import (
    append_jsonl,
    read_jsonl_dicts,
    read_tasks_jsonl,
)


def _task(prompt_id: str, sample_n: int = 0) -> Task:
    return Task(
        run_id="2026-04-29-manual",
        prompt_id=prompt_id,
        model_id="openai:gpt-5.1:ungrounded",
        sample_n=sample_n,
        status=TaskStatus.SUCCESS,
        text="hello",
        tokens_in=10,
        tokens_out=5,
        cost_usd_estimate=0.001,
        latency_ms=100,
    )


class TestAppendJsonl:
    def test_appends_a_line(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        append_jsonl(path, _task("p1").model_dump(mode="json"))
        text = path.read_text()
        assert text.count("\n") == 1
        assert "p1" in text

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "more" / "tasks.jsonl"
        append_jsonl(path, {"a": 1})
        assert path.exists()

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"b": 2})
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2


class TestReadJsonlDicts:
    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.touch()
        assert read_jsonl_dicts(path) == []

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_jsonl_dicts(tmp_path / "nope.jsonl") == []

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"b": 2})
        assert read_jsonl_dicts(path) == [{"a": 1}, {"b": 2}]


class TestReadTasksJsonl:
    def test_parses_back_to_task_models(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        append_jsonl(path, _task("p1").model_dump(mode="json"))
        append_jsonl(path, _task("p2", sample_n=1).model_dump(mode="json"))
        tasks = read_tasks_jsonl(path)
        assert len(tasks) == 2
        assert {t.prompt_id for t in tasks} == {"p1", "p2"}

    def test_invalid_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        path.write_text("{not json\n")
        with pytest.raises(ValueError):
            read_tasks_jsonl(path)


class TestReadTasksJsonlDedupes:
    def test_keeps_last_entry_per_key(self, tmp_path: Path) -> None:
        # Same task key written twice — first as FAILED, then as SUCCESS
        # (a retry-after-failure scenario). Reader should return the SUCCESS one.
        path = tmp_path / "tasks.jsonl"
        failed = _task("p1").model_copy(update={"status": TaskStatus.FAILED, "error": "rate limit"})
        success = _task("p1")  # status=SUCCESS by default in the helper
        append_jsonl(path, failed.model_dump(mode="json"))
        append_jsonl(path, success.model_dump(mode="json"))

        tasks = read_tasks_jsonl(path)
        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.SUCCESS

    def test_distinct_keys_all_returned(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        append_jsonl(path, _task("p1").model_dump(mode="json"))
        append_jsonl(path, _task("p2").model_dump(mode="json"))
        tasks = read_tasks_jsonl(path)
        assert len(tasks) == 2
        assert {t.prompt_id for t in tasks} == {"p1", "p2"}
