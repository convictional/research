from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from geo_analyzer.runtime import (
    Run,
    RunStatus,
    Score,
    Task,
    TaskStatus,
)


class TestRun:
    def test_minimal(self) -> None:
        r = Run(
            id="2026-04-29-manual",
            trigger="manual",
            started_at=datetime(2026, 4, 29, 9, 0, tzinfo=UTC),
        )
        assert r.id == "2026-04-29-manual"
        assert r.trigger == "manual"
        assert r.status == RunStatus.IN_PROGRESS  # default

    def test_id_must_match_format(self) -> None:
        with pytest.raises(ValidationError):
            Run(
                id="not-a-date",
                trigger="manual",
                started_at=datetime.now(UTC),
            )

    def test_unknown_trigger_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Run(
                id="2026-04-29-manual",
                trigger="bogus",  # type: ignore[arg-type]
                started_at=datetime.now(UTC),
            )


class TestTask:
    def test_minimal(self) -> None:
        t = Task(
            run_id="2026-04-29-manual",
            prompt_id="prompt.broad.l1.companies-in-age-of-ai",
            model_id="openai:gpt-5.1:ungrounded",
            sample_n=0,
            status=TaskStatus.SUCCESS,
            text="Convictional helps...",
            tokens_in=100,
            tokens_out=50,
            cost_usd_estimate=0.003,
            latency_ms=500,
        )
        assert t.sample_n == 0
        assert t.status == TaskStatus.SUCCESS

    def test_failure_task_can_have_error(self) -> None:
        t = Task(
            run_id="2026-04-29-manual",
            prompt_id="prompt.broad.l1.companies-in-age-of-ai",
            model_id="openai:gpt-5.1:ungrounded",
            sample_n=0,
            status=TaskStatus.FAILED,
            error="rate limit",
            text="",
            tokens_in=0,
            tokens_out=0,
            cost_usd_estimate=0.0,
            latency_ms=0,
        )
        assert t.error == "rate limit"

    def test_negative_sample_n_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Task(
                run_id="2026-04-29-manual",
                prompt_id="x",
                model_id="x:x:grounded",
                sample_n=-1,
                status=TaskStatus.SUCCESS,
                text="",
                tokens_in=0,
                tokens_out=0,
                cost_usd_estimate=0.0,
                latency_ms=0,
            )

    def test_task_key_is_stable(self) -> None:
        t1 = Task(
            run_id="r",
            prompt_id="p",
            model_id="m:n:grounded",
            sample_n=1,
            status=TaskStatus.SUCCESS,
            text="",
            tokens_in=0,
            tokens_out=0,
            cost_usd_estimate=0.0,
            latency_ms=0,
        )
        t2 = Task(
            run_id="r",
            prompt_id="p",
            model_id="m:n:grounded",
            sample_n=1,
            status=TaskStatus.SUCCESS,
            text="different",
            tokens_in=1,
            tokens_out=1,
            cost_usd_estimate=0.0,
            latency_ms=0,
        )
        assert t1.key() == t2.key()
        assert t1.key() == ("r", "p", "m:n:grounded", 1)


class TestScore:
    def test_minimal(self) -> None:
        s = Score(
            run_id="2026-04-29-manual",
            prompt_id="p",
            model_id="m:n:grounded",
            subject_id="convictional_brand",
            metric="mention_presence",
            value=True,
            scoring_method="deterministic",
            sample_aggregation="majority_vote",
        )
        assert s.value is True
        assert s.scoring_method == "deterministic"

    def test_value_can_be_int_float_bool_or_none(self) -> None:
        # ordinal_rank can be int|None; SoV can be float|None; presence bool.
        for v in (None, 0, 1, 0.5, True, False):
            Score(
                run_id="r",
                prompt_id="p",
                model_id="m:n:grounded",
                subject_id="s",
                metric="x",
                value=v,
                scoring_method="deterministic",
                sample_aggregation="single",
            )
