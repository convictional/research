from __future__ import annotations

from geo_analyzer.runner.scoring import score_run
from geo_analyzer.runtime import Task, TaskStatus
from geo_analyzer.types import Subject, SubjectKind

_RUN_ID = "r1"


def _task(text: str, prompt_id: str = "p1", model_id: str = "openai:gpt-5.1:ungrounded", sample_n: int = 0) -> Task:
    return Task(
        run_id="r",
        prompt_id=prompt_id,
        model_id=model_id,
        sample_n=sample_n,
        status=TaskStatus.SUCCESS,
        text=text,
        tokens_in=10,
        tokens_out=10,
        cost_usd_estimate=0.0,
        latency_ms=0,
    )


def _brand() -> Subject:
    return Subject(
        id="convictional_brand",
        kind=SubjectKind.BRAND,
        aliases=["Convictional"],
        definition="x",
        competitors=["lattice"],
    )


def _lattice() -> Subject:
    return Subject(
        id="lattice",
        kind=SubjectKind.BRAND,
        aliases=["Lattice"],
        definition="x",
    )


def _legacy() -> Subject:
    return Subject(
        id="convictional_legacy_dropship",
        kind=SubjectKind.ANTI_BRAND,
        aliases=["dropship"],
        definition="x",
        legacy_of="convictional_brand",
    )


class TestScoreRun:
    def test_mention_presence_emitted(self) -> None:
        tasks = [_task("Convictional is great.")]
        # prompt p1 targets convictional_brand only
        prompt_targets = {"p1": ["convictional_brand"]}
        subjects = {s.id: s for s in [_brand(), _lattice()]}
        scores = score_run(tasks, run_id=_RUN_ID, prompt_targets=prompt_targets, subjects=subjects)
        metrics = {(s.subject_id, s.metric, s.value) for s in scores}
        assert ("convictional_brand", "mention_presence", True) in metrics

    def test_ungrounded_does_not_emit_rate(self) -> None:
        tasks = [_task("Convictional helps.")]
        prompt_targets = {"p1": ["convictional_brand"]}
        subjects = {"convictional_brand": _brand(), "lattice": _lattice()}
        scores = score_run(tasks, run_id=_RUN_ID, prompt_targets=prompt_targets, subjects=subjects)
        metrics = {s.metric for s in scores}
        assert "mention_presence_rate" not in metrics

    def test_grounded_three_samples_emits_rate_and_majority(self) -> None:
        tasks = [
            _task("Convictional helps.", model_id="openai:gpt-5.1:grounded", sample_n=0),
            _task("Convictional helps.", model_id="openai:gpt-5.1:grounded", sample_n=1),
            _task("nothing relevant", model_id="openai:gpt-5.1:grounded", sample_n=2),
        ]
        prompt_targets = {"p1": ["convictional_brand"]}
        subjects = {"convictional_brand": _brand(), "lattice": _lattice()}
        scores = score_run(tasks, run_id=_RUN_ID, prompt_targets=prompt_targets, subjects=subjects)
        by_metric = {s.metric: s for s in scores if s.subject_id == "convictional_brand"}
        # 2/3 samples have presence → majority True, rate = 2/3
        assert by_metric["mention_presence"].value is True
        assert abs((by_metric["mention_presence_rate"].value or 0.0) - (2 / 3)) < 1e-9

    def test_brand_legacy_conflation_emitted_only_when_anti_brand_present(self) -> None:
        tasks = [_task("Convictional was a dropship platform.")]
        # Even though prompt targets only the brand, conflation is keyed on the
        # presence of an anti_brand subject in the catalog, not in targets.
        prompt_targets = {"p1": ["convictional_brand"]}
        subjects = {
            "convictional_brand": _brand(),
            "lattice": _lattice(),
            "convictional_legacy_dropship": _legacy(),
        }
        scores = score_run(tasks, run_id=_RUN_ID, prompt_targets=prompt_targets, subjects=subjects)
        metrics = {s.metric for s in scores}
        assert "brand_legacy_conflation" in metrics

    def test_failed_tasks_skipped(self) -> None:
        from geo_analyzer.runtime import TaskStatus as TS

        tasks = [_task("ok").model_copy(update={"status": TS.FAILED})]
        prompt_targets = {"p1": ["convictional_brand"]}
        subjects = {"convictional_brand": _brand(), "lattice": _lattice()}
        scores = score_run(tasks, run_id=_RUN_ID, prompt_targets=prompt_targets, subjects=subjects)
        assert scores == []
