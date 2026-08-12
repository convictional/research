"""geo-analyzer CLI.

Phase 1 wired `catalog validate`. Phase 2 adds `probe`. Phase 3 adds `run`.
Phase 4 adds `report` and `status`.
"""

from __future__ import annotations

import asyncio
import os
import webbrowser
from datetime import date
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from geo_analyzer.catalog import CatalogError, load_catalog
from geo_analyzer.goals import load_goals
from geo_analyzer.providers import (
    ProbeRequest,
    ProviderError,
    ProviderResponse,
    get_provider,
)
from geo_analyzer.providers.pricing import estimate_cost
from geo_analyzer.reports import (
    SummaryInputs,
    latest_run_id,
    list_run_ids,
    read_scores_jsonl,
    render_html,
    render_summary,
)
from geo_analyzer.runner import RunSummary, expand_matrix, filter_catalog
from geo_analyzer.runner.orchestrator import run as orchestrator_run
from geo_analyzer.runtime import RunTrigger
from geo_analyzer.storage import read_manifest, read_tasks_jsonl, run_paths_for

app = typer.Typer(
    name="geo-analyzer",
    help="GEO Analyzer — measure how Convictional shows up in generative AI answers.",
    no_args_is_help=True,
    add_completion=False,
)

catalog_app = typer.Typer(help="Catalog inspection and validation.")
app.add_typer(catalog_app, name="catalog")

_console = Console()
_err_console = Console(stderr=True)


def _load_env() -> None:
    """Load env vars from .env.secrets (gitignored — holds the API keys)."""
    load_dotenv(".env.secrets")


# --- catalog --------------------------------------------------------------


@catalog_app.command("validate")
def catalog_validate(
    catalog_dir: Annotated[
        Path,
        typer.Option("--catalog-dir", help="Path to the catalog/ directory containing subjects/models/prompts YAML."),
    ] = Path("catalog"),
) -> None:
    """Validate the catalog (subjects, models, prompts) and report a one-line summary."""
    try:
        cat = load_catalog(catalog_dir)
    except CatalogError as e:
        _err_console.print(f"[red]catalog error:[/red] {e}")
        raise typer.Exit(code=1) from e
    except ValidationError as e:
        _err_console.print(f"[red]schema error:[/red]\n{e}")
        raise typer.Exit(code=1) from e

    _console.print(
        f"[green]ok[/green] subjects={len(cat.subjects)} "
        f"prompts={len(cat.prompts)} models={len(cat.models)} "
        f"providers={len(cat.providers)}"
    )


# --- probe ----------------------------------------------------------------

_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

_DEFAULT_COST_WARNING_USD = 5.0
_DRY_RUN_AVG_TOKENS_IN = 100
_DRY_RUN_AVG_TOKENS_OUT = 200


@app.command("probe")
def probe(
    prompt: Annotated[str, typer.Argument(help="The prompt to send.")],
    model_id: Annotated[
        str,
        typer.Option("--model", "-m", help="Model id from models.yaml (e.g., openai:gpt-5.1:grounded)."),
    ],
    catalog_dir: Annotated[
        Path,
        typer.Option("--catalog-dir", help="Path to the catalog/ directory."),
    ] = Path("catalog"),
    sensitivity_samples: Annotated[
        int,
        typer.Option(
            "--sensitivity-samples",
            help="Run N samples and print the distribution. Requires --temperature when N>1.",
            min=1,
        ),
    ] = 1,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--temperature",
            help="Override sampling temperature for this probe (required for sensitivity mode).",
        ),
    ] = None,
) -> None:
    """Run a single prompt (or N samples) against one model and print the response(s).

    Reads API keys from `.env` (via python-dotenv) or process environment.
    With --sensitivity-samples N > 1, runs N samples at --temperature and
    prints all responses plus a one-line summary. Results are not persisted.
    """
    _load_env()

    if sensitivity_samples > 1 and temperature is None:
        _err_console.print(
            "[red]error:[/red] --sensitivity-samples > 1 requires --temperature "
            "(otherwise every sample would be identical)."
        )
        raise typer.Exit(code=1)

    try:
        cat = load_catalog(catalog_dir)
    except (CatalogError, ValidationError) as e:
        _err_console.print(f"[red]catalog error:[/red] {e}")
        raise typer.Exit(code=1) from e

    try:
        model = next(m for m in cat.models if m.id == model_id)
    except StopIteration as e:
        known = sorted(m.id for m in cat.models)
        _err_console.print(f"[red]error:[/red] model {model_id!r} not found.")
        _err_console.print(f"  known models: {known}")
        raise typer.Exit(code=1) from e

    env_var = _API_KEY_ENV.get(model.provider)
    if env_var is None:
        _err_console.print(f"[red]error:[/red] no API key env var configured for provider {model.provider!r}.")
        raise typer.Exit(code=1)
    api_key = os.environ.get(env_var)
    if not api_key:
        _err_console.print(f"[red]error:[/red] {env_var} is not set. Add it to .env or your shell.")
        raise typer.Exit(code=1)

    try:
        provider = get_provider(model.provider, api_key=api_key)
    except ProviderError as e:
        _err_console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    async def _run_all() -> list[ProviderResponse]:
        responses: list[ProviderResponse] = []
        for _ in range(sensitivity_samples):
            req = ProbeRequest(
                model=model,
                prompt=prompt,
                temperature_override=temperature,
            )
            try:
                resp = await provider.call(req)
            except ProviderError as e:
                _err_console.print(f"[red]provider error:[/red] {e}")
                raise typer.Exit(code=1) from e
            responses.append(resp)
        return responses

    responses = asyncio.run(_run_all())

    if len(responses) == 1:
        _print_probe_result(model_id, responses[0])
    else:
        _print_sensitivity_summary(model_id, responses, temperature)


def _print_probe_result(model_id: str, response: ProviderResponse) -> None:
    _console.print(f"[bold]model:[/bold] {model_id}")
    _console.print(f"[bold]tokens:[/bold] in={response.tokens_in} out={response.tokens_out}")
    _console.print(f"[bold]cost:[/bold]   ${response.cost_usd_estimate:.6f}")
    _console.print(f"[bold]latency:[/bold] {response.latency_ms} ms")
    _console.rule("response")
    _console.print(response.text)


def _print_sensitivity_summary(
    model_id: str,
    responses: list[ProviderResponse],
    temperature: float | None,
) -> None:
    n = len(responses)
    total_in = sum(r.tokens_in for r in responses)
    total_out = sum(r.tokens_out for r in responses)
    total_cost = sum(r.cost_usd_estimate for r in responses)
    mean_latency = sum(r.latency_ms for r in responses) / n

    _console.print(f"[bold]model:[/bold] {model_id}")
    _console.print(f"[bold]samples:[/bold] {n} @ temperature={temperature}")
    _console.print(f"[bold]tokens (sum):[/bold] in={total_in} out={total_out}")
    _console.print(f"[bold]cost (sum):[/bold] ${total_cost:.6f}")
    _console.print(f"[bold]latency (mean):[/bold] {mean_latency:.0f} ms")
    for i, resp in enumerate(responses):
        _console.rule(f"sample {i + 1}/{n}")
        _console.print(resp.text)


@app.command("run")
def run_command(
    catalog_dir: Annotated[Path, typer.Option("--catalog-dir")] = Path("catalog"),
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    trigger: Annotated[
        str,
        typer.Option("--trigger", help="manual | launchd-weekly | launchd-monthly | ci"),
    ] = "manual",
    tiers: Annotated[
        list[str] | None,
        typer.Option("--tier", help="Filter to specific tiers (repeatable). Default: all."),
    ] = None,
    subjects: Annotated[
        list[str] | None,
        typer.Option("--subject", help="Filter prompts to those targeting these subject ids (repeatable)."),
    ] = None,
    model_ids: Annotated[
        list[str] | None,
        typer.Option("--model", help="Filter to specific model ids (repeatable). Default: all active."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print matrix size + estimated cost; do not call providers."),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume any partial run on the same date."),
    ] = True,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip cost confirmation prompt."),
    ] = False,
) -> None:
    """Run the full catalog x matrix and persist artifacts under data/runs/<run-id>/."""
    _load_env()

    try:
        full_catalog = load_catalog(catalog_dir)
    except (CatalogError, ValidationError) as e:
        _err_console.print(f"[red]catalog error:[/red] {e}")
        raise typer.Exit(code=1) from e

    cat = filter_catalog(
        full_catalog,
        tiers=tiers,  # type: ignore[arg-type]
        subjects=subjects,
        model_ids=model_ids,
    )
    if not cat.prompts or not cat.models:
        _err_console.print("[red]error:[/red] filter produced empty matrix (no prompts or no models).")
        raise typer.Exit(code=1)

    pending = expand_matrix(cat, run_id="dry-run")  # run_id only used in PendingTask
    estimated_cost = _estimate_run_cost(cat, n_tasks=len(pending))

    _console.print(
        f"[bold]matrix:[/bold] prompts={len(cat.prompts)} models={len(cat.models)} "
        f"tasks={len(pending)} est_cost=${estimated_cost:.2f}"
    )

    if dry_run:
        return

    if estimated_cost > _DEFAULT_COST_WARNING_USD and not yes:
        confirm = typer.confirm(
            f"Estimated cost ${estimated_cost:.2f} exceeds threshold " f"${_DEFAULT_COST_WARNING_USD:.2f}. Continue?",
            default=False,
        )
        if not confirm:
            _err_console.print("[yellow]aborted.[/yellow]")
            raise typer.Exit(code=1)

    # Only require API keys for providers that have models in the filtered catalog.
    used_providers = {m.provider for m in cat.models}
    api_keys: dict[str, str] = {}
    for provider_name in used_providers:
        env_var = _API_KEY_ENV.get(provider_name)
        if env_var is None:
            _err_console.print(f"[red]error:[/red] no API key env var configured for {provider_name!r}.")
            raise typer.Exit(code=1)
        key = os.environ.get(env_var)
        if not key:
            _err_console.print(f"[red]error:[/red] {env_var} is not set.")
            raise typer.Exit(code=1)
        api_keys[provider_name] = key

    with Progress(
        TextColumn("[bold]running[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=_console,
    ) as progress:
        bar_id: TaskID | None = None

        def _on_start(n_pending: int) -> None:
            nonlocal bar_id
            if n_pending == 0:
                _console.print("[yellow]nothing to do — all tasks already complete.[/yellow]")
                return
            bar_id = progress.add_task("", total=n_pending)

        def _on_complete() -> None:
            if bar_id is not None:
                progress.update(bar_id, advance=1)

        summary: RunSummary = asyncio.run(
            orchestrator_run(
                catalog=cat,
                data_dir=data_dir,
                run_date=date.today(),
                trigger=cast(RunTrigger, trigger),
                api_keys=api_keys,
                resume=resume,
                on_run_start=_on_start,
                on_task_complete=_on_complete,
            )
        )

    _console.print(
        f"[green]done[/green] run_id={summary.run.id} " f"success={summary.n_success} failed={summary.n_failed}"
    )
    if summary.n_failed > 0:
        raise typer.Exit(code=2)


@app.command("report")
def report_command(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Run id (e.g. 2026-04-29-manual). Defaults to latest run."),
    ] = None,
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    catalog_dir: Annotated[Path, typer.Option("--catalog-dir")] = Path("catalog"),
    open_latest: Annotated[
        bool,
        typer.Option("--open-latest", help="Open the rendered summary.md in a browser."),
    ] = False,
    since: Annotated[
        str | None,
        typer.Option("--since", help="Render a trend report across all runs since this YYYY-MM-DD."),
    ] = None,
) -> None:
    """Render summary.md for a run (default: latest)."""
    if since is not None:
        from datetime import date as _date

        from geo_analyzer.reports import compute_multi_run_trends

        try:
            since_date = _date.fromisoformat(since)
        except ValueError as e:
            _err_console.print(f"[red]error:[/red] --since must be YYYY-MM-DD; got {since!r}")
            raise typer.Exit(code=1) from e
        rows = compute_multi_run_trends(data_dir, since=since_date)
        if not rows:
            _err_console.print(f"[yellow]no runs found since {since_date}.[/yellow]")
            return
        _console.print(f"[bold]trend since {since_date}[/bold] — {len(rows)} rows")
        _console.print(
            "run_id              subject              metric              n  prompt-rate  interaction-rate  mean"
        )
        for r in rows:
            _console.print(
                f"{r.run_id:<20} {r.subject_id:<20} {r.metric:<20} {r.n:>2}  "
                f"{_fmt_or_dash(r.prompt_level_rate):>11}  "
                f"{_fmt_or_dash(r.interaction_level_rate):>16}  "
                f"{_fmt_or_dash(r.mean_value):>5}"
            )
        return

    try:
        cat = load_catalog(catalog_dir)
    except (CatalogError, ValidationError) as e:
        _err_console.print(f"[red]catalog error:[/red] {e}")
        raise typer.Exit(code=1) from e

    if run_id is None:
        resolved = latest_run_id(data_dir)
        if resolved is None:
            _err_console.print(f"[red]error:[/red] no runs found under {data_dir / 'runs'}.")
            raise typer.Exit(code=1)
        run_id = resolved

    if run_id not in list_run_ids(data_dir):
        _err_console.print(f"[red]error:[/red] run {run_id!r} not found under {data_dir / 'runs'}.")
        raise typer.Exit(code=1)

    rp = run_paths_for(data_dir, run_id)
    manifest = read_manifest(rp.manifest)
    tasks = read_tasks_jsonl(rp.tasks_jsonl)
    scores = read_scores_jsonl(rp.scores_jsonl)
    goals = load_goals(catalog_dir / "goals.yaml")

    summary_md = render_summary(
        SummaryInputs(
            run=manifest.run,
            tasks=tasks,
            scores=scores,
            catalog=cat,
            goals=goals,
            today=date.today(),
        )
    )
    rp.run_dir.joinpath("summary.md").write_text(summary_md, encoding="utf-8")
    summary_html = render_html(summary_md, title=f"Run {run_id}")
    rp.run_dir.joinpath("summary.html").write_text(summary_html, encoding="utf-8")
    _console.print(f"[green]wrote[/green] {rp.run_dir / 'summary.md'} + summary.html")

    if open_latest:
        # Open the HTML version — browsers don't render raw markdown.
        # Path.as_uri() requires an absolute path and handles URL encoding correctly.
        webbrowser.open(rp.run_dir.joinpath("summary.html").resolve().as_uri())


@app.command("status")
def status_command(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    catalog_dir: Annotated[Path, typer.Option("--catalog-dir")] = Path("catalog"),
) -> None:
    """Print goal traffic-lights against the latest run.

    Exit codes:
      0 — all goals green/yellow/pending
      1 — operational error (no runs, catalog invalid)
      3 — at least one goal is RED
    """
    from datetime import date as _date

    from geo_analyzer.reports import GoalStatus, evaluate_goal

    try:
        cat = load_catalog(catalog_dir)
    except (CatalogError, ValidationError) as e:
        _err_console.print(f"[red]catalog error:[/red] {e}")
        raise typer.Exit(code=1) from e

    goals = load_goals(catalog_dir / "goals.yaml")
    if not goals:
        _console.print("no goals defined.")
        return

    run_id = latest_run_id(data_dir)
    if run_id is None:
        _err_console.print(f"[red]error:[/red] no runs found under {data_dir / 'runs'}.")
        raise typer.Exit(code=1)

    rp = run_paths_for(data_dir, run_id)
    scores = read_scores_jsonl(rp.scores_jsonl)
    today = _date.today()

    any_red = False
    light_names = {
        GoalStatus.GREEN: "[GREEN]",
        GoalStatus.YELLOW: "[YELLOW]",
        GoalStatus.RED: "[RED]",
        GoalStatus.PENDING: "[PENDING]",
    }
    for goal in goals:
        ev = evaluate_goal(goal, scores=scores, catalog=cat, today=today)
        light = light_names[ev.status]
        actual = "-" if ev.actual is None else f"{ev.actual:.3f}"
        expected = "-" if ev.expected is None else f"{ev.expected:.3f}"
        _console.print(
            f"{light}  {goal.id}  ({goal.subject}/{goal.metric}/{goal.tier})  "
            f"actual={actual} expected={expected} target={goal.target}"
        )
        if ev.status == GoalStatus.RED:
            any_red = True

    if any_red:
        raise typer.Exit(code=3)


def _fmt_or_dash(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def _estimate_run_cost(catalog: Any, *, n_tasks: int) -> float:
    """Crude estimate using DEFAULT_AVG_TOKENS_{IN,OUT} and the pricing table."""
    if not catalog.models:
        return 0.0
    per_task = sum(
        estimate_cost(m.model_name, tokens_in=_DRY_RUN_AVG_TOKENS_IN, tokens_out=_DRY_RUN_AVG_TOKENS_OUT)
        for m in catalog.models
    ) / len(catalog.models)
    return per_task * n_tasks


if __name__ == "__main__":
    app()
