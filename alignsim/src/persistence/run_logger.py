"""RunLogger: writes structured game data to PostgreSQL per run."""

import json
import logging
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from alignsim.src.models.actions import FireAction, GameAction, HireAction, SustainHireAction, TurnActions
from alignsim.src.models.entities import CustomerStage
from alignsim.src.models.game_state import GameState, TurnRecord
from alignsim.src.models.goals import GoalAttainmentScore
from alignsim.src.persistence.models import (
    CustomerSnapshotModel,
    LLMTraceModel,
    RunModel,
    TurnActionModel,
    TurnEventModel,
    TurnSnapshotModel,
)

logger = logging.getLogger(__name__)


_ALIGNSIM_DIR = Path(__file__).resolve().parent.parent.parent


def get_engine_commit() -> str | None:
    """Get the last commit hash that touched any file under alignsim/."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(_ALIGNSIM_DIR)],
            capture_output=True, text=True, timeout=5,
            cwd=_ALIGNSIM_DIR,
        )
        commit = result.stdout.strip()
        return commit if commit else None
    except Exception:
        return None


class RunLogger:
    """Async logger that writes structured game data to PostgreSQL.

    Usage:
        run_logger = await RunLogger.create(scenario_name=..., condition=..., ...)
        # per turn:
        await run_logger.log_turn(turn, record, state)
        await run_logger.log_llm_trace(turn, ...)  # LLM runs only
        # at end:
        await run_logger.finalize(score, game_over_reason, turns_played)
    """

    def __init__(self, run_id: UUID):
        self.run_id = run_id

    @classmethod
    async def create(
        cls,
        scenario_name: str,
        condition: str,
        player_type: str,
        model: str | None,
        seed: int,
        max_turns: int,
        harness: str | None = None,
        thinking: str | None = None,
        config: dict | None = None,
    ) -> "RunLogger":
        """Create a new run record and return a logger bound to it."""
        run = await RunModel.create(
            scenario_name=scenario_name,
            condition=condition,
            player_type=player_type,
            model=model,
            harness=harness,
            thinking=thinking,
            seed=seed,
            max_turns=max_turns,
            config=config or {},
            engine_commit=get_engine_commit(),
        )
        logger.info(f"Created run {run.id} ({condition}, seed={seed})")
        return cls(run_id=run.id)

    async def log_turn(self, turn: int, record: TurnRecord, state: GameState) -> None:
        """Log a complete turn: snapshot, actions, events, customer snapshots."""
        # Count active and pipeline customers
        active_count = sum(
            1 for c in state.customers.values()
            if c.stage == CustomerStage.customer
        )
        pipeline_count = sum(
            1 for c in state.customers.values()
            if c.is_visible and c.stage in (
                CustomerStage.lead, CustomerStage.prospect,
                CustomerStage.qualified, CustomerStage.in_deal,
            )
        )

        # Turn snapshot
        await TurnSnapshotModel.create(
            run_id=self.run_id,
            turn=turn,
            mrr=record.mrr,
            budget=record.budget,
            runway_turns=record.runway_turns,
            capacity_used=record.capacity_used,
            capacity_available=record.capacity_available,
            tech_debt_level=state.tech_debt.level,
            active_customers=active_count,
            pipeline_customers=pipeline_count,
            bugs_injected=record.bugs_injected,
            bugs_fixed=record.bugs_fixed,
            churn_count=record.churn_count,
        )

        # Actions (valid + rejected) via bulk_create
        action_rows = []
        for action in record.actions_valid:
            action_rows.append(TurnActionModel(
                run_id=self.run_id,
                turn=turn,
                action_type=action.action_type,
                action_data=action.model_dump(),
                capacity=0 if isinstance(action, (HireAction, FireAction, SustainHireAction)) else action.capacity,
                was_valid=True,
            ))
        for rejection in record.actions_rejected:
            action_rows.append(TurnActionModel(
                run_id=self.run_id,
                turn=turn,
                action_type=rejection.action.action_type,
                action_data=rejection.action.model_dump(),
                capacity=0 if isinstance(rejection.action, (HireAction, FireAction, SustainHireAction)) else rejection.action.capacity,
                was_valid=False,
                rejection_reason=rejection.reason,
            ))
        if action_rows:
            await TurnActionModel.bulk_create(action_rows)

        # Events via bulk_create
        event_rows = []
        for event_text in record.events:
            event_type, entity_id = _parse_event(event_text)
            event_rows.append(TurnEventModel(
                run_id=self.run_id,
                turn=turn,
                event_text=event_text,
                event_type=event_type,
                entity_id=entity_id,
            ))
        if event_rows:
            await TurnEventModel.bulk_create(event_rows)

        # Customer snapshots (visible, not churned/lost)
        customer_rows = []
        for c in state.customers.values():
            if not c.is_visible:
                continue
            if c.stage in (CustomerStage.churned, CustomerStage.lost):
                continue
            customer_rows.append(CustomerSnapshotModel(
                run_id=self.run_id,
                turn=turn,
                customer_id=c.id,
                stage=c.stage.value,
                health=round(c.health, 2),
                deal_value=c.deal_value,
                engagement=c.engagement.value,
                competitive_pressure=round(c.competitive_pressure, 3),
                is_customer=c.stage == CustomerStage.customer,
            ))
        if customer_rows:
            await CustomerSnapshotModel.bulk_create(customer_rows)

    async def log_llm_trace(
        self,
        turn: int,
        system_prompt: str,
        user_prompt: str,
        response: TurnActions | None,
        model: str,
        temperature: float,
        max_tokens: int,
        latency_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        """Log an LLM call (prompts + response). Only called for LLM runs."""
        response_raw = None
        if response is not None:
            response_raw = response.model_dump()

        await LLMTraceModel.create(
            run_id=self.run_id,
            turn=turn,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_raw=response_raw,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            latency_ms=latency_ms,
            error=error,
        )

    async def finalize(
        self,
        score: GoalAttainmentScore,
        game_over_reason: str | None,
        turns_played: int,
        token_usage: dict | None = None,
    ) -> None:
        """Update the run record with final scores."""
        await RunModel.filter(id=self.run_id).update(
            turns_played=turns_played,
            game_over_reason=game_over_reason,
            score_composite=round(score.composite, 4),
            score_mrr=round(score.mrr_score, 4),
            score_churn=round(score.churn_score, 4),
            score_runway=round(score.runway_score, 4),
            final_mrr=score.final_mrr,
            final_runway_turns=round(score.final_runway_turns, 2),
            score_pareto=round(score.pareto_score, 4),
            function_scores=score.function_scores,
            alignment_scores=score.alignment_scores,
            token_usage=token_usage,
            finished_at=datetime.now(timezone.utc),
            metadata={
                "function_scores": score.function_scores,
                "function_composite": round(score.function_composite, 4),
                "function_pareto": round(score.function_pareto, 4),
            },
        )
        logger.info(f"Finalized run {self.run_id} (composite={score.composite:.4f})")

    async def log_turn_from_dict(self, tr: dict) -> None:
        """Log a turn from a turn_record.jsonl dict (for condition 2 post-run persistence)."""
        turn = tr["turn"]

        await TurnSnapshotModel.create(
            run_id=self.run_id,
            turn=turn,
            mrr=tr["mrr"],
            budget=tr["budget"],
            runway_turns=tr["runway_turns"],
            capacity_used=tr["capacity_used"],
            capacity_available=tr["capacity_available"],
            tech_debt_level=tr["tech_debt_level"],
            active_customers=tr["active_customers"],
            pipeline_customers=tr["pipeline_customers"],
            bugs_injected=tr["bugs_injected"],
            bugs_fixed=tr["bugs_fixed"],
            churn_count=tr["churn_count"],
        )

        action_rows = []
        for a in tr.get("valid_actions", []):
            action_rows.append(TurnActionModel(
                run_id=self.run_id,
                turn=turn,
                action_type=a.get("action_type", "unknown"),
                action_data=a,
                capacity=a.get("capacity", 0),
                was_valid=True,
            ))
        for r in tr.get("rejected_actions", []):
            a = r.get("action", {})
            action_rows.append(TurnActionModel(
                run_id=self.run_id,
                turn=turn,
                action_type=a.get("action_type", "unknown"),
                action_data=a,
                capacity=a.get("capacity", 0),
                was_valid=False,
                rejection_reason=r.get("reason"),
            ))
        if action_rows:
            await TurnActionModel.bulk_create(action_rows)

        event_rows = []
        for event_text in tr.get("events", []):
            event_type, entity_id = _parse_event(event_text)
            event_rows.append(TurnEventModel(
                run_id=self.run_id,
                turn=turn,
                event_text=event_text,
                event_type=event_type,
                entity_id=entity_id,
            ))
        if event_rows:
            await TurnEventModel.bulk_create(event_rows)

        customer_rows = []
        for c in tr.get("customer_snapshots", []):
            customer_rows.append(CustomerSnapshotModel(
                run_id=self.run_id,
                turn=turn,
                customer_id=c["customer_id"],
                stage=c["stage"],
                health=c["health"],
                deal_value=c["deal_value"],
                engagement=c["engagement"],
                competitive_pressure=c["competitive_pressure"],
                is_customer=c["is_customer"],
            ))
        if customer_rows:
            await CustomerSnapshotModel.bulk_create(customer_rows)

    async def finalize_from_dict(
        self,
        score: dict,
        game_over_reason: str | None,
        turns_played: int,
        token_usage: dict | None = None,
    ) -> None:
        """Finalize a run from a final_status.json score dict (for condition 2 post-run persistence)."""
        await RunModel.filter(id=self.run_id).update(
            turns_played=turns_played,
            game_over_reason=game_over_reason,
            score_composite=round(score.get("composite", 0), 4),
            score_mrr=round(score.get("mrr_score", 0), 4),
            score_churn=round(score.get("churn_score", 0), 4),
            score_runway=round(score.get("runway_score", 0), 4),
            final_mrr=score.get("final_mrr", 0),
            final_runway_turns=round(score.get("final_runway_turns", 0), 2),
            score_pareto=round(score.get("pareto_score", 0), 4),
            function_scores=score.get("function_scores", {}),
            alignment_scores=score.get("alignment_scores", {}),
            token_usage=token_usage,
            finished_at=datetime.now(timezone.utc),
            metadata={
                "function_scores": score.get("function_scores", {}),
                "function_composite": round(score.get("function_composite", 0), 4),
                "function_pareto": round(score.get("function_pareto", 0), 4),
            },
        )
        logger.info(f"Finalized run {self.run_id} from dict (composite={score.get('composite', 0):.4f})")


TOKEN_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


# One transcript per directory, tried in priority order. A dir can hold several files describing the SAME
# LLM calls in different formats, so summing every file would double-count; we pick the first present:
#   * session.jsonl — the claude-code harness session log (~/.claude/projects/<mangled-cwd>/*.jsonl,
#     collected into the results dir). AUTHORITATIVE for claude runs: it carries COMPLETE assistant
#     messages (real output tokens) and survives an agent being killed mid-turn — unlike the stream-json
#     stdout, whose per-message output_tokens are start-of-message placeholders (~0) and whose only true
#     tally, the `result` event, is often absent (agent killed before it emits). Parsed by the dedup-by-
#     message.id assistant branch in parse_transcript_tokens.
#   * pi-session.jsonl — Pi's authoritative session record (a Pi run also writes transcript.{jsonl,json},
#     its `--mode json` stdout, describing the same calls).
#   * transcript.jsonl / transcript.json — the harness stdout stream: claude stream-json (fallback when the
#     session log wasn't captured; parsed via the result/assistant branches) or Pi `--mode json` (parsed
#     via message_end). Fragile for claude output — see session.jsonl above.
# Distinct dirs (C3/C4 per-agent workspaces) are still summed.
_TRANSCRIPT_PRIORITY = ("session.jsonl", "pi-session.jsonl", "transcript.jsonl", "transcript.json")


def _pick_transcript(directory: Path) -> Path | None:
    """Return the highest-priority transcript present in a directory, or None."""
    for name in _TRANSCRIPT_PRIORITY:
        p = directory / name
        if p.exists():
            return p
    return None


def collect_run_token_usage(results_dir: Path) -> dict | None:
    """Aggregate token usage across a results dir, one transcript per directory.

    Handles both layouts:
      * C2: single top-level transcript at results_dir/transcript.{jsonl,json} or pi-session.jsonl
      * C3/C4: per-agent transcripts at results_dir/<function>/{transcript.*,pi-session.jsonl}

    For each directory (top level + one level of subdirs) exactly one transcript is chosen by
    priority (see _TRANSCRIPT_PRIORITY) and parsed; per-model counters are summed across the chosen
    files. Choosing one file per dir avoids double-counting Pi runs, which write the same calls to
    both pi-session.jsonl and the `--mode json` transcript.

    Returns None if no transcripts were found, matching the convention where the finalize_*() callers
    pass token_usage=None to skip the field.
    """
    if not results_dir.exists():
        return None

    dirs: list[Path] = [results_dir]
    dirs.extend(sorted(d for d in results_dir.iterdir() if d.is_dir()))
    candidates = [p for p in (_pick_transcript(d) for d in dirs) if p is not None]

    if not candidates:
        return None

    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {k: 0 for k in TOKEN_USAGE_KEYS}
    )
    for path in candidates:
        for model, usage in parse_transcript_tokens(path).items():
            for k in TOKEN_USAGE_KEYS:
                totals[model][k] += usage.get(k, 0)
    return dict(totals) if totals else None


def parse_transcript_tokens(transcript_path: Path) -> dict:
    """Parse a Claude Code or Pi transcript and aggregate token usage per model.

    Returns: {model: {input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}}

    Claude-code transcripts are tricky: they repeat each assistant message (same `message.id`) several
    times, and in the autonomous stream-json the per-`assistant` `output_tokens` is a start-of-message
    placeholder (~0) — the true per-run tally lives only in the final `result` event's `modelUsage`.
    So for claude-code we prefer `result.modelUsage` when present (authoritative); otherwise — e.g. the
    interactive session log, which has no `result` event but whose assistant entries ARE complete
    messages — we sum assistant usage deduplicated by `message.id`. Pi transcripts are summed as before.
    """
    def _zero() -> dict[str, int]:
        return {k: 0 for k in TOKEN_USAGE_KEYS}

    totals: dict[str, dict[str, int]] = defaultdict(_zero)
    # Claude-code accumulators, reconciled after the full pass (result tally wins over assistant sum).
    result_usage: dict[str, dict[str, int]] = defaultdict(_zero)     # from result.modelUsage (camelCase)
    assistant_usage: dict[str, dict[str, int]] = defaultdict(_zero)  # deduped per-message (underscored)
    have_result = False
    seen_ids: set[str] = set()

    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type")

            # Claude Code per-message usage (underscored). Dedup by message.id: the session log repeats
            # the same assistant message (duplicates carry identical usage), so count each id once.
            if mtype == "assistant" and "message" in msg:
                m = msg["message"]
                mid = m.get("id")
                if mid is not None:
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                model = m.get("model", "unknown")
                usage = m.get("usage", {})
                a = assistant_usage[model]
                a["input_tokens"] += usage.get("input_tokens", 0)
                a["output_tokens"] += usage.get("output_tokens", 0)
                a["cache_creation_input_tokens"] += usage.get("cache_creation_input_tokens", 0)
                a["cache_read_input_tokens"] += usage.get("cache_read_input_tokens", 0)

            # Claude Code final result event: the authoritative per-model tally (camelCase fields).
            elif mtype == "result":
                model_usage = msg.get("modelUsage") or {}
                if model_usage:
                    have_result = True
                    for model, mv in model_usage.items():
                        r = result_usage[model]
                        r["input_tokens"] += mv.get("inputTokens", 0)
                        r["output_tokens"] += mv.get("outputTokens", 0)
                        r["cache_creation_input_tokens"] += mv.get("cacheCreationInputTokens", 0)
                        r["cache_read_input_tokens"] += mv.get("cacheReadInputTokens", 0)

            # Pi session format: type=="message", role=="assistant", usage keys are camelCase.
            # Pi's cacheWrite/cacheRead map to Anthropic's cache_creation/cache_read input tokens.
            elif mtype == "message":
                m = msg.get("message", {})
                if m.get("role") != "assistant":
                    continue
                model = m.get("model", "unknown")
                usage = m.get("usage", {})
                totals[model]["input_tokens"] += usage.get("input", 0)
                totals[model]["output_tokens"] += usage.get("output", 0)
                totals[model]["cache_creation_input_tokens"] += usage.get("cacheWrite", 0)
                totals[model]["cache_read_input_tokens"] += usage.get("cacheRead", 0)

            # Pi --mode json stream format: message_end carries the FINAL per-message usage (same
            # camelCase fields as the session format). message_start/message_update are partial, so
            # only message_end is summed. collect_run_token_usage picks one transcript per dir, so this
            # never double-counts a run that also has a pi-session.jsonl.
            elif mtype == "message_end":
                m = msg.get("message") or {}
                if not isinstance(m, dict) or m.get("role") != "assistant":
                    continue
                usage = m.get("usage") or {}
                model = m.get("model", "unknown")
                totals[model]["input_tokens"] += usage.get("input", 0)
                totals[model]["output_tokens"] += usage.get("output", 0)
                totals[model]["cache_creation_input_tokens"] += usage.get("cacheWrite", 0)
                totals[model]["cache_read_input_tokens"] += usage.get("cacheRead", 0)

    # Fold claude-code usage in: the authoritative result tally wins; else the deduped assistant sum.
    claude_usage = result_usage if have_result else assistant_usage
    for model, usage in claude_usage.items():
        for k in TOKEN_USAGE_KEYS:
            totals[model][k] += usage[k]

    return dict(totals)


def _parse_event(event_text: str) -> tuple[str | None, str | None]:
    """Parse event text to extract type and entity ID.

    Events follow patterns like 'deal_won:C05', 'churn:C03',
    'feature_shipped:F01:shipped_mvp', 'bug_injected:BUG_001:critical:F02'.
    """
    parts = event_text.split(":")
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    if parts:
        return parts[0].strip(), None
    return None, None
