"""Condition 3: FastAPI orchestrator for multi-agent games.

Sits between agents and the game engine. Enforces information asymmetry,
validates per-function actions, synchronizes turns, and manages the chat room.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from alignsim.src.engine.game import GameEngine
from alignsim.src.game_cli import _parse_actions, _persist_internal_scores
from alignsim.src.harness.condition3_filters import (
    ALL_FUNCTIONS,
    filter_events,
    filter_observation,
    is_compute_allowed,
    is_query_allowed,
    validate_function_actions,
)
from alignsim.src.harness.inspector import GameInspector
from alignsim.src.models.actions import TurnActions
from alignsim.src.models.goals import score_to_player_dict

logger = logging.getLogger("alignsim.orchestrator")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentSlot:
    function: str
    active: bool = True
    last_chat_read_seq: int = 0
    warned_at_seq: int | None = None


@dataclass
class ChatMessage:
    seq: int
    turn: int
    agent: str
    ts: str
    message: str

    def to_dict(self) -> dict:
        return {"seq": self.seq, "turn": self.turn, "agent": self.agent, "ts": self.ts, "message": self.message}


# ---------------------------------------------------------------------------
# Turn synchronizer
# ---------------------------------------------------------------------------


class TurnSynchronizer:
    """Block-on-submit until every active agent has submitted, then resolve.

    Uses per-call futures instead of a shared condition variable to prevent
    late-waking waiters from reading the wrong turn's result after a reset.
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.active_functions: set[str] = set()
        self._submissions: dict[str, list[dict]] = {}
        self._futures: dict[str, asyncio.Future] = {}

    @property
    def pending_submissions(self) -> set[str]:
        return set(self._submissions.keys())

    async def submit_and_wait(self, function: str, actions: list[dict], resolve_callback) -> dict:
        async with self.lock:
            future = self._futures.get(function)
            if future is None or future.done():
                future = asyncio.get_running_loop().create_future()
                self._futures[function] = future
            self._submissions[function] = actions

            if set(self._submissions.keys()) >= self.active_functions:
                submissions_snapshot = dict(self._submissions)
                futures_snapshot = dict(self._futures)
                self._submissions = {}
                self._futures = {}
            else:
                submissions_snapshot = None
                futures_snapshot = None

        if submissions_snapshot is not None:
            try:
                results = await resolve_callback(submissions_snapshot)
            except BaseException as exc:
                for fut in futures_snapshot.values():
                    if not fut.done():
                        fut.set_exception(exc)
                raise
            for fn, res in results.items():
                fut = futures_snapshot.get(fn)
                if fut and not fut.done():
                    fut.set_result(res)

        return await future


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    def __init__(self, engine: GameEngine, submit_timeout_s: float = 600.0, output_dir: str | None = None) -> None:
        self.engine = engine
        self.inspector = GameInspector(engine)
        self.agents: dict[str, AgentSlot] = {}
        self.synchronizer = TurnSynchronizer()
        self.chat_messages: list[ChatMessage] = []
        self.chat_seq: int = 0
        self.onboarding_queue: list[str] = []
        self.game_log: list[dict] = []
        self.hire_owners: dict[str, str] = {}
        self.submit_timeout_s = submit_timeout_s
        self._current_obs = engine.get_initial_observation()
        self._pre_turn_pools: dict[str, int] | None = None
        self._turn_start_time: float = time.monotonic()
        self._auto_submitted: set[str] = set()
        self._chat_log_written: bool = False
        self._output_dir = Path(output_dir) if output_dir else None
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_functions(self) -> set[str]:
        return {fn for fn, slot in self.agents.items() if slot.active}

    # -- Agent management --

    def _register_agent_unlocked(self, function: str) -> None:
        if function not in ALL_FUNCTIONS:
            raise ValueError(f"Unknown function: {function}")
        self.agents[function] = AgentSlot(function=function)
        self.synchronizer.active_functions = self.active_functions
        self._log("agent_registered", function=function)

    async def register_agent(self, function: str) -> None:
        async with self.synchronizer.lock:
            self._register_agent_unlocked(function)

    # -- Observation --

    def get_filtered_observation(self, function: str) -> dict:
        return filter_observation(self._current_obs, function, self.engine)

    # -- Status --

    def get_status(self, function: str) -> dict:
        status = self.inspector.get_status()
        status["function"] = function
        status["active_agents"] = sorted(self.active_functions)
        return status

    # -- Chat room --

    def post_chat(self, agent: str, message: str) -> int:
        self.chat_seq += 1
        self.chat_messages.append(ChatMessage(
            seq=self.chat_seq,
            turn=self.engine.state.turn,
            agent=agent,
            ts=datetime.now(timezone.utc).isoformat(),
            message=message,
        ))
        return self.chat_seq

    def read_chat(self, function: str, since: int = 0) -> list[dict]:
        messages = [m.to_dict() for m in self.chat_messages if m.seq > since]
        if messages:
            self.agents[function].last_chat_read_seq = messages[-1]["seq"]
        return messages

    def _check_unread_messages(self, function: str) -> dict | None:
        slot = self.agents[function]
        unread = [m for m in self.chat_messages if m.seq > slot.last_chat_read_seq and m.agent != function]
        if not unread:
            return None

        if slot.warned_at_seq is not None:
            max_other_seq = max(
                (m.seq for m in self.chat_messages if m.agent != function),
                default=0,
            )
            if max_other_seq <= slot.warned_at_seq:
                did_read = slot.last_chat_read_seq >= slot.warned_at_seq
                self._log("unread_warning_resolved",
                          function=function, read_after_warning=did_read)
                slot.warned_at_seq = None
                return None

        slot.warned_at_seq = self.chat_messages[-1].seq if self.chat_messages else 0
        self._log("unread_warning_issued", function=function, unread_count=len(unread))
        return {
            "status": "unread_messages",
            "unread_count": len(unread),
            "preview": [{"seq": m.seq, "agent": m.agent, "message": m.message} for m in unread[-5:]],
            "message": (
                f"You have {len(unread)} unread chat messages. "
                "Read them with ./game chat read, then resubmit. "
                "Your actions will be accepted on next submit attempt."
            ),
        }

    # -- Submit --

    async def submit_actions(self, function: str, actions: list[dict], force: bool = False) -> dict:
        if not force:
            unread = self._check_unread_messages(function)
            if unread is not None:
                return unread

        validation = validate_function_actions(actions, function, self.engine)

        for rej in validation.rejected_actions:
            # A rejected "action" can be a non-dict (e.g. a bare string) when the model emits a
            # malformed action item — validate_function_actions rejects it but keeps the raw value.
            # Guard the .get() so recording the rejection can't raise (was an unhandled 500).
            rejected = rej["action"]
            action_type = rejected.get("action_type") if isinstance(rejected, dict) else None
            self.inspector.record_rejections(
                self.engine.state.turn,
                [{"action_type": action_type, "reason": rej["reason"]}],
            )

        result = await self.synchronizer.submit_and_wait(
            function, validation.valid_actions, self._resolve_turn,
        )

        if validation.rejected_actions:
            final = dict(result)
            final["function_rejections"] = (
                list(result.get("function_rejections", [])) + list(validation.rejected_actions)
            )
            return final

        return result

    # -- Turn resolution --

    async def _resolve_turn(self, submissions: dict[str, list[dict]]) -> dict[str, dict]:
        turn = self.engine.state.turn

        self._pre_turn_pools = {
            "support": self.engine.state.resources.support_capacity,
            "ops": self.engine.state.resources.ops_capacity,
        }

        all_parsed = []
        action_to_fn: dict[int, str] = {}
        fn_parse_errors: dict[str, list[str]] = {}

        for fn, fn_actions in submissions.items():
            fn_turn, fn_errors = _parse_actions(fn_actions, turn)
            fn_parse_errors[fn] = fn_errors
            for action in fn_turn.actions:
                all_parsed.append(action)
                action_to_fn[id(action)] = fn

        merged = TurnActions(turn=turn, actions=all_parsed)
        result, next_obs = self.engine.step(merged)

        if next_obs:
            self._current_obs = next_obs

        for event in result.record.events:
            if event.startswith("hire_started:"):
                self._register_hire_from_event(event)

        fn_engine_rej: dict[str, list[dict]] = {fn: [] for fn in submissions}
        for rej in result.validation.rejected_actions:
            fn = action_to_fn.get(id(rej.action))
            entry = {"action_type": rej.action.action_type, "reason": rej.reason}
            if fn is not None:
                fn_engine_rej[fn].append(entry)
            else:
                logger.warning(
                    "Unattributable rejection (action %s, reason %s)",
                    rej.action.action_type, rej.reason,
                )

        for rej in result.validation.rejected_actions:
            self.inspector.record_rejections(
                turn,
                [{"action_type": rej.action.action_type, "reason": rej.reason}],
            )

        self._check_onboarding()

        per_function: dict[str, dict] = {}
        for fn in self.active_functions:
            fn_result: dict[str, Any] = {
                "turn": turn,
                "game_over": result.game_over,
                "game_over_reason": result.game_over_reason,
                "events": filter_events(result.record.events, fn, self.hire_owners),
                "engine_rejections": fn_engine_rej.get(fn, []),
            }

            if fn_parse_errors.get(fn):
                fn_result["parse_errors"] = fn_parse_errors[fn]

            if next_obs:
                fn_result["next_observation"] = filter_observation(next_obs, fn, self.engine)

            if result.game_over:
                score = self.engine.get_final_score()
                fn_result["final_score"] = score_to_player_dict(score)

            per_function[fn] = fn_result

        self._log(
            "turn_resolved",
            turn=turn,
            actions_valid=len(result.validation.valid_actions),
            actions_rejected=len(result.validation.rejected_actions),
            game_over=result.game_over,
        )

        self._write_turn_record(turn, result, submissions)
        if result.game_over:
            self.write_chat_log()
            if self._output_dir:
                _persist_internal_scores(self._output_dir, self.engine.get_final_score())
        self._turn_start_time = time.monotonic()
        self._auto_submitted.clear()
        return per_function

    def _register_hire_from_event(self, event: str) -> None:
        parts = event.split(":")
        if len(parts) < 3:
            return
        hire_id = parts[1]
        if ":cross_hire_from_" in event:
            for part in parts:
                if part.startswith("cross_hire_from_"):
                    self.hire_owners[hire_id] = part[len("cross_hire_from_"):]
                    return
        self.hire_owners[hire_id] = parts[2]

    # -- Onboarding --

    def _check_onboarding(self) -> None:
        if self._pre_turn_pools is None:
            return
        pool_map = {
            "support": self.engine.state.resources.support_capacity,
            "ops": self.engine.state.resources.ops_capacity,
        }
        for fn, post_cap in pool_map.items():
            pre_cap = self._pre_turn_pools.get(fn, 0)
            if pre_cap == 0 and post_cap > 0 and fn not in self.agents:
                self.onboarding_queue.append(fn)
                self._log("onboarding_queued", function=fn, capacity=post_cap)

    def get_pending_onboarding(self) -> list[str]:
        return list(self.onboarding_queue)

    def ack_onboarding(self, function: str) -> None:
        if function in self.onboarding_queue:
            self.onboarding_queue.remove(function)

    # -- Game over --

    def is_game_over(self) -> bool:
        return self.engine.state.game_over

    # -- Orchestrator status --

    def get_orchestrator_status(self) -> dict:
        status: dict[str, Any] = {
            "turn": self.engine.state.turn,
            "game_over": self.engine.state.game_over,
            "game_over_reason": self.engine.state.game_over_reason,
            "active_agents": sorted(self.active_functions),
            "pending_submissions": sorted(self.synchronizer.pending_submissions),
            "onboarding_queue": list(self.onboarding_queue),
            "chat_message_count": len(self.chat_messages),
        }
        if self.engine.state.game_over:
            score = self.engine.get_final_score()
            status["final_score"] = score_to_player_dict(score)
        return status

    # -- Timeout watchdog --

    async def run_watchdog(self) -> None:
        """Background loop: auto-submit empty actions for agents that exceed the timeout."""
        while not self.engine.state.game_over:
            await asyncio.sleep(30)
            elapsed = time.monotonic() - self._turn_start_time
            if elapsed < self.submit_timeout_s:
                continue
            missing = (
                self.active_functions
                - self.synchronizer.pending_submissions
                - self._auto_submitted
            )
            for fn in missing:
                logger.warning("Timeout: auto-submitting empty actions for %s", fn)
                self._auto_submitted.add(fn)
                task = asyncio.create_task(self.submit_actions(fn, [], force=True))
                task.add_done_callback(
                    lambda t, _fn=fn: (
                        logger.error("Watchdog auto-submit failed for %s: %s", _fn, t.exception())
                        if not t.cancelled() and t.exception() else None
                    )
                )

    def _log(self, event_name: str, **kwargs) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_name, **kwargs}
        self.game_log.append(entry)
        logger.info("%s: %s", event_name, kwargs)

    def _write_turn_record(self, turn: int, result, submissions: dict[str, list[dict]]) -> None:
        if not self._output_dir:
            return
        state = self.engine.state
        active = sum(1 for c in state.customers.values() if c.stage.value == "customer")
        pipeline = sum(
            1 for c in state.customers.values()
            if c.is_visible and c.stage.value in ("lead", "prospect", "qualified", "in_deal")
        )
        record = {
            "turn": turn,
            "mrr": result.record.mrr,
            "budget": result.record.budget,
            "runway_turns": result.record.runway_turns,
            "capacity_used": result.record.capacity_used,
            "capacity_available": result.record.capacity_available,
            "tech_debt_level": state.tech_debt.level,
            "active_customers": active,
            "pipeline_customers": pipeline,
            "bugs_injected": result.record.bugs_injected,
            "bugs_fixed": result.record.bugs_fixed,
            "churn_count": result.record.churn_count,
            "valid_actions": [a.model_dump() for a in result.validation.valid_actions],
            "rejected_actions": [
                {"action": r.action.model_dump(), "reason": r.reason}
                for r in result.validation.rejected_actions
            ],
            "events": result.record.events,
            "customer_snapshots": [
                {
                    "customer_id": c.id,
                    "stage": c.stage.value,
                    "health": round(c.health, 2),
                    "deal_value": c.deal_value,
                    "engagement": c.engagement.value,
                    "competitive_pressure": round(c.competitive_pressure, 3),
                    "is_customer": c.stage.value == "customer",
                }
                for c in state.customers.values()
                if c.is_visible and c.stage.value not in ("churned", "lost")
            ],
            "actions_by_function": {fn: actions for fn, actions in submissions.items()},
            "game_over": result.game_over,
        }
        with open(self._output_dir / "turn_record.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def write_chat_log(self) -> None:
        if not self._output_dir or self._chat_log_written:
            return
        with open(self._output_dir / "chat_log.jsonl", "w", encoding="utf-8") as f:
            for msg in self.chat_messages:
                f.write(json.dumps(msg.to_dict()) + "\n")
        self._chat_log_written = True


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


# -- Shared request helpers (module level so C4 factories can reuse them) --


def _orch(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


def _require_agent(request: Request, fn: str) -> AgentSlot:
    o = _orch(request)
    if fn not in o.agents:
        raise HTTPException(404, f"Agent '{fn}' not registered")
    return o.agents[fn]


async def _json_body(request: Request) -> dict:
    """Parse a JSON request body, returning a clean 400 instead of an unhandled 500.

    A malformed body (e.g. an agent's empty/truncated actions file yielding `{"actions": }`)
    raises JSONDecodeError; a valid-but-non-object body (e.g. a bare list) would break the
    downstream `body.get(...)` with an AttributeError. Both become a 400 with a `detail` message.
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(400, "Request body must be valid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object")
    return body


# -- Composable route registrars (reused verbatim by the C4 app factories) --


def _register_agent_routes(app: FastAPI) -> None:
    """Observe, submit, status, query, and compute endpoints (substrate-independent)."""

    @app.get("/agents/{fn}/observe")
    async def observe(request: Request, fn: str):
        _require_agent(request, fn)
        o = _orch(request)
        if o.engine.state.game_over:
            return {"error": "Game is over", "reason": o.engine.state.game_over_reason}
        return o.get_filtered_observation(fn)

    @app.post("/agents/{fn}/submit")
    async def submit(request: Request, fn: str):
        _require_agent(request, fn)
        o = _orch(request)
        if o.engine.state.game_over:
            return JSONResponse(
                {"error": "Game is over", "reason": o.engine.state.game_over_reason},
                status_code=400,
            )
        body = await _json_body(request)
        actions = body.get("actions", [])
        if not isinstance(actions, list):
            raise HTTPException(400, "'actions' must be a JSON array of action objects")
        result = await o.submit_actions(fn, actions)
        if result.get("status") == "unread_messages":
            return JSONResponse(result, status_code=409)
        return result

    @app.get("/agents/{fn}/status")
    async def agent_status(request: Request, fn: str):
        _require_agent(request, fn)
        return _orch(request).get_status(fn)

    @app.get("/agents/{fn}/query/{query_type}")
    async def query(request: Request, fn: str, query_type: str, id: str = Query(None)):
        _require_agent(request, fn)
        o = _orch(request)
        if not is_query_allowed(fn, query_type):
            raise HTTPException(403, f"Query '{query_type}' not allowed for {fn}")
        if query_type == "customer":
            if not id:
                raise HTTPException(400, "Customer ID required")
            details = o.inspector.get_customer_details(id)
            if fn == "support" and "error" not in details:
                _CS_HIDDEN_FIELDS = {"known_needs", "deal_value", "competitive_pressure",
                                     "timeline_remaining", "min_sell_capacity"}
                details = {k: v for k, v in details.items() if k not in _CS_HIDDEN_FIELDS}
            return details
        elif query_type == "feature":
            if not id:
                raise HTTPException(400, "Feature ID required")
            return o.inspector.get_feature_status(id)
        elif query_type == "bugs":
            return o.inspector.list_bugs()
        elif query_type == "rejections":
            return o.inspector.get_rejection_history()
        else:
            raise HTTPException(400, f"Unknown query type: {query_type}")

    @app.get("/agents/{fn}/compute/{compute_type}")
    async def compute_get(
        request: Request,
        fn: str,
        compute_type: str,
        id: str = Query(None),
        quality: str = Query(None),
    ):
        _require_agent(request, fn)
        o = _orch(request)
        if not is_compute_allowed(fn, compute_type):
            raise HTTPException(403, f"Compute '{compute_type}' not allowed for {fn}")
        if compute_type == "maturity":
            return o.inspector.compute_maturity()
        elif compute_type == "satisfaction":
            if not id:
                raise HTTPException(400, "Customer ID required")
            return o.inspector.estimate_satisfaction(id)
        elif compute_type == "maturity-if":
            if not id or not quality:
                raise HTTPException(400, "Feature ID and quality required")
            return o.inspector.simulate_maturity_change(id, quality)
        elif compute_type == "capacity-cost":
            raise HTTPException(400, "capacity-cost requires POST to /agents/{fn}/compute/capacity-cost")
        else:
            raise HTTPException(400, f"Unknown compute type: {compute_type}")

    @app.post("/agents/{fn}/compute/capacity-cost")
    async def compute_capacity_cost(request: Request, fn: str):
        _require_agent(request, fn)
        o = _orch(request)
        if not is_compute_allowed(fn, "capacity-cost"):
            raise HTTPException(403, f"Compute 'capacity-cost' not allowed for {fn}")
        body = await _json_body(request)
        return o.inspector.compute_capacity_cost(body.get("actions", []))


def _register_flat_chat_routes(app: FastAPI) -> None:
    """C3's flat single-room chat. Reused verbatim by C4b (its chat IS C3's chat)."""

    @app.get("/chat")
    async def read_chat_global(request: Request, since: int = 0):
        o = _orch(request)
        return [m.to_dict() for m in o.chat_messages if m.seq > since]

    @app.get("/agents/{fn}/chat")
    async def read_chat(request: Request, fn: str, since: int = 0):
        _require_agent(request, fn)
        return _orch(request).read_chat(fn, since)

    @app.post("/agents/{fn}/chat")
    async def post_chat(request: Request, fn: str):
        _require_agent(request, fn)
        o = _orch(request)
        body = await _json_body(request)
        message = body.get("message", "")
        if not message:
            raise HTTPException(400, "Message required")
        seq = o.post_chat(fn, message)
        return {"seq": seq}


def _register_orchestrator_routes(app: FastAPI) -> None:
    """Orchestrator management + health (substrate-independent)."""

    @app.get("/orchestrator/status")
    async def orchestrator_status(request: Request):
        return _orch(request).get_orchestrator_status()

    @app.post("/orchestrator/register-agent")
    async def register_agent(request: Request):
        o = _orch(request)
        body = await _json_body(request)
        function = body.get("function", "")
        if function not in ALL_FUNCTIONS:
            raise HTTPException(400, f"Unknown function: {function}")
        if function in o.agents:
            raise HTTPException(409, f"Agent '{function}' already registered")
        await o.register_agent(function)
        return {"status": "registered", "function": function}

    @app.get("/orchestrator/pending-onboarding")
    async def pending_onboarding(request: Request):
        return {"pending": _orch(request).get_pending_onboarding()}

    @app.post("/orchestrator/ack-onboarding")
    async def ack_onboarding(request: Request):
        o = _orch(request)
        body = await _json_body(request)
        function = body.get("function", "")
        o.ack_onboarding(function)
        return {"status": "acknowledged", "function": function}

    @app.get("/orchestrator/game-over")
    async def game_over(request: Request):
        o = _orch(request)
        resp: dict[str, Any] = {"game_over": o.is_game_over()}
        if o.is_game_over():
            resp["reason"] = o.engine.state.game_over_reason
            score = o.engine.get_final_score()
            resp["final_score"] = score_to_player_dict(score)
        return resp

    @app.get("/health")
    async def health():
        return {"status": "ok"}


def create_app(orchestrator: Orchestrator) -> FastAPI:
    app = FastAPI(title="AlignSim Condition 3 Orchestrator")
    app.state.orchestrator = orchestrator
    _register_agent_routes(app)
    _register_flat_chat_routes(app)
    _register_orchestrator_routes(app)
    return app
