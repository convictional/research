"""Condition 4: two orthogonal substrate extensions of the Condition 3 orchestrator.

C4 holds the game and the agents fixed and varies only *how the functions coordinate*
(the collaboration substrate). Two sibling subclasses of the C3 ``Orchestrator``:

- ``ChannelOrchestrator`` (C4a — "Slack model"): C3's flat chat organised into named,
  public channels. Seeded with ``everyone`` plus one channel per starting function;
  agents create more at will. No durable artifacts.
- ``ConvictionalOrchestrator`` (C4b — "Convictional model"): C3's flat ``everyone`` chat
  UNCHANGED, plus two durable artifacts — Posts (topic-scoped threads with comments and
  recorded decisions) and a shared Goals hierarchy (seeded from the scenario's real goals,
  with live progress; agents create/own/sub-goal and self-report updates).

Both reuse ``condition3_filters`` verbatim (information bounds are substrate-independent)
and the C3 engine unchanged. The substrate is the treatment; game strategy never leaks
into it — the goal tree is seeded with only the real game goals every condition already has.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request

from alignsim.src.engine.game import GameEngine
from alignsim.src.engine.scoring import _extract_metric, compute_goal_attainment
from alignsim.src.harness.condition3_filters import ALL_FUNCTIONS, STARTING_FUNCTIONS
from alignsim.src.harness.condition3_orchestrator import (
    ChatMessage,
    Orchestrator,
    _orch,
    _register_agent_routes,
    _register_orchestrator_routes,
    _require_agent,
)
from alignsim.src.models.goals import GoalAttainmentScore

# ---------------------------------------------------------------------------
# C4a — Channel data structures
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass
class ChannelChatMessage(ChatMessage):
    """A chat message tagged with the channel it was posted to.

    ``to_dict`` is a superset of C3's format (adds ``channel``), so the C3 chat-log
    writer and readers accept it unchanged.
    """

    channel: str = "everyone"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["channel"] = self.channel
        return d


@dataclass
class ChannelMeta:
    name: str
    created_by: str
    created_turn: int
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_by": self.created_by,
            "created_turn": self.created_turn,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# C4b — Post + Goal data structures
# ---------------------------------------------------------------------------


@dataclass
class PostComment:
    author: str
    text: str
    created_turn: int

    def to_dict(self) -> dict:
        return {"author": self.author, "text": self.text, "created_turn": self.created_turn}


@dataclass
class Post:
    id: str
    author: str
    title: str
    body: str
    created_turn: int
    comments: list[PostComment] = field(default_factory=list)
    decision: str | None = None
    decided_by: str | None = None
    decided_turn: int | None = None
    # Monotonic activity stamp (create/comment/decision) for per-post unread tracking.
    last_activity: int = 0

    def to_summary(self) -> dict:
        return {
            "id": self.id,
            "author": self.author,
            "title": self.title,
            "created_turn": self.created_turn,
            "comment_count": len(self.comments),
            "has_decision": self.decision is not None,
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "author": self.author,
            "title": self.title,
            "body": self.body,
            "created_turn": self.created_turn,
            "comments": [c.to_dict() for c in self.comments],
            "decision": self.decision,
            "decided_by": self.decided_by,
            "decided_turn": self.decided_turn,
        }


@dataclass
class GoalUpdate:
    author: str
    status: str
    progress: float
    note: str
    created_turn: int

    def to_dict(self) -> dict:
        return {
            "author": self.author,
            "status": self.status,
            "progress": self.progress,
            "note": self.note,
            "created_turn": self.created_turn,
        }


@dataclass
class Goal:
    id: str
    title: str
    description: str
    owner: str | None
    parent_id: str | None
    status: str  # on_track | at_risk | off_track
    progress: float
    native: bool
    updates: list[GoalUpdate] = field(default_factory=list)
    created_by: str = "system"
    created_turn: int = 0
    # Native-only metadata (used to compute live progress from engine state).
    native_key: str | None = None  # "mrr" | "churn" | "runway" | <role> for function goals
    metric: str | None = None
    target: float | None = None


_GOAL_STATUSES = frozenset({"on_track", "at_risk", "off_track"})


def _status_from_progress(progress: float) -> str:
    """Derive a display status from an attainment ratio (1.0 = par).

    Presentation-only convenience derived from the engine's own scoring — not a tactic.
    """
    if progress >= 1.0:
        return "on_track"
    if progress >= 0.6:
        return "at_risk"
    return "off_track"


# ---------------------------------------------------------------------------
# C4a — Channel Orchestrator
# ---------------------------------------------------------------------------


class ChannelOrchestrator(Orchestrator):
    """C4a: C3's flat chat organised into named, public channels.

    Seeded with ``everyone`` plus one channel per STARTING function. Agents create the
    rest (including support/ops channels) at will; every channel is public (only 5
    players). The unread hard-gate is inherited from C3 and fires across ALL channels —
    a reader must keep up with every channel or be blocked, matching C3's "read
    everything" mechanic.
    """

    # everyone + one channel per STARTING function (derived from the shared constant so
    # C4a can't silently diverge from the other conditions' starting-function set).
    DEFAULT_CHANNELS = ["everyone", *sorted(STARTING_FUNCTIONS)]

    def __init__(self, engine: GameEngine, **kwargs: Any) -> None:
        super().__init__(engine, **kwargs)
        self.channels: dict[str, ChannelMeta] = {
            c: ChannelMeta(name=c, created_by="system", created_turn=0) for c in self.DEFAULT_CHANNELS
        }

    def create_channel(self, name: str, creator: str, description: str = "") -> dict:
        if not _SLUG_RE.fullmatch(name):
            raise ValueError(
                f"Invalid channel name '{name}'. Use lowercase letters, digits, '-' and '_' "
                "(must start with a letter or digit)."
            )
        if name in self.channels:
            raise ValueError(f"Channel '{name}' already exists")
        meta = ChannelMeta(
            name=name,
            created_by=creator,
            created_turn=self.engine.state.turn,
            description=description,
        )
        self.channels[name] = meta
        self._log("channel_created", channel=name, created_by=creator)
        return meta.to_dict()

    def list_channels(self) -> list[dict]:
        return [m.to_dict() for m in self.channels.values()]

    # -- Chat (channel-aware overrides of C3) --

    def post_chat(self, agent: str, message: str, channel: str = "everyone") -> int:
        if channel not in self.channels:
            raise ValueError(
                f"Channel '{channel}' does not exist. Create it first with "
                "'./game channel create <name>', or use an existing channel."
            )
        self.chat_seq += 1
        self.chat_messages.append(
            ChannelChatMessage(
                seq=self.chat_seq,
                turn=self.engine.state.turn,
                agent=agent,
                ts=datetime.now(timezone.utc).isoformat(),
                message=message,
                channel=channel,
            )
        )
        return self.chat_seq

    def read_chat(self, function: str, since: int = 0, channel: str | None = None) -> list[dict]:
        messages = [m for m in self.chat_messages if m.seq > since]
        if channel is not None:
            # A single-channel read is a PARTIAL view. It must NOT advance the unread
            # gate cursor (which spans all channels) — otherwise reading one channel
            # would silently clear the 409 gate for unread messages in other channels,
            # defeating the "read everything or be blocked" invariant. Only a full read
            # (channel is None) advances the cursor.
            return [m.to_dict() for m in messages if getattr(m, "channel", "everyone") == channel]
        dicts = [m.to_dict() for m in messages]
        if dicts:
            # Advance monotonically: never rewind below what a prior read already cleared.
            self.agents[function].last_chat_read_seq = max(
                self.agents[function].last_chat_read_seq,
                dicts[-1]["seq"],
            )
        return dicts


# ---------------------------------------------------------------------------
# C4b — Convictional Orchestrator
# ---------------------------------------------------------------------------


class ConvictionalOrchestrator(Orchestrator):
    """C4b: Durable Posts + a shared Goals tree.

    Chat is inherited verbatim from C3 (its treatment is the artifacts, not chat structure).
    Posts and Goals are pull-based: agents check them via ``./game`` commands and see a soft
    unread counter in ``./game status``. The chat 409 hard-gate (C3's exact mechanism) is the
    only gate; async, work-in-public artifacts are pull-based by nature.
    """

    def __init__(self, engine: GameEngine, **kwargs: Any) -> None:
        super().__init__(engine, **kwargs)
        self.posts: dict[str, Post] = {}
        self._post_id_seq: int = 0
        self.goals: dict[str, Goal] = {}
        self._goal_id_seq: int = 0
        # Soft unread counters (surfaced in status; NOT a hard gate).
        # Posts: a monotonic activity stamp per post + per-agent per-post last-seen, so a
        # glance at the summary list never clears an unread comment on a post you didn't open.
        self._post_activity_seq: int = 0
        self._post_seen: dict[str, dict[str, int]] = {}
        # Goals: a single counter suffices — get_goals returns the full tree (every update),
        # so viewing it genuinely consumes all goal activity.
        self.goal_activity: int = 0
        self._seen_goal_activity: dict[str, int] = {}
        # Substrate 409 gate: per-function "warned at this activity level" watermark. Mirrors the
        # base chat gate's warn-then-allow so a read-gate + turn-barrier can't livelock (Phase 1).
        self._substrate_warned: dict[str, int | None] = {}
        self._seed_native_goals()

    async def _resolve_turn(self, submissions: dict[str, list[dict]]) -> dict[str, dict]:
        # Reuse C3's turn resolution verbatim, then flush the durable artifacts on game
        # over so they survive a hard kill (mirrors how C3 writes chat_log on game over).
        per_function = await super()._resolve_turn(submissions)
        if self.engine.state.game_over:
            self.write_posts_log()
            self.write_goals_log()
        return per_function

    # -- Goal seeding (native game goals only — no coaching) --

    def _seed_native_goals(self) -> None:
        """Seed the shared goal tree from the scenario's REAL goals (flat — 8 top-level goals).

        The 3 primary constraints (MRR / churn / runway) are shared — everyone contributes — so
        they are seeded ``owner=None``. Each of the 5 function goals is seeded owned by its
        function IF that function exists at the start (engineering / sales / marketing); CS + Ops
        don't exist yet, so their goals start ``owner=None`` — the honest state, which lets the
        team notice the gap and self-organise. Goals are flat (no parent nesting): the scenario's
        sub-goals hang off the whole primary goal, not any single constraint, so nesting them
        under MRR would wrongly imply they ladder to MRR over churn/runway. No invented goals
        that name measured behaviours.
        """
        goal = self.engine.scenario.primary_goal

        self.goals["NG-mrr"] = Goal(
            id="NG-mrr",
            title=f"Reach ${goal.mrr_target:,} MRR by turn {goal.target_turn}",
            description="Company monthly recurring revenue target — the primary win condition.",
            owner=None,
            parent_id=None,
            status="on_track",
            progress=0.0,
            native=True,
            native_key="mrr",
            target=float(goal.mrr_target),
        )
        self.goals["NG-churn"] = Goal(
            id="NG-churn",
            title=f"Keep churn rate below {goal.max_churn_rate:.0%}",
            description="Cumulative churn-rate constraint — retention guardrail.",
            owner=None,
            parent_id=None,
            status="on_track",
            progress=0.0,
            native=True,
            native_key="churn",
            target=float(goal.max_churn_rate),
        )
        self.goals["NG-runway"] = Goal(
            id="NG-runway",
            title=f"Maintain runway above {goal.min_runway_turns:g} turns",
            description="Cash-runway constraint — solvency guardrail.",
            owner=None,
            parent_id=None,
            status="on_track",
            progress=0.0,
            native=True,
            native_key="runway",
            target=float(goal.min_runway_turns),
        )
        for sg in goal.sub_goals:
            # Own each function goal by its function IF that function exists at the start
            # (engineering/sales/marketing). CS + Ops don't exist yet, so their goals stay
            # unowned — the honest state; the team can notice the gap and self-organise.
            owner = sg.role if sg.role in STARTING_FUNCTIONS else None
            self.goals[f"NG-{sg.role}"] = Goal(
                id=f"NG-{sg.role}",
                title=sg.description,
                description=f"{sg.role} function goal (metric: {sg.metric}, target: {sg.target_value:g}).",
                owner=owner,
                parent_id=None,
                status="on_track",
                progress=0.0,
                native=True,
                native_key=sg.role,
                metric=sg.metric,
                target=float(sg.target_value),
            )

    # -- Posts --

    def create_post(self, agent: str, title: str, body: str) -> dict:
        if not title:
            raise ValueError("Post title required")
        self._post_id_seq += 1
        post_id = f"P{self._post_id_seq:02d}"
        post = Post(
            id=post_id,
            author=agent,
            title=title,
            body=body,
            created_turn=self.engine.state.turn,
        )
        self.posts[post_id] = post
        self._touch_post(agent, post)
        self._log("post_created", post_id=post_id, author=agent, title=title)
        return post.to_dict()

    def comment_on_post(self, agent: str, post_id: str, text: str) -> dict:
        post = self.posts.get(post_id)
        if post is None:
            raise ValueError(f"Post '{post_id}' not found")
        if not text:
            raise ValueError("Comment text required")
        post.comments.append(PostComment(author=agent, text=text, created_turn=self.engine.state.turn))
        self._touch_post(agent, post)
        self._log("post_commented", post_id=post_id, author=agent)
        return post.to_dict()

    def record_decision(self, agent: str, post_id: str, text: str) -> dict:
        post = self.posts.get(post_id)
        if post is None:
            raise ValueError(f"Post '{post_id}' not found")
        if not text:
            raise ValueError("Decision text required")
        post.decision = text
        post.decided_by = agent
        post.decided_turn = self.engine.state.turn
        self._touch_post(agent, post)
        self._log("post_decision_recorded", post_id=post_id, decided_by=agent)
        return post.to_dict()

    def list_posts(self, function: str | None = None) -> list[dict]:
        # A summary list is a glance, NOT reading content — it must not clear the unread
        # counter, or a new comment on an unopened post would be silently marked read.
        return [p.to_summary() for p in self.posts.values()]

    def list_decisions(self, function: str | None = None) -> list[dict]:
        # The running log of every decision recorded on a Post, oldest first. Like
        # list_posts this is a glance — it must NOT mark anything seen, or it would
        # silently clear the unread gate. `function` is accepted for signature parity.
        decisions = [
            {
                "post_id": p.id,
                "title": p.title,
                "decision": p.decision,
                "decided_by": p.decided_by,
                "decided_turn": p.decided_turn,
            }
            for p in self.posts.values()
            if p.decision is not None
        ]
        decisions.sort(key=lambda d: (d["decided_turn"], d["post_id"]))
        return decisions

    def read_post(self, post_id: str, function: str | None = None) -> dict:
        post = self.posts.get(post_id)
        if post is None:
            raise ValueError(f"Post '{post_id}' not found")
        if function is not None:
            self._mark_post_seen(function, post)
        return post.to_dict()

    def _touch_post(self, agent: str, post: Post) -> None:
        """Stamp new activity on a post and mark it seen for the acting agent."""
        self._post_activity_seq += 1
        post.last_activity = self._post_activity_seq
        self._mark_post_seen(agent, post)

    def _mark_post_seen(self, function: str, post: Post) -> None:
        self._post_seen.setdefault(function, {})[post.id] = post.last_activity

    # -- Goals --

    def _native_progress(
        self,
        g: Goal,
        attain: GoalAttainmentScore,
    ) -> tuple[float, Any, float | None, str]:
        """Compute (progress, current_raw_value, target, status) for a native goal.

        Reuses the engine's own scoring (``compute_goal_attainment`` + ``_extract_metric``)
        so the surfaced numbers are consistent with the final score.
        """
        key = g.native_key
        state = self.engine.state
        if key == "mrr":
            return attain.mrr_score, attain.final_mrr, g.target, _status_from_progress(attain.mrr_score)
        if key == "churn":
            # churn_score's par is ZERO churn, not the target — so status must be judged
            # against the actual constraint (avg_churn_rate vs max_churn_rate = g.target),
            # or a healthy sub-target churn rate would wrongly read as at_risk/off_track.
            rate = attain.avg_churn_rate
            target = g.target if g.target is not None else 0.0
            if rate <= target:
                status = "on_track"
            elif rate <= 2 * target:
                status = "at_risk"
            else:
                status = "off_track"
            return attain.churn_score, rate, g.target, status
        if key == "runway":
            return attain.runway_score, attain.final_runway_turns, g.target, _status_from_progress(attain.runway_score)
        # Function sub-goal keyed by role. Guard _extract_metric: an unknown metric string
        # (e.g. a future/renamed scenario sub-goal) must not 500 the whole goals view.
        prog = attain.function_scores.get(key, 0.0)
        current: Any = None
        if g.metric:
            try:
                current = _extract_metric(state, g.metric)
            except ValueError:
                current = None
        return prog, current, g.target, _status_from_progress(prog)

    def _goal_to_dict(self, g: Goal, attain: GoalAttainmentScore | None = None) -> dict:
        d: dict[str, Any] = {
            "id": g.id,
            "title": g.title,
            "description": g.description,
            "owner": g.owner,
            "parent_id": g.parent_id,
            "native": g.native,
            "created_by": g.created_by,
            "updates": [u.to_dict() for u in g.updates],
        }
        if g.native:
            # Native progress needs the live attainment score; compute it lazily so callers
            # serializing only agent-authored goals (create/update) pay nothing.
            if attain is None:
                attain = compute_goal_attainment(self.engine.state, self.engine.scenario.primary_goal)
            prog, current, target, status = self._native_progress(g, attain)
            d["progress"] = round(prog, 4)
            d["status"] = status
            d["target"] = target
            d["current"] = round(current, 4) if isinstance(current, float) else current
            if g.metric:
                d["metric"] = g.metric
        else:
            d["progress"] = round(g.progress, 4)
            d["status"] = g.status
        return d

    def get_goals(self, function: str | None = None) -> list[dict]:
        if function is not None:
            self._seen_goal_activity[function] = self.goal_activity
        attain = compute_goal_attainment(self.engine.state, self.engine.scenario.primary_goal)
        return [self._goal_to_dict(g, attain) for g in self.goals.values()]

    def create_goal(
        self,
        agent: str,
        title: str,
        description: str = "",
        owner: str | None = None,
        parent_id: str | None = None,
    ) -> dict:
        if not title:
            raise ValueError("Goal title required")
        if parent_id is not None and parent_id not in self.goals:
            raise ValueError(f"Parent goal '{parent_id}' not found")
        if owner is not None and owner not in ALL_FUNCTIONS:
            raise ValueError(f"Unknown owner function '{owner}'. Valid: {sorted(ALL_FUNCTIONS)}")
        self._goal_id_seq += 1
        goal_id = f"G{self._goal_id_seq:02d}"
        goal = Goal(
            id=goal_id,
            title=title,
            description=description,
            owner=owner,
            parent_id=parent_id,
            status="on_track",
            progress=0.0,
            native=False,
            created_by=agent,
            created_turn=self.engine.state.turn,
        )
        self.goals[goal_id] = goal
        self._bump_goal_activity(agent)
        self._log("goal_created", goal_id=goal_id, created_by=agent, owner=owner, parent_id=parent_id)
        # Agent goals are always non-native, so no attainment computation is needed.
        return self._goal_to_dict(goal)

    def update_goal(
        self,
        agent: str,
        goal_id: str,
        status: str,
        progress: float,
        note: str = "",
    ) -> dict:
        goal = self.goals.get(goal_id)
        if goal is None:
            raise ValueError(f"Goal '{goal_id}' not found")
        if goal.native:
            raise ValueError(
                f"Goal '{goal_id}' is a native game goal — its progress is computed live from "
                "game state and cannot be written directly. Create your own goal to track work."
            )
        if status not in _GOAL_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Valid: {sorted(_GOAL_STATUSES)}")
        try:
            progress_val = float(progress)
        except (TypeError, ValueError):
            raise ValueError(f"progress must be a number (e.g. 0.5), got {progress!r}")
        goal.status = status
        goal.progress = progress_val
        goal.updates.append(
            GoalUpdate(
                author=agent,
                status=status,
                progress=progress_val,
                note=note,
                created_turn=self.engine.state.turn,
            )
        )
        self._bump_goal_activity(agent)
        self._log("goal_updated", goal_id=goal_id, author=agent, status=status)
        # The updated goal is agent-authored (non-native), so no attainment computation.
        return self._goal_to_dict(goal)

    def comment_on_goal(self, agent: str, goal_id: str, note: str) -> dict:
        """Attach a note to ANY goal (native or agent-created) without changing its progress.

        Native goals' progress/status are computed live from game state (``_native_progress``)
        and stay authoritative — a comment records the agent's plan/assessment alongside a
        snapshot of the current computed values, but never overwrites them. This is the goal
        analogue of ``comment_on_post`` and mirrors the product's goal comments; it lets the
        team coordinate on the seeded company/function objectives, which are otherwise read-only.
        """
        goal = self.goals.get(goal_id)
        if goal is None:
            raise ValueError(f"Goal '{goal_id}' not found")
        if not note or not note.strip():
            raise ValueError("Comment note required")
        if goal.native:
            attain = compute_goal_attainment(self.engine.state, self.engine.scenario.primary_goal)
            progress, _current, _target, status = self._native_progress(goal, attain)
        else:
            progress, status = goal.progress, goal.status
        goal.updates.append(
            GoalUpdate(
                author=agent,
                status=status,
                progress=round(progress, 4),
                note=note,
                created_turn=self.engine.state.turn,
            )
        )
        self._bump_goal_activity(agent)
        self._log("goal_commented", goal_id=goal_id, author=agent)
        return self._goal_to_dict(goal)

    def _bump_goal_activity(self, agent: str) -> None:
        self.goal_activity += 1
        self._seen_goal_activity[agent] = self.goal_activity

    # -- Soft unread summary + status --

    def get_unread_summary(self, function: str) -> dict:
        seen = self._post_seen.get(function, {})
        unread_posts = sum(1 for p in self.posts.values() if p.last_activity > seen.get(p.id, 0))
        return {
            "unread_posts": unread_posts,
            "unread_goal_updates": max(0, self.goal_activity - self._seen_goal_activity.get(function, 0)),
        }

    def get_status(self, function: str) -> dict:
        status = super().get_status(function)
        status["artifacts"] = self.get_unread_summary(function)
        return status

    def _check_unread_messages(self, function: str) -> dict | None:
        """C4b gate: 409 on unread *substrate* activity (Posts/comments/goal-updates), not chat.

        C4b has no chat, so the base cursor gate never fires. This mirrors the base gate's one-shot
        warn-then-allow (see condition3_orchestrator._check_unread_messages): block once per new
        activity batch, then let the retry through. That keeps the read-gate + turn-barrier from
        livelocking under concurrent authorship; run_watchdog is the final backstop. An agent clears
        the gate by opening each unread Post (./game post read <id>) and viewing Goals (./game goals).
        """
        summary = self.get_unread_summary(function)
        unread = summary["unread_posts"] + summary["unread_goal_updates"]
        if unread == 0:
            self._substrate_warned[function] = None
            return None

        # Monotonic activity watermark (writes bump these; reads do not, so the read-then-resubmit
        # path resolves cleanly). New activity from anyone pushes it past the warned level → re-warn.
        activity = self._post_activity_seq + self.goal_activity
        warned_at = self._substrate_warned.get(function)
        if warned_at is not None and activity <= warned_at:
            self._substrate_warned[function] = None
            self._log("unread_warning_resolved", function=function)
            return None

        self._substrate_warned[function] = activity
        self._log(
            "unread_warning_issued",
            function=function,
            unread_posts=summary["unread_posts"],
            unread_goal_updates=summary["unread_goal_updates"],
        )
        return {
            "status": "unread_messages",
            "unread_count": unread,
            "artifacts": summary,
            "message": (
                f"You have {summary['unread_posts']} unread Post update(s) and "
                f"{summary['unread_goal_updates']} unread Goal update(s). Read them "
                "(./game post read <id> for each new Post, ./game goals), then resubmit. "
                "Your actions will be accepted on the next submit attempt."
            ),
        }

    # -- Persistence --

    def write_posts_log(self) -> None:
        if not self._output_dir:
            return
        with open(self._output_dir / "posts.jsonl", "w", encoding="utf-8") as f:
            for post in self.posts.values():
                f.write(json.dumps(post.to_dict()) + "\n")

    def write_goals_log(self) -> None:
        if not self._output_dir:
            return
        attain = compute_goal_attainment(self.engine.state, self.engine.scenario.primary_goal)
        with open(self._output_dir / "goals.jsonl", "w", encoding="utf-8") as f:
            for goal in self.goals.values():
                f.write(json.dumps(self._goal_to_dict(goal, attain)) + "\n")


# ---------------------------------------------------------------------------
# C4a route registrar
# ---------------------------------------------------------------------------


def _register_channel_chat_routes(app: FastAPI) -> None:
    """Channel-aware chat + channel management, all under /agents/{fn}/... (plus global /chat)."""

    @app.get("/chat")
    async def read_chat_global(request: Request, since: int = 0, channel: str = Query(None)):
        o = _orch(request)
        messages = [m for m in o.chat_messages if m.seq > since]
        if channel is not None:
            messages = [m for m in messages if getattr(m, "channel", "everyone") == channel]
        return [m.to_dict() for m in messages]

    @app.get("/agents/{fn}/chat")
    async def read_chat(request: Request, fn: str, since: int = 0, channel: str = Query(None)):
        _require_agent(request, fn)
        return _orch(request).read_chat(fn, since, channel)

    @app.post("/agents/{fn}/chat")
    async def post_chat(request: Request, fn: str):
        _require_agent(request, fn)
        o = _orch(request)
        body = await request.json()
        message = body.get("message", "")
        channel = body.get("channel", "everyone")
        if not message:
            raise HTTPException(400, "Message required")
        try:
            seq = o.post_chat(fn, message, channel)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"seq": seq, "channel": channel}

    @app.get("/agents/{fn}/channels")
    async def list_channels(request: Request, fn: str):
        _require_agent(request, fn)
        return _orch(request).list_channels()

    @app.post("/agents/{fn}/channels")
    async def create_channel(request: Request, fn: str):
        _require_agent(request, fn)
        o = _orch(request)
        body = await request.json()
        name = body.get("name", "")
        description = body.get("description", "")
        if not name:
            raise HTTPException(400, "Channel name required")
        try:
            meta = o.create_channel(name, fn, description)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return meta


# ---------------------------------------------------------------------------
# C4b route registrars
# ---------------------------------------------------------------------------


def _register_post_routes(app: FastAPI) -> None:
    """Convictional Posts, all under /agents/{fn}/posts."""

    @app.get("/agents/{fn}/posts")
    async def list_posts(request: Request, fn: str):
        _require_agent(request, fn)
        return _orch(request).list_posts(fn)

    @app.get("/agents/{fn}/decisions")
    async def list_decisions(request: Request, fn: str):
        _require_agent(request, fn)
        return _orch(request).list_decisions(fn)

    @app.post("/agents/{fn}/posts")
    async def create_post(request: Request, fn: str):
        _require_agent(request, fn)
        o = _orch(request)
        body = await request.json()
        title = body.get("title", "")
        post_body = body.get("body", "")
        try:
            return o.create_post(fn, title, post_body)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/agents/{fn}/posts/{post_id}")
    async def read_post(request: Request, fn: str, post_id: str):
        _require_agent(request, fn)
        try:
            return _orch(request).read_post(post_id, fn)
        except ValueError as exc:
            raise HTTPException(404, str(exc))

    @app.post("/agents/{fn}/posts/{post_id}/comments")
    async def comment_on_post(request: Request, fn: str, post_id: str):
        _require_agent(request, fn)
        o = _orch(request)
        body = await request.json()
        text = body.get("text", "")
        try:
            return o.comment_on_post(fn, post_id, text)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/agents/{fn}/posts/{post_id}/decision")
    async def record_decision(request: Request, fn: str, post_id: str):
        _require_agent(request, fn)
        o = _orch(request)
        body = await request.json()
        text = body.get("text", "")
        try:
            return o.record_decision(fn, post_id, text)
        except ValueError as exc:
            raise HTTPException(400, str(exc))


def _register_goal_routes(app: FastAPI) -> None:
    """Shared Goals hierarchy, all under /agents/{fn}/goals."""

    @app.get("/agents/{fn}/goals")
    async def get_goals(request: Request, fn: str):
        _require_agent(request, fn)
        return _orch(request).get_goals(fn)

    @app.post("/agents/{fn}/goals")
    async def create_goal(request: Request, fn: str):
        _require_agent(request, fn)
        o = _orch(request)
        body = await request.json()
        title = body.get("title", "")
        description = body.get("description", "")
        owner = body.get("owner")
        parent_id = body.get("parent_id")
        try:
            return o.create_goal(fn, title, description, owner, parent_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/agents/{fn}/goals/{goal_id}/update")
    async def update_goal(request: Request, fn: str, goal_id: str):
        _require_agent(request, fn)
        o = _orch(request)
        body = await request.json()
        status = body.get("status", "")
        progress = body.get("progress", 0.0)
        note = body.get("note", "")
        try:
            return o.update_goal(fn, goal_id, status, progress, note)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/agents/{fn}/goals/{goal_id}/comment")
    async def comment_on_goal(request: Request, fn: str, goal_id: str):
        _require_agent(request, fn)
        o = _orch(request)
        body = await request.json()
        note = body.get("note", "")
        try:
            return o.comment_on_goal(fn, goal_id, note)
        except ValueError as exc:
            raise HTTPException(400, str(exc))


# ---------------------------------------------------------------------------
# FastAPI application factories
# ---------------------------------------------------------------------------


def create_c4a_app(orchestrator: ChannelOrchestrator) -> FastAPI:
    app = FastAPI(title="AlignSim Condition 4a Orchestrator (channels)")
    app.state.orchestrator = orchestrator
    _register_agent_routes(app)
    _register_channel_chat_routes(app)
    _register_orchestrator_routes(app)
    return app


def create_c4b_app(orchestrator: ConvictionalOrchestrator) -> FastAPI:
    app = FastAPI(title="AlignSim Condition 4b Orchestrator (convictional)")
    app.state.orchestrator = orchestrator
    # C4b (Phase 1): NO chat. All coordination flows through Posts + Goals; the submit gate 409s on
    # unread substrate activity (see ConvictionalOrchestrator._check_unread_messages).
    _register_agent_routes(app)
    _register_post_routes(app)
    _register_goal_routes(app)
    _register_orchestrator_routes(app)
    return app
