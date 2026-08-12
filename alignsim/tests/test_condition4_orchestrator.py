"""Integration tests for the Condition 4 orchestrators (C4a channels, C4b convictional).

Mirrors the C3 orchestrator test patterns (httpx AsyncClient + ASGITransport) and pins the
research-integrity invariants: the substrate is the treatment, and the reused no-leak filter
layer (condition3_filters) must behave identically through both C4 orchestrators.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from alignsim.src.engine.game import GameEngine
from alignsim.src.engine.scoring import compute_goal_attainment
from alignsim.src.harness import condition3_filters
from alignsim.src.harness.condition3_filters import filter_observation
from alignsim.src.harness.condition4_orchestrator import (
    ChannelChatMessage,
    ChannelOrchestrator,
    ConvictionalOrchestrator,
    create_c4a_app,
    create_c4b_app,
)
from alignsim.src.models.goals import GoalAttainmentScore
from alignsim.src.scenarios.seed_stage import create_seed_stage_scenario


def _make_engine() -> GameEngine:
    scenario = create_seed_stage_scenario(seed=42)
    scenario.max_turns = 12
    return GameEngine(scenario)


# ---------------------------------------------------------------------------
# Fixtures — C4a (channels)
# ---------------------------------------------------------------------------


@pytest.fixture
def c4a_orch() -> ChannelOrchestrator:
    orch = ChannelOrchestrator(_make_engine())
    orch._register_agent_unlocked("engineering")
    orch._register_agent_unlocked("sales")
    orch._register_agent_unlocked("marketing")
    return orch


@pytest.fixture
def c4a_client(c4a_orch: ChannelOrchestrator) -> AsyncClient:
    app = create_c4a_app(c4a_orch)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Fixtures — C4b (convictional)
# ---------------------------------------------------------------------------


@pytest.fixture
def c4b_orch() -> ConvictionalOrchestrator:
    orch = ConvictionalOrchestrator(_make_engine())
    orch._register_agent_unlocked("engineering")
    orch._register_agent_unlocked("sales")
    orch._register_agent_unlocked("marketing")
    return orch


@pytest.fixture
def c4b_client(c4b_orch: ConvictionalOrchestrator) -> AsyncClient:
    app = create_c4b_app(c4b_orch)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ===========================================================================
# C4a — Channels
# ===========================================================================


class TestC4aChannels:
    def test_default_channels_seeded(self, c4a_orch: ChannelOrchestrator):
        # everyone + one channel per starting function (derived from STARTING_FUNCTIONS).
        assert c4a_orch.channels["everyone"].created_by == "system"
        assert set(c4a_orch.channels) == {"everyone", "engineering", "sales", "marketing"}

    @pytest.mark.anyio
    async def test_list_channels(self, c4a_client: AsyncClient):
        r = await c4a_client.get("/agents/engineering/channels")
        assert r.status_code == 200
        names = {c["name"] for c in r.json()}
        assert names == {"everyone", "sales", "engineering", "marketing"}

    @pytest.mark.anyio
    async def test_post_defaults_to_everyone(self, c4a_client: AsyncClient):
        r = await c4a_client.post("/agents/engineering/chat", json={"message": "hi team"})
        assert r.status_code == 200
        assert r.json()["channel"] == "everyone"

        r = await c4a_client.get("/chat", params={"since": 0})
        msgs = r.json()
        assert len(msgs) == 1
        assert msgs[0]["channel"] == "everyone"

    @pytest.mark.anyio
    async def test_post_to_named_channel(self, c4a_client: AsyncClient):
        r = await c4a_client.post(
            "/agents/engineering/chat", json={"message": "build note", "channel": "engineering"})
        assert r.status_code == 200
        assert r.json()["channel"] == "engineering"

    @pytest.mark.anyio
    async def test_channel_filter_on_read(self, c4a_client: AsyncClient):
        await c4a_client.post("/agents/engineering/chat", json={"message": "eng msg", "channel": "engineering"})
        await c4a_client.post("/agents/sales/chat", json={"message": "sales msg", "channel": "sales"})

        r = await c4a_client.get("/agents/marketing/chat", params={"since": 0, "channel": "engineering"})
        msgs = r.json()
        assert len(msgs) == 1
        assert msgs[0]["message"] == "eng msg"
        assert msgs[0]["channel"] == "engineering"

    @pytest.mark.anyio
    async def test_unknown_channel_rejected(self, c4a_client: AsyncClient):
        r = await c4a_client.post(
            "/agents/engineering/chat", json={"message": "x", "channel": "nonexistent"})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_create_channel(self, c4a_client: AsyncClient, c4a_orch: ChannelOrchestrator):
        r = await c4a_client.post("/agents/sales/channels", json={"name": "customer-c01"})
        assert r.status_code == 200
        assert "customer-c01" in c4a_orch.channels
        assert c4a_orch.channels["customer-c01"].created_by == "sales"
        # Can now post to it.
        r = await c4a_client.post(
            "/agents/sales/chat", json={"message": "deal notes", "channel": "customer-c01"})
        assert r.status_code == 200

    @pytest.mark.anyio
    async def test_create_channel_duplicate_rejected(self, c4a_client: AsyncClient):
        r = await c4a_client.post("/agents/sales/channels", json={"name": "everyone"})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_create_channel_invalid_slug_rejected(self, c4a_client: AsyncClient):
        r = await c4a_client.post("/agents/sales/channels", json={"name": "Bad Name!"})
        assert r.status_code == 400

    def test_channel_message_to_dict_superset_of_c3(self):
        """ChannelChatMessage.to_dict is C3's format plus a channel key."""
        m = ChannelChatMessage(seq=1, turn=1, agent="sales", ts="t", message="hi", channel="sales")
        d = m.to_dict()
        for key in ("seq", "turn", "agent", "ts", "message"):
            assert key in d
        assert d["channel"] == "sales"

    @pytest.mark.anyio
    async def test_unread_gate_fires_across_channels(self, c4a_client: AsyncClient):
        """A message in ANY channel blocks submit with 409 (must read everything)."""
        await c4a_client.post(
            "/agents/sales/chat", json={"message": "need input", "channel": "sales"})
        r = await c4a_client.post("/agents/engineering/submit", json={"actions": []})
        assert r.status_code == 409
        assert r.json()["unread_count"] == 1

    @pytest.mark.anyio
    async def test_channel_read_does_not_clear_gate_for_other_channels(self, c4a_client: AsyncClient):
        """Reading ONE channel must not clear the cross-channel unread 409 gate.

        Regression guard: a channel-filtered read used to advance the global cursor to that
        channel's max seq, silently clearing unread messages in other channels.
        """
        await c4a_client.post("/agents/sales/chat", json={"message": "in sales", "channel": "sales"})
        await c4a_client.post("/agents/marketing/chat", json={"message": "in eng", "channel": "engineering"})
        # Engineering reads ONLY the engineering channel (sees the higher-seq message there)...
        await c4a_client.get("/agents/engineering/chat", params={"since": 0, "channel": "engineering"})
        # ...but the lower-seq #sales message is still unread, so submit is still gated.
        r = await c4a_client.post("/agents/engineering/submit", json={"actions": []})
        assert r.status_code == 409

    @pytest.mark.anyio
    async def test_full_read_clears_gate(self, c4a_client: AsyncClient):
        """A full read (no channel filter) does advance the cursor and clear the gate."""
        await c4a_client.post("/agents/sales/chat", json={"message": "in sales", "channel": "sales"})
        await c4a_client.get("/agents/engineering/chat", params={"since": 0})
        await c4a_client.get("/agents/marketing/chat", params={"since": 0})
        results = await asyncio.gather(
            c4a_client.post("/agents/engineering/submit", json={"actions": []}),
            c4a_client.post("/agents/sales/submit", json={"actions": []}),
            c4a_client.post("/agents/marketing/submit", json={"actions": []}),
        )
        assert results[0].status_code == 200


# ===========================================================================
# C4b — Posts
# ===========================================================================


class TestC4bPosts:
    @pytest.mark.anyio
    async def test_create_list_read(self, c4b_client: AsyncClient):
        r = await c4b_client.post(
            "/agents/engineering/posts", json={"title": "Hire CS?", "body": "churn showing up"})
        assert r.status_code == 200
        post_id = r.json()["id"]

        r = await c4b_client.get("/agents/sales/posts")
        summaries = r.json()
        assert len(summaries) == 1
        assert summaries[0]["id"] == post_id
        assert summaries[0]["title"] == "Hire CS?"

        r = await c4b_client.get(f"/agents/sales/posts/{post_id}")
        full = r.json()
        assert full["body"] == "churn showing up"
        assert full["author"] == "engineering"
        assert full["comments"] == []
        assert full["decision"] is None

    @pytest.mark.anyio
    async def test_comment_threading(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/engineering/posts", json={"title": "T", "body": "B"})
        post_id = r.json()["id"]

        await c4b_client.post(f"/agents/sales/posts/{post_id}/comments", json={"text": "I agree"})
        await c4b_client.post(f"/agents/marketing/posts/{post_id}/comments", json={"text": "me too"})

        r = await c4b_client.get(f"/agents/engineering/posts/{post_id}")
        comments = r.json()["comments"]
        assert [c["author"] for c in comments] == ["sales", "marketing"]
        assert comments[0]["text"] == "I agree"

    @pytest.mark.anyio
    async def test_record_decision(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/engineering/posts", json={"title": "T", "body": "B"})
        post_id = r.json()["id"]

        r = await c4b_client.post(
            f"/agents/sales/posts/{post_id}/decision", json={"text": "Sales cross-hires CS"})
        assert r.status_code == 200
        data = r.json()
        assert data["decision"] == "Sales cross-hires CS"
        assert data["decided_by"] == "sales"

    @pytest.mark.anyio
    async def test_comment_on_missing_post_400(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/sales/posts/P99/comments", json={"text": "x"})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_read_missing_post_404(self, c4b_client: AsyncClient):
        r = await c4b_client.get("/agents/sales/posts/P99")
        assert r.status_code == 404

    @pytest.mark.anyio
    async def test_posts_gate_submit(self, c4b_client: AsyncClient):
        """An unread Post hard-gates submit (409). Reading it clears the gate; once every
        agent is caught up the turn resolves. (C4b has no chat — Posts/Goals are the gate.)"""
        r = await c4b_client.post("/agents/sales/posts", json={"title": "T", "body": "B"})
        post_id = r.json()["id"]
        # sales authored it (0 unread); engineering has it unread → its submit is gated.
        r = await c4b_client.post("/agents/engineering/submit", json={"actions": []})
        assert r.status_code == 409
        assert r.json()["artifacts"]["unread_posts"] == 1
        # Everyone reads the post to clear their gate, then all submit → the turn resolves.
        await c4b_client.get(f"/agents/engineering/posts/{post_id}")
        await c4b_client.get(f"/agents/marketing/posts/{post_id}")
        results = await asyncio.gather(
            c4b_client.post("/agents/engineering/submit", json={"actions": []}),
            c4b_client.post("/agents/sales/submit", json={"actions": []}),
            c4b_client.post("/agents/marketing/submit", json={"actions": []}),
        )
        for r in results:
            assert r.status_code == 200

    @pytest.mark.anyio
    async def test_unread_summary_in_status(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/sales/posts", json={"title": "T", "body": "B"})
        post_id = r.json()["id"]
        r = await c4b_client.get("/agents/engineering/status")
        assert r.json()["artifacts"]["unread_posts"] == 1
        # A summary LIST is a glance — it must NOT clear the unread counter.
        await c4b_client.get("/agents/engineering/posts")
        r = await c4b_client.get("/agents/engineering/status")
        assert r.json()["artifacts"]["unread_posts"] == 1
        # Reading the post's full content clears it.
        await c4b_client.get(f"/agents/engineering/posts/{post_id}")
        r = await c4b_client.get("/agents/engineering/status")
        assert r.json()["artifacts"]["unread_posts"] == 0

    @pytest.mark.anyio
    async def test_unread_posts_tracked_per_post(self, c4b_client: AsyncClient):
        """A new comment on an unopened post re-raises the unread count (per-post granularity)."""
        r = await c4b_client.post("/agents/sales/posts", json={"title": "T", "body": "B"})
        post_id = r.json()["id"]
        await c4b_client.get(f"/agents/engineering/posts/{post_id}")  # eng reads it → 0 unread
        assert (await c4b_client.get("/agents/engineering/status")).json()["artifacts"]["unread_posts"] == 0
        # sales comments on the already-read post → eng has 1 unread again.
        await c4b_client.post(f"/agents/sales/posts/{post_id}/comments", json={"text": "new info"})
        assert (await c4b_client.get("/agents/engineering/status")).json()["artifacts"]["unread_posts"] == 1

    def test_gate_warns_once_then_allows(self, c4b_orch: ConvictionalOrchestrator):
        """Warn-then-allow: an unread Post gates the first submit check, then lets the retry
        through even without a read — the anti-livelock nudge (mirrors the chat gate)."""
        c4b_orch.create_post("sales", "T", "B")
        assert c4b_orch._check_unread_messages("engineering") is not None  # first: warned (409)
        assert c4b_orch._check_unread_messages("engineering") is None      # retry: allowed


class TestC4bDecisionLog:
    """The read-only decision-log tool (`./game decisions` → GET /agents/{fn}/decisions):
    an aggregated view of every decision recorded on a Post, oldest first."""

    @pytest.mark.anyio
    async def test_lists_recorded_decisions_only(self, c4b_client: AsyncClient):
        r = await c4b_client.post(
            "/agents/engineering/posts", json={"title": "Hire CS?", "body": "churn"})
        decided_id = r.json()["id"]
        await c4b_client.post(
            f"/agents/sales/posts/{decided_id}/decision", json={"text": "Sales cross-hires CS"})
        # A second post with no decision must be excluded from the log.
        await c4b_client.post("/agents/marketing/posts", json={"title": "No decision", "body": "b"})

        log = (await c4b_client.get("/agents/engineering/decisions")).json()
        assert len(log) == 1
        d = log[0]
        assert d["post_id"] == decided_id
        assert d["title"] == "Hire CS?"
        assert d["decision"] == "Sales cross-hires CS"
        assert d["decided_by"] == "sales"
        assert d["decided_turn"] is not None

    @pytest.mark.anyio
    async def test_decision_log_is_read_only(self, c4b_client: AsyncClient):
        """Like list_posts, the log is a glance — it must NOT clear the unread gate."""
        r = await c4b_client.post("/agents/sales/posts", json={"title": "T", "body": "B"})
        post_id = r.json()["id"]
        await c4b_client.post(f"/agents/sales/posts/{post_id}/decision", json={"text": "settled"})
        # engineering has unread activity from the post + decision.
        assert (await c4b_client.get("/agents/engineering/status")).json()["artifacts"]["unread_posts"] == 1
        # Viewing the decision log does not clear it (only reading the Post's content does).
        await c4b_client.get("/agents/engineering/decisions")
        assert (await c4b_client.get("/agents/engineering/status")).json()["artifacts"]["unread_posts"] == 1

    def test_ordered_by_decided_turn(self, c4b_orch: ConvictionalOrchestrator):
        """Log is oldest-first by decided_turn, independent of post creation order."""
        a = c4b_orch.create_post("sales", "A", "b")["id"]
        b = c4b_orch.create_post("sales", "B", "b")["id"]
        c4b_orch.engine.state.turn = 5
        c4b_orch.record_decision("sales", a, "decided later")
        c4b_orch.engine.state.turn = 2
        c4b_orch.record_decision("sales", b, "decided earlier")
        log = c4b_orch.list_decisions()
        assert [d["post_id"] for d in log] == [b, a]
        assert [d["decided_turn"] for d in log] == [2, 5]


class TestC4bHasNoChat:
    """C4b (Phase 1) removes chat entirely — coordination is Posts + Goals only, and the
    submit gate fires on unread substrate activity instead of chat."""

    @pytest.mark.anyio
    async def test_chat_send_route_absent(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/engineering/chat", json={"message": "hi"})
        assert r.status_code == 404

    @pytest.mark.anyio
    async def test_chat_read_routes_absent(self, c4b_client: AsyncClient):
        assert (await c4b_client.get("/chat", params={"since": 0})).status_code == 404
        assert (await c4b_client.get("/agents/engineering/chat", params={"since": 0})).status_code == 404

    @pytest.mark.anyio
    async def test_goal_update_gates_submit(self, c4b_client: AsyncClient):
        """Unread goal activity hard-gates submit (409), mirroring the Post gate."""
        await c4b_client.post("/agents/sales/goals", json={
            "title": "Assign a CS owner", "description": "d",
            "parent_id": "NG-support", "owner": "support"})
        r = await c4b_client.post("/agents/engineering/submit", json={"actions": []})
        assert r.status_code == 409
        assert r.json()["artifacts"]["unread_goal_updates"] >= 1


# ===========================================================================
# C4b — Goals
# ===========================================================================


class TestC4bGoals:
    @pytest.mark.anyio
    async def test_native_tree_seeded(self, c4b_client: AsyncClient):
        r = await c4b_client.get("/agents/engineering/goals")
        goals = {g["id"]: g for g in r.json()}
        # Flat: 8 native goals, all top-level (no parent nesting — nesting under a single
        # constraint would wrongly imply function goals ladder to it over the others).
        assert len(goals) == 8
        assert all(g["parent_id"] is None for g in goals.values())
        assert all(g["native"] for g in goals.values())
        # The 3 company constraints are shared → unowned.
        for gid in ("NG-mrr", "NG-churn", "NG-runway"):
            assert goals[gid]["owner"] is None
        # The 3 starting functions own their own function goals.
        assert goals["NG-engineering"]["owner"] == "engineering"
        assert goals["NG-sales"]["owner"] == "sales"
        assert goals["NG-marketing"]["owner"] == "marketing"
        # CS + Ops don't exist yet → their function goals start unowned.
        assert goals["NG-support"]["owner"] is None
        assert goals["NG-ops"]["owner"] is None

    def test_native_progress_matches_engine_scoring(self, c4b_orch: ConvictionalOrchestrator):
        """Native progress is the engine's own attainment score, not a recompute."""
        goals = {g["id"]: g for g in c4b_orch.get_goals()}
        attain = compute_goal_attainment(
            c4b_orch.engine.state, c4b_orch.engine.scenario.primary_goal)
        assert goals["NG-mrr"]["progress"] == round(attain.mrr_score, 4)
        assert goals["NG-churn"]["progress"] == round(attain.churn_score, 4)
        assert goals["NG-runway"]["progress"] == round(attain.runway_score, 4)
        assert goals["NG-engineering"]["progress"] == round(
            attain.function_scores["engineering"], 4)

    def test_mrr_progress_is_live(self, c4b_orch: ConvictionalOrchestrator):
        """Mutating MRR is reflected in native goal progress (progress = mrr/target)."""
        target = c4b_orch.engine.scenario.primary_goal.mrr_target
        c4b_orch.engine.state.resources.mrr = target // 2
        mrr_goal = {g["id"]: g for g in c4b_orch.get_goals()}["NG-mrr"]
        assert mrr_goal["progress"] == pytest.approx(0.5, abs=1e-4)
        assert mrr_goal["current"] == target // 2
        assert mrr_goal["target"] == float(target)

    def test_churn_score_at_turn_one(self, c4b_orch: ConvictionalOrchestrator):
        """No churn yet → churn score is 1.0 (max(0, 1 - 0))."""
        churn_goal = {g["id"]: g for g in c4b_orch.get_goals()}["NG-churn"]
        assert churn_goal["progress"] == 1.0

    def test_churn_status_judged_against_target_not_zero_churn(
        self, c4b_orch: ConvictionalOrchestrator):
        """Churn status compares avg_churn_rate to the target, not churn_score to 1.0.

        A sub-target churn rate (e.g. 1% vs a 2% target) must read on_track even though its
        churn_score (0.99) is below 1.0.
        """
        churn_goal = c4b_orch.goals["NG-churn"]
        target = churn_goal.target  # max_churn_rate, e.g. 0.02

        def status_for(rate: float) -> str:
            attain = GoalAttainmentScore(
                mrr_score=0.0, churn_score=max(0.0, 1 - rate), runway_score=1.0,
                composite=0.0, avg_churn_rate=rate,
            )
            return c4b_orch._native_progress(churn_goal, attain)[3]

        assert status_for(target / 2) == "on_track"     # well under target
        assert status_for(target) == "on_track"         # exactly at target
        assert status_for(target * 1.5) == "at_risk"    # over target, under 2x
        assert status_for(target * 3) == "off_track"    # well over target

    @pytest.mark.anyio
    async def test_create_sub_goal_with_parent_and_owner(
        self, c4b_client: AsyncClient, c4b_orch: ConvictionalOrchestrator):
        r = await c4b_client.post("/agents/sales/goals", json={
            "title": "Assign a CS owner",
            "description": "Give the support goal an owner",
            "parent_id": "NG-support",
            "owner": "support",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["parent_id"] == "NG-support"
        assert data["owner"] == "support"
        assert data["native"] is False
        assert data["created_by"] == "sales"
        assert data["id"] in c4b_orch.goals

    @pytest.mark.anyio
    async def test_create_goal_bad_parent_400(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/sales/goals", json={"title": "X", "parent_id": "NOPE"})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_create_goal_bad_owner_400(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/sales/goals", json={"title": "X", "owner": "hr"})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_update_goal_appends_history(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/sales/goals", json={"title": "My goal"})
        goal_id = r.json()["id"]

        r = await c4b_client.post(
            f"/agents/sales/goals/{goal_id}/update",
            json={"status": "at_risk", "progress": 0.3, "note": "slow start"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "at_risk"
        assert data["progress"] == 0.3
        assert len(data["updates"]) == 1
        assert data["updates"][0]["note"] == "slow start"

    @pytest.mark.anyio
    async def test_update_goal_bad_status_400(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/sales/goals", json={"title": "My goal"})
        goal_id = r.json()["id"]
        r = await c4b_client.post(
            f"/agents/sales/goals/{goal_id}/update", json={"status": "great", "progress": 1.0})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_native_goal_rejects_direct_write(self, c4b_client: AsyncClient):
        r = await c4b_client.post(
            "/agents/sales/goals/NG-mrr/update", json={"status": "on_track", "progress": 5.0})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_update_missing_goal_400(self, c4b_client: AsyncClient):
        r = await c4b_client.post(
            "/agents/sales/goals/G99/update", json={"status": "on_track", "progress": 1.0})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_update_non_numeric_progress_400(self, c4b_client: AsyncClient):
        """Non-numeric progress is rejected server-side with a clean 400, not a crash."""
        r = await c4b_client.post("/agents/sales/goals", json={"title": "My goal"})
        goal_id = r.json()["id"]
        r = await c4b_client.post(
            f"/agents/sales/goals/{goal_id}/update", json={"status": "on_track", "progress": "50%"})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_update_progress_accepts_numeric_string(self, c4b_client: AsyncClient):
        """The CLI sends progress as a string; the server coerces it to float."""
        r = await c4b_client.post("/agents/sales/goals", json={"title": "My goal"})
        goal_id = r.json()["id"]
        r = await c4b_client.post(
            f"/agents/sales/goals/{goal_id}/update", json={"status": "on_track", "progress": "0.75"})
        assert r.status_code == 200
        assert r.json()["progress"] == 0.75

    @pytest.mark.anyio
    async def test_comment_on_native_goal_appends_note_without_touching_progress(
        self, c4b_client: AsyncClient, c4b_orch: ConvictionalOrchestrator):
        """A comment on a native goal records a note but leaves computed progress untouched."""
        before = {g["id"]: g for g in c4b_orch.get_goals()}["NG-mrr"]["progress"]
        r = await c4b_client.post(
            "/agents/marketing/goals/NG-mrr/comment",
            json={"note": "plan: accelerate C04/C06 to close the MRR gap"})
        assert r.status_code == 200
        data = r.json()
        assert data["native"] is True
        # Progress is still the engine-computed value — the comment did NOT overwrite it.
        attain = compute_goal_attainment(
            c4b_orch.engine.state, c4b_orch.engine.scenario.primary_goal)
        assert data["progress"] == round(attain.mrr_score, 4) == before
        # The note is on the goal's history, authored by the commenter.
        assert len(data["updates"]) == 1
        assert data["updates"][0]["author"] == "marketing"
        assert data["updates"][0]["note"] == "plan: accelerate C04/C06 to close the MRR gap"

    @pytest.mark.anyio
    async def test_comment_allowed_on_any_goal_regardless_of_owner(self, c4b_client: AsyncClient):
        """No owner restriction on comments — cross-functional coordination is the point."""
        r = await c4b_client.post(
            "/agents/marketing/goals/NG-sales/comment", json={"note": "leads for F03 incoming"})
        assert r.status_code == 200
        assert r.json()["updates"][-1]["author"] == "marketing"

    def test_comment_bumps_goal_activity_for_others(self, c4b_orch: ConvictionalOrchestrator):
        """A comment on a shared goal is a coordination signal — it marks others' goal feed unread."""
        before = c4b_orch.get_unread_summary("engineering")["unread_goal_updates"]
        c4b_orch.comment_on_goal("marketing", "NG-mrr", "watching the MRR gap")
        after = c4b_orch.get_unread_summary("engineering")["unread_goal_updates"]
        assert after == before + 1

    @pytest.mark.anyio
    async def test_comment_empty_note_400(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/sales/goals/NG-mrr/comment", json={"note": "   "})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_comment_missing_goal_400(self, c4b_client: AsyncClient):
        r = await c4b_client.post("/agents/sales/goals/G99/comment", json={"note": "hi"})
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_comment_on_created_goal_preserves_tracked_state(self, c4b_client: AsyncClient):
        """Commenting on an agent-created goal adds a note without altering its status/progress."""
        r = await c4b_client.post("/agents/sales/goals", json={"title": "My goal"})
        goal_id = r.json()["id"]
        await c4b_client.post(
            f"/agents/sales/goals/{goal_id}/update",
            json={"status": "at_risk", "progress": 0.3, "note": "slow"})
        r = await c4b_client.post(
            f"/agents/sales/goals/{goal_id}/comment", json={"note": "still working it"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "at_risk"
        assert data["progress"] == 0.3
        assert len(data["updates"]) == 2
        assert data["updates"][-1]["note"] == "still working it"


# ===========================================================================
# Research-integrity: only native game goals are seeded (no coaching)
# ===========================================================================


class TestGoalSeedingIntegrity:
    _FORBIDDEN = (
        "full-function coverage", "cross-functional intelligence", "ops_analysis",
        "hire cs early", "stand up cs", "stand up ops", "orphan",
    )

    def test_seed_contains_only_native_game_goals(self, c4b_orch: ConvictionalOrchestrator):
        goal = c4b_orch.engine.scenario.primary_goal
        native = [g for g in c4b_orch.goals.values() if g.native]
        # Exactly the 3 primary constraints + the scenario's function sub-goals.
        assert len(native) == 3 + len(goal.sub_goals)
        # No agent-authored goals exist at seed time.
        assert all(g.native for g in c4b_orch.goals.values())
        keys = {g.native_key for g in native}
        assert keys == {"mrr", "churn", "runway"} | {sg.role for sg in goal.sub_goals}

    def test_native_goal_text_is_derived_from_scenario(self, c4b_orch: ConvictionalOrchestrator):
        """Function-goal titles come from the scenario's RoleSubGoal.description verbatim."""
        goal = c4b_orch.engine.scenario.primary_goal
        for sg in goal.sub_goals:
            assert c4b_orch.goals[f"NG-{sg.role}"].title == sg.description

    def test_no_coaching_language_in_seeded_goals(self, c4b_orch: ConvictionalOrchestrator):
        for g in c4b_orch.goals.values():
            text = f"{g.title} {g.description}".lower()
            for phrase in self._FORBIDDEN:
                assert phrase not in text, f"Coaching phrase '{phrase}' leaked into goal {g.id}"


# ===========================================================================
# No-leak invariant: the reused condition3_filters layer behaves identically
# ===========================================================================


class TestNoLeakThroughC4:
    @pytest.mark.anyio
    @pytest.mark.parametrize("factory", ["c4a", "c4b"])
    async def test_observation_filtering_identical_to_c3(self, factory: str):
        engine = _make_engine()
        if factory == "c4a":
            orch = ChannelOrchestrator(engine)
            app = create_c4a_app(orch)
        else:
            orch = ConvictionalOrchestrator(engine)
            app = create_c4b_app(orch)
        orch._register_agent_unlocked("engineering")
        orch._register_agent_unlocked("sales")
        orch._register_agent_unlocked("marketing")

        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        r = await client.get("/agents/engineering/observe")
        obs = r.json()
        # Same partition as C3: engineering sees global + product_eng, never sales/cs.
        assert "product_eng" in obs
        assert "sales" not in obs
        assert "cs" not in obs
        # And it equals what the shared filter layer produces directly (not forked).
        expected = filter_observation(orch._current_obs, "engineering", engine)
        assert obs == expected

    @pytest.mark.anyio
    @pytest.mark.parametrize("factory", ["c4a", "c4b"])
    async def test_function_restricted_action_rejected(self, factory: str):
        engine = _make_engine()
        if factory == "c4a":
            orch = ChannelOrchestrator(engine)
            app = create_c4a_app(orch)
        else:
            orch = ConvictionalOrchestrator(engine)
            app = create_c4b_app(orch)
        for fn in ("engineering", "sales", "marketing"):
            orch._register_agent_unlocked(fn)
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

        async def submit_eng():
            return await client.post("/agents/engineering/submit", json={
                "actions": [{"action_type": "sell", "customer_id": "C01",
                             "sell_action": "outbound", "capacity": 2}]})

        eng_r, _, _ = await asyncio.gather(
            submit_eng(),
            client.post("/agents/sales/submit", json={"actions": []}),
            client.post("/agents/marketing/submit", json={"actions": []}),
        )
        data = eng_r.json()
        assert "function_rejections" in data
        assert len(data["function_rejections"]) == 1

    def test_filters_module_is_imported_not_forked(self):
        """C4 reuses condition3_filters verbatim — it does not define its own filter functions."""
        import alignsim.src.harness.condition4_orchestrator as c4
        assert not hasattr(c4, "filter_observation") or c4.filter_observation is filter_observation
        # The base Orchestrator's observation path is inherited unchanged.
        assert ChannelOrchestrator.get_filtered_observation is ConvictionalOrchestrator.get_filtered_observation

    def test_analyses_received_never_leaks_via_c4(self):
        """C4 analog of the C3 no-leak invariant: analyses go to the requester only."""
        for fn, perms in condition3_filters.FUNCTION_PERMISSIONS.items():
            assert "analyses_received_this_turn" not in perms.obs_sections


# ===========================================================================
# Persistence
# ===========================================================================


class TestPersistence:
    def test_c4a_chat_log_includes_channel(self, tmp_path):
        orch = ChannelOrchestrator(_make_engine(), output_dir=str(tmp_path))
        orch._register_agent_unlocked("engineering")
        orch.post_chat("engineering", "hi", channel="engineering")
        orch.post_chat("engineering", "yo", channel="everyone")
        orch.write_chat_log()

        lines = (tmp_path / "chat_log.jsonl").read_text().splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert records[0]["channel"] == "engineering"
        assert records[1]["channel"] == "everyone"

    def test_c4b_persists_posts_and_goals(self, tmp_path):
        orch = ConvictionalOrchestrator(_make_engine(), output_dir=str(tmp_path))
        orch._register_agent_unlocked("engineering")
        orch.create_post("engineering", "Hire CS?", "churn")
        orch.create_goal("engineering", "My goal", "desc", owner="support", parent_id="NG-support")
        orch.write_posts_log()
        orch.write_goals_log()

        post_lines = (tmp_path / "posts.jsonl").read_text().splitlines()
        assert len(post_lines) == 1
        assert json.loads(post_lines[0])["title"] == "Hire CS?"

        goal_lines = [json.loads(line) for line in (tmp_path / "goals.jsonl").read_text().splitlines()]
        # 3 primary + 5 function native + 1 agent goal.
        assert len(goal_lines) == 9
        agent_goals = [g for g in goal_lines if not g["native"]]
        assert len(agent_goals) == 1
        assert agent_goals[0]["owner"] == "support"
