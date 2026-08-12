"""Integration tests for the Condition 3 orchestrator server."""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from alignsim.src.engine.game import GameEngine
from alignsim.src.harness.condition3_orchestrator import Orchestrator, create_app
from alignsim.src.scenarios.seed_stage import create_seed_stage_scenario


@pytest.fixture
def engine() -> GameEngine:
    scenario = create_seed_stage_scenario(seed=42)
    scenario.max_turns = 12
    return GameEngine(scenario)


@pytest.fixture
def orchestrator(engine: GameEngine) -> Orchestrator:
    orch = Orchestrator(engine)
    orch._register_agent_unlocked("engineering")
    orch._register_agent_unlocked("sales")
    orch._register_agent_unlocked("marketing")
    return orch


@pytest.fixture
def client(orchestrator: Orchestrator) -> AsyncClient:
    app = create_app(orchestrator)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Basic endpoint tests
# ---------------------------------------------------------------------------


class TestHealthAndStatus:
    @pytest.mark.anyio
    async def test_health(self, client: AsyncClient):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    @pytest.mark.anyio
    async def test_orchestrator_status(self, client: AsyncClient):
        r = await client.get("/orchestrator/status")
        assert r.status_code == 200
        data = r.json()
        assert set(data["active_agents"]) == {"engineering", "sales", "marketing"}
        assert data["turn"] == 1
        assert data["game_over"] is False

    @pytest.mark.anyio
    async def test_agent_status(self, client: AsyncClient):
        r = await client.get("/agents/engineering/status")
        assert r.status_code == 200
        data = r.json()
        assert data["function"] == "engineering"
        assert data["turn"] == 1

    @pytest.mark.anyio
    async def test_unregistered_agent_404(self, client: AsyncClient):
        r = await client.get("/agents/support/observe")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Observation filtering via HTTP
# ---------------------------------------------------------------------------


class TestObserve:
    @pytest.mark.anyio
    async def test_engineering_observe(self, client: AsyncClient):
        r = await client.get("/agents/engineering/observe")
        assert r.status_code == 200
        data = r.json()
        assert "global" in data
        assert "product_eng" in data
        assert "sales" not in data
        assert "cs" not in data

    @pytest.mark.anyio
    async def test_sales_observe(self, client: AsyncClient):
        r = await client.get("/agents/sales/observe")
        data = r.json()
        assert "sales" in data
        assert "product_eng" not in data

    @pytest.mark.anyio
    async def test_marketing_observe(self, client: AsyncClient):
        r = await client.get("/agents/marketing/observe")
        data = r.json()
        assert "marketing_history" in data
        assert "sales" not in data


# ---------------------------------------------------------------------------
# Query / compute restrictions
# ---------------------------------------------------------------------------


class TestQueryRestrictions:
    @pytest.mark.anyio
    async def test_engineering_can_query_feature(self, client: AsyncClient):
        r = await client.get("/agents/engineering/query/feature", params={"id": "F01"})
        assert r.status_code == 200

    @pytest.mark.anyio
    async def test_engineering_cannot_query_customer(self, client: AsyncClient):
        r = await client.get("/agents/engineering/query/customer", params={"id": "C01"})
        assert r.status_code == 403

    @pytest.mark.anyio
    async def test_sales_can_query_customer(self, client: AsyncClient):
        r = await client.get("/agents/sales/query/customer", params={"id": "C01"})
        assert r.status_code == 200

    @pytest.mark.anyio
    async def test_sales_cannot_query_feature(self, client: AsyncClient):
        r = await client.get("/agents/sales/query/feature", params={"id": "F01"})
        assert r.status_code == 403

    @pytest.mark.anyio
    async def test_marketing_can_only_query_rejections(self, client: AsyncClient):
        r = await client.get("/agents/marketing/query/rejections")
        assert r.status_code == 200

        r = await client.get("/agents/marketing/query/customer", params={"id": "C01"})
        assert r.status_code == 403

    @pytest.mark.anyio
    async def test_support_customer_query_hides_sales_fields(
        self, client: AsyncClient, orchestrator: Orchestrator,
    ):
        """Support sees customer health but not sales-specific fields."""
        await orchestrator.register_agent("support")
        r = await client.get("/agents/support/query/customer", params={"id": "C01"})
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert "segment" in data
        for hidden in ("known_needs", "deal_value", "competitive_pressure",
                       "timeline_remaining", "min_sell_capacity"):
            assert hidden not in data

    @pytest.mark.anyio
    async def test_sales_customer_query_includes_all_fields(self, client: AsyncClient):
        """Sales sees full customer details including pipeline fields."""
        r = await client.get("/agents/sales/query/customer", params={"id": "C01"})
        assert r.status_code == 200
        data = r.json()
        assert "known_needs" in data
        assert "deal_value" in data


class TestComputeRestrictions:
    @pytest.mark.anyio
    async def test_engineering_can_compute_maturity(self, client: AsyncClient):
        r = await client.get("/agents/engineering/compute/maturity")
        assert r.status_code == 200

    @pytest.mark.anyio
    async def test_engineering_cannot_compute_satisfaction(self, client: AsyncClient):
        r = await client.get("/agents/engineering/compute/satisfaction", params={"id": "C01"})
        assert r.status_code == 403

    @pytest.mark.anyio
    async def test_sales_can_compute_satisfaction(self, client: AsyncClient):
        r = await client.get("/agents/sales/compute/satisfaction", params={"id": "C01"})
        assert r.status_code == 200

    @pytest.mark.anyio
    async def test_marketing_no_computes(self, client: AsyncClient):
        r = await client.get("/agents/marketing/compute/maturity")
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Malformed request bodies -> 400, not an unhandled 500
# ---------------------------------------------------------------------------


class TestMalformedBody:
    @pytest.mark.anyio
    async def test_capacity_cost_malformed_json_is_400(self, client: AsyncClient):
        # A truncated actions file yields a body like `{"actions": ` — invalid JSON.
        r = await client.post(
            "/agents/engineering/compute/capacity-cost",
            content='{"actions": ',
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400
        assert "detail" in r.json()

    @pytest.mark.anyio
    async def test_capacity_cost_non_object_body_is_400(self, client: AsyncClient):
        # Valid JSON but not an object — `body.get(...)` would otherwise 500 with AttributeError.
        r = await client.post("/agents/engineering/compute/capacity-cost", json=[1, 2, 3])
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_submit_malformed_json_is_400(self, client: AsyncClient):
        # The guard is shared across every POST endpoint, not just capacity-cost.
        r = await client.post(
            "/agents/engineering/submit",
            content="not json at all",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_submit_non_list_actions_is_400(self, client: AsyncClient):
        # `actions` present but not a list (a model emitting `{"actions": "build F01"}`) would make
        # the validator iterate a scalar (TypeError 500) or a string char-by-char — reject cleanly.
        r = await client.post("/agents/engineering/submit", json={"actions": "build feature F01"})
        assert r.status_code == 400
        assert "detail" in r.json()

    @pytest.mark.anyio
    async def test_submit_string_action_is_rejected_not_500(self, engine: GameEngine):
        # A malformed action ITEM — a bare string inside a valid list — is rejected by the validator,
        # but recording that rejection used to call .get() on the string, an unhandled 500. A solo
        # orchestrator lets the turn barrier resolve immediately so the full submit path is exercised.
        orch = Orchestrator(engine)
        orch._register_agent_unlocked("marketing")
        app = create_app(orch)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/agents/marketing/submit", json={"actions": ["not an action object"]})
        assert r.status_code == 200
        rejections = r.json().get("function_rejections", [])
        assert any(rej.get("action") == "not an action object" for rej in rejections)


# ---------------------------------------------------------------------------
# Chat room
# ---------------------------------------------------------------------------


class TestChatRoom:
    @pytest.mark.anyio
    async def test_post_and_read(self, client: AsyncClient):
        r = await client.post("/agents/engineering/chat", json={"message": "Building F02 this turn"})
        assert r.status_code == 200
        seq1 = r.json()["seq"]

        r = await client.post("/agents/sales/chat", json={"message": "Need F02 for C01"})
        seq2 = r.json()["seq"]
        assert seq2 > seq1

        r = await client.get("/chat", params={"since": 0})
        messages = r.json()
        assert len(messages) == 2
        assert messages[0]["agent"] == "engineering"
        assert messages[1]["agent"] == "sales"

    @pytest.mark.anyio
    async def test_read_since(self, client: AsyncClient):
        await client.post("/agents/engineering/chat", json={"message": "msg1"})
        r = await client.post("/agents/sales/chat", json={"message": "msg2"})
        seq1 = r.json()["seq"]

        await client.post("/agents/marketing/chat", json={"message": "msg3"})

        r = await client.get("/chat", params={"since": seq1})
        messages = r.json()
        assert len(messages) == 1
        assert messages[0]["agent"] == "marketing"

    @pytest.mark.anyio
    async def test_read_updates_last_read_seq(self, client: AsyncClient, orchestrator: Orchestrator):
        await client.post("/agents/engineering/chat", json={"message": "hello"})
        await client.get("/agents/sales/chat", params={"since": 0})
        assert orchestrator.agents["sales"].last_chat_read_seq == 1

    @pytest.mark.anyio
    async def test_unknown_agent_rejected(self, client: AsyncClient):
        r = await client.post("/agents/nonexistent/chat", json={"message": "test"})
        assert r.status_code == 404

    @pytest.mark.anyio
    async def test_empty_message_rejected(self, client: AsyncClient):
        r = await client.post("/agents/engineering/chat", json={"message": ""})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------


class TestRegistration:
    @pytest.mark.anyio
    async def test_register_new_agent(self, client: AsyncClient, orchestrator: Orchestrator):
        r = await client.post("/orchestrator/register-agent", json={"function": "support"})
        assert r.status_code == 200
        assert "support" in orchestrator.active_functions

    @pytest.mark.anyio
    async def test_register_duplicate_409(self, client: AsyncClient):
        r = await client.post("/orchestrator/register-agent", json={"function": "engineering"})
        assert r.status_code == 409

    @pytest.mark.anyio
    async def test_register_unknown_function_400(self, client: AsyncClient):
        r = await client.post("/orchestrator/register-agent", json={"function": "hr"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Turn flow: 3-agent submit + synchronization
# ---------------------------------------------------------------------------


class TestTurnFlow:
    @pytest.mark.anyio
    async def test_three_agent_turn(self, client: AsyncClient, orchestrator: Orchestrator):
        """All three agents submit, last one triggers resolution."""

        async def submit_engineering():
            return await client.post("/agents/engineering/submit", json={
                "actions": [
                    {"action_type": "build", "feature_id": "F02", "quality": "mvp", "capacity": 5},
                ]
            })

        async def submit_sales():
            return await client.post("/agents/sales/submit", json={
                "actions": [
                    {"action_type": "sell", "customer_id": "C01", "sell_action": "outbound", "capacity": 2},
                ]
            })

        async def submit_marketing():
            return await client.post("/agents/marketing/submit", json={
                "actions": [
                    {"action_type": "market", "channel": "content", "capacity": 3},
                ]
            })

        results = await asyncio.gather(
            submit_engineering(),
            submit_sales(),
            submit_marketing(),
        )

        for r in results:
            assert r.status_code == 200
            data = r.json()
            assert data["turn"] == 1
            assert "events" in data
            assert "next_observation" in data

    @pytest.mark.anyio
    async def test_engineering_sees_own_events_only(self, client: AsyncClient):
        """Engineering should only see engineering-related events."""

        async def submit_eng():
            return await client.post("/agents/engineering/submit", json={
                "actions": [
                    {"action_type": "build", "feature_id": "F02", "quality": "mvp", "capacity": 5},
                ]
            })

        async def submit_sales():
            return await client.post("/agents/sales/submit", json={
                "actions": [
                    {"action_type": "sell", "customer_id": "C01", "sell_action": "outbound", "capacity": 2},
                ]
            })

        async def submit_mktg():
            return await client.post("/agents/marketing/submit", json={
                "actions": [{"action_type": "market", "channel": "content", "capacity": 3}]
            })

        eng_r, sales_r, mktg_r = await asyncio.gather(
            submit_eng(), submit_sales(), submit_mktg(),
        )

        eng_events = eng_r.json()["events"]
        sales_events = sales_r.json()["events"]

        # Engineering should not see sales events
        for e in eng_events:
            assert not e.startswith("deal_won:")
            assert not e.startswith("stage_advanced:")
            assert not e.startswith("timeline_started:")

        # Sales should not see engineering events
        for e in sales_events:
            assert not e.startswith("feature_shipped:")
            assert not e.startswith("bug_fixed:")

    @pytest.mark.anyio
    async def test_empty_submission(self, client: AsyncClient):
        """All agents can submit empty action lists."""
        results = await asyncio.gather(
            client.post("/agents/engineering/submit", json={"actions": []}),
            client.post("/agents/sales/submit", json={"actions": []}),
            client.post("/agents/marketing/submit", json={"actions": []}),
        )
        for r in results:
            assert r.status_code == 200
            assert r.json()["turn"] == 1

    @pytest.mark.anyio
    async def test_function_restricted_action_rejected(self, client: AsyncClient):
        """Engineering submitting a sell action is rejected at the function level."""

        async def submit_eng():
            return await client.post("/agents/engineering/submit", json={
                "actions": [
                    {"action_type": "sell", "customer_id": "C01", "sell_action": "outbound", "capacity": 2},
                ]
            })

        async def submit_sales():
            return await client.post("/agents/sales/submit", json={"actions": []})

        async def submit_mktg():
            return await client.post("/agents/marketing/submit", json={"actions": []})

        eng_r, _, _ = await asyncio.gather(submit_eng(), submit_sales(), submit_mktg())

        data = eng_r.json()
        assert "function_rejections" in data
        assert len(data["function_rejections"]) == 1

    @pytest.mark.anyio
    async def test_next_observation_is_filtered(self, client: AsyncClient):
        """After turn resolution, next_observation is function-filtered."""
        results = await asyncio.gather(
            client.post("/agents/engineering/submit", json={"actions": []}),
            client.post("/agents/sales/submit", json={"actions": []}),
            client.post("/agents/marketing/submit", json={"actions": []}),
        )

        eng_data = results[0].json()
        next_obs = eng_data["next_observation"]
        assert "global" in next_obs
        assert "product_eng" in next_obs
        assert "sales" not in next_obs

    @pytest.mark.anyio
    async def test_multi_turn(self, client: AsyncClient, orchestrator: Orchestrator):
        """Play 2 consecutive turns."""
        for expected_turn in [1, 2]:
            results = await asyncio.gather(
                client.post("/agents/engineering/submit", json={
                    "actions": [{"action_type": "build", "feature_id": "F02", "quality": "mvp", "capacity": 3}]
                }),
                client.post("/agents/sales/submit", json={
                    "actions": [{"action_type": "sell", "customer_id": "C01", "sell_action": "outbound", "capacity": 2}]
                }),
                client.post("/agents/marketing/submit", json={
                    "actions": [{"action_type": "market", "channel": "content", "capacity": 2}]
                }),
            )
            for r in results:
                assert r.status_code == 200
                assert r.json()["turn"] == expected_turn

        assert orchestrator.engine.state.turn == 3


# ---------------------------------------------------------------------------
# Chat unread-message warning (409 on submit)
# ---------------------------------------------------------------------------


class TestUnreadMessageWarning:
    @pytest.mark.anyio
    async def test_submit_warns_on_unread(self, client: AsyncClient):
        """Submit returns 409 when agent has unread messages."""
        await client.post("/agents/sales/chat", json={"message": "Need F02"})

        r = await client.post("/agents/engineering/submit", json={"actions": []})
        assert r.status_code == 409
        data = r.json()
        assert data["status"] == "unread_messages"
        assert data["unread_count"] == 1

    @pytest.mark.anyio
    async def test_submit_accepted_after_reading(self, client: AsyncClient):
        """After reading chat, submit is accepted."""
        await client.post("/agents/sales/chat", json={"message": "Need F02"})

        # First submit → 409
        r = await client.post("/agents/engineering/submit", json={"actions": []})
        assert r.status_code == 409

        # All non-posting agents read chat to clear unread status
        await client.get("/agents/engineering/chat", params={"since": 0})
        await client.get("/agents/marketing/chat", params={"since": 0})

        # Second submit → accepted (blocks until all submit)
        results = await asyncio.gather(
            client.post("/agents/engineering/submit", json={"actions": []}),
            client.post("/agents/sales/submit", json={"actions": []}),
            client.post("/agents/marketing/submit", json={"actions": []}),
        )
        assert results[0].status_code == 200

    @pytest.mark.anyio
    async def test_submit_accepted_on_retry_without_reading(self, client: AsyncClient):
        """Second submit attempt is accepted even without reading (agent chose to ignore)."""
        await client.post("/agents/sales/chat", json={"message": "Need F02"})

        # First submit for engineering and marketing → both get 409
        r = await client.post("/agents/engineering/submit", json={"actions": []})
        assert r.status_code == 409
        r = await client.post("/agents/marketing/submit", json={"actions": []})
        assert r.status_code == 409

        # Retry without reading → accepted (no new messages since warning)
        results = await asyncio.gather(
            client.post("/agents/engineering/submit", json={"actions": []}),
            client.post("/agents/sales/submit", json={"actions": []}),
            client.post("/agents/marketing/submit", json={"actions": []}),
        )
        assert results[0].status_code == 200

    @pytest.mark.anyio
    async def test_force_submit_bypasses_unread(self, orchestrator: Orchestrator):
        """Force submit (watchdog path) skips unread check."""
        orchestrator.post_chat("sales", "Need F02")

        result = await orchestrator.submit_actions("engineering", [], force=False)
        assert result.get("status") == "unread_messages"

        # force=True bypasses unread check (simulates watchdog auto-submit)
        results = await asyncio.gather(
            orchestrator.submit_actions("engineering", [], force=True),
            orchestrator.submit_actions("sales", [], force=True),
            orchestrator.submit_actions("marketing", [], force=True),
        )
        assert results[0]["turn"] == 1

    @pytest.mark.anyio
    async def test_self_message_does_not_re_warn(self, client: AsyncClient):
        """Posting a message yourself should not cause re-warning about old unread messages."""
        # Sales sends a message
        await client.post("/agents/sales/chat", json={"message": "Need F02"})

        # Engineering gets warned
        r = await client.post("/agents/engineering/submit", json={"actions": []})
        assert r.status_code == 409

        # Engineering posts its own message (should not reset the warning)
        await client.post("/agents/engineering/chat", json={"message": "Ack, building F02"})

        # Other agents read chat to clear their unread status
        await client.get("/agents/sales/chat", params={"since": 0})
        await client.get("/agents/marketing/chat", params={"since": 0})

        # Engineering retries — should be accepted (no NEW messages from others since warning)
        results = await asyncio.gather(
            client.post("/agents/engineering/submit", json={"actions": []}),
            client.post("/agents/sales/submit", json={"actions": []}),
            client.post("/agents/marketing/submit", json={"actions": []}),
        )
        assert results[0].status_code == 200


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------


class TestOnboarding:
    @pytest.mark.anyio
    async def test_pending_onboarding_empty_initially(self, client: AsyncClient):
        r = await client.get("/orchestrator/pending-onboarding")
        assert r.json()["pending"] == []

    @pytest.mark.anyio
    async def test_ack_onboarding(self, client: AsyncClient, orchestrator: Orchestrator):
        orchestrator.onboarding_queue.append("support")
        r = await client.post("/orchestrator/ack-onboarding", json={"function": "support"})
        assert r.status_code == 200
        assert orchestrator.onboarding_queue == []


# ---------------------------------------------------------------------------
# Game over
# ---------------------------------------------------------------------------


class TestGameOver:
    @pytest.mark.anyio
    async def test_game_over_not_yet(self, client: AsyncClient):
        r = await client.get("/orchestrator/game-over")
        data = r.json()
        assert data["game_over"] is False

    @pytest.mark.anyio
    async def test_submit_after_game_over(self, client: AsyncClient, orchestrator: Orchestrator):
        orchestrator.engine.state.game_over = True
        orchestrator.engine.state.game_over_reason = "test"
        r = await client.post("/agents/engineering/submit", json={"actions": []})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Hire ownership tracking
# ---------------------------------------------------------------------------


class TestHireOwnership:
    @pytest.mark.anyio
    async def test_native_hire_registers_owner_from_event(self, orchestrator: Orchestrator):
        """A native hire (engineering→engineering) records ownership correctly."""
        results = await asyncio.gather(
            orchestrator.submit_actions("engineering", [{
                "action_type": "hire",
                "hiring_function": "engineering",
                "target_function": "engineering",
            }]),
            orchestrator.submit_actions("sales", []),
            orchestrator.submit_actions("marketing", []),
        )
        eng_result = results[0]
        hire_events = [e for e in eng_result.get("events", []) if e.startswith("hire_started:")]
        assert len(hire_events) == 1, f"Expected 1 hire_started event, got: {eng_result.get('events', [])}"
        hire_id = hire_events[0].split(":")[1]
        assert orchestrator.hire_owners[hire_id] == "engineering"

    @pytest.mark.anyio
    async def test_cross_hire_registers_hiring_function(self, orchestrator: Orchestrator):
        """A cross-hire (sales→cs) records the hiring function (sales), not the target."""
        results = await asyncio.gather(
            orchestrator.submit_actions("engineering", []),
            orchestrator.submit_actions("sales", [{
                "action_type": "hire",
                "hiring_function": "sales",
                "target_function": "cs",
            }]),
            orchestrator.submit_actions("marketing", []),
        )
        sales_result = results[1]
        hire_events = [e for e in sales_result.get("events", []) if e.startswith("hire_started:")]
        assert len(hire_events) == 1, f"Expected 1 hire_started event, got: {sales_result.get('events', [])}"
        hire_id = hire_events[0].split(":")[1]
        assert orchestrator.hire_owners[hire_id] == "sales"
