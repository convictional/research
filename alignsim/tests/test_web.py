"""Tests for alignsim.src.web.app — FastAPI HTTP endpoints.

Uses FastAPI TestClient. The DB layer degrades gracefully when unreachable
(see persistence.database.try_init_db), so no DB is required.
"""

import json

import pytest
from fastapi.testclient import TestClient

from alignsim.src.web import app as web_app
from alignsim.src.web.app import app


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset the module-level _session/_run_logger between tests so each test starts fresh."""
    web_app._session = None
    web_app._run_logger = None
    yield
    web_app._session = None
    web_app._run_logger = None


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_setup_page_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_start_creates_session_and_redirects(client):
    r = client.post("/start", data={"seed": "42", "max_turns": "4",
                                    "scenario": "seed_stage"},
                    follow_redirects=False)
    assert r.status_code in (302, 303)
    assert r.headers["location"] == "/turn"
    assert web_app._session is not None
    assert web_app._session.scenario.name == "seed_stage"


def test_turn_redirects_to_setup_without_session(client):
    r = client.get("/turn", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/"


def test_turn_renders_after_start(client):
    client.post("/start", data={"seed": "42", "max_turns": "4",
                                "scenario": "seed_stage"})
    r = client.get("/turn")
    assert r.status_code == 200


def test_submit_processes_turn_and_advances(client):
    client.post("/start", data={"seed": "42", "max_turns": "4",
                                "scenario": "seed_stage"})
    initial_turn = web_app._session.turn
    actions_payload = json.dumps([{"action_type": "infrastructure", "capacity": 1}])
    r = client.post("/submit", data={"actions_json": actions_payload},
                    follow_redirects=False)
    assert r.status_code in (302, 303)
    assert web_app._session.turn == initial_turn + 1


def test_submit_redirects_to_results_when_game_over(client):
    """Reaching max_turns ends the game and redirects to /results."""
    client.post("/start", data={"seed": "42", "max_turns": "1",
                                "scenario": "seed_stage"})
    r = client.post("/submit", data={"actions_json": "[]"},
                    follow_redirects=False)
    assert r.status_code in (302, 303)
    assert r.headers["location"] == "/results"


def test_history_page(client):
    client.post("/start", data={"seed": "42", "max_turns": "4",
                                "scenario": "seed_stage"})
    client.post("/submit", data={"actions_json": "[]"})
    r = client.get("/history")
    assert r.status_code == 200


def test_rules_page_renders_with_or_without_session(client):
    """/rules renders even without an active session, then again with one."""
    r1 = client.get("/rules")
    assert r1.status_code == 200
    client.post("/start", data={"seed": "42", "max_turns": "4",
                                "scenario": "seed_stage"})
    r2 = client.get("/rules")
    assert r2.status_code == 200


def test_customer_detail_unknown_redirects_to_turn(client):
    client.post("/start", data={"seed": "42", "max_turns": "4",
                                "scenario": "seed_stage"})
    r = client.get("/customer/MISSING", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/turn"


def test_customer_detail_visible_renders(client):
    client.post("/start", data={"seed": "42", "max_turns": "4",
                                "scenario": "seed_stage"})
    visible = next(
        (c for c in web_app._session.state.customers.values() if c.is_visible),
        None,
    )
    assert visible is not None
    r = client.get(f"/customer/{visible.id}")
    assert r.status_code == 200
