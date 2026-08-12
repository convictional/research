"""FastAPI web app for interactive AlignSim play."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from alignsim.src.persistence.database import close_db, try_init_db
from alignsim.src.persistence.run_logger import RunLogger
from alignsim.src.web.game_session import GameSession, parse_action

logger = logging.getLogger(__name__)

# DB and session state
_db_available = False
_run_logger: RunLogger | None = None
_session: GameSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_available
    _db_available = await try_init_db()
    yield
    if _db_available:
        await close_db()


app = FastAPI(title="AlignSim", lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def setup_page(request: Request):
    return templates.TemplateResponse(request, "setup.html")


@app.post("/start")
async def start_game(
    request: Request,
    seed: int = Form(default=42),
    max_turns: int = Form(default=48),
    scenario: str = Form(default="playtest"),
):
    global _session, _run_logger
    _session = GameSession(seed=seed, max_turns=max_turns, scenario=scenario)
    _run_logger = None
    if _db_available:
        _run_logger = await RunLogger.create(
            scenario_name=_session.scenario.name,
            condition="human_web",
            player_type="human",
            model=None,
            seed=seed,
            max_turns=max_turns,
        )
    return RedirectResponse(url="/turn", status_code=303)


@app.get("/turn", response_class=HTMLResponse)
async def turn_page(request: Request):
    if _session is None:
        return RedirectResponse(url="/")
    if _session.game_over:
        return RedirectResponse(url="/results")
    ctx = _session.get_context()
    return templates.TemplateResponse(request, "turn.html", ctx)


@app.post("/submit", response_class=HTMLResponse)
async def submit_actions(request: Request):
    if _session is None:
        return RedirectResponse(url="/")

    form = await request.form()
    actions_json = form.get("actions_json", "[]")

    try:
        action_dicts = json.loads(actions_json)
    except json.JSONDecodeError:
        action_dicts = []

    actions = []
    for ad in action_dicts:
        action = parse_action(ad)
        if action is not None:
            actions.append(action)

    _session.submit_actions(actions)

    # Log turn to DB
    if _run_logger and _session.state.turn_history:
        last_record = _session.state.turn_history[-1]
        await _run_logger.log_turn(last_record.turn, last_record, _session.state)

    if _session.game_over:
        # Finalize run in DB
        if _run_logger and _session.score:
            await _run_logger.finalize(
                _session.score, _session.game_over_reason, _session.state.turn - 1
            )
        return RedirectResponse(url="/results", status_code=303)
    return RedirectResponse(url="/turn", status_code=303)


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    if _session is None:
        return RedirectResponse(url="/")
    history = _session.get_history()
    return templates.TemplateResponse(request, "history.html", {
        "history": history,
        "turn": _session.turn,
        "game_over": _session.game_over,
    })


@app.get("/customer/{customer_id}", response_class=HTMLResponse)
async def customer_detail_page(request: Request, customer_id: str):
    if _session is None:
        return RedirectResponse(url="/")
    detail = _session.get_customer_detail(customer_id)
    if detail is None:
        return RedirectResponse(url="/turn")
    return templates.TemplateResponse(request, "customer.html", {
        "c": detail,
        "turn": _session.turn,
        "game_over": _session.game_over,
        "min_rubric": _session.scenario.calibration.min_rubric_for_close,
    })


@app.get("/rules", response_class=HTMLResponse)
async def rules_page(request: Request):
    calibration = None
    if _session:
        calibration = _session.scenario.calibration
    return templates.TemplateResponse(request, "rules.html", {
        "calibration": calibration,
        "game_active": _session is not None,
    })


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    if _session is None:
        return RedirectResponse(url="/")
    ctx = _session.get_context()
    return templates.TemplateResponse(request, "results.html", ctx)
