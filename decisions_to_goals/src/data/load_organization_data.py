from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit
from ..models import ActivityEvent, Decision, StatedGoal
from ..settings import settings
from .load_organization_data_postgres import load_from_postgres

ACTIVITY_EVENTS_CACHE = settings.shared_output_path / "activity_events.pkl"
DECISIONS_CACHE = settings.shared_output_path / "decisions.pkl"
C1_STATED_GOALS_CACHE = settings.shared_output_path / "c1_stated_goals.pkl"


async def load_organization_data(
    load_from_cache: bool = True,
) -> tuple[list[ActivityEvent], list[Decision], list[StatedGoal]]:
    """Fetch and cache the shared activity corpus, decision corpus, and C1 stated goals.

    Data is read from the local Postgres DB populated by `make research_load`
    (see settings.postgres_dsn). All three conditions read from the same cached
    files; only the stated-goals layer and decision author_stated_goals differ
    across conditions.
    """
    if load_from_cache and ACTIVITY_EVENTS_CACHE.exists() and DECISIONS_CACHE.exists() and C1_STATED_GOALS_CACHE.exists():
        log_cache_hit(ACTIVITY_EVENTS_CACHE)
        log_cache_hit(DECISIONS_CACHE)
        log_cache_hit(C1_STATED_GOALS_CACHE)
        activity_events = load_pickle_file(ACTIVITY_EVENTS_CACHE)
        decisions = load_pickle_file(DECISIONS_CACHE)
        c1_stated_goals = load_pickle_file(C1_STATED_GOALS_CACHE)
        print(f"Loaded from cache: {len(activity_events)} events, {len(decisions)} decisions, {len(c1_stated_goals)} goals")
        return activity_events, decisions, c1_stated_goals

    # Canonical ingress: read from the local Postgres DB populated by
    # `make research_load` (see settings.postgres_dsn).
    activity_events, decisions, c1_stated_goals = await load_from_postgres()

    dump_to_pickle_file(activity_events, ACTIVITY_EVENTS_CACHE)
    dump_to_pickle_file(decisions, DECISIONS_CACHE)
    dump_to_pickle_file(c1_stated_goals, C1_STATED_GOALS_CACHE)

    print(f"\nShared corpus cached:")
    print(f"  Activity events : {len(activity_events)} → {ACTIVITY_EVENTS_CACHE}")
    print(f"  Decisions       : {len(decisions)}       → {DECISIONS_CACHE}")
    print(f"  C1 stated goals : {len(c1_stated_goals)} → {C1_STATED_GOALS_CACHE}")

    return activity_events, decisions, c1_stated_goals
