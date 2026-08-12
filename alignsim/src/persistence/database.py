"""Database connection and initialization for AlignSim run persistence."""

import logging

import asyncpg
from tortoise import Tortoise

from alignsim.src.settings import settings

logger = logging.getLogger(__name__)


def get_db_url() -> str:
    """Get database URL from settings."""
    return (
        f"postgres://{settings.local_postgres_user}:{settings.local_postgres_password}"
        f"@{settings.local_postgres_host}:{settings.local_postgres_port}/{settings.local_postgres_db}"
    )


TORTOISE_ORM = {
    "connections": {"default": get_db_url()},
    "apps": {
        "models": {
            "models": ["alignsim.src.persistence.models"],
            "default_connection": "default",
        }
    },
}


async def ensure_database_exists() -> None:
    """Ensure the database exists, create it if not."""
    try:
        conn = await asyncpg.connect(
            host=settings.local_postgres_host,
            port=settings.local_postgres_port,
            user=settings.local_postgres_user,
            password=settings.local_postgres_password,
            database="postgres",
        )

        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1)", settings.local_postgres_db
        )

        if not exists:
            logger.info(f"Database '{settings.local_postgres_db}' does not exist. Creating...")
            await conn.execute(f'CREATE DATABASE "{settings.local_postgres_db}"')
            logger.info(f"Database '{settings.local_postgres_db}' created successfully")
        else:
            logger.debug(f"Database '{settings.local_postgres_db}' already exists")

        await conn.close()

    except asyncpg.exceptions.InvalidPasswordError:
        logger.error(f"Invalid password for PostgreSQL user '{settings.local_postgres_user}'")
        raise
    except asyncpg.exceptions.ConnectionDoesNotExistError:
        logger.error(
            f"Could not connect to PostgreSQL at {settings.local_postgres_host}:{settings.local_postgres_port}"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to ensure database exists: {e}")
        raise


# Columns added to RunModel after alignsim_runs was first created. generate_schemas() only issues
# CREATE TABLE IF NOT EXISTS and never ALTERs, so a pre-existing table never gains new columns on its
# own — and inserts that reference them (e.g. harness) then fail. ADD COLUMN IF NOT EXISTS reconciles
# the drift idempotently: it fills any gap on an existing table and is a no-op on a fresh/correct one.
# Append a row here whenever a nullable column is added to RunModel.
# NOTE: this is a stopgap. The intended long-term fix is a real migration tool (aerich) with versioned
# migrations; until then this hand-rolled reconciler keeps existing DBs in sync.
_RUNS_MIGRATION_COLUMNS = {
    "harness": "VARCHAR(20)",
    "alignment_scores": "JSONB",
    "thinking": "VARCHAR(20)",
}


async def _reconcile_run_columns() -> None:
    """Add any RunModel columns missing from an existing alignsim_runs table (idempotent)."""
    conn = Tortoise.get_connection("default")
    for column, sql_type in _RUNS_MIGRATION_COLUMNS.items():
        await conn.execute_query(
            f"ALTER TABLE alignsim_runs ADD COLUMN IF NOT EXISTS {column} {sql_type}"
        )
    logger.debug("Reconciled alignsim_runs columns")


async def init_db() -> None:
    """Initialize database connection, create tables, and reconcile column drift."""
    await ensure_database_exists()

    try:
        await Tortoise.init(config=TORTOISE_ORM)
        logger.info(f"Connected to database: {settings.local_postgres_db}")
        await Tortoise.generate_schemas()
        await _reconcile_run_columns()
        logger.debug("Database tables ensured")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


async def close_db() -> None:
    """Close database connections."""
    await Tortoise.close_connections()
    logger.debug("Database connections closed")


async def try_init_db() -> bool:
    """Try to initialize DB. Returns True if successful, False if unavailable.

    On failure, logs a warning so the caller can continue without persistence.
    """
    try:
        await init_db()
        return True
    except Exception as e:
        logger.warning(f"Database unavailable, persistence disabled: {e}")
        return False
