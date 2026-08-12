"""Database connection and initialization for Decision DAGs."""

import logging
import asyncpg
from tortoise import Tortoise

from ..settings import settings

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
            "models": ["decision_dags.src.persistence.models"],
            "default_connection": "default",
        }
    },
}


async def ensure_database_exists():
    """Ensure the database exists, create it if not."""
    try:
        # Connect to postgres database to check/create our database
        conn = await asyncpg.connect(
            host=settings.local_postgres_host,
            port=settings.local_postgres_port,
            user=settings.local_postgres_user,
            password=settings.local_postgres_password,
            database="postgres",
        )

        # Check if database exists
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1)", settings.local_postgres_db
        )

        if not exists:
            logger.info(f"Database '{settings.local_postgres_db}' does not exist. Creating...")
            # Create the database
            await conn.execute(f'CREATE DATABASE "{settings.local_postgres_db}"')
            logger.info(f"Database '{settings.local_postgres_db}' created successfully")
        else:
            logger.info(f"Database '{settings.local_postgres_db}' already exists")

        await conn.close()

    except asyncpg.exceptions.InvalidPasswordError:
        logger.error(f"Invalid password for PostgreSQL user '{settings.local_postgres_user}'")
        logger.error("Please check your database credentials in settings.py or environment variables")
        raise
    except asyncpg.exceptions.ConnectionDoesNotExistError:
        logger.error(
            f"Could not connect to PostgreSQL at {settings.local_postgres_host}:{settings.local_postgres_port}"
        )
        logger.error("Please ensure PostgreSQL is running and accessible")
        raise
    except asyncpg.exceptions.InsufficientPrivilegeError:
        logger.error(f"User '{settings.local_postgres_user}' does not have permission to create databases")
        logger.error("Please grant CREATE permission or create the database manually")
        raise
    except Exception as e:
        logger.error(f"Failed to ensure database exists: {e}")
        raise


async def init_db():
    """Initialize database connection and create tables if needed."""
    # First ensure the database exists
    await ensure_database_exists()

    try:
        await Tortoise.init(config=TORTOISE_ORM)
        logger.info(f"Connected to database: {settings.local_postgres_db}")

        # Create tables if they don't exist
        await create_tables()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


async def close_db():
    """Close database connections."""
    await Tortoise.close_connections()
    logger.info("Database connections closed")


async def create_tables():
    """Create database tables if they don't exist."""
    try:
        # Generate schema
        await Tortoise.generate_schemas()
        logger.info("Database tables created successfully")

        # Create indexes for vector similarity search if using raw SQL
        connection = Tortoise.get_connection("default")

        # Check if vector extension exists
        try:
            await connection.execute_query("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("Vector extension ensured")
        except Exception as e:
            logger.warning(f"Could not create vector extension: {e}")

    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


async def drop_tables():
    """Drop all database tables. WARNING: This will delete all data!"""
    try:
        connection = Tortoise.get_connection("default")

        # Drop tables in reverse order of dependencies
        await connection.execute_query("DROP TABLE IF EXISTS decision_edges CASCADE;")
        await connection.execute_query("DROP TABLE IF EXISTS decision_nodes CASCADE;")
        await connection.execute_query("DROP TABLE IF EXISTS decision_dags CASCADE;")

        logger.info("Database tables dropped")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise
