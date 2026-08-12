import asyncpg
import logging


async def query_local_postgres(
    query: str,
    logger: logging.Logger,
    user: str = "postgres",
    password: str = "",
    database: str = "decide_development",
    host: str = "127.0.0.1",
    port: int = 5432
) -> list[dict]:
    """
    Execute a query on a PostgreSQL database and return the results as a list of dictionaries.

    Args:
        query: The SQL query to execute
        user: PostgreSQL username
        password: PostgreSQL password
        database: PostgreSQL database name
        host: PostgreSQL host
        port: PostgreSQL port

    Returns:
        List of dictionaries, where each dictionary represents a row in the result set

    Raises:
        Exception: If connection or query execution fails
    """
    try:
        conn = await asyncpg.connect(
            user=user,
            password=password,
            database=database,
            host=host,
            port=port
        )
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

    try:
        records = await conn.fetch(query)
        return [dict(row) for row in records]
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        raise
    finally:
        await conn.close()
        logger.info("Database connection closed.")
