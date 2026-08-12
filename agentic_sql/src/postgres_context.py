import numpy as np
from openai import AsyncOpenAI
from tortoise import Tortoise
from typing import List, Dict
from uuid import UUID

from .models import SQLQueryPair, SQLQuerySearch, TableSchemaInfo
from .settings import settings, logger
from common.embeddings import aembed_query

# Create and configure OpenAI client for embeddings
async_openai_client = AsyncOpenAI(
    api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key.get_secret_value() else "",
    organization=settings.openai_organization,
)

# Constants for search and embeddings
MAX_NUMBER_OF_SEARCH_RESULTS = 10
EMBEDDING_DIMENSION = 1536
EMBEDDING_TOKENS = 8100
RESULTS_PER_TYPE_LIMIT = 10


class ContentConfig:
    """Content model configuration."""

    tortoise_orm = {
        "connections": {
            "default": f"postgres://{settings.local_postgres_user}:{settings.local_postgres_password}@{settings.local_postgres_host}:{settings.local_postgres_port}/decide_development"
        },
        "apps": {
            "models": {
                "models": ["src.models"],
                "default_connection": "default",
            }
        },
    }


# Create migrations for SQLQueryPair and BigQuerySchemaCache tables
async def create_cache_tables():
    """Create the required tables if they don't exist."""
    connection = Tortoise.get_connection("default")

    # Create sql_query_pairs table
    try:
        # Get a count of rows to see if the table exists
        await connection.execute_query("SELECT COUNT(*) FROM sql_query_pairs LIMIT 1")
        # If we get here, table exists
        sql_query_pairs_exists = True
        logger.info("Table exists check successful - sql_query_pairs table exists")
    except Exception as e:
        # If we get an error, table doesn't exist
        sql_query_pairs_exists = False
        logger.info(f"Table exists check failed with error: {e}")

    if not sql_query_pairs_exists:
        logger.info("Creating sql_query_pairs table")
        await connection.execute_script(
            """
            CREATE TABLE sql_query_pairs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                question TEXT NOT NULL,
                sql TEXT NOT NULL,
                result_sample TEXT NOT NULL,
                verified BOOLEAN NOT NULL DEFAULT FALSE,
                embedding vector(1536) NOT NULL DEFAULT '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB NOT NULL DEFAULT '{}'
            );

            CREATE INDEX idx_sql_query_pairs_verified ON sql_query_pairs(verified);
            """
        )
        logger.info("Created sql_query_pairs table")
    else:
        logger.info("sql_query_pairs table already exists")

    # Create bigquery_schema_cache table
    try:
        # Get a count of rows to see if the table exists
        await connection.execute_query("SELECT COUNT(*) FROM bigquery_schema_cache LIMIT 1")
        # If we get here, table exists
        bq_schema_cache_exists = True
        logger.info("Table exists check successful - bigquery_schema_cache table exists")
    except Exception as e:
        # If we get an error, table doesn't exist
        bq_schema_cache_exists = False
        logger.info(f"Table exists check failed with error: {e}")

    if not bq_schema_cache_exists:
        logger.info("Creating bigquery_schema_cache table")
        await connection.execute_script(
            """
            CREATE TABLE bigquery_schema_cache (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                project_id VARCHAR(100) NOT NULL,
                dataset_id VARCHAR(100) NOT NULL,
                table_id VARCHAR(100) NOT NULL,
                schema_json JSONB NOT NULL,
                schema_text TEXT NOT NULL,
                table_description TEXT,
                embedding vector(1536) NOT NULL DEFAULT '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector,
                column_count INTEGER NOT NULL DEFAULT 0,
                row_count BIGINT,
                sample_data TEXT,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                cache_version INTEGER NOT NULL DEFAULT 1
            );

            -- Create indexes to improve query performance
            CREATE INDEX idx_bigquery_schema_cache_project ON bigquery_schema_cache(project_id);
            CREATE INDEX idx_bigquery_schema_cache_dataset ON bigquery_schema_cache(dataset_id);
            CREATE INDEX idx_bigquery_schema_cache_table ON bigquery_schema_cache(table_id);
            CREATE INDEX idx_bigquery_schema_cache_active ON bigquery_schema_cache(is_active);
            CREATE INDEX idx_bigquery_schema_cache_updated ON bigquery_schema_cache(updated_at);

            -- Add full-text search index on schema_text
            CREATE INDEX idx_bigquery_schema_cache_text_search ON bigquery_schema_cache USING GIN (to_tsvector('english', schema_text));
            """
        )
        logger.info("Created bigquery_schema_cache table")
    else:
        logger.info("bigquery_schema_cache table already exists")


async def clear_sql_query_pairs():
    """Delete all queries from the sql_query_pairs table."""
    try:
        connection = Tortoise.get_connection("default")
        # Check if the table exists first
        try:
            await connection.execute_query("SELECT COUNT(*) FROM sql_query_pairs LIMIT 1")
            table_exists = True
        except Exception:
            table_exists = False

        if table_exists:
            # Get count of rows to be deleted
            count, _ = await connection.execute_query("SELECT COUNT(*) FROM sql_query_pairs")

            # Delete all rows
            await connection.execute_query("DELETE FROM sql_query_pairs")
            logger.info(f"Cleared {count} entries from sql_query_pairs table")
            return count
        else:
            logger.info("sql_query_pairs table does not exist - nothing to clear")
            return 0
    except Exception as e:
        logger.error(f"Error clearing sql_query_pairs table: {e}")
        return 0


async def clear_bigquery_schema_cache():
    """Delete all queries from the bigquery_schema_cache table."""
    try:
        connection = Tortoise.get_connection("default")
        # Check if the table exists first
        try:
            await connection.execute_query("SELECT COUNT(*) FROM bigquery_schema_cache LIMIT 1")
            table_exists = True
        except Exception:
            table_exists = False

        if table_exists:
            # Get count of rows to be deleted
            count, _ = await connection.execute_query("SELECT COUNT(*) FROM bigquery_schema_cache")

            # Delete all rows
            await connection.execute_query("DELETE FROM bigquery_schema_cache")
            logger.info(f"Cleared {count} entries from bigquery_schema_cache table")
            return count
        else:
            logger.info("bigquery_schema_cache table does not exist - nothing to clear")
            return 0
    except Exception as e:
        logger.error(f"Error clearing bigquery_schema_cache table: {e}")
        return 0


async def get_table_schemas() -> List[TableSchemaInfo]:
    """
    Get schema information about database tables.

    This function queries the database for table and column information
    to provide context to the LLM for SQL generation.

    It prioritizes BigQuery tables based on the query, and falls back to
    PostgreSQL tables if no BigQuery tables are found or if BigQuery isn't configured.

    Returns:
        List of TableSchemaInfo objects with table metadata
    """

    # Initialize the database connection if needed
    if not Tortoise._inited:
        await Tortoise.init(config=ContentConfig.tortoise_orm)

    connection = Tortoise.get_connection("default")

    # Get table schemas
    result = await connection.execute_query("""
        SELECT
            t.table_name,
            json_agg(json_build_object(
                'column_name', c.column_name,
                'data_type', c.data_type,
                'is_nullable', c.is_nullable
            )) AS columns
        FROM
            information_schema.tables t
        JOIN
            information_schema.columns c ON t.table_name = c.table_name
        WHERE
            t.table_schema = 'public' AND
            t.table_type = 'BASE TABLE'
        GROUP BY
            t.table_name
        ORDER BY
            t.table_name;
    """)

    schemas = []
    # Check if result[0] is iterable (list or tuple)
    if isinstance(result[0], (list, tuple)):
        for row in result[0]:
            # Get sample data for the table (limit to 5 rows)
            try:
                sample_query = f"SELECT * FROM {row[0]} LIMIT 5"
                sample_result = await connection.execute_query(sample_query)

                # Handle different types of results
                if isinstance(sample_result[0], (list, tuple)):
                    sample_data = "\n".join([str(r) for r in sample_result[0]]) if sample_result[0] else ""
                elif isinstance(sample_result[0], int):
                    sample_data = str(sample_result[0])
                else:
                    sample_data = str(sample_result[0])
            except Exception as e:
                # If error occurs (e.g., permissions), skip sample data
                print(f"Error getting sample data: {e}")
                sample_data = ""

            schemas.append(
                TableSchemaInfo(
                    table_name=row[0],
                    columns=row[1],
                    description=f"Table containing {row[0]} data",
                    sample_data=sample_data if sample_data else None,
                )
            )
    else:
        # If result[0] is not iterable (e.g., an int), log and return empty schema list
        print(f"Unexpected result type from schema query: {type(result[0])}, value: {result[0]}")
        # We'll continue with an empty schema list

    return schemas


async def format_schemas_for_prompt(schemas: List[TableSchemaInfo]) -> str:
    """
    Format table schemas into a string for the prompt.

    Args:
        schemas: List of TableSchemaInfo objects

    Returns:
        Formatted string describing tables and their schemas
    """
    formatted = ""

    for schema in schemas:
        formatted += f"Table: {schema.table_name}\n"
        formatted += "Columns:\n"

        for column in schema.columns:
            nullable = "NULL" if column.get("is_nullable", "YES") == "YES" else "NOT NULL"
            formatted += f"  - {column.get('column_name')}: {column.get('data_type')} {nullable}\n"

        if schema.sample_data:
            formatted += "Sample data (5 rows max):\n"
            formatted += f"{schema.sample_data}\n"

        formatted += "\n"

    return formatted


async def get_similar_sql_examples(question: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve similar question-SQL pairs from the database.

    Args:
        question: The natural language question to find similar examples for
        limit: Maximum number of examples to return

    Returns:
        List of dictionaries with question and SQL pairs
    """

    logger.info(f"Searching for similar SQL examples for question: '{question}'")

    # Initialize the database connection if needed
    if not Tortoise._inited:
        await Tortoise.init(config=ContentConfig.tortoise_orm)

    try:
        # Execute the search
        search = SQLQuerySearch(query=question, limit=limit)
        results = await search.execute()

        logger.info(f"Found {len(results)} similar SQL examples in database")

        # Log individual results at debug level
        for i, result in enumerate(results):
            logger.debug(f"Similar example #{i + 1}:\nQuestion: {result.question}\nSQL: {result.sql[:100]}...")

        # Format results as question/sql pairs
        return [{"question": result.question, "sql": result.sql} for result in results]
    except Exception as e:
        logger.error(f"Error getting similar SQL examples: {e}")
        return []


async def get_similar_queries_context(question: str) -> Dict[str, any]:
    """
    Fetch similar SQL queries for context.

    Args:
        question: The natural language question to get context for

    Returns:
        Dictionary containing the similar queries
    """
    # Get similar question-SQL pairs for few-shot examples
    question_sql_pairs = await get_similar_sql_examples(question)
    logger.info(f"Found {len(question_sql_pairs)} similar SQL examples from previous queries")

    # Format the similar queries for logging
    formatted_similar_queries = ""
    if question_sql_pairs:
        formatted_similar_queries = "\n\n".join(
            [f"Q: {ex['question']}\nSQL: {ex['sql']}" for ex in question_sql_pairs]
        )
        logger.debug(
            f"Similar examples found: {', '.join([ex['question'][:50] + '...' for ex in question_sql_pairs])}"
        )

    return {
        "similar_queries": question_sql_pairs,
        "formatted_similar_queries": formatted_similar_queries,
    }


async def save_verified_query(question: str, sql: str, result_sample: str, verified: bool = True) -> UUID:
    """
    Save a verified query to the database.

    Args:
        question: The natural language question
        sql: The verified SQL query
        result_sample: Sample of the results returned by the query
        verified: Whether this query has been verified (True by default)

    Returns:
        UUID of the saved query pair
    """

    # Initialize the database connection if needed
    if not Tortoise._inited:
        await Tortoise.init(config=ContentConfig.tortoise_orm)

    try:
        # Get embeddings for the question
        embedding = await aembed_query(
            async_openai_client, question, settings.embedding_model, settings.embedding_dimension
        )

        # Convert from numpy array to list if needed - PostgreSQL vector fields expect lists
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            logger.info("Converted numpy array embedding to list for database storage")

        # Create a new SQL query pair
        query_pair = await SQLQueryPair.create(
            question=question,
            sql=sql,
            result_sample=result_sample,
            verified=verified,
            embedding=embedding,
            metadata={},
        )

        logger.info(f"Successfully saved query pair with ID: {query_pair.id}")
        return query_pair.id

    except Exception as e:
        logger.error(f"Error saving verified query: {str(e)}")
        logger.error(f"Question: {question}")
        logger.error(f"SQL: {sql}")
        raise e
