import asyncio
import pandas as pd
import warnings

from datetime import datetime, timezone, timedelta
from openai import AsyncOpenAI
from tortoise import Tortoise
from typing import Dict, List, Any


from common.bigquery import query_bq
from common.embeddings import aembed_query
from .models import BigQuerySchemaCache, BigQuerySchemaSearch, TableSchemaInfo
from .settings import settings, logger


# Tortoise ORM configuration
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


# Suppress the specific Google Auth warning about user credentials
warnings.filterwarnings(
    "ignore",
    message="Your application has authenticated using end user credentials from Google Cloud SDK without a quota project.*",
    module="google.auth._default",
)

# Create OpenAI client for embeddings
async_openai_client = AsyncOpenAI(
    api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key.get_secret_value() else "",
    organization=settings.openai_organization,
)

# For caching results to avoid repeated querying
_bq_schema_cache = {}
_bq_table_sample_cache = {}
_bq_datasets_tables_cache = None

# Cache freshness threshold (24 hours)
CACHE_FRESHNESS_THRESHOLD = timedelta(hours=24)


async def get_bigquery_datasets_and_tables() -> Dict[str, List[str]]:
    """
    Get a dictionary of all datasets and their tables in the project.

    Returns:
        Dictionary mapping dataset IDs to lists of table IDs
    """
    global _bq_datasets_tables_cache

    if _bq_datasets_tables_cache is not None:
        return _bq_datasets_tables_cache

    # Use specified datasets from settings
    requested_datasets = []
    if settings.bigquery_datasets:
        requested_datasets = [d.strip() for d in settings.bigquery_datasets.split(",")]

    if not requested_datasets:
        logger.warning("No BigQuery datasets specified in settings")
        return {}

    # Get tables for each specified dataset
    result = {}

    for dataset_id in requested_datasets:
        try:
            # Query for tables in this dataset
            tables_query = f"""
            SELECT table_name
            FROM `{dataset_id}.INFORMATION_SCHEMA.TABLES`
            WHERE table_type = 'BASE TABLE'
            """

            tables_df = await asyncio.to_thread(query_bq, tables_query, settings.gcp_project)

            if not tables_df.empty:
                result[dataset_id] = tables_df["table_name"].tolist()
                logger.info(f"Found {len(tables_df)} tables in dataset {dataset_id}")
            else:
                logger.warning(f"No tables found in dataset {dataset_id}")
                result[dataset_id] = []

        except Exception as e:
            logger.error(f"Error getting tables for dataset {dataset_id}: {str(e)}")
            result[dataset_id] = []

    _bq_datasets_tables_cache = result
    return result


async def get_bigquery_table_schema(dataset_id: str, table_id: str) -> Dict[str, Any]:
    """
    Get schema information for a BigQuery table.

    Args:
        dataset_id: The dataset ID containing the table (can be fully qualified with project)
        table_id: The table ID

    Returns:
        Dictionary with table schema information
    """
    cache_key = f"{dataset_id}.{table_id}"

    if cache_key in _bq_schema_cache:
        return _bq_schema_cache[cache_key]

    schema_query = f"""
    SELECT
        column_name,
        data_type,
        is_nullable,
        ordinal_position,
        is_hidden,
        is_system_defined,
    FROM
        `{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
    WHERE
        table_name = '{table_id}'
        -- Skip pseudo and system-defined columns
        AND IFNULL(is_system_defined, 'NO') = 'NO'
        AND IFNULL(is_hidden, 'NO') = 'NO'
    ORDER BY
        ordinal_position
    """

    table_query = f"""
    SELECT
        table_catalog,
        table_schema,
        table_name,
        table_type,
        creation_time,
        ddl
    FROM
        `{dataset_id}.INFORMATION_SCHEMA.TABLES`
    WHERE
        table_name = '{table_id}'
    """

    try:
        # Get column information
        columns_df = await asyncio.to_thread(query_bq, schema_query, settings.gcp_project)

        # Get table information
        table_df = await asyncio.to_thread(query_bq, table_query, settings.gcp_project)

        if table_df.empty:
            logger.warning(f"Table {dataset_id}.{table_id} not found")
            return {
                "table_name": f"{dataset_id}.{table_id}",
                "description": f"Table not found in {dataset_id}",
                "columns": [],
            }

        # Process columns
        columns = []
        for _, row in columns_df.iterrows():
            column_info = {
                "column_name": row["column_name"],
                "data_type": row["data_type"],
                "is_nullable": row.get("is_nullable", "YES"),
                "description": f"Column {row['column_name']}",
            }
            columns.append(column_info)

        # Sort columns by ordinal position if available
        if not columns_df.empty and "ordinal_position" in columns_df.columns:
            columns_df = columns_df.sort_values(by="ordinal_position")

        # Get table description and metadata
        table_type = (
            table_df["table_type"].iloc[0] if not table_df.empty and "table_type" in table_df.columns else "BASE TABLE"
        )
        table_description = f"Table {dataset_id}.{table_id}"

        # Get DDL if available
        ddl = (
            table_df["ddl"].iloc[0]
            if not table_df.empty and "ddl" in table_df.columns and pd.notna(table_df["ddl"].iloc[0])
            else ""
        )

        # Construct full qualified table name with project if available
        full_table_name = f"{dataset_id}.{table_id}"

        schema_info = {
            "table_name": full_table_name,
            "table_type": table_type,
            "description": table_description,
            "columns": columns,
            "ddl": ddl,
        }

        _bq_schema_cache[cache_key] = schema_info
        return schema_info

    except Exception as e:
        logger.error(f"Error getting schema for {dataset_id}.{table_id}: {str(e)}")
        logger.error(f"Schema query: {schema_query}")
        logger.error(f"Table query: {table_query}")
        return {
            "table_name": f"{dataset_id}.{table_id}",
            "description": f"Error retrieving schema: {str(e)}",
            "columns": [],
        }


async def get_bigquery_table_sample(dataset_id: str, table_id: str, limit: int = 5) -> str:
    """
    Get sample data from a BigQuery table.

    Args:
        dataset_id: The dataset ID containing the table (can be fully qualified with project)
        table_id: The table ID
        limit: Maximum number of rows to return (default 5)

    Returns:
        Sample data as a formatted string
    """
    cache_key = f"{dataset_id}.{table_id}.{limit}"

    if cache_key in _bq_table_sample_cache:
        return _bq_table_sample_cache[cache_key]

    # First, check if the table exists and get column information
    # to avoid SELECT * which might be problematic for large tables
    try:
        # Get column information
        columns_query = f"""
        SELECT
            column_name
        FROM
            `{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
        WHERE
            table_name = '{table_id}'
            -- Skip pseudo and system-defined columns
            AND IFNULL(is_system_defined, 'NO') = 'NO'
            AND IFNULL(is_hidden, 'NO') = 'NO'
        ORDER BY
            ordinal_position
        LIMIT 100
        """

        columns_df = await asyncio.to_thread(query_bq, columns_query, settings.gcp_project)

        if columns_df.empty:
            logger.warning(f"No columns found for table {dataset_id}.{table_id}")
            return "No columns available for this table"

        # Get column names for the SELECT statement
        # Limit to first 15 columns to avoid very wide outputs
        column_names = columns_df["column_name"].tolist()[:15]

        # Build sample query with specific columns
        column_list = ", ".join([f"`{col}`" for col in column_names])
        sample_query = f"""
        SELECT {column_list} FROM `{dataset_id}.{table_id}`
        LIMIT {limit}
        """

        # Execute the sample query
        sample_df = await asyncio.to_thread(query_bq, sample_query, settings.gcp_project)

        if sample_df.empty:
            sample_data = "No data available in this table"
        else:
            # Format the data with limited column width
            with pd.option_context("display.max_colwidth", 30):
                sample_data = sample_df.to_string(index=False)

            # Add a note if we didn't include all columns
            if len(column_names) < len(columns_df):
                sample_data += f"\n\nNote: Only showing {len(column_names)} of {len(columns_df)} columns"

        _bq_table_sample_cache[cache_key] = sample_data
        logger.info(f"Retrieved {len(sample_df)} sample rows from {dataset_id}.{table_id}")
        return sample_data

    except Exception as e:
        logger.error(f"Error getting sample data for {dataset_id}.{table_id}: {str(e)}")

        # Try a fallback approach with SELECT * and limited columns
        try:
            fallback_query = f"""
            SELECT * FROM `{dataset_id}.{table_id}`
            LIMIT {limit}
            """

            fallback_df = await asyncio.to_thread(query_bq, fallback_query, settings.gcp_project)

            if fallback_df.empty:
                return "No data available"

            # Limit to 15 columns max
            if len(fallback_df.columns) > 15:
                fallback_df = fallback_df.iloc[:, :15]
                sample_data = fallback_df.to_string(index=False)
                sample_data += f"\n\nNote: Only showing 15 of {len(fallback_df.columns)} columns"
            else:
                sample_data = fallback_df.to_string(index=False)

            _bq_table_sample_cache[cache_key] = sample_data
            logger.info(f"Retrieved {len(fallback_df)} sample rows using fallback from {dataset_id}.{table_id}")
            return sample_data

        except Exception as fallback_e:
            logger.error(f"Fallback also failed for {dataset_id}.{table_id}: {str(fallback_e)}")
            return f"Error retrieving sample data: {str(e)}"


async def search_bigquery_information_schema(search_term: str) -> List[Dict[str, Any]]:
    """
    Search BigQuery information schema for tables and columns matching a search term.

    Args:
        search_term: The term to search for in table and column names/descriptions

    Returns:
        List of matching table information
    """
    datasets_tables = await get_bigquery_datasets_and_tables()

    if not datasets_tables:
        logger.warning("No datasets to search in BigQuery")
        return []

    search_results = []
    logger.info(f"Searching for term '{search_term}' across {len(datasets_tables)} datasets")

    # Process each dataset in parallel
    async def search_dataset(dataset_id):
        # Query to search for tables and columns matching the search term
        # Based on the BQ documentation, we need to be careful about nullable fields
        # and make sure to use INFORMATION_SCHEMA correctly
        search_query = f"""
        SELECT
            '{dataset_id}' as full_dataset_id,
            c.table_catalog,
            c.table_schema,
            c.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable,
            t.table_type,
            t.ddl
        FROM
            `{dataset_id}.INFORMATION_SCHEMA.COLUMNS` c
        JOIN
            `{dataset_id}.INFORMATION_SCHEMA.TABLES` t
        ON
            c.table_name = t.table_name AND
            c.table_schema = t.table_schema
        WHERE
            (
                LOWER(c.table_name) LIKE LOWER('%{search_term}%') OR
                LOWER(c.column_name) LIKE LOWER('%{search_term}%') OR
                LOWER(IFNULL(t.ddl, '')) LIKE LOWER('%{search_term}%')
            )
            -- Exclude system tables and views that might cause issues
            AND t.table_type IN ('BASE TABLE', 'VIEW', 'MATERIALIZED VIEW', 'EXTERNAL')
            -- Skip pseudo and system-defined columns
            AND IFNULL(c.is_system_defined, 'NO') = 'NO'
            AND IFNULL(c.is_hidden, 'NO') = 'NO'
        ORDER BY
            c.table_name, c.ordinal_position
        """

        try:
            results_df = await asyncio.to_thread(query_bq, search_query, settings.gcp_project)

            if results_df.empty:
                logger.info(f"No matches found for '{search_term}' in dataset {dataset_id}")
                return []

            logger.info(f"Found {len(results_df)} matches for '{search_term}' in dataset {dataset_id}")

            # Group results by table
            grouped_tables = {}
            for _, row in results_df.iterrows():
                # Use full table name including project ID if available
                table_key = f"{dataset_id}.{row['table_name']}"

                if table_key not in grouped_tables:
                    grouped_tables[table_key] = {
                        "table_catalog": row.get("table_catalog", ""),
                        "table_schema": dataset_id,
                        "table_name": row["table_name"],
                        "table_type": row.get("table_type", "BASE TABLE"),
                        "table_description": row["ddl"] if pd.notna(row["ddl"]) else f"Table {table_key}",
                        "columns": [],
                    }

                # Only add columns that aren't already in the list
                column_names = [col["column_name"] for col in grouped_tables[table_key]["columns"]]
                if row["column_name"] not in column_names:
                    grouped_tables[table_key]["columns"].append(
                        {
                            "column_name": row["column_name"],
                            "data_type": row["data_type"],
                            "is_nullable": row.get("is_nullable", "YES"),
                        }
                    )

            # Convert to list and return tables with at least one column
            result_tables = []
            for table_key, table_info in grouped_tables.items():
                if table_info["columns"]:
                    result_tables.append(table_info)
                else:
                    logger.warning(f"Table {table_key} has no valid columns, skipping")

            return result_tables

        except Exception as e:
            logger.error(f"Error searching information schema for dataset {dataset_id}: {str(e)}")
            logger.error(f"Query that failed: {search_query}")
            return []

    # Execute searches in parallel
    tasks = [search_dataset(dataset_id) for dataset_id in datasets_tables.keys()]
    results = await asyncio.gather(*tasks)

    # Flatten results
    for dataset_results in results:
        search_results.extend(dataset_results)

    logger.info(f"Total matching tables found across all datasets: {len(search_results)}")
    return search_results


async def convert_to_table_schema_info(tables: List[Dict[str, Any]]) -> List[TableSchemaInfo]:
    """
    Convert BigQuery table information to TableSchemaInfo objects.

    Args:
        tables: List of table information from BigQuery queries

    Returns:
        List of TableSchemaInfo objects for use with LLM prompts
    """
    result = []

    # Process each table
    async def process_table(table):
        dataset_id, table_id = table["table_schema"], table["table_name"]
        full_table_name = f"{dataset_id}.{table_id}"

        # Get sample data
        sample_data = await get_bigquery_table_sample(dataset_id, table_id, limit=5)

        # Properly format columns for TableSchemaInfo
        formatted_columns = []
        for col in table["columns"]:
            formatted_column = {
                "column_name": col["column_name"],
                "data_type": col["data_type"],
                "is_nullable": col.get("is_nullable", "YES"),  # Default to nullable if not specified
            }
            formatted_columns.append(formatted_column)

        # Create TableSchemaInfo object
        return TableSchemaInfo(
            table_name=full_table_name,
            columns=formatted_columns,
            description=table["table_description"],
            sample_data=sample_data,
        )

    # Process tables in parallel
    tasks = [process_table(table) for table in tables]
    result = await asyncio.gather(*tasks)

    # Log results
    logger.info(f"Converted {len(result)} tables to TableSchemaInfo format")

    return result


async def get_bigquery_tables_for_query(question: str) -> List[TableSchemaInfo]:
    """
    Get relevant BigQuery tables for a natural language query.

    This function:
    1. Checks for fresh schema cache in PostgreSQL
    2. If cache is stale or doesn't exist, updates it synchronously (we're in dev, so waiting is fine)
    3. Uses vector similarity search to find relevant tables
    4. Falls back to keyword search only if vector search finds nothing

    Args:
        question: Natural language question to find relevant tables for

    Returns:
        List of TableSchemaInfo objects with relevant tables
    """
    # First, check if the schema cache is fresh
    is_cache_fresh = await check_schema_cache_freshness()

    # If cache is not fresh, update it synchronously
    if not is_cache_fresh:
        logger.info("Schema cache is stale or doesn't exist, updating synchronously...")
        await update_schema_cache()
        logger.info("Schema cache update completed")
        is_cache_fresh = True

    logger.info("Using vector similarity search on schema cache")
    schema_tables = await get_tables_from_schema_cache(question)
    if schema_tables:
        return schema_tables
    logger.info("No matches found in schema cache, falling back to keyword search")


async def check_schema_cache_freshness() -> bool:
    """
    Check if there's a recent schema cache entry in the database.

    Returns:
        True if fresh cache exists (created within the last 24 hours), False otherwise
    """
    if not Tortoise._inited:
        logger.warning("Database not initialized when checking schema cache freshness")
        return False

    try:
        # Get the most recent cache entry
        latest_cache = await BigQuerySchemaCache.filter(is_active=True).order_by("-updated_at").first()

        if not latest_cache:
            logger.info("No schema cache entries found in database")
            return False

        # Check if the cache is fresh (less than 24 hours old)
        cache_age = datetime.now(timezone.utc) - latest_cache.updated_at
        is_fresh = cache_age < CACHE_FRESHNESS_THRESHOLD

        if is_fresh:
            logger.info(f"Found fresh schema cache ({cache_age.total_seconds() / 3600:.1f} hours old)")
        else:
            logger.info(f"Schema cache is stale ({cache_age.total_seconds() / 3600:.1f} hours old)")

        return is_fresh

    except Exception as e:
        logger.error(f"Error checking schema cache freshness: {e}")
        return False


async def create_schema_text_representation(table_info: Dict[str, Any]) -> str:
    """
    Create a rich text representation of the schema for text search.

    Args:
        table_info: The table information dictionary

    Returns:
        A string representation of the schema with all metadata
    """
    # Safely construct the table name, handling missing keys
    table_schema = table_info.get("table_schema", "")
    table_name_part = table_info.get("table_name", "")

    if table_schema and table_name_part:
        table_name = f"{table_schema}.{table_name_part}"
    else:
        # Use the full table name if already formatted that way
        table_name = table_name_part if table_name_part else "Unknown Table"

    # Start with table name and type
    result = f"Table: {table_name}\n"
    result += f"Type: {table_info.get('table_type', 'TABLE')}\n"

    # Add table description if available
    description = table_info.get("table_description", table_info.get("description", ""))
    if description:
        result += f"Description: {description}\n"

    # Add columns section - safely handle missing 'columns' key
    columns = table_info.get("columns", [])
    result += "Columns:\n"

    if not columns:
        result += "  No column information available\n"
    else:
        for column in columns:
            col_name = column.get("column_name", "Unknown Column")
            col_type = column.get("data_type", "Unknown Type")
            is_nullable = column.get("is_nullable", "YES")
            col_description = column.get("description", "")

            result += f"  - {col_name}: {col_type} {is_nullable}\n"
            if col_description:
                result += f"    Description: {col_description}\n"

    # Add DDL if available
    ddl = table_info.get("ddl", "")
    if ddl:
        result += f"DDL: {ddl}\n"

    return result


async def update_schema_cache() -> None:
    """
    Update the schema cache in PostgreSQL with fresh BigQuery schema information.

    This function:
    1. Retrieves all datasets and tables from BigQuery
    2. For each table, gets its schema and sample data
    3. Creates rich text representation and vector embeddings
    4. Stores everything in PostgreSQL cache with the current timestamp
    """

    logger.info("Starting schema cache update process")

    # Ensure Tortoise is initialized
    if not Tortoise._inited:
        await Tortoise.init(config=ContentConfig.tortoise_orm)

    # Get all datasets and tables
    datasets_tables = await get_bigquery_datasets_and_tables()

    if not datasets_tables:
        logger.warning("No datasets/tables found to cache")
        return

    # Mark all existing cache entries as inactive
    await BigQuerySchemaCache.filter(is_active=True).update(is_active=False)

    # Cache version to group this update batch
    cache_version = int(datetime.now(timezone.utc).timestamp())

    # Process each dataset and table
    total_tables = sum(len(tables) for tables in datasets_tables.values())
    cached_tables = 0

    for dataset_id, tables in datasets_tables.items():
        logger.info(f"Processing dataset {dataset_id} with {len(tables)} tables")

        for table_id in tables:
            try:
                # Get schema info
                schema_info = await get_bigquery_table_schema(dataset_id, table_id)

                # Get sample data
                sample_data = await get_bigquery_table_sample(dataset_id, table_id, limit=5)

                # Log schema info keys for debugging
                logger.debug(f"Schema info keys: {list(schema_info.keys())}")

                # Create text representation - fix the table_schema key issue
                # Ensure schema_info has the expected structure for create_schema_text_representation
                schema_for_text = {
                    "table_name": schema_info.get("table_name", f"{dataset_id}.{table_id}"),
                    "table_schema": dataset_id,  # Add table_schema explicitly
                    "table_type": schema_info.get("table_type", "BASE TABLE"),
                    "description": schema_info.get("description", ""),
                    "columns": schema_info.get("columns", []),
                }
                schema_text = await create_schema_text_representation(schema_for_text)

                # Project ID from the schema info
                project_id = settings.gcp_project
                if schema_info.get("table_name", "").count(".") >= 2:
                    # Extract project ID from fully qualified name if available
                    parts = schema_info["table_name"].split(".")
                    if len(parts) >= 3:
                        project_id = parts[0]

                # Generate embedding
                embedding_input = (
                    f"Table {schema_info.get('table_name', f'{dataset_id}.{table_id}')} "
                    f"with columns: {', '.join([col.get('column_name', '') for col in schema_info.get('columns', [])])}"
                )
                embedding = await aembed_query(
                    async_openai_client, embedding_input, settings.embedding_model, settings.embedding_dimension
                )

                # Save to database with more robust error handling
                try:
                    await BigQuerySchemaCache.create(
                        project_id=project_id,
                        dataset_id=dataset_id,
                        table_id=table_id,
                        schema_json=schema_info,
                        schema_text=schema_text,
                        table_description=schema_info.get("description", ""),
                        embedding=embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                        column_count=len(schema_info.get("columns", [])),
                        sample_data=sample_data,
                        is_active=True,
                        cache_version=cache_version,
                    )
                    logger.debug(f"Successfully cached schema for {dataset_id}.{table_id}")
                except Exception as e:
                    logger.error(f"Error creating cache record for {dataset_id}.{table_id}: {str(e)}")
                    logger.debug(f"Schema info: {schema_info}")
                    continue

                cached_tables += 1
                logger.debug(f"Cached schema for {dataset_id}.{table_id}")

            except Exception as e:
                logger.error(f"Error caching schema for {dataset_id}.{table_id}: {e}")

    logger.info(f"Schema cache update completed: {cached_tables}/{total_tables} tables cached")


async def get_tables_from_schema_cache(question: str, limit: int = 5) -> List[TableSchemaInfo]:
    """
    Get relevant tables from the schema cache based on vector similarity.

    Args:
        question: The natural language question to find tables for
        limit: Maximum number of tables to return

    Returns:
        List of TableSchemaInfo objects with relevant tables
    """

    logger.info(f"Searching schema cache for tables relevant to: '{question}'")

    try:
        # Count active cache entries
        cache_count = await BigQuerySchemaCache.filter(is_active=True).count()
        logger.info(f"Found {cache_count} active records in schema cache")

        if cache_count == 0:
            logger.warning("Schema cache is empty, cannot perform similarity search")
            return []

        # Run the similarity search
        search = BigQuerySchemaSearch(query=question, limit=limit)
        matching_tables = await search.execute()

        if not matching_tables:
            logger.info("No matching tables found in schema cache")
            return []

        logger.info(f"Found {len(matching_tables)} matching tables in schema cache")

        # Convert to TableSchemaInfo format
        result = []
        for table in matching_tables:
            try:
                # Extract schema info from JSON
                schema_info = table.schema_json
                logger.debug(f"Schema JSON keys for {table.fully_qualified_table_name}: {list(schema_info.keys())}")

                # Create TableSchemaInfo object
                table_schema = TableSchemaInfo(
                    table_name=table.fully_qualified_table_name,
                    columns=schema_info.get("columns", []),
                    description=table.table_description
                    or schema_info.get("description", f"Table {table.fully_qualified_table_name}"),
                    sample_data=table.sample_data,
                )
                result.append(table_schema)

                logger.debug(f"Selected table from cache: {table.fully_qualified_table_name}")
            except Exception as e:
                logger.error(f"Error processing cached table {table.fully_qualified_table_name}: {e}")
                continue

        return result
    except Exception as e:
        logger.error(f"Error searching schema cache: {e}")
        return []
