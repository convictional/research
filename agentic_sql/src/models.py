import functools
from dataclasses import dataclass
from datetime import datetime, date
from json import JSONDecoder, JSONEncoder
from json import dumps as json_dumps
from json import loads as json_loads
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Type
from uuid import UUID, uuid4
from tortoise import Model, fields
from tortoise.fields.data import T


@dataclass
class GlobalID:
    """A global ID for a resource."""

    type: str
    id: UUID

    @classmethod
    def parse(cls, global_id: str) -> "GlobalID":
        """Parse a global ID from a string."""
        try:
            type_part, id_part = global_id.split(":")
            return cls(type=type_part, id=UUID(id_part))
        except (ValueError, AttributeError):
            return cls(type="unknown", id=UUID(int=0))

    @property
    def is_internal(self) -> bool:
        """Returns True if this is an internal ID."""
        internal_types = {
            "decision",
            "discussion",
            "meeting",
            "goal",
            "workspace",
            "task",
            "option",
            "update",
            "comment",
        }
        return self.type in internal_types


class RecordModel(Model):
    """Base model with ID, created_at, and updated_at fields."""

    id = fields.UUIDField(pk=True, default=uuid4)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)


class SQLQueryPairRequest(BaseModel):
    """Request model for storing SQL query pairs."""

    question: str = Field(..., description="The natural language question")
    sql: str = Field(..., description="The SQL query generated to answer the question")
    result_sample: str = Field(..., description="Sample of the results returned by the query")
    verified: bool = Field(default=False, description="Whether this query has been verified")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the query")


class SQLQueryPairResponse(BaseModel):
    """Response model for SQL query pairs."""

    id: UUID = Field(..., description="The unique ID for this query pair")
    question: str = Field(..., description="The natural language question")
    sql: str = Field(..., description="The verified SQL query")
    result_sample: str = Field(..., description="Sample of the results returned by the query")
    verified: bool = Field(..., description="Whether this query has been verified")
    created_at: datetime = Field(..., description="When this query pair was created")
    updated_at: datetime = Field(..., description="When this query pair was last updated")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the query")


class SQLQueryRequest(BaseModel):
    """Request model for generating SQL queries."""

    question: str = Field(..., description="The natural language question to convert to SQL")
    limit: int = Field(default=10, description="Maximum number of similar examples to retrieve")


class SQLQueryResponse(BaseModel):
    """Response model for generated SQL queries."""

    sql: str = Field(..., description="The generated SQL query")
    explanation: str = Field(..., description="Explanation of what the SQL query does")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the query, including database type"
    )


class SQLQueryFeedbackRequest(BaseModel):
    """Request model for SQL query feedback."""

    question: str = Field(..., description="The original natural language question")
    sql: str = Field(..., description="The SQL query that was generated")
    results: str = Field(..., description="The results returned by executing the query")


class SQLQueryFeedbackResponse(BaseModel):
    """Response model for SQL query feedback."""

    verified: bool = Field(..., description="Whether the query correctly answers the question")
    feedback: str = Field(..., description="Feedback about the query, including any improvements")
    suggested_sql: Optional[str] = Field(None, description="An improved SQL query if the original had issues")


class TableSchemaInfo(BaseModel):
    """Information about a database table schema."""

    table_name: str = Field(..., description="Name of the table")
    columns: List[Dict[str, str]] = Field(..., description="List of columns and their data types")
    description: Optional[str] = Field(None, description="Description of the table's purpose")
    sample_data: Optional[str] = Field(None, description="Sample data from the table (if available)")


# Custom Fields for Tortoise ORM
class Encoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return {"__uuid__": True, "value": str(obj)}
        elif isinstance(obj, datetime):
            return {"__datetime__": True, "value": obj.isoformat()}
        elif isinstance(obj, date):
            return {"__date__": True, "value": obj.isoformat()}
        return JSONEncoder.default(self, obj)


class Decoder(JSONDecoder):
    def __init__(self):
        JSONDecoder.__init__(self, object_hook=self.object_hook)

    def object_hook(self, obj):
        if "__uuid__" in obj:
            return UUID(obj["value"])
        elif "__datetime__" in obj:
            return datetime.fromisoformat(obj["value"])
        elif "__date__" in obj:
            return date.fromisoformat(obj["value"])
        return obj


JSONDumps = functools.partial(json_dumps, separators=(",", ":"), cls=Encoder)
JSONLoads = functools.partial(json_loads, cls=Decoder)


class JSONField(fields.JSONField[T]):  # type: ignore
    def __init__(self, **kwargs):
        super().__init__(encoder=JSONDumps, decoder=JSONLoads, **kwargs)


class VectorField(fields.Field[list[float]]):
    def __init__(self, vector_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vector_size = vector_size

    @property
    def SQL_TYPE(self) -> str:  # type: ignore # noqa: N802, RUF100
        return f"vector({self._vector_size})"

    def to_db_value(self, value: list[float], instance: Type[Model] | Model) -> str:
        if isinstance(value, list):
            return "[" + ",".join([str(item) for item in value]) + "]"
        return value

    def to_python_value(self, value: Any) -> list[float]:
        if isinstance(value, str):
            value = value.removeprefix("[").removesuffix("]")
            return list([float(item) for item in value.split(",")])
        return value


# This is a workaround for the fact that Tortoise does not support generated fields
# that are not primary keys.
class TSVectorField(fields.Field[str]):
    SQL_TYPE = "TSVECTOR"
    field_type = str
    allows_generated = True


# Database models
# SQL Query models
class SQLQueryPair(Model):
    id = fields.UUIDField(pk=True, default=uuid4)
    question = fields.TextField(description="The natural language question")
    sql = fields.TextField(description="The verified SQL query")
    result_sample = fields.TextField(description="Sample of the results returned by the query")
    verified = fields.BooleanField(default=False, description="Whether this query has been verified")
    embedding = VectorField(vector_size=1536, default=[0.0] * 1536)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    metadata: dict[str, Any] = JSONField(default={})

    class Meta:
        table = "sql_query_pairs"


# Search class for SQL query pairs
class SQLQuerySearch:
    def __init__(self, query: str, limit: int = 5):
        self.query = query
        self.limit = limit

    async def execute(self) -> list[SQLQueryPair]:
        """Find similar questions and their verified SQL queries"""
        from tortoise import Tortoise
        from common.embeddings import aembed_query
        from .settings import settings

        connection = Tortoise.get_connection("default")
        table_name = SQLQueryPair._meta.db_table

        # Create OpenAI client
        from openai import AsyncOpenAI

        async_openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            organization=settings.openai_organization,
        )

        # Get embeddings for the query
        embedding_values = await aembed_query(
            async_openai_client, self.query, settings.embedding_model, settings.embedding_dimension
        )
        embedding_query = "[" + ", ".join(str(value) for value in embedding_values) + "]"

        # Run similarity search with a threshold
        raw_results = await connection.execute_query_dict(
            f"""
            SELECT *, (1 - (embedding <=> $1) / 2) AS similarity
            FROM {table_name}
            WHERE verified = true
            ORDER BY similarity DESC
            LIMIT $2;
            """,
            [embedding_query, self.limit],
        )

        return [
            SQLQueryPair(**{k: v for k, v in result.items() if k != "similarity"})
            for result in raw_results
            if result.get("similarity", 0) > 0.7
        ]


# BigQuery schema models
class BigQuerySchemaCache(Model):
    id = fields.UUIDField(pk=True, default=uuid4)
    # Basic metadata
    project_id = fields.CharField(max_length=100, index=True)
    dataset_id = fields.CharField(max_length=100, index=True)
    table_id = fields.CharField(max_length=100, index=True)
    # Schema information
    schema_json = JSONField(description="Complete schema information in JSON format")
    schema_text = fields.TextField(description="Text representation for full-text search")
    table_description = fields.TextField(description="Description of the table", null=True)
    # Vector similarity search
    embedding = VectorField(vector_size=1536, default=[0.0] * 1536)
    # Additional metadata
    column_count = fields.IntField(default=0)
    row_count = fields.BigIntField(default=0, null=True)
    sample_data = fields.TextField(null=True)
    # Cache management
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    is_active = fields.BooleanField(default=True)
    cache_version = fields.IntField(default=1)

    class Meta:
        table = "bigquery_schema_cache"

    @property
    def fully_qualified_table_name(self) -> str:
        """Return the fully qualified table name in format project_id.dataset_id.table_id"""
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"

    @property
    def is_fresh(self) -> bool:
        """Check if the schema cache is fresh (less than 24 hours old)"""
        from datetime import datetime, timezone, timedelta

        return (datetime.now(timezone.utc) - self.updated_at) < timedelta(hours=24)


# Search class for BigQuery schema cache
class BigQuerySchemaSearch:
    def __init__(self, query: str, limit: int = 5):
        self.query = query
        self.limit = limit

    async def execute(self) -> list[BigQuerySchemaCache]:
        """Find similar tables based on schema information"""
        from tortoise import Tortoise
        from common.embeddings import aembed_query
        from .settings import settings

        connection = Tortoise.get_connection("default")
        table_name = BigQuerySchemaCache._meta.db_table

        # Create OpenAI client
        from openai import AsyncOpenAI

        async_openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            organization=settings.openai_organization,
        )

        # Get embeddings for the query
        embedding_values = await aembed_query(
            async_openai_client, self.query, settings.embedding_model, settings.embedding_dimension
        )
        embedding_query = "[" + ", ".join(str(value) for value in embedding_values) + "]"

        # Run similarity search with a threshold
        raw_results = await connection.execute_query_dict(
            f"""
            WITH vector_matches AS (
                SELECT
                    *,
                    (1 - (embedding <=> $1) / 2) AS vector_similarity,
                    -- Also add text search similarity with lower weight
                    ts_rank(to_tsvector('english', schema_text), plainto_tsquery('english', $2)) AS text_similarity
                FROM {table_name}
                WHERE is_active = true
                -- Filter to the most recent version for each table
                AND id IN (
                    SELECT id FROM (
                        SELECT id, ROW_NUMBER() OVER (
                            PARTITION BY project_id, dataset_id, table_id
                            ORDER BY updated_at DESC, cache_version DESC
                        ) as rn
                        FROM {table_name}
                        WHERE is_active = true
                    ) as latest
                    WHERE rn = 1
                )
            )
            SELECT *,
                   (vector_similarity * 0.8 + text_similarity * 0.2) AS total_similarity
            FROM vector_matches
            ORDER BY total_similarity DESC
            LIMIT $3;
            """,
            [embedding_query, self.query, self.limit],
        )

        # Filter to only include tables with good similarity score
        return [
            BigQuerySchemaCache(
                **{
                    k: v
                    for k, v in result.items()
                    if k not in ["vector_similarity", "text_similarity", "total_similarity"]
                }
            )
            for result in raw_results
            if result.get("total_similarity", 0) > 0.5
        ]
