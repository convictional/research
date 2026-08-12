import functools
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, UTC
from enum import Enum
from json import JSONDecoder, JSONEncoder
from json import dumps as json_dumps
from json import loads as json_loads
from pydantic import BaseModel, Field
from tortoise import Tortoise, connections, fields, Model
from tortoise.fields.data import T
from typing import Any, List, Optional, ClassVar, Type, Annotated, Protocol
from uuid import UUID

from ..utils.embeddings import aembed_query
from ..utils.models import GlobalID, RecordModel
from ..settings import settings

logger = logging.getLogger(__name__)

# Constants from production
MAX_NUMBER_OF_SEARCH_RESULTS = 10
EMBEDDING_DIMENSION = 1536
EMBEDDING_TOKENS = 8100
RESULTS_PER_TYPE_LIMIT = 10

#
# Enums
#
class Sharing(str, Enum):
    PRIVATE = "private"
    ORGANIZATION = "organization"

    @property
    def is_private(self):
        return self == Sharing.PRIVATE

    @property
    def is_organization(self):
        return self == Sharing.ORGANIZATION


class ContextSource(str, Enum):
    INTERNAL = "internal"
    GITHUB = "github"
    GOOGLE_DRIVE = "google_drive"
    MICROSOFT_ONE_DRIVE = "microsoft_one_drive"
    SLACK = "slack"
    WEBSITE = "website"
    SEARCH_ENGINE = "search_engine"


class ContextCategory(str, Enum):
    DOCUMENT = "document"
    ACTIVITY = "activity"
    PERSON = "person"


class ContextContent(str, Enum):
    GITHUB_ISSUE = "github_issue"
    GITHUB_COMMENT = "github_comment"
    GITHUB_PULL_REQUEST = "github_pull_request"
    GITHUB_DISCUSSION = "github_discussion"
    GITHUB_DISCUSSION_COMMENT = "github_discussion_comment"
    GOOGLE_DOC = "google_doc"
    SLACK_CHANNEL = "slack_channel"
    SLACK_THREAD = "slack_thread"
    WEBSITE_PAGE = "website_page"
    DECISION_PROCESS = "decision_process"
    MEETING = "meeting"
    DECISION = "decision"
    DISCUSSION = "discussion"
    DISCUSSION_COMMENT = "discussion_comment"
    USER = "user"
    COMMENT = "comment"
    GOAL = "goal"
    TASK = "task"
    FILE = "file"


#
# Custom Fields
#
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


@dataclass
class SearchableData:
    title: str
    index_content: str
    url: str
    author: str | None = None
    preview_content: str | None = None


class ContentFilters:
    private: ClassVar[dict[str, Any]] = {"sharing": Sharing.PRIVATE}

    @classmethod
    def by_source(cls, source_type: ContextSource):
        return {"source_type": source_type}


class Content(RecordModel):
    category = fields.CharEnumField(ContextCategory, max_length=255)
    source_type = fields.CharEnumField(ContextSource, max_length=255)
    source_id = fields.TextField()
    source_url = fields.TextField()
    content_type = fields.CharEnumField(ContextContent, max_length=255)
    sharing = fields.CharEnumField(Sharing, default=Sharing.PRIVATE, max_length=255)
    allowed_user_ids: list[str] = JSONField[list[str]](default=list)
    title = fields.TextField()
    author = fields.TextField(null=True)
    index_content = fields.TextField()
    preview_content: str | None = fields.TextField(null=True)
    metadata: dict[str, Any] = JSONField(default={})
    embedding = VectorField(vector_size=EMBEDDING_DIMENSION, default=[0.0] * EMBEDDING_DIMENSION)
    text_search = TSVectorField(generated=True, store=True)
    last_indexed_at = fields.DatetimeField(auto_now_add=True)
    organization_id: Annotated[UUID, "foreign key to organization"]
    filters = ContentFilters()

    class Meta:
        ordering = ["-last_indexed_at"]

    @property
    def source_global_id(self):
        return GlobalID.parse(self.source_id)

    @property
    def is_internal(self):
        return self.source_type == ContextSource.INTERNAL or self.source_global_id.is_internal


@dataclass
class ContentSearch:
    query: str
    limit: int = RESULTS_PER_TYPE_LIMIT
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    sources: list[ContextSource] = field(default_factory=list)
    exclude_source_urls: list[str] = field(default_factory=list)

    async def execute(self) -> list[Content]:
        self.connection = Tortoise.get_connection("default")
        self.table_name = Content._meta.db_table

        self.embedding_query = await self._get_embedding_query()

        text_query = " OR ".join(self.query.split())
        self._params = [text_query, self.embedding_query, self.limit]
        self._where_conditions: list[str] = []

        # self._apply_access()
        self._apply_sources()
        # self._apply_exclusions()
        self._apply_exact_match_phrases()
        self._apply_date_range()

        where_clause = f"WHERE {' AND '.join(self._where_conditions)}" if self._where_conditions else ""

        raw_results = await self.connection.execute_query_dict(
            f"""
            WITH vector_matches AS (
                SELECT id, (1 - (embedding <=> $2) / 2) AS similarity
                FROM {self.table_name} c
                {where_clause}
            ),
            ranked_results AS (
                SELECT
                    c.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.category
                        ORDER BY (
                            vm.similarity * 0.7 +
                            ts_rank(c.text_search, websearch_to_tsquery('english', $1)) * 0.3
                        ) DESC
                    ) AS rn
                FROM vector_matches vm
                JOIN {self.table_name} c ON c.id = vm.id
            )
            SELECT *
            FROM ranked_results
            WHERE rn <= $3;
            """,
            self._params,
        )

        return [Content(**result) for result in raw_results]

    async def _get_embedding_query(self) -> str:
        embedding_values = await aembed_query(self.query)
        return "[" + ", ".join(str(value) for value in embedding_values) + "]"

    def _apply_access(self) -> None:
        self._where_conditions.append(f"c.sharing = '{Sharing.ORGANIZATION.value}'")

    def _apply_sources(self) -> None:
        if self.sources:
            param_sources = [f"'{source.value}'" for source in self.sources]
            self._where_conditions.append(f"c.source_type IN ({','.join(param_sources)})")

    def _apply_exclusions(self) -> None:
        for excluded_url in self.exclude_source_urls:
            self._params.append(excluded_url)
            self._where_conditions.append(f"c.source_url <> ${len(self._params)}")

    def _apply_exact_match_phrases(self) -> None:
        exact_match_phrases: list[str] = re.findall(r'"[^"]+?"', self.query)
        for phrase in exact_match_phrases:
            self._params.append(phrase.strip('"'))
            self._where_conditions.append(f"c.text_search @@ websearch_to_tsquery('english', ${len(self._params)})")

    def _apply_date_range(self) -> None:
        if self.starts_at:
            if self.starts_at.tzinfo is None:
                self.starts_at = self.starts_at.replace(tzinfo=UTC)
            else:
                self.starts_at = self.starts_at.astimezone(UTC)
            self._params.append(self.starts_at)
            self._where_conditions.append(f"c.created_at >= ${len(self._params)}")

        if self.ends_at:
            if self.ends_at.tzinfo is None:
                self.ends_at = self.ends_at.replace(tzinfo=UTC)
            else:
                self.ends_at = self.ends_at.astimezone(UTC)
            self._params.append(self.ends_at)
            self._where_conditions.append(f"c.created_at <= ${len(self._params)}")


class ContentConfig:
    """Content model configuration."""

    tortoise_orm = {
        "connections": {"default": "postgres://localhost/decide_development"},
        "apps": {
            "models": {
                "models": ["src.deep_research.get_context"],
                "default_connection": "default",
            }
        },
    }


class SearchResultItem(BaseModel):
    """Individual search result with required fields."""

    url: str = Field(..., description="Source URL of the content")
    markdown: str = Field(..., description="Markdown formatted content combining title and preview")
    source: str = Field(default="internal", description="Source type of the content")


class SearchResults(BaseModel):
    """Container for search results matching SERP expected format."""

    data: list[SearchResultItem] = Field(..., description="List of search results")


class SearchProvider(Protocol):
    async def search(
        self,
        query: str,
        limit: int = MAX_NUMBER_OF_SEARCH_RESULTS,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        sources: Optional[List[ContextSource]] = None,
        **kwargs,
    ) -> SearchResults: ...


class InternalContentSearch(SearchProvider):
    """Search provider implementation using our content database."""

    sources: List[ContextSource] = [ContextSource.INTERNAL, ContextSource.GITHUB, ContextSource.GOOGLE_DRIVE]

    def __init__(self):
        self.organization_id = UUID(settings.organization_id)

    async def initialize(self):
        """Initialize the database connection."""
        await Tortoise.init(config=ContentConfig.tortoise_orm)

    async def cleanup(self):
        """Clean up database connections."""
        try:
            if Tortoise._inited:
                await connections.close_all()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def search(
        self,
        query: str,
        limit: int = MAX_NUMBER_OF_SEARCH_RESULTS,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        sources: Optional[List[ContextSource]] = [],
        **kwargs,
    ) -> SearchResults:
        """Perform vector similarity search against our database."""
        # Initialize connection if needed
        try:
            if not Tortoise._inited:
                await self.initialize()
        except Exception as e:
            logger.warning(f"Error initializing database: {e}")
            await self.initialize()

        # Use provided sources or the default sources for this instance
        search_sources = sources if sources else self.sources

        # Create a ContentSearch object to execute the search
        content_search = ContentSearch(
            query=query,
            limit=limit,
            starts_at=created_after,
            ends_at=created_before,
            sources=search_sources,
        )

        # Execute the search
        results = await content_search.execute()

        # Transform results into SearchResultItems format
        formatted_results = [
            SearchResultItem(
                url=result.source_url,
                markdown=(
                    f"# {result.title}\n\n"
                    f"Source: {result.category} ({result.source_type})\n"
                    f"Author: {result.author} | Created: {result.created_at}\n\n"
                    f"{result.index_content}"
                ),
                source=result.source_type.value,  # Set the source property from source_type
            )
            for result in results
        ]

        return SearchResults(data=formatted_results)
