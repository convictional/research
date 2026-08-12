from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from datetime import datetime


class SourceContentBase(BaseModel):
    content_id: str = Field(..., title="Content ID, created by concatenating repository_full_name and issue_number for github and id for app tasks")
    title: str = Field(..., title="Title of the content")
    content: str = Field(..., title="Original unchunked content of the source data")
    created_at: datetime = Field(..., title="When the content was created")
    last_comment_at: datetime | None = Field(..., title="When the last comment was made")
    closed_at: datetime | None = Field(..., title="When the content was closed/marked completed")
    url: str = Field(..., title="URL to the content")


class SourceContent(SourceContentBase):
    chunk_index: int = Field(..., title="Index of the text chunk")
    text_chunk: str = Field(
        ...,
        title="Text chunk of the content. This is the content split into chunks. Each text chunk has the content title and the text chunk content.",
    )
    type: str = Field(..., title="Type of the source content, e.g., 'github_issue', 'github_pull_request', 'app_task'")


class ResearchAndDevelopmentProject(BaseModel):
    name: str = Field(..., title="Name of the project")
    description: str = Field(..., title="Description of the project.")
    source_content_ids: list[str] = Field(
        ..., title="List of source content ids of the content that the project came from."
    )


class ResearchAndDevelopmentProjects(BaseModel):
    projects: list[ResearchAndDevelopmentProject] = Field(..., title="List of research and development projects.")


class HighLevelResearchAndDevelopmentProject(BaseModel):
    name: str = Field(..., title="Name of the high-level research and development project.")
    description: str = Field(
        ...,
        title="Description of the high-level research and development project.",
    )
    source_content_ids: SkipJsonSchema[list[str]] = Field(
        [], title="List of source content ids of the content that the project came from."
    )
    source_content_urls: SkipJsonSchema[str] = Field(
        "", title="List of source content urls of the content that the project came from."
    )
    project_owner: SkipJsonSchema[str] = Field("", title="The person who owns the project.")
    project_owner_reason_summary: SkipJsonSchema[str] = Field(
        "", title="A summary of the reasons why the project owner is the project owner."
    )
    other_people_involved: SkipJsonSchema[list[str]] = Field([], title="List of other people involved in the project.")
    project_start: SkipJsonSchema[datetime] = Field(datetime(2000, 1, 1), title="When the project started.")
    project_end: SkipJsonSchema[datetime] = Field(datetime(2000, 1, 1), title="When the project ended.")
