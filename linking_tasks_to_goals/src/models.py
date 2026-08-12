from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class WorkspaceComment(BaseModel):
    """
    Represents a comment on a workspace.

    Note, this doesn't quite resemble the platform model, and is just really used to store workspace comments.
    """

    id: str = Field(..., description="The unique identifier for the comment")
    workspace_id: str = Field(..., description="The ID of the workspace to which the comment belongs")
    user_id: str = Field(..., description="The ID of the user who made the comment")
    user_name: str = Field(..., description="The name of the user who made the comment")
    content: str = Field(..., description="The content of the comment")
    created_at: datetime = Field(..., description="The datetime when the comment was created")


class Workspace(BaseModel):
    """
    Represents a workspace with its attributes.

    Note, this doesn't quite resemble the platform model, and is jsut really used to store workspace comments.
    """

    id: str = Field(..., description="The unique identifier for the workspace")
    comments: list[WorkspaceComment] = Field(..., description="The list of comments associated with the workspace")


class SuccessCondition(BaseModel):
    """
    Represents a success condition for a goal.

    Note, there are less fields in this model than in the platform model - just the ones we need for this experiment.
    """

    id: str = Field(..., description="The unique identifier for the success condition")
    goal_id: str = Field(..., description="The ID of the goal to which this success condition belongs")
    description: str = Field(..., description="The description of the success condition")
    # Note, status = None is for older success conditions
    status: str | None = Field(..., description="The current status of the success condition")
    tracking_url: str | None = Field(None, description="The tracking URL for the success condition")
    created_at: datetime = Field(..., description="The datetime when the success condition was created")


class Goal(BaseModel):
    """
    Represents a goal with its attributes.

    Note, there are less fields in this model than in the platform model - just the ones we need for this experiment.
    """

    id: str = Field(..., description="The unique identifier for the goal")
    organization_id: str = Field(..., description="The ID of the organization to which the goal belongs")
    workspace_id: str = Field(..., description="The ID of the workspace attached to the goal")
    title: str = Field(..., description="The title of the goal")
    status: str = Field(..., description="The current status of the goal")
    sharing: str = Field(..., description="The sharing setting for the goal")
    created_at: datetime = Field(..., description="The datetime when the goal was created")
    success_conditions: list[SuccessCondition] = Field(
        ..., description="The list of success conditions associated with the goal"
    )
    workspace: Workspace = Field(..., description="The workspace associated with the goal")
    # URL is not part of the platform model, but is here for convenience
    goal_url: str = Field(..., description="The platform URL for the goal")


class LinkedGoal(BaseModel):
    goal_id: str = Field(..., description="The unique identifier for the linked goal")
    title: str = Field(..., description="The title of the linked goal")
    created_at: datetime = Field(..., description="The datetime when the linked goal was created")


class Task(BaseModel):
    """
    Represents a task with its attributes.

    Note, there are less fields in this model than in the platform model - just the ones we need for this experiment.
    """

    id: str = Field(..., description="The unique identifier for the task")
    organization_id: str = Field(..., description="The ID of the organization to which the task belongs")
    workspace_id: str = Field(..., description="The ID of the workspace attached to the task")
    title: str = Field(..., description="The title of the task")
    description: str = Field(None, description="The description of the task")
    sharing: str = Field(..., description="The sharing setting for the task")
    created_at: datetime = Field(..., description="The datetime when the task was created")
    workspace: Workspace = Field(..., description="The workspace associated with the task")
    linked_goals: Optional[list[LinkedGoal]] = Field(
        [], description="The list of linked goals associated with the task"
    )
    # URL is not part of the platform model, but is here for convenience
    task_url: str = Field(..., description="The platform URL for the task")
