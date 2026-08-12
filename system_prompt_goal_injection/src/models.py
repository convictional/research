from pydantic import BaseModel, Field


class ResponseComparison(BaseModel):
    db_id: str = Field(..., title="The id from the database")
    db_created_at: str = Field(..., title="The time the request was created from the database")
    db_url: str = Field(..., title="The url from the database")
    db_response_model: str | None = Field(..., title="The response model from the database")
    db_request_body_with_goals: dict = Field(..., title="The request body from the database")
    db_response_body_with_goals: dict = Field(..., title="The response body from the database")
    request_with_goals_system_prompt: str = Field(..., title="The system prompt from the request with goals")
    request_without_goals_system_prompt: str = Field(..., title="The system prompt from the request without goals")
    request_body_without_goals: dict = Field(..., title="The request body without goals")
    response_body_without_goals: dict = Field(..., title="The response body without goals")
    main_response_with_goals: str = Field(
        ..., title="The main response text with goals, i.e. this is the text to use for comparison"
    )
    main_response_without_goals: str = Field(
        ..., title="The main response text without goals, i.e. this is the text to use for comparison"
    )
