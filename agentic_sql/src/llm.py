from datetime import datetime
from openai import AsyncOpenAI
from pydantic import Field, BaseModel
from typing import Dict, List, Optional

from .instruct_llm import ainstruct_llm
from .models import SQLQueryResponse, SQLQueryFeedbackResponse
from .settings import settings, logger
from common.prompt_template_engine import build_prompt

# Create and configure Anthropic and OpenAI clients
async_openai_client = AsyncOpenAI(
    api_key=settings.openai_api_key.get_secret_value(),
    organization=settings.openai_organization,
)


class SQLGenerationResponse(BaseModel):
    """Response model for SQL generation."""

    sql: str = Field(..., description="The generated SQL query")
    explanation: str = Field(..., description="Explanation of what the SQL query does")


class SQLFeedbackResponse(BaseModel):
    """Response model for SQL feedback."""

    verified: bool = Field(..., description="Whether the query correctly answers the question")
    feedback: str = Field(..., description="Feedback about the query, including any improvements")
    suggested_sql: Optional[str] = Field(None, description="An improved SQL query if the original had issues")


class SQLSelfReflectionResponse(BaseModel):
    """Response model for SQL self-reflection."""

    improved_sql: str = Field(..., description="The improved SQL query based on feedback")
    explanation: str = Field(..., description="Explanation of changes made to improve the query")


class SQLAnswerResponse(BaseModel):
    """Response model for generating a concise answer to the user's question based on SQL results."""

    answer: str = Field(..., description="Concise answer to the user's original question")
    caveats: Optional[str] = Field(None, description="Any important caveats or assumptions")
    follow_up_questions: Optional[str] = Field(None, description="Potential follow-up questions")


async def generate_sql(
    question: str, table_schemas: Optional[str] = None, similar_queries: Optional[List[Dict[str, str]]] = None
) -> SQLQueryResponse:
    """
    Generate a SQL query from a natural language question.

    Args:
        question: The natural language question to convert to SQL
        table_schemas: Formatted schema information for relevant tables
        similar_queries: List of similar questions and their SQL queries

    Returns:
        SQLQueryResponse with generated SQL and explanation
    """
    # Get the current datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format the similar queries for logging
    formatted_similar_queries = ""
    if similar_queries:
        formatted_similar_queries = "\n\n".join([f"Q: {ex['question']}\nSQL: {ex['sql']}" for ex in similar_queries])
        logger.debug(f"Similar examples found: {', '.join([ex['question'][:50] + '...' for ex in similar_queries])}")

    # Build the prompt
    user_prompt = build_prompt("sql_generation/user.txt.jinja", question=question)
    logger.debug(f"User prompt for LLM:\n{user_prompt}")

    # Always use BigQuery
    db_type = "BigQuery"

    system_prompt = build_prompt(
        "sql_generation/system.txt.jinja",
        ddl_list=table_schemas,
        question_sql_pairs=similar_queries,
        current_datetime=current_datetime,
    )
    logger.debug(f"System prompt for LLM (truncated):\n{system_prompt[:500]}...")

    # Call the LLM
    logger.info(f"Sending prompt to LLM (model: {settings.llm_model})")
    response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=SQLGenerationResponse,
        llm_model=settings.llm_model,
    )

    # Log LLM response
    logger.info("Received response from LLM")
    logger.debug(f"Generated SQL:\n{response.sql}")
    logger.debug(f"Explanation: {response.explanation}")

    # Return SQL with database type context and similar queries data for proper execution later
    return SQLQueryResponse(
        sql=response.sql,
        explanation=response.explanation,
        metadata={
            "database_type": db_type,
            "similar_queries_count": len(similar_queries) if similar_queries else 0,
            "similar_queries": formatted_similar_queries,
        },
    )


async def get_sql_feedback(
    question: str,
    sql: str,
    results: str,
    table_schemas: Optional[str] = None,
    similar_queries: Optional[List[Dict[str, str]]] = None,
) -> SQLQueryFeedbackResponse:
    """
    Get feedback on a SQL query and its results.

    Args:
        question: The original natural language question
        sql: The SQL query
        results: The results of executing the query
        table_schemas: Optional formatted schema information for relevant tables
        similar_queries: Optional list of similar questions and their SQL queries

    Returns:
        SQLQueryFeedbackResponse with verification and feedback
    """
    # Get the current datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Getting feedback on SQL query for question: '{question}'")
    logger.debug(f"SQL being evaluated:\n{sql}")
    logger.debug(
        f"Query results (truncated):\n{results[:500]}..." if len(results) > 500 else f"Query results:\n{results}"
    )

    # Build the prompt
    prompt_args = {"question": question, "sql": sql, "results": results}

    # Add table schemas if available
    if table_schemas:
        prompt_args["table_schemas"] = table_schemas
        logger.debug("Including table schemas in feedback generation")

    # Add similar queries if available
    if similar_queries:
        prompt_args["similar_queries"] = similar_queries
        logger.debug(f"Including {len(similar_queries)} similar queries in feedback generation")

    user_prompt = build_prompt("sql_generation/feedback_user.txt.jinja", **prompt_args)
    logger.debug(f"Feedback user prompt:\n{user_prompt[:500]}...")

    system_prompt = build_prompt("sql_generation/feedback_system.txt.jinja", current_datetime=current_datetime)
    logger.debug(f"Feedback system prompt (truncated):\n{system_prompt[:500]}...")

    # Call the LLM
    logger.info(f"Sending feedback prompt to LLM (model: {settings.llm_model})")
    response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=SQLFeedbackResponse,
        llm_model=settings.llm_model,
    )

    # Log feedback response
    logger.info(f"Received feedback from LLM - Query verified: {response.verified}")
    logger.debug(f"Feedback: {response.feedback}")
    if response.suggested_sql:
        logger.debug(f"Suggested SQL: {response.suggested_sql}")

    return SQLQueryFeedbackResponse(
        verified=response.verified, feedback=response.feedback, suggested_sql=response.suggested_sql
    )


async def get_sql_self_reflection(
    question: str,
    sql: str,
    results: str,
    feedback: str,
    attempt_counter: int,
    suggested_sql: Optional[str] = None,
    table_schemas: Optional[str] = None,
    similar_queries: Optional[List[Dict[str, str]]] = None,
) -> SQLSelfReflectionResponse:
    """
    Generate an improved SQL query based on feedback through self-reflection.

    Args:
        question: The original natural language question
        sql: The previous SQL query that needs improvement
        results: The results of executing the previous query
        feedback: The feedback about the previous query
        attempt_counter: The current attempt number
        suggested_sql: Optional suggested SQL from the feedback
        table_schemas: Optional formatted schema information for relevant tables
        similar_queries: Optional list of similar questions and their SQL queries

    Returns:
        SQLSelfReflectionResponse with an improved SQL query and explanation
    """
    # Get the current datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Performing self-reflection for SQL query (attempt #{attempt_counter}) for question: '{question}'")
    logger.debug(f"Previous SQL:\n{sql}")
    logger.debug(f"Feedback: {feedback}")

    # Build the prompt
    prompt_args = {
        "question": question,
        "sql": sql,
        "results": results,
        "feedback": feedback,
        "attempt_counter": attempt_counter,
        "current_datetime": current_datetime,
    }

    if suggested_sql:
        prompt_args["suggested_sql"] = suggested_sql
        logger.debug(f"Including suggested SQL in self-reflection:\n{suggested_sql}")

    # Add table schemas if available
    if table_schemas:
        prompt_args["table_schemas"] = table_schemas
        logger.debug("Including table schemas in self-reflection")

    # Add similar queries if available
    if similar_queries:
        prompt_args["similar_queries"] = similar_queries
        logger.debug(f"Including {len(similar_queries)} similar queries in self-reflection")

    user_prompt = build_prompt("sql_generation/self_reflection_user.txt.jinja", **prompt_args)
    logger.debug(f"Self-reflection user prompt:\n{user_prompt[:500]}...")

    system_prompt = build_prompt("sql_generation/self_reflection_system.txt.jinja", current_datetime=current_datetime)
    logger.debug(f"Self-reflection system prompt (truncated):\n{system_prompt[:500]}...")

    # Call the LLM
    logger.info(f"Sending self-reflection prompt to LLM (model: {settings.llm_model})")
    response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=SQLSelfReflectionResponse,
        llm_model=settings.llm_model,
    )

    # Log response
    logger.info("Received self-reflection response from LLM")
    logger.debug(f"Improved SQL:\n{response.improved_sql}")
    logger.debug(f"Explanation: {response.explanation}")

    return SQLSelfReflectionResponse(improved_sql=response.improved_sql, explanation=response.explanation)


async def generate_answer(
    question: str, sql: str, results: str, table_schemas: Optional[str] = None
) -> SQLAnswerResponse:
    """
    Generate a concise answer to the user's original question based on the SQL query and results.

    Args:
        question: The original natural language question
        sql: The SQL query used to answer the question
        results: The results of executing the query
        table_schemas: Optional formatted schema information for relevant tables

    Returns:
        SQLAnswerResponse with a concise answer, caveats, and follow-up questions
    """
    # Get the current datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Generating concise answer for question: '{question}'")

    # Build the prompt
    prompt_args = {"question": question, "sql": sql, "results": results, "current_datetime": current_datetime}

    if table_schemas:
        prompt_args["table_schemas"] = table_schemas
        logger.debug("Including table schemas in answer generation")

    user_prompt = build_prompt("sql_generation/answer_user.txt.jinja", **prompt_args)
    logger.debug(f"Answer user prompt:\n{user_prompt[:500]}...")

    system_prompt = build_prompt("sql_generation/answer_system.txt.jinja", current_datetime=current_datetime)
    logger.debug(f"Answer system prompt (truncated):\n{system_prompt[:500]}...")

    # Call the LLM
    logger.info(f"Sending answer prompt to LLM (model: {settings.llm_model})")
    response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=SQLAnswerResponse,
        llm_model=settings.llm_model,
    )

    # Log response
    logger.info("Received answer response from LLM")
    logger.debug(f"Answer: {response.answer}")
    if response.caveats:
        logger.debug(f"Caveats: {response.caveats}")
    if response.follow_up_questions:
        logger.debug(f"Follow-up questions: {response.follow_up_questions}")

    return SQLAnswerResponse(
        answer=response.answer, caveats=response.caveats, follow_up_questions=response.follow_up_questions
    )
