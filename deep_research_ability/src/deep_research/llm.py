from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List

from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt
from ..settings import OPENAI_O3_MINI, OPENAI_GPT4O, logger
from ..deep_research.get_context import SearchResults


class Query(BaseModel):
    model_config = ConfigDict(validate_assignment=True)  # Enable validation on assignment

    query: str = Field(..., description="Generated serp query; search uses both sparse and dense representations.")
    research_goal: str = Field(
        ...,
        description="First talk about the goal of the research that this query is meant to accomplish, then go deeper into how to advance the research once the results are found, mention additional research directions. Be as specific as possible, especially for additional research directions.",
    )
    created_after: datetime | None = Field(
        None, description="Only include content created after this date (ISO format)"
    )
    created_before: datetime | None = Field(
        None, description="Only include content created before this date (ISO format)"
    )

    @field_validator("created_after", "created_before", mode="before")
    @classmethod
    def parse_datetime(cls, value):
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            # Handle both date-only and datetime strings
            if "T" not in str(value) and " " not in str(value):
                value = f"{value}T00:00:00"
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    @field_validator("created_before")
    @classmethod
    def ensure_after_before_order(cls, v, info):
        if v and info.data.get("created_after") and v < info.data["created_after"]:
            # Swap dates if they're in wrong order
            return info.data["created_after"]
        return v


class GeneratedResearchQuery(BaseModel):
    """Models the GeneratedResearchQuery class from production."""

    title: str = Field(
        ..., description="A user-facing title for the query, it'll be prefaced with 'Reading about' in the UI"
    )
    terms: str = Field(..., description="The search terms")
    starts_at: datetime | None = Field(None, description="The earliest datetime to allow results")
    ends_at: datetime | None = Field(None, description="The latest datetime to allow results")
    search_type: str = Field("internal", description="Whether to search internal content or the web")
    goals: str = Field(
        ...,
        description=(
            "First talk about the goal of the research that this query is meant to accomplish, "
            "then go deeper into how to advance the research once the results are found, mention additional research "
            "directions. Be as specific as possible, especially for additional research directions."
        ),
    )

    @property
    def created_after(self) -> datetime | None:
        return self.starts_at

    @property
    def created_before(self) -> datetime | None:
        return self.ends_at


class QueryGenResponse(BaseModel):
    queries: List[Query] = Field(..., description="Generated search queries for the SERP.")


class SearchResultReviewResponse(BaseModel):
    learnings: List[str] = Field(..., description="Learnings extracted from the search results.")
    follow_up_questions: List[str] = Field(
        ..., description="Follow-up questions to refine the research direction and answer open questions."
    )


class GeneratedResearchQueryReview(BaseModel):
    """Models the GeneratedResearchQueryReview class from production."""

    learnings: list[str] = Field(..., description="List of learnings extracted from search results")
    follow_up_questions: list[str] = Field(
        ..., description="List of follow-up questions to research the topic further"
    )
    follow_up_title: str = Field(
        ..., description="A one or two word user-facing title for the follow-up questions, shown as progress in the UI"
    )


class ReportResponse(BaseModel):
    title: str = Field(..., description="Title of the research report; should be succint and informative.")
    report: str = Field(..., description="Markdown formatted final report")
    key_takeaways: list[str] = Field(
        ..., description="Key takeaways from the research to provide to an Executive reader."
    )
    further_research: list[str] = Field(
        ..., description="Suggestions for further research to provide to a Researcher."
    )


class FeedbackResponse(BaseModel):
    questions: list[str] = Field(
        ...,
        description="Follow-up questions to refine the research direction. Use these to help target both the content and tone of the research report.",
    )


class ReRankResponse(BaseModel):
    """Response model for re-ranking search results."""

    reranked_indices: list[int] = Field(..., description="Indices of the original results in preferred order")


class ProofreadResponse(BaseModel):
    """Response model for the proofread report."""

    revised_report: str = Field(..., description="The revised, more focused report")
    revision_notes: list[str] = Field(..., description="Notes about what was changed and why")


async def generate_serp_queries(
    query: str,
    num_queries: int = 3,
    learnings: List[str] | None = None,
) -> List[dict]:
    """Generate search queries based on the input query and previous learnings."""
    user_prompt = build_prompt(
        "deep_research/generate_queries.txt.jinja",
        query=query,
        num_queries=num_queries,
        learnings=learnings,
    )

    system_prompt = build_prompt(
        "deep_research/global_system_prompt.txt.jinja",
        now=datetime.now().isoformat(),
    )

    response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=QueryGenResponse,
        llm_model=OPENAI_O3_MINI,
        temperature=0.15,  # Note: Reasoning models such as o1 and o3 ignore this parameter as it is not allowed
    )

    return response


async def process_serp_result(
    query: str,
    result: SearchResults,
    num_learnings: int = 3,
    num_follow_up_questions: int = 3,
) -> GeneratedResearchQueryReview:
    """
    Process search results and extract learnings and follow-up questions.

    Args:
        query: The search query
        result: Search results
        num_learnings: Maximum number of learnings to extract
        num_follow_up_questions: Maximum number of follow-up questions to generate

    Returns:
        GeneratedResearchQueryReview with learnings and follow-up questions
    """
    # Import here to avoid circular imports
    from .framework import MAX_NUMBER_OF_LEARNINGS, MAX_NUMBER_OF_FOLLOW_UP_QUESTIONS

    # Use constants from framework but allow overrides
    final_num_learnings = num_learnings or MAX_NUMBER_OF_LEARNINGS
    final_num_follow_up = num_follow_up_questions or MAX_NUMBER_OF_FOLLOW_UP_QUESTIONS

    # Log what we're processing
    logger.info(f"Processing {len(result.data)} search results for query: '{query}'")

    # If we have no results, return an empty review with a placeholder question
    if not result.data:
        logger.warning(f"No results to process for query: '{query}'")
        return GeneratedResearchQueryReview(
            learnings=[],
            follow_up_questions=[f"Can you provide more specific information about {query}?"],
            follow_up_title=f"No Results {datetime.now().strftime('%H:%M')}",
        )

    user_prompt = build_prompt(
        "deep_research/process_results.txt.jinja",
        query=query,
        results=result.data,
        num_learnings=final_num_learnings,
        num_follow_up_questions=final_num_follow_up,
    )

    system_prompt = build_prompt(
        "deep_research/global_system_prompt.txt.jinja",
        now=datetime.now().isoformat(),
    )

    # First get the standard response format
    standard_response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=SearchResultReviewResponse,
        llm_model=OPENAI_O3_MINI,
        temperature=0.15,
    )

    # Convert to production format with a title
    follow_up_title = f"Follow-up {datetime.now().strftime('%H:%M')}"

    # Log the results
    logger.info(f"Extracted {len(standard_response.learnings)} learnings")
    if standard_response.learnings:
        logger.info(f"Sample learning: {standard_response.learnings[0][:100]}...")

    logger.info(f"Generated {len(standard_response.follow_up_questions)} follow-up questions")
    if standard_response.follow_up_questions:
        logger.info(f"Sample question: {standard_response.follow_up_questions[0]}")

    # Return in the production format
    return GeneratedResearchQueryReview(
        learnings=standard_response.learnings[:final_num_learnings],
        follow_up_questions=standard_response.follow_up_questions[:final_num_follow_up],
        follow_up_title=follow_up_title,
    )


async def generate_research_report(
    query: str,
    learnings: list[str],
    feedback_qa: list[tuple[str, str]] | None = None,
) -> dict[str, str | list[str]]:
    """Generate a comprehensive research report from findings."""
    prompt = build_prompt(
        "deep_research/generate_report.txt.jinja",
        query=query,
        learnings=learnings,
        feedback_qa=feedback_qa,
    )

    system_prompt = build_prompt(
        "deep_research/global_system_prompt.txt.jinja",
        now=datetime.now().isoformat(),
    )

    response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=prompt,
        response_model=ReportResponse,
        llm_model=OPENAI_O3_MINI,
        max_tokens=4000,  # Ensure enough tokens for a detailed report
        temperature=0.15,  # Note: Reasoning models such as o1 and o3 ignore this parameter as it is not allowed
    )

    return response


async def generate_feedback(
    query: str,
    num_questions: int = 3,
) -> list[str]:
    """Generate follow-up questions to refine the research direction."""
    prompt = build_prompt(
        "deep_research/generate_feedback.txt.jinja",
        query=query,
        num_questions=num_questions,
    )

    system_prompt = build_prompt(
        "deep_research/global_system_prompt.txt.jinja",
        now=datetime.now().isoformat(),
    )

    response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=prompt,
        response_model=FeedbackResponse,
        llm_model=OPENAI_O3_MINI,
        temperature=0.15,  # Note: Reasoning models such as o1 and o3 ignore this parameter as it is not allowed
    )

    return response


async def rerank_results(
    query: str,
    results: SearchResults,
    top_k: int,
) -> SearchResults:
    """Re-rank search results based on relevance to the query."""

    # Log the reranking operation
    logger.info(f"Reranking {len(results.data)} results for query: '{query}'")
    logger.info(f"Will keep top {top_k} results after reranking")

    # Check if we have results to rerank
    if not results.data:
        logger.warning("No results to rerank")
        return results
    user_prompt = build_prompt(
        "deep_research/rerank_results.txt.jinja",
        query=query,
        results=[
            {
                "index": i,
                "url": item.url,
                "content": item.markdown,
            }
            for i, item in enumerate(results.data)
        ],
        top_k=top_k,
    )

    system_prompt = build_prompt(
        "deep_research/global_system_prompt.txt.jinja",
        now=datetime.now().isoformat(),
    )

    response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ReRankResponse,
        llm_model=OPENAI_O3_MINI,
        temperature=0.15,
    )

    # Return only the top_k reranked results in the original format
    reranked_data = [results.data[i] for i in response.reranked_indices[:top_k]]
    return SearchResults(data=reranked_data)


async def combine_query_with_feedback(
    initial_query: str,
    feedback_questions: list[str],
    feedback_answers: list[str],
) -> str:
    """Combine initial query with feedback Q&A to create a refined query."""
    qa_pairs = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(feedback_questions, feedback_answers))

    return f"Initial Query: {initial_query}\n\nFollow-up Questions and Answers:\n{qa_pairs}".strip()


async def proofread_report(
    report: str,
    original_query: str,
    feedback_qa: list[tuple[str, str]] | None = None,
) -> ProofreadResponse:
    """
    Proofread and revise the report to ensure relevance to the original research goal.

    Args:
        report: The original report to revise
        original_query: The user's original research query
        feedback_qa: Optional list of (question, answer) tuples from user feedback
    """
    qa_context = ""
    if feedback_qa:
        qa_pairs = "\n".join(f"Q: {q}\nA: {a}" for q, a in feedback_qa)
        qa_context = f"\nUser Feedback Context:\n{qa_pairs}"

    user_prompt = build_prompt(
        "deep_research/proofread_report.txt.jinja",
        report=report,
        original_query=original_query,
        feedback_context=qa_context,
    )

    system_prompt = build_prompt(
        "deep_research/global_system_prompt.txt.jinja",
        now=datetime.now().isoformat(),
    )

    return await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ProofreadResponse,
        llm_model=OPENAI_GPT4O,
        temperature=0.15,
    )
