from dataclasses import dataclass, field
from typing import List, Optional, TypedDict, Callable, Awaitable, Dict, Any, Set
import asyncio
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .get_context import SearchProvider, ContextSource
from .llm import (
    Query,
    GeneratedResearchQuery,
    GeneratedResearchQueryReview,
    generate_serp_queries,
    process_serp_result,
    generate_research_report,
    generate_feedback,
    combine_query_with_feedback,
    rerank_results,
    proofread_report,
)
from ..settings import settings, logger  # Corrected import
from .tree_visualizer import ResearchTree

# Constants from production
MAX_NUMBER_OF_SEARCH_RESULTS = 10
MAX_NUMBER_OF_LEARNINGS = 3
MAX_NUMBER_OF_FOLLOW_UP_QUESTIONS = 3


@dataclass
class ResearchProgress:
    """Track and report research progress for UI display."""

    current_depth: int
    total_depth: int
    current_breadth: int
    total_breadth: int
    current_query: Optional[str]
    total_queries: int
    completed_queries: int


@dataclass
class ResearchIteration:
    """Models the ResearchIteration class from production."""

    id: str
    title: str
    directions: str
    queries_count: int
    depth: int = 0
    parent_iteration_id: Optional[str] = None
    parent_query_id: Optional[str] = None


@dataclass
class ResearchQuery:
    """Models the ResearchQuery class from production."""

    id: str
    iteration_id: str
    title: str
    query: str
    goals: str
    urls: List[str] = field(default_factory=list)
    learnings: List[str] = field(default_factory=list)
    completed: bool = False
    search_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Research:
    """Models the Research class from production."""

    id: str
    topic: str
    max_breadth: int
    max_depth: int
    sources: List[ContextSource] = field(default_factory=list)
    iterations: Dict[str, ResearchIteration] = field(default_factory=dict)
    queries: Dict[str, ResearchQuery] = field(default_factory=dict)

    @property
    def learnings(self) -> List[str]:
        """Aggregate all learnings from queries."""
        all_learnings: Set[str] = set()
        for query in self.queries.values():
            all_learnings.update(query.learnings)
        return list(all_learnings)

    @property
    def visited_urls(self) -> List[str]:
        """Aggregate all URLs from queries."""
        all_urls: Set[str] = set()
        for query in self.queries.values():
            all_urls.update(query.urls)
        return list(all_urls)

    @property
    def is_completed(self) -> bool:
        """Check if the research is completed."""
        # Research is complete when we've reached the max depth
        max_depth_iterations = [it for it in self.iterations.values() if it.depth == (self.max_depth - 1)]
        return len(max_depth_iterations) > 0


class ResearchResult(TypedDict):
    """Result of a research operation."""

    learnings: List[str]
    visited_urls: List[str]


@dataclass
class ResearchSearch:
    """Base class for search operations."""

    iteration: ResearchIteration
    generated_query: GeneratedResearchQuery
    research_query: Optional[ResearchQuery] = None
    results: Any = None

    async def perform(self):
        """Perform the search operation."""
        raise NotImplementedError


@dataclass
class InternalResearch(ResearchSearch):
    """Internal search implementation."""

    async def perform(self, search_provider: SearchProvider):
        """Perform internal content search."""
        # Create a research query
        self.research_query = ResearchQuery(
            id=str(uuid4()),
            iteration_id=self.iteration.id,
            title=self.generated_query.title,
            query=self.generated_query.terms,
            goals=self.generated_query.goals,
        )

        # Execute search
        self.results = await search_provider.search(
            query=self.generated_query.terms,
            limit=MAX_NUMBER_OF_SEARCH_RESULTS,
            created_after=self.generated_query.created_after,
            created_before=self.generated_query.created_before,
        )

        # Store URLs from results
        if self.results and self.results.data:
            self.research_query.urls = [item.url for item in self.results.data]

        return self.research_query, self.results


class ResearchOutput(TypedDict):
    """Final output of the research process."""

    title: str
    learnings: list[str]
    visited_urls: list[str]
    report: str
    key_takeaways: list[str]
    further_research: list[str]
    revised_report: str
    revision_notes: list[str]
    csv_path: str  # Path to the CSV file with detailed research data
    tree_viz_path: str  # Path to the tree visualization image
    tree_json_path: str  # Path to the tree visualization JSON
    research_json_path: str  # Path to the detailed research JSON data


async def research_iteration(
    research: Research,
    iteration: ResearchIteration,
    search_provider: SearchProvider,
    on_progress: Optional[Callable[[ResearchProgress], None]] = None,
    concurrency_limit: int = 2,
    top_k: int = 5,
) -> List[ResearchQuery]:
    """
    Perform a single research iteration, similar to ResearchIterationJob in production.
    """
    # Generate queries for this iteration
    logger.info(f"Generating queries for iteration: {iteration.title}")
    serp_queries = await generate_serp_queries(
        query=iteration.directions, num_queries=iteration.queries_count, learnings=research.learnings
    )

    if not serp_queries or not serp_queries.queries:
        logger.warning("No queries generated for this iteration!")
        return []

    # Progress tracking
    progress = ResearchProgress(
        current_depth=iteration.depth,
        total_depth=research.max_depth,
        current_breadth=iteration.queries_count,
        total_breadth=research.max_breadth,
        current_query=None,
        total_queries=len(serp_queries.queries),
        completed_queries=0,
    )

    def report_progress(**kwargs):
        if on_progress:
            for k, v in kwargs.items():
                setattr(progress, k, v)
            on_progress(progress)

    # Store the iteration in the research object
    research.iterations[iteration.id] = iteration

    # Process each query in parallel with concurrency limit
    semaphore = asyncio.Semaphore(concurrency_limit)
    completed_queries: List[ResearchQuery] = []

    async def process_query(query_idx: int, serp_query: Query) -> Optional[ResearchQuery]:
        async with semaphore:
            try:
                # Update progress
                report_progress(current_query=serp_query.query)

                # Convert to GeneratedResearchQuery format for compatibility
                gen_query = GeneratedResearchQuery(
                    title=f"Query {query_idx + 1}",
                    terms=serp_query.query,
                    search_type="internal",
                    goals=serp_query.research_goal,
                    starts_at=serp_query.created_after,
                    ends_at=serp_query.created_before,
                )

                # Create research search
                search = InternalResearch(iteration=iteration, generated_query=gen_query)

                # Perform search
                research_query, results = await search.perform(search_provider)  # Get results

                if not results or not results.data:
                    logger.warning(f"No results found for query: {serp_query.query}")
                    research_query.completed = True
                    research.queries[research_query.id] = research_query
                    return research_query

                # Log original search results
                logger.info(f"Query '{serp_query.query}' returned {len(results.data)} results")
                if results.data:
                    for idx, item in enumerate(results.data[:2]):  # Log first 2 results only
                        logger.info(f"  Result {idx + 1}: {item.url}")

                # Rerank results
                reranked_results = await rerank_results(query=serp_query.research_goal, results=results, top_k=top_k)

                # Log reranked results
                logger.info(f"After reranking, kept {len(reranked_results.data)} out of {len(results.data)} results")

                # Process search results
                follow_up_count = min(MAX_NUMBER_OF_FOLLOW_UP_QUESTIONS, max(1, iteration.queries_count // 2))
                query_review = await process_serp_result(
                    query=serp_query.query,
                    result=reranked_results,
                    num_learnings=MAX_NUMBER_OF_LEARNINGS,
                    num_follow_up_questions=follow_up_count,
                )

                # Update research query with learnings
                research_query.learnings = query_review.learnings
                research_query.completed = True

                # *** COLLECT SEARCH RESULT METADATA ***
                search_metadata = {}
                if results:  # Check if results exist
                    search_metadata["total_results"] = len(results.data) if results.data else 0
                    search_metadata["results_by_source"] = {}
                    if results.data:  # Check if results.data exists
                        for item in results.data:
                            source = item.source  # Get the source from the result item
                            search_metadata["results_by_source"][source] = (
                                search_metadata["results_by_source"].get(source, 0) + 1
                            )
                    if serp_query.created_after:
                        search_metadata["created_after"] = serp_query.created_after.isoformat()
                    if serp_query.created_before:
                        search_metadata["created_before"] = serp_query.created_before.isoformat()

                # Store metadata in the research query
                research_query.search_results = search_metadata

                # Store in research object
                research.queries[research_query.id] = research_query

                # Update progress
                report_progress(completed_queries=progress.completed_queries + 1)

                return research_query

            except Exception as e:
                logger.error(f"Error processing query {serp_query.query}: {e}")  # Added error to log
                return None

    # Process all queries in parallel
    query_tasks = [process_query(i, query) for i, query in enumerate(serp_queries.queries)]
    results = await asyncio.gather(*query_tasks)

    # Filter out None results
    completed_queries = [query for query in results if query]

    return completed_queries


async def start_next_iteration(
    research: Research,
    previous_iteration: ResearchIteration,
    query: ResearchQuery,
    follow_up_questions: List[str],
    follow_up_title: str,
) -> ResearchIteration:
    """Create the next research iteration."""
    next_directions = (
        f"Previous research goal: {query.goals}\n\n"
        f"Follow-up research directions: \n- "
        f"{'\n- '.join(follow_up_questions)}"
    )

    next_iteration = ResearchIteration(
        id=str(uuid4()),
        title=follow_up_title,
        directions=next_directions,
        queries_count=max(1, previous_iteration.queries_count // 2),
        depth=previous_iteration.depth + 1,
        parent_iteration_id=previous_iteration.id,
        parent_query_id=query.id,
    )

    return next_iteration


async def deep_research(
    query: str,
    breadth: int,
    depth: int,
    search_provider: SearchProvider,
    on_progress: Optional[Callable[[ResearchProgress], None]] = None,
    concurrency_limit: int = 2,
    top_k: int = 5,
) -> Research:
    """
    Perform deep research on a topic using iteration-based approach.

    This implementation aligns with the production Research flow, using
    explicit iterations and queries.

    Args:
        query: The research query
        breadth: Number of parallel search paths to explore (max_breadth)
        depth: How deep to go in the research tree (max_depth)
        search_provider: Provider for search functionality
        on_progress: Callback for progress updates
        concurrency_limit: Maximum concurrent searches
        top_k: Number of top results to keep after reranking
    """
    # Create research object
    research = Research(
        id=str(uuid4()),
        topic=query,
        max_breadth=breadth,
        max_depth=depth,
        sources=search_provider.sources,
    )

    # Create first iteration
    first_iteration = ResearchIteration(
        id=str(uuid4()), title="Planning", directions=query, queries_count=breadth, depth=0
    )

    # Start with the first iteration
    current_depth = 0
    next_iteration = first_iteration
    while current_depth < depth:
        # Get the current iteration
        current_iteration = first_iteration if current_depth == 0 else next_iteration

        # Execute the iteration
        completed_queries = await research_iteration(
            research=research,
            iteration=current_iteration,
            search_provider=search_provider,
            on_progress=on_progress,
            concurrency_limit=concurrency_limit,
            top_k=top_k,
        )

        # If no queries were completed or we've reached max depth, break
        if not completed_queries or current_depth >= depth - 1:
            break

        # Find latest completed query to base the next iteration on
        latest_query = max(completed_queries, key=lambda q: q.id)

        # Get the query result for follow-up questions
        query_result = None
        for q in research.queries.values():
            if q.id == latest_query.id:
                query_result = q
                break

        if not query_result or not query_result.learnings:
            break

        # In production, the review would be stored from the query process
        # Here we simulate it based on the stored learnings
        # For brevity, create a plausible title and use follow-up questions from previous iteration
        follow_up_count = min(MAX_NUMBER_OF_FOLLOW_UP_QUESTIONS, max(1, current_iteration.queries_count // 2))
        review = GeneratedResearchQueryReview(
            learnings=query_result.learnings,
            follow_up_questions=[
                f"How does {query_result.learnings[i % len(query_result.learnings)]} impact the overall research topic?"
                if query_result.learnings
                else f"What additional information is needed about {query_result.query}?"
                for i in range(follow_up_count)
            ],
            follow_up_title=f"Depth {current_depth + 1}",
        )

        # Create next iteration
        next_iteration = await start_next_iteration(
            research=research,
            previous_iteration=current_iteration,
            query=query_result,
            follow_up_questions=review.follow_up_questions,
            follow_up_title=review.follow_up_title,
        )

        # Increment depth
        current_depth += 1

    return research


def export_research_to_csv(
    research: Research,
    output_path: Path,
    report_title: str,
    key_takeaways: List[str],
    further_research: List[str],
    revision_notes: List[str],
) -> Path:
    """
    Export detailed research data to CSV file.

    Includes:
    - Overall research metadata
    - Iterations information
    - Queries and results
    - Report details
    """
    # Define CSV filenames
    sanitized_title = "".join(c if c.isalnum() else "_" for c in report_title)[:100]
    csv_filename = f"{sanitized_title}_research_data.csv"
    csv_path = output_path / csv_filename

    # Create a dict of data to export
    data = []

    # Add overall research data
    data.append(
        {
            "section": "metadata",
            "topic": research.topic,
            "max_breadth": research.max_breadth,
            "max_depth": research.max_depth,
            "total_iterations": len(research.iterations),
            "total_queries": len(research.queries),
            "total_urls": len(research.visited_urls),
            "total_learnings": len(research.learnings),
            "title": report_title,
            "timestamp": datetime.now().isoformat(),
            "sources": ", ".join([source.value for source in research.sources]),
        }
    )

    # Add iterations data
    for it_id, iteration in research.iterations.items():
        data.append(
            {
                "section": "iteration",
                "iteration_id": it_id,
                "title": iteration.title,
                "directions": iteration.directions,
                "queries_count": iteration.queries_count,
                "depth": iteration.depth,
                "parent_iteration_id": iteration.parent_iteration_id,  # Added parent_iteration_id
                "parent_query_id": iteration.parent_query_id,  # Added parent_query_id
            }
        )

    # Add queries data
    for query_id, query in research.queries.items():
        iteration = research.iterations.get(query.iteration_id, None)
        depth = iteration.depth if iteration else "unknown"

        data.append(
            {
                "section": "query",
                "query_id": query_id,
                "iteration_id": query.iteration_id,
                "iteration_depth": depth,
                "title": query.title,
                "query": query.query,
                "goals": query.goals,
                "urls": ", ".join(query.urls) if query.urls else "",
                "url_count": len(query.urls) if query.urls else 0,
                "learnings": ", ".join(query.learnings) if query.learnings else "",
                "completed": query.completed,
            }
        )

    # Add source URLs as individual entries for easy analysis
    for idx, url in enumerate(research.visited_urls):
        data.append({"section": "source_url", "index": idx, "url": url})

    # Add report details
    for idx, takeaway in enumerate(key_takeaways):
        data.append({"section": "key_takeaway", "index": idx, "content": takeaway})

    for idx, research_topic in enumerate(further_research):
        data.append({"section": "further_research", "index": idx, "content": research_topic})

    for idx, note in enumerate(revision_notes):
        data.append({"section": "revision_note", "index": idx, "content": note})

    # Add a list of learnings for easier analysis
    for idx, learning in enumerate(research.learnings):
        data.append({"section": "learning", "index": idx, "content": learning})

    # Get all possible field names from data
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())

    # Write data to CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(list(fieldnames)))
        writer.writeheader()
        writer.writerows(data)

    logger.info(f"Research data exported to CSV: {csv_path}")
    return csv_path


async def research_with_report(
    query: str,
    breadth: int,
    depth: int,
    search_provider: SearchProvider,
    on_progress: Optional[Callable[[ResearchProgress], None]] = None,
    feedback_qa: list[tuple[str, str]] | None = None,
) -> ResearchOutput:
    """
    Perform deep research and generate a comprehensive report.
    """
    # Use the new iteration-based approach
    research = await deep_research(
        query=query,
        breadth=breadth,
        depth=depth,
        search_provider=search_provider,
        on_progress=on_progress,
    )

    # Generate final research report
    report_results = await generate_research_report(
        query=query,
        learnings=research.learnings,
        feedback_qa=feedback_qa,
    )

    # Add proofreading step
    logger.info("Proofreading and revising report for relevance...")
    proofread_results = await proofread_report(
        report=report_results.report,
        original_query=query,
        feedback_qa=feedback_qa,
    )

    # Add sources section to the report
    sources_section = "\n\n## Sources\n\n" + "\n".join(
        f"- {url}" for url in research.visited_urls
    )  # Corrected f-string
    report_results.report = report_results.report + sources_section

    # Log revision notes
    for note in proofread_results.revision_notes:
        logger.debug(f"Revision note: {note}")  # Added note to log

    # Create filename from the LLM-generated title
    sanitized_title = "".join(c if c.isalnum() else "_" for c in report_results.title)[:100]
    output_filename = f"{sanitized_title}.md"  # Added .md extension

    # Write report to file
    output_path = settings.output_path / output_filename
    output_path.write_text(report_results.report, encoding="utf-8")

    revised_output_path = settings.output_path / f"{sanitized_title}_revised.md"  # Added sanitized title
    revised_output_path.write_text(proofread_results.revised_report, encoding="utf-8")

    # Export detailed research data to CSV
    csv_path = export_research_to_csv(
        research=research,
        output_path=settings.output_path,
        report_title=report_results.title,
        key_takeaways=report_results.key_takeaways,
        further_research=report_results.further_research,
        revision_notes=proofread_results.revision_notes,
    )

    # Convert research object to a dictionary for the visualizer
    research_data = {
        "metadata": {
            "id": research.id,
            "topic": research.topic,
            "max_breadth": research.max_breadth,
            "max_depth": research.max_depth,
            "sources": [source.value for source in research.sources],
            "total_iterations": len(research.iterations),
            "total_queries": len(research.queries),
            "timestamp": datetime.now().isoformat(),
        },
        "iterations": {
            it_id: {
                "id": it_id,
                "title": iteration.title,
                "directions": iteration.directions,
                "queries_count": iteration.queries_count,
                "depth": iteration.depth,
                "parent_iteration_id": iteration.parent_iteration_id,
                "parent_query_id": iteration.parent_query_id,
                "queries": [],  # Will be populated with query IDs
            }
            for it_id, iteration in research.iterations.items()
        },
        "query_results": {
            query_id: {
                "id": query_id,
                "iteration_id": query.iteration_id,
                "title": query.title,
                "query": query.query,
                "goals": query.goals,
                "urls": query.urls,
                "learnings": query.learnings,
                "completed": query.completed,
            }
            for query_id, query in research.queries.items()
        },
    }
    # Populate query IDs in iterations
    for query_id, query in research.queries.items():
        if query.iteration_id in research_data["iterations"]:
            research_data["iterations"][query.iteration_id]["queries"].append(query_id)

    tree = ResearchTree(research_data)  # Pass the dictionary
    tree_json_path = tree.save_json(settings.output_path)
    flowchart_path = tree.visualize(settings.output_path)
    research_json_path = tree.export_research_json(settings.output_path)

    logger.info(f"Report written to: {output_path}")
    logger.info(f"Research data exported to CSV: {csv_path}")
    logger.info(f"Research data exported to JSON: {research_json_path}")
    logger.info(f"Research flowchart saved to: {flowchart_path}")
    logger.info(f"Tree JSON structure saved to: {tree_json_path}")

    return ResearchOutput(
        title=report_results.title,
        learnings=research.learnings,
        visited_urls=research.visited_urls,
        report=report_results.report,
        key_takeaways=report_results.key_takeaways,
        further_research=report_results.further_research,
        revised_report=proofread_results.revised_report,
        revision_notes=proofread_results.revision_notes,
        csv_path=str(csv_path),
        tree_viz_path=str(flowchart_path),
        tree_json_path=str(tree_json_path),
        research_json_path=str(research_json_path),
    )


async def research_with_feedback(
    query: str,
    breadth: int,
    depth: int,
    search_provider: SearchProvider,
    on_progress: Optional[Callable[[ResearchProgress], None]] = None,
    get_feedback_answer: Callable[[str], Awaitable[str]] | None = None,
) -> ResearchOutput:
    """
    Perform deep research with initial query refinement through feedback.

    Args:
        query: Initial research query
        breadth: Number of parallel search paths to explore
        depth: How deep to go in the research tree
        search_provider: Search provider implementation
        on_progress: Optional callback for progress updates
        get_feedback_answer: Optional callback to get answers to feedback questions.
                           If not provided, feedback step will be skipped.
    """
    feedback_qa = None
    if get_feedback_answer:
        logger.info("Generating feedback questions...")
        feedback_response = await generate_feedback(query)

        logger.info("\nTo better understand your research needs, please answer these questions:")
        feedback_qa = []
        for question in feedback_response.questions:
            answer = await get_feedback_answer(question)
            feedback_qa.append((question, answer))

        # Combine original query with feedback
        refined_query = await combine_query_with_feedback(
            query,
            [q for q, _ in feedback_qa],
            [a for _, a in feedback_qa],
        )
    else:
        refined_query = query

    # Call the main research function with the refined query
    result = await research_with_report(
        query=refined_query,
        breadth=breadth,
        depth=depth,
        search_provider=search_provider,
        on_progress=on_progress,
        feedback_qa=feedback_qa,
    )

    return result
