from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from .helpers.json import to_json
from .prompts.engine import register_prompt_templates, build_prompt
from .settings import settings
from .get_user_data import UsersData, User
from .helpers.io import dump_list_of_objects_to_csv
from .helpers.instruct_llm import ainstruct_llm
from .helpers.async_helper import execute_tasks_with_manual_pbar
from .get_google_search_results import get_google_search_results
from .webpage_content import fetch_webpage_content


# NOTE: The user input CSV contained real user records and was removed before open-sourcing.
# Supply your own CSV at this path; see `get_user_data.py` for the expected columns.
user_data_input_csv_path = settings.input_path / "tailored_web_search_users.csv"
results_csv_path = settings.output_path / "tailored_web_search_results.csv"
num_search_results = 5


LLM_TEMPERATURE = 0.0


class PrintableResults(BaseModel):
    user_name: str = Field(..., title="The name of the user")
    org_profile: str = Field(..., title="The organization profile of the user")
    user_profile: str = Field(..., title="The user profile of the user")
    google_search_query: str = Field(..., title="The Google search query for the user")
    google_search_results: str = Field(..., title="The Google search results for the user")
    google_search_results_with_summaries: str = Field(
        ..., title="The Google search results with summaries for the user"
    )


class GoogleSearchResult(BaseModel):
    title: str = Field(..., title="The title of the search result")
    link: str = Field(..., title="The link of the search result")
    date: str | None = Field(..., title="The date of the search result")
    llm_summary: str = Field("", title="The summary of the search result")


async def get_google_search_queries(users_data: list[User]) -> list[str]:
    """
    Get Google search query strings for each user, using an LLM.
    Give an LLM a user's profile and organization profile, and have it generate a search query for recent events tailored to that user.
    """
    print("Getting Google search queries for users...")

    current_date = datetime.now()
    one_week_ago = current_date - timedelta(days=7)

    system_prompt = build_prompt(
        "generate_google_search_query_system.txt.jinja", starting_date=one_week_ago.strftime("%Y-%m-%d")
    )
    user_prompts = [build_prompt("generate_google_search_query_user.txt.jinja", user=user) for user in users_data]

    tasks = [
        ainstruct_llm(
            system_prompt,
            user_prompt,
            response_model=str,
            temperature=LLM_TEMPERATURE,
        )
        for user_prompt in user_prompts
    ]

    results = await execute_tasks_with_manual_pbar(tasks)

    return [f"{r} source:news OR site:news OR inurl:news" for r in results]


def get_search_results(google_search_queries: list[str]) -> list[list[GoogleSearchResult]]:
    """
    Use search query strings to get top search results, for each user.
    """
    print("Getting search results...")

    search_results_all_users = []

    for index, query in enumerate(google_search_queries):
        print(f"Getting search results for user {index + 1}...")

        search_result = get_google_search_results(query, num_search_results)

        try:
            search_items = search_result["items"]

            search_results_for_user = [
                GoogleSearchResult(
                    title=item["title"],
                    link=item["link"],
                    date=get_search_item_date(item),
                )
                for item in search_items
            ]
        except KeyError:
            print("Error getting search results.")
            search_results_for_user = []

        search_results_all_users.append(search_results_for_user)

    return search_results_all_users


def get_search_item_date(item: dict) -> str:
    """
    Get the date of the search result item.
    Note, this was produced 100% from Claude.
    """
    # Check pagemap
    pagemap = item.get("pagemap", {})

    # Try metatags first
    metatags = pagemap.get("metatags", [{}])[0]
    date = (
        metatags.get("article:published_time")
        or metatags.get("date")
        or metatags.get("og:published_time")
        or metatags.get("datePublished")
    )

    if date:
        return date

    # Try article
    article = pagemap.get("article", [{}])[0]
    date = article.get("datepublished") or article.get("datemodified")

    if date:
        return date

    # Try newsarticle
    newsarticle = pagemap.get("newsarticle", [{}])[0]
    date = newsarticle.get("datepublished") or newsarticle.get("datemodified")

    return date or None


async def get_search_results_summaries(
    users_data: list[User], search_results_all_users: list[list[GoogleSearchResult]]
) -> list[list[GoogleSearchResult]]:
    """
    This function goes to the search result URLs, and summarizes the content using an LLM.
    """
    print("Getting search result summaries using LLM...")

    search_results_for_all_users = []

    for index, (user, search_results_per_user) in enumerate(zip(users_data, search_results_all_users)):
        print(f"Getting search result summaries for user {index + 1}...")

        print("Fetching web page content...")
        web_page_content: list[str | None] = [
            fetch_webpage_content(search_result.link) for search_result in search_results_per_user
        ]

        print("Summarizing web page content...")
        tasks = [
            get_summary_from_llm(
                user=user,
                web_page_content=content,
            )
            for content in web_page_content
        ]

        summaries = await execute_tasks_with_manual_pbar(tasks)

        search_results_for_user = [
            GoogleSearchResult(
                title=search_result.title,
                link=search_result.link,
                date=search_result.date,
                llm_summary=summary,
            )
            for search_result, summary in zip(search_results_per_user, summaries)
        ]

        search_results_for_all_users.append(search_results_for_user)

    return search_results_for_all_users


async def get_summary_from_llm(user: User, web_page_content: str | None) -> str:
    """
    This function gets a summary from an LLM.
    """
    if not web_page_content:
        return ""

    system_prompt = build_prompt("summarize_web_page_content_system.txt.jinja")
    user_prompt = build_prompt(
        "summarize_web_page_content_user.txt.jinja", user=user, web_page_content=web_page_content
    )

    try:
        summary = await ainstruct_llm(
            system_prompt,
            user_prompt,
            response_model=str,
            temperature=LLM_TEMPERATURE,
        )
    except Exception as e:
        print(f"Error summarizing: {str(e)}")
        summary = ""

    return summary


def dump_results_to_csv(
    users_data: list[User],
    google_search_queries: list[str],
    search_results_all_users: list[list[GoogleSearchResult]],
    search_results_with_summaries_all_users: list[list[GoogleSearchResult]],
):
    """
    Dump the results to a csv file.
    """
    print("Dumping results to csv...")

    results = [
        PrintableResults(
            user_name=user.user_name,
            org_profile=user.org_profile,
            user_profile=user.user_profile,
            google_search_query=google_search_query,
            google_search_results="\n".join(
                [
                    f"Title: {result.title}\nLink: {result.link}\nDate: {result.date}\n"
                    for result in search_results_per_user
                ]
            ),
            google_search_results_with_summaries="\n".join(
                [
                    f"Title: {result.title}\nLink: {result.link}\nDate: {result.date}\nLLM Summary: {result.llm_summary}\n"
                    for result in search_results_with_summaries_per_user
                ]
            ),
        )
        for user, google_search_query, search_results_per_user, search_results_with_summaries_per_user in zip(
            users_data, google_search_queries, search_results_all_users, search_results_with_summaries_all_users
        )
    ]

    dump_list_of_objects_to_csv(results, results_csv_path)


async def main():
    """
    This routine gets top search results tailored to a user and org.
    Steps:
    1. Get users and profiles.
    2. Get Google search query string for each user.
    3. Get top search results for each user.
    4. Go to search result URLs, and summarize using an LLM.
    5. Dump results to csv.
    """
    print("Starting tailored web search and summarization...")
    # prompt templates
    prompt_templates = Environment(loader=FileSystemLoader(searchpath=settings.root / "src" / "prompts"))
    prompt_templates.filters["to_json"] = to_json
    register_prompt_templates(prompt_templates)

    # Get user and org data
    users: list[User] = UsersData(user_data_input_csv_path).users

    # TODO: remove after testing
    # users = users[:2]

    # Get Google search query string for each user
    google_search_queries: list[str] = await get_google_search_queries(users)

    # Do google search for each query
    search_results_all_users: list[list[GoogleSearchResult]] = get_search_results(google_search_queries)

    # Summarize search results
    search_results_with_summaries_all_users: list[list[GoogleSearchResult]] = await get_search_results_summaries(
        users, search_results_all_users
    )

    # Dump results to csv
    dump_results_to_csv(
        users, google_search_queries, search_results_all_users, search_results_with_summaries_all_users
    )
