from googleapiclient.discovery import build

from .settings import settings


def get_google_search_results(query_string: str, num_search_results: int):
    # Initialize the service
    service = build("customsearch", "v1", developerKey=settings.google_custom_search_engine_api_key.get_secret_value())

    result = (
        service.cse()
        .list(q=query_string, cx=settings.google_custom_search_engine_id.get_secret_value(), num=num_search_results)
        .execute()
    )

    return result
