import requests
from bs4 import BeautifulSoup
import trafilatura


def fetch_webpage_content(url: str) -> str:
    """
    Fetch the content of a webpage.
    Note, this was generated 100% using Claude.
    """
    try:
        # Download webpage
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()

        # Try trafilatura first (best for article content)
        content = trafilatura.extract(response.text, include_comments=False, include_tables=False)

        # Fallback to BeautifulSoup if needed
        if not content:
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            for tag in ["script", "style", "nav", "header", "footer"]:
                for element in soup.find_all(tag):
                    element.decompose()

            content = soup.get_text(separator=" ", strip=True)

        return content
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return None
