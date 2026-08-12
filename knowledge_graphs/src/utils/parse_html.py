from bs4 import BeautifulSoup


def clean_html_content(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ")
