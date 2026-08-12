import re


def clean_content(content: str) -> str:
    """
    Clean the content by removing or replacing special characters.
    """
    # Example: Remove non-ASCII characters
    cleaned_content = re.sub(r"[^\x00-\x7F]+", " ", content)
    return cleaned_content
