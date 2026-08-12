import asyncio
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import httpx

from .utils.tokens import truncate_to_tokens

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_HTML_BASE = "https://arxiv.org/html"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
ATOM_NS = "http://www.w3.org/2005/Atom"

MAX_FULL_TEXT_TOKENS = 128_000
PAGE_SIZE = 25
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 15


@dataclass
class ArxivPaper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    updated: datetime
    categories: list[str]
    pdf_url: str
    html_url: str | None = None
    full_text: str | None = None


async def fetch_recent_papers(
    categories: list[str],
    max_results: int = 100,
    days_back: int = 1,
) -> list[ArxivPaper]:
    cat_query = "+OR+".join(f"cat:{quote(c)}" for c in categories)
    query = f"({cat_query})"
    cutoff = datetime.now(UTC) - timedelta(days=days_back)

    all_papers: list[ArxivPaper] = []
    offset = 0

    async with httpx.AsyncClient(timeout=60) as client:
        while offset < max_results:
            page_size = min(PAGE_SIZE, max_results - offset)
            url = (
                f"{ARXIV_API_URL}?search_query={query}"
                f"&sortBy=submittedDate&sortOrder=descending"
                f"&start={offset}&max_results={page_size}"
            )

            response = await _fetch_with_backoff(client, url)
            if response is None:
                print(f"  ArXiv page at offset {offset} failed after retries, continuing with {len(all_papers)} papers")
                break

            page_papers = _parse_atom_feed(response.text)
            if not page_papers:
                break

            recent = [p for p in page_papers if p.published >= cutoff]
            all_papers.extend(recent)

            if len(recent) < len(page_papers):
                break

            offset += page_size

            # Respect ArXiv's 1-request-per-3-seconds rate limit
            if offset < max_results:
                await asyncio.sleep(4)

    return all_papers


def _backoff_seconds(attempt: int) -> float:
    return INITIAL_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 2)


async def _fetch_with_backoff(client: httpx.AsyncClient, url: str) -> httpx.Response | None:
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(url)
            if response.status_code == 429:
                backoff = _backoff_seconds(attempt)
                print(f"  ArXiv rate limited, retrying in {backoff:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(backoff)
                continue
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError:
            if attempt < MAX_RETRIES - 1:
                backoff = _backoff_seconds(attempt)
                print(f"  ArXiv request failed, retrying in {backoff:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(backoff)
            else:
                return None
    return None


def _parse_atom_feed(xml_text: str) -> list[ArxivPaper]:
    root = ET.fromstring(xml_text)
    papers = []

    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        paper_id = _text(entry, f"{{{ATOM_NS}}}id").split("/abs/")[-1]
        title = _text(entry, f"{{{ATOM_NS}}}title").replace("\n", " ").strip()
        abstract = _text(entry, f"{{{ATOM_NS}}}summary").strip()
        published = datetime.fromisoformat(_text(entry, f"{{{ATOM_NS}}}published").replace("Z", "+00:00"))
        updated = datetime.fromisoformat(_text(entry, f"{{{ATOM_NS}}}updated").replace("Z", "+00:00"))

        authors = [
            _text(author, f"{{{ATOM_NS}}}name")
            for author in entry.findall(f"{{{ATOM_NS}}}author")
        ]

        categories = [
            cat.get("term", "")
            for cat in entry.findall("{http://arxiv.org/schemas/atom}primary_category")
        ] + [
            cat.get("term", "")
            for cat in entry.findall("{http://arxiv.org/schemas/atom}category")
            if cat.get("scheme") == "http://arxiv.org/schemas/atom"
        ]
        categories = list(dict.fromkeys(c for c in categories if c))

        pdf_url = ""
        html_url = None
        for link in entry.findall(f"{{{ATOM_NS}}}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
            elif link.get("type") == "text/html":
                html_url = link.get("href")

        if not html_url:
            html_url = f"{ARXIV_HTML_BASE}/{paper_id}"

        papers.append(ArxivPaper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            published=published,
            updated=updated,
            categories=categories,
            pdf_url=pdf_url,
            html_url=html_url,
        ))

    return papers


async def fetch_paper_full_text(paper: ArxivPaper) -> str | None:
    text = await _fetch_html_text(paper)
    if not text:
        text = await _fetch_pdf_text(paper)
    if text:
        return truncate_to_tokens(text, MAX_FULL_TEXT_TOKENS)
    return None


async def _fetch_html_text(paper: ArxivPaper) -> str | None:
    if not paper.html_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(paper.html_url)
            if response.status_code != 200:
                return None

        from html.parser import HTMLParser

        class ArticleExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_article = False
                self.depth = 0
                self.text_parts: list[str] = []

            def handle_starttag(self, tag, attrs):
                if tag == "article":
                    self.in_article = True
                    self.depth = 1
                elif self.in_article:
                    self.depth += 1

            def handle_endtag(self, tag):
                if self.in_article:
                    self.depth -= 1
                    if self.depth <= 0:
                        self.in_article = False

            def handle_data(self, data):
                if self.in_article:
                    stripped = data.strip()
                    if stripped:
                        self.text_parts.append(stripped)

        extractor = ArticleExtractor()
        extractor.feed(response.text)
        text = "\n".join(extractor.text_parts)
        return text if len(text) > 500 else None
    except Exception:
        return None


async def _fetch_pdf_text(paper: ArxivPaper) -> str | None:
    if not paper.pdf_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(paper.pdf_url)
            if response.status_code != 200:
                return None

        import pymupdf

        doc = pymupdf.Document(stream=response.content, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()

        text = "\n".join(text_parts)
        return text if len(text) > 500 else None
    except Exception:
        return None


def _text(element: ET.Element, tag: str) -> str:
    child = element.find(tag)
    return child.text if child is not None and child.text else ""
