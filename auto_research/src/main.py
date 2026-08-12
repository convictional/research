import asyncio
import re
from datetime import UTC, datetime

from jinja2 import Environment, FileSystemLoader

from .arxiv import ArxivPaper, fetch_paper_full_text, fetch_recent_papers
from .context import build_codebase_context
from .email import send_report_email
from .models import FilteredPapers, PaperRelevance
from .prompts.engine import build_prompt, register_prompt_templates
from .settings import settings
from .utils.instruct_llm import ainstruct_llm
from .utils.llm import astring_completion

RELEVANCE_THRESHOLD = 6
MAX_LOOKBACK_DAYS = 7


def _days_since_last_report() -> int:
    report_dates = []
    for path in settings.output_path.glob("*.md"):
        match = re.match(r"(\d{4}-\d{2}-\d{2})\.md$", path.name)
        if match:
            report_dates.append(datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=UTC))

    if not report_dates:
        return MAX_LOOKBACK_DAYS

    latest = max(report_dates)
    delta = (datetime.now(UTC) - latest).days
    return min(max(delta, 1), MAX_LOOKBACK_DAYS)


async def main():
    _initialize_prompts()

    print("Stage 1: Building codebase context...")
    context = await build_codebase_context()
    print(f"  Context loaded ({len(context)} chars)")

    days_back = _days_since_last_report()
    print(f"Stage 2: Fetching ArXiv papers (last {days_back} days)...")
    papers = await fetch_recent_papers(settings.arxiv_categories, max_results=100, days_back=days_back)
    print(f"  Found {len(papers)} papers")

    if not papers:
        print("No papers found. Exiting.")
        return

    print("Stage 3: Filtering papers for relevance...")
    relevant = await _filter_papers(papers, context)
    top_papers = [p for p, r in relevant if r.relevance_score >= RELEVANCE_THRESHOLD]
    print(f"  {len(top_papers)} papers scored >= {RELEVANCE_THRESHOLD}")

    if not top_papers:
        print("No sufficiently relevant papers found. Writing minimal report.")
        report_content = _write_report(papers, relevant, [])
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        print("Stage 7: Sending email...")
        await send_report_email(report_content, today)
        return

    print("Stage 4: Fetching full text for top papers...")
    for i, paper in enumerate(top_papers):
        if i > 0:
            await asyncio.sleep(4)  # Rate limit: arxiv.org full-text fetches
        paper.full_text = await fetch_paper_full_text(paper)
        status = f"{len(paper.full_text)} chars" if paper.full_text else "failed"
        print(f"  {paper.title[:60]}... → {status}")

    print("Stage 5: Deep research on top papers...")
    plans: list[tuple[ArxivPaper, str]] = []
    for paper in top_papers:
        plan = await _research_paper(paper, context)
        plans.append((paper, plan))
        print(f"  Completed: {paper.title[:60]}...")

    print("Stage 6: Writing daily report...")
    report_content = _write_report(papers, relevant, plans)

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    print("Stage 7: Sending email...")
    await send_report_email(report_content, today)
    print("Done!")


def _initialize_prompts():
    prompt_templates = Environment(loader=FileSystemLoader(searchpath=settings.root / "src" / "prompts"))
    register_prompt_templates(prompt_templates)


async def _filter_papers(
    papers: list[ArxivPaper],
    context: str,
) -> list[tuple[ArxivPaper, PaperRelevance]]:
    system_prompt = build_prompt("filter_papers.md.jinja", codebase_context=context)

    paper_summaries = "\n\n".join(
        f"Paper ID: {p.id}\nTitle: {p.title}\nCategories: {', '.join(p.categories)}\nAbstract: {p.abstract}"
        for p in papers
    )

    result: FilteredPapers | None = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=f"Evaluate these papers:\n\n{paper_summaries}",
        response_model=FilteredPapers,
        llm_model=settings.filter_model,
        max_tokens=8000,
    )

    if not result:
        return []

    relevance_map = {r.paper_id: r for r in result.papers}
    paired = []
    for paper in papers:
        if paper.id in relevance_map:
            paired.append((paper, relevance_map[paper.id]))

    paired.sort(key=lambda x: x[1].relevance_score, reverse=True)
    return paired


async def _research_paper(paper: ArxivPaper, context: str) -> str:
    system_prompt = build_prompt("deep_research.md.jinja", codebase_context=context)

    content = paper.full_text or paper.abstract
    user_prompt = (
        f"# {paper.title}\n\n"
        f"**Authors:** {', '.join(paper.authors)}\n"
        f"**Categories:** {', '.join(paper.categories)}\n"
        f"**Published:** {paper.published.strftime('%Y-%m-%d')}\n\n"
        f"## Full Text\n\n{content}"
    )

    return await astring_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=2000,
    )


def _write_report(
    all_papers: list[ArxivPaper],
    relevant: list[tuple[ArxivPaper, PaperRelevance]],
    plans: list[tuple[ArxivPaper, str]],
) -> str:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    report_path = settings.output_path / f"{today}.md"

    lines = [
        f"# Research Ideation Report — {today}\n",
        "## Summary\n",
        f"- **Papers scanned:** {len(all_papers)}",
        f"- **Papers evaluated:** {len(relevant)}",
        f"- **Papers researched in depth:** {len(plans)}\n",
    ]

    if relevant:
        table_min_score = RELEVANCE_THRESHOLD - 1
        visible = [(p, r) for p, r in relevant if r.relevance_score >= table_min_score]
        if visible:
            lines.append("## Relevance Scores\n")
        else:
            lines.append("## Relevance Scores\n")
            lines.append("No papers passing threshold found today. Top 3 for reference:\n")
            visible = relevant[:3]
        lines.append("| Score | Paper | Reason |")
        lines.append("|-------|-------|--------|")
        for paper, rel in visible:
            title = paper.title[:80]
            lines.append(
                f"| {rel.relevance_score} | [{title}](https://arxiv.org/abs/{paper.id}) | {rel.relevance_reason} |"
            )
        lines.append("")

    for paper, plan in plans:
        lines.append("---\n")
        lines.append(f"## [{paper.title}](https://arxiv.org/abs/{paper.id})\n")
        lines.append(plan)
        lines.append("")

    content = "\n".join(lines)
    report_path.write_text(content)
    print(f"  Report written to {report_path}")
    return content
