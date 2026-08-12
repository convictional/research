"""Load experiment data into DSPy Example format."""

import json
from pathlib import Path

import dspy


DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
TEAM_DATA_DIR = Path(__file__).parent.parent.parent / "data_team"


def load_examples(filepath: str | Path) -> list[dspy.Example]:
    """Load examples from JSONL file into DSPy format.

    Args:
        filepath: Path to JSONL file (train.jsonl, dev.jsonl, or test.jsonl)

    Returns:
        List of DSPy Examples with context, target_date, ground_truth, and context_summary
    """
    examples = []
    filepath = Path(filepath)

    with open(filepath) as f:
        for line in f:
            data = json.loads(line)

            # Format context as text
            context = _format_context(data["context"])

            # Extract ground truth priorities
            ground_truth = [p["description"] for p in data["standup_entry"]["priorities"]]

            # Create context summary for judge
            ctx = data["context"]
            context_summary = (
                f"{len(ctx.get('emails', []))} emails, "
                f"{len(ctx.get('meetings', []))} meetings, "
                f"{len(ctx.get('tasks', []))} tasks, "
                f"{len(ctx.get('discussions', []))} discussions"
            )

            # Extract target date (just the date portion)
            target_date = data["standup_entry"]["date"][:10]  # YYYY-MM-DD

            example = dspy.Example(
                context=context,
                target_date=target_date,
                ground_truth=ground_truth,
                context_summary=context_summary,
            ).with_inputs("context", "target_date")

            examples.append(example)

    return examples


def _format_context(context: dict) -> str:
    """Format context dict into readable text for the LLM."""
    sections = []

    # Emails
    emails = context.get("emails", [])
    if emails:
        sections.append("EMAILS:")
        for item in emails[:5]:
            created = item.get("created_at", "")[:10]
            title = item.get("title", "No title")
            content = item.get("content", "")[:200]
            sections.append(f"- [{created}] {title}\n  {content}...")
        sections.append("")

    # Meetings
    meetings = context.get("meetings", [])
    if meetings:
        sections.append("MEETINGS/CALENDAR:")
        for item in meetings[:5]:
            created = item.get("created_at", "")[:10]
            title = item.get("title", "No title")
            content = item.get("content", "")[:200]
            sections.append(f"- [{created}] {title}\n  {content}...")
        sections.append("")

    # Tasks
    tasks = context.get("tasks", [])
    if tasks:
        sections.append("TASKS/ISSUES:")
        for item in tasks[:10]:
            created = item.get("created_at", "")[:10]
            title = item.get("title", "No title")
            content = item.get("content", "")[:200]
            sections.append(f"- [{created}] {title}\n  {content}...")
        sections.append("")

    # Discussions
    discussions = context.get("discussions", [])
    if discussions:
        sections.append("DISCUSSIONS/COMMENTS:")
        for item in discussions[:10]:
            created = item.get("created_at", "")[:10]
            title = item.get("title", "No title")
            content = item.get("content", "")[:200]
            sections.append(f"- [{created}] {title}\n  {content}...")
        sections.append("")

    if not sections:
        return "No relevant context found."

    return "\n".join(sections)
