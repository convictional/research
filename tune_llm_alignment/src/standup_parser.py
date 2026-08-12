"""Parse standup markdown files to extract ground truth priorities."""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .models import Priority, StandupEntry


class StandupParser:
    """Parse markdown standup documents to extract user entries."""

    def __init__(self, username: str = "Person A"):
        self.username = username

    def parse_file(self, file_path: str | Path) -> List[StandupEntry]:
        """Parse a standup markdown file and extract all entries for the user."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Standup file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.parse_content(content)

    def parse_content(self, content: str) -> List[StandupEntry]:
        """Parse standup markdown content."""
        entries = []

        # Split by date sections
        date_sections = self._split_by_dates(content)

        for date, section_content in date_sections:
            # Extract user's entry from this date section
            user_entry = self._extract_user_entry(section_content)
            if user_entry:
                entry = self._create_standup_entry(date, user_entry)
                if entry:
                    entries.append(entry)

        return entries

    def _split_by_dates(self, content: str) -> List[tuple[datetime, str]]:
        """Split content into sections by date headers."""
        # Match date headers: ### **Date:** Nov 19, 2025
        date_pattern = r"###\s+\*\*Date:\*\*\s+(.+?)$"

        sections = []
        current_date = None
        current_content = []

        for line in content.split("\n"):
            date_match = re.match(date_pattern, line)
            if date_match:
                # Save previous section
                if current_date and current_content:
                    sections.append((current_date, "\n".join(current_content)))

                # Start new section
                date_str = date_match.group(1).strip()
                try:
                    current_date = datetime.strptime(date_str, "%b %d, %Y")
                except ValueError:
                    # Try alternative formats
                    try:
                        current_date = datetime.strptime(date_str, "%B %d, %Y")
                    except ValueError:
                        print(f"Warning: Could not parse date: {date_str}")
                        current_date = None

                current_content = []
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_date and current_content:
            sections.append((current_date, "\n".join(current_content)))

        return sections

    def _extract_user_entry(self, section_content: str) -> Optional[str]:
        """Extract the user's entry from a date section."""
        # Find the user's entry: **Person A**
        user_pattern = rf"\*\*{re.escape(self.username)}\*\*\s*\n(.*?)(?=\n\*\*[A-Za-z]+\*\*|\Z)"

        match = re.search(user_pattern, section_content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _create_standup_entry(
        self, date: datetime, entry_text: str
    ) -> Optional[StandupEntry]:
        """Create a StandupEntry from extracted text."""
        # Extract main focus
        main_focus = self._extract_main_focus(entry_text)

        # Skip if main focus is empty or too short (< 10 chars likely not real content)
        if not main_focus or len(main_focus.strip()) < 10:
            return None

        # Extract asks/blockers for additional context
        asks_blockers = self._extract_asks_blockers(entry_text)

        # Create priority from main focus
        priorities = [
            Priority(
                description=main_focus,
                rationale=None,
                rank=1,
            )
        ]

        return StandupEntry(
            date=date,
            priorities=priorities,
            context_signals=asks_blockers if asks_blockers else None,
            raw_text=entry_text,
        )

    def _extract_main_focus(self, entry_text: str) -> Optional[str]:
        """Extract the main focus text."""
        # Match: Main focus: <text>
        # Continue until we hit Asks/Blockers or end of entry
        pattern = r"Main focus:\s*(.+?)(?=\nAsks/Blockers:|$)"

        match = re.search(pattern, entry_text, re.DOTALL)
        if match:
            focus = match.group(1).strip()
            # Make sure we didn't accidentally capture "Asks/Blockers:" or minimal content
            if focus and len(focus) > 10 and "Asks/Blockers" not in focus:
                return focus
        return None

    def _extract_asks_blockers(self, entry_text: str) -> Optional[str]:
        """Extract asks/blockers content if present."""
        # Match: Asks/Blockers: followed by bullet points or text
        pattern = r"Asks/Blockers:\s*(.+?)$"

        match = re.search(pattern, entry_text, re.DOTALL)
        if match:
            blockers = match.group(1).strip()
            return blockers if blockers else None
        return None


def parse_standup_file(
    file_path: str | Path, username: str = "Person A"
) -> List[StandupEntry]:
    """Convenience function to parse a standup file."""
    parser = StandupParser(username=username)
    return parser.parse_file(file_path)


def parse_all_standups(
    username: str = "Person A",
    h1_file: str | Path = "Product Standups 2025H1.md",
    h2_file: str | Path = "Product Standups 2025H2.md",
    before_date: Optional[datetime] = None,
) -> List[StandupEntry]:
    """
    Parse both H1 and H2 standup files and return all entries.

    Args:
        username: Username to extract entries for
        h1_file: Path to H1 standup file
        h2_file: Path to H2 standup file
        before_date: Optional cutoff date - only return entries before this date

    Returns:
        List of StandupEntry objects, sorted by date ascending
    """
    parser = StandupParser(username=username)

    entries = []

    # Parse H1 file if it exists
    h1_path = Path(h1_file)
    if h1_path.exists():
        entries.extend(parser.parse_file(h1_path))

    # Parse H2 file if it exists
    h2_path = Path(h2_file)
    if h2_path.exists():
        entries.extend(parser.parse_file(h2_path))

    # Filter by date if specified
    if before_date:
        entries = [e for e in entries if e.date < before_date]

    # Sort by date ascending
    entries.sort(key=lambda e: e.date)

    return entries


# Team members to extract. Each name must match the bold heading used for that
# person in the standup document (see StandupParser.user_pattern). The real
# roster was removed before open-sourcing — set this to your own, or pass
# `team_members=[...]` to parse_team_standups().
TEAM_MEMBERS = ["Person A", "Person B", "Person C"]


def parse_team_standups(
    h1_file: str | Path = "Product Standups 2025H1.md",
    h2_file: str | Path = "Product Standups 2025H2.md",
    before_date: Optional[datetime] = None,
    team_members: List[str] | None = None,
) -> List[StandupEntry]:
    """
    Parse standup files and return combined team entries for each date.

    Each returned StandupEntry contains ALL team members' priorities for that date,
    allowing for pooled recall/precision measurement.

    Args:
        h1_file: Path to H1 standup file
        h2_file: Path to H2 standup file
        before_date: Optional cutoff date
        team_members: List of usernames to include (defaults to TEAM_MEMBERS)

    Returns:
        List of StandupEntry objects with pooled team priorities, sorted by date
    """
    if team_members is None:
        team_members = TEAM_MEMBERS

    # Collect entries by date from all team members
    entries_by_date: dict[datetime, list[Priority]] = {}
    raw_text_by_date: dict[datetime, list[str]] = {}

    for username in team_members:
        parser = StandupParser(username=username)

        for file_path in [h1_file, h2_file]:
            path = Path(file_path)
            if not path.exists():
                continue

            user_entries = parser.parse_file(path)

            for entry in user_entries:
                if before_date and entry.date >= before_date:
                    continue

                if entry.date not in entries_by_date:
                    entries_by_date[entry.date] = []
                    raw_text_by_date[entry.date] = []

                # Add priorities with username prefix for tracking
                for priority in entry.priorities:
                    entries_by_date[entry.date].append(
                        Priority(
                            description=priority.description,
                            rationale=f"[{username}]",  # Track who said this
                            rank=len(entries_by_date[entry.date]) + 1,
                        )
                    )

                if entry.raw_text:
                    raw_text_by_date[entry.date].append(f"[{username}] {entry.raw_text}")

    # Convert to StandupEntry objects
    result = []
    for date in sorted(entries_by_date.keys()):
        priorities = entries_by_date[date]
        if not priorities:
            continue

        result.append(
            StandupEntry(
                date=date,
                priorities=priorities,
                context_signals=None,
                raw_text="\n---\n".join(raw_text_by_date.get(date, [])),
            )
        )

    return result
