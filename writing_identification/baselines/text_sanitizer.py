"""Text sanitization utilities for removing identifying metadata from evaluation data."""

import re
from typing import Tuple, Set


class TextSanitizer:
    """Sanitizes text to remove identifying metadata that could leak author information.

    Uses a two-pass approach:
    1. Extract names from metadata patterns
    2. Replace those specific names throughout the text
    """

    @staticmethod
    def extract_names_from_metadata(text: str) -> Set[str]:
        """Extract author names from known metadata patterns.

        Returns a set of names found in metadata contexts.
        """
        names = set()

        # Pattern 1: "Author: [Name]" or "Authors: [Names]" with optional Labels
        # More specific pattern to avoid matching issue titles
        author_patterns = re.findall(
            r"\bAuthors?:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)(?:\s+Labels|\s*\n|\s*$)",
            text,
            re.MULTILINE,
        )
        for match in author_patterns:
            # Split on commas and ampersands for multiple authors
            for name in re.split(r"[,&]", match):
                name = name.strip()
                if name and len(name) > 2:  # Avoid single letters
                    names.add(name)

        # Pattern 2: "# Github Comment/Discussion Comment: [Name] - [Date]"
        # More flexible pattern to catch names before dates
        github_patterns = re.findall(
            r"#\s*Github\s+(?:Comment|Issue|Discussion|Discussion Comment):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*-\s*\d{4}",
            text,
        )
        names.update(github_patterns)

        # Pattern 3: "# Google Doc: [Title] - [Date] Authors: [Names]"
        doc_patterns = re.findall(
            r"#\s*Google\s+Doc:.*?Authors?:\s*([A-Za-z][A-Za-z\s,&]+?)(?:\s*\n|$)", text, re.IGNORECASE
        )
        for match in doc_patterns:
            for name in re.split(r"[,&]", match):
                name = name.strip()
                if name and len(name) > 2:
                    names.add(name)

        # Pattern 4: "Submitted update: [Name], [Date]"
        submitted_patterns = re.findall(
            r"Submitted\s+(?:update|by|from):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),\s*\d{4}", text, re.IGNORECASE
        )
        names.update(submitted_patterns)

        # Clean up extracted names (remove trailing punctuation, etc.)
        cleaned_names = set()
        for name in names:
            # Remove trailing punctuation and whitespace
            name = name.rstrip(".,;: \t")
            # Skip if it's too short or looks like noise
            if len(name) > 2 and not name.startswith("[") and name != "Labels":
                cleaned_names.add(name)

        return cleaned_names

    # Hardcoded names to always redact in metadata (first 100 chars).
    #
    # This is the belt to `extract_names_from_metadata`'s braces: any author whose
    # name must never reach an LLM baseline goes here, whether or not the metadata
    # parser catches it. The real roster was removed before open-sourcing —
    # populate this with the authors in your own corpus.
    HARDCODED_NAMES = [
        "Person A",
        "Person B",
        "Person C",
    ]

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Remove identifying metadata from a text sample.

        Uses a targeted approach:
        1. Replace hardcoded names in first 100 chars
        2. Extract names from metadata contexts
        3. Replace those names throughout the text
        4. Redact obvious metadata lines
        """
        original_text = text

        # First pass: Replace hardcoded names throughout the entire text
        # This is more aggressive but ensures no names slip through
        # Use case-insensitive replacement
        for name in TextSanitizer.HARDCODED_NAMES:
            # Create a regex pattern for case-insensitive replacement
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            text = pattern.sub("[REDACTED]", text)

        # Second pass: Extract names from metadata
        names_to_redact = TextSanitizer.extract_names_from_metadata(text)

        # Third pass: Replace the extracted names
        for name in names_to_redact:
            # Use word boundaries to avoid partial matches
            # But be careful with names that might appear in actual content
            # Only replace in obvious metadata contexts

            # Replace in "Author: [name]" patterns (including with Labels)
            text = re.sub(
                rf"\bAuthor:\s*{re.escape(name)}(?=\s*(?:Labels|\n|$))",
                "Author: [REDACTED]",
                text,
                flags=re.IGNORECASE,
            )

            # Replace in "Authors: ...[name]..." patterns
            text = re.sub(rf"(\bAuthors?:.*?)\b{re.escape(name)}\b", r"\1[REDACTED]", text, flags=re.IGNORECASE)

            # Replace in Github headers (both formats)
            # Format 1: "# Github Comment: [Name] - [Date]"
            text = re.sub(
                rf"(#\s*Github\s+(?:Comment|Issue|Discussion|Discussion Comment|PR):\s*){re.escape(name)}(\s*-)",
                r"\1[REDACTED]\2",
                text,
                flags=re.IGNORECASE,
            )

            # Format 2: "# Github Issue: [Title] Author: [Name]"
            text = re.sub(
                rf"(#\s*Github[^\n]*Author:\s*){re.escape(name)}\b", r"\1[REDACTED]", text, flags=re.IGNORECASE
            )

            # Replace in Google Doc headers
            text = re.sub(
                rf"(#\s*Google\s+Doc:.*?Authors?:\s*[^:\n]*?)\b{re.escape(name)}\b",
                r"\1[REDACTED]",
                text,
                flags=re.IGNORECASE,
            )

            # Replace in "Submitted update: [name]" patterns
            text = re.sub(
                rf"(Submitted\s+(?:update|by|from):\s*){re.escape(name)}\b", r"\1[REDACTED]", text, flags=re.IGNORECASE
            )

        # Fourth pass: Redact dates in specific metadata contexts (but not in content)
        # Only redact the date portion, not the entire line
        text = re.sub(
            r"(#\s*Github[^\n]*?)\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[+\-]\d{2}:\d{2}",
            r"\1[DATE]",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"(#\s*Google\s+Doc[^\n]*?)\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?[+\-]\d{2}:\d{2}",
            r"\1[DATE]",
            text,
            flags=re.IGNORECASE,
        )

        # Clean up "Labels: []" metadata
        text = re.sub(r"^\s*Labels:\s*\[.*?\]\s*$", "", text, flags=re.MULTILINE)

        # Clean up multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # If we accidentally removed everything, return the original
        # (better to leak names than to have no content)
        if not text or len(text) < 10:
            return original_text

        return text

    @staticmethod
    def sanitize_pairs(text_pairs: list[Tuple[str, str]]) -> list[Tuple[str, str]]:
        """Sanitize a list of text pairs.

        Args:
            text_pairs: list of (text1, text2) tuples

        Returns:
            list of sanitized (text1, text2) tuples
        """
        sanitizer = TextSanitizer()
        sanitized_pairs = []

        for text1, text2 in text_pairs:
            sanitized_text1 = sanitizer.sanitize_text(text1)
            sanitized_text2 = sanitizer.sanitize_text(text2)
            sanitized_pairs.append((sanitized_text1, sanitized_text2))

        return sanitized_pairs


def test_sanitization():
    """Test the sanitization with sample texts."""
    # Synthetic fixtures. These use the placeholder names in HARDCODED_NAMES so the
    # test exercises the same code paths without embedding a real roster.
    samples = [
        # Test case 1: Name in Author field
        "# Github Issue: Add eval script - 1911 - 2024-10-09 16:34:31+00:00 Author: Person A Labels: []",
        # Test case 2: Name appearing multiple times in content
        "# Github Issue: Adoption Bets - 1938 - 2024-10-15 20:30:05+00:00 Author: Person B Labels: []\n\nSome comments:\nPerson B: I think we should try this approach.\nPerson A: Agreed with Person B here.",
        # Test case 3: Names in various contexts
        "Author: Person C\nThis PR implements the OAuth flow.\n\nPerson C mentioned that we need to consider security.\nPerson A and Person B reviewed this.",
        # Test case 4: Case sensitivity check
        "person b, PERSON A, and Person C were discussing this.",
        # Test case 5: Partial matches (shouldn't be replaced)
        "Personable approach was good. Personalized thinking helps.",
    ]

    sanitizer = TextSanitizer()

    print("SANITIZATION TEST RESULTS")
    print("=" * 80)

    for i, sample in enumerate(samples):
        print(f"\n[Test {i + 1}]")

        # First show what names we extracted
        names = sanitizer.extract_names_from_metadata(sample)
        if names:
            print(f"Extracted names: {names}")

        sanitized = sanitizer.sanitize_text(sample)

        print(f"\nORIGINAL:\n{sample}")
        print(f"\nSANITIZED:\n{sanitized}")
        print("-" * 80)


if __name__ == "__main__":
    test_sanitization()
