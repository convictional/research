"""Email-specific pattern extraction for authorship verification."""

import re
import torch


class EmailPatternExtractor:
    """Extracts email-specific stylistic patterns."""

    def __init__(self):
        self.greeting_patterns = self._build_greeting_patterns()
        self.closing_patterns = self._build_closing_patterns()
        self.reply_patterns = self._build_reply_patterns()

    def _build_greeting_patterns(self) -> list[str]:
        """Build regex patterns for common email greetings."""
        return [
            r"^hi\b",
            r"^hello\b",
            r"^hey\b",
            r"^dear\b",
            r"^good morning\b",
            r"^good afternoon\b",
            r"^good evening\b",
            r"^greetings\b",
            r"^salutations\b",
            r"^\w+,",  # Name followed by comma
        ]

    def _build_closing_patterns(self) -> list[str]:
        """Build regex patterns for common email closings."""
        return [
            r"\bbest\s*regards?\b",
            r"\bbest\b$",
            r"\bthanks?\b$",
            r"\bthank\s+you\b$",
            r"\bsincerely\b",
            r"\byours?\b$",
            r"\bcheers\b$",
            r"\btalk\s+soon\b",
            r"\bsee\s+you\b",
            r"\btake\s+care\b",
        ]

    def _build_reply_patterns(self) -> list[str]:
        """Build patterns for reply formatting."""
        return [
            r"^>\s*",  # Quoted text with >
            r"^on\s+.*wrote:",  # "On [date] [person] wrote:"
            r"^-----original message-----",
            r"^from:\s*",
            r"^to:\s*",
            r"^sent:\s*",
            r"^subject:\s*",
        ]

    def extract_features(self, text: str) -> torch.Tensor:
        """Extract all email-specific features."""
        features = []

        # Preprocessing
        text_lower = text.lower()
        lines = text.split("\n")

        # Greeting features
        greeting_features = self._extract_greeting_features(text_lower, lines)
        features.extend(greeting_features)

        # Closing features
        closing_features = self._extract_closing_features(text_lower, lines)
        features.extend(closing_features)

        # Reply formatting features
        reply_features = self._extract_reply_features(text_lower, lines)
        features.extend(reply_features)

        # Structure features
        structure_features = self._extract_structure_features(text, lines)
        features.extend(structure_features)

        return torch.tensor(features, dtype=torch.float32)

    def _extract_greeting_features(self, text_lower: str, lines: list[str]) -> list[float]:
        """Extract greeting-related features."""
        features = []
        first_lines = lines[:3]  # Check first few lines for greetings

        for pattern in self.greeting_patterns:
            # Check if pattern appears in first few lines
            found = any(re.search(pattern, line.strip().lower()) for line in first_lines)
            features.append(1.0 if found else 0.0)

        # Additional greeting features
        has_greeting = any(features)
        features.append(1.0 if has_greeting else 0.0)

        return features

    def _extract_closing_features(self, text_lower: str, lines: list[str]) -> list[float]:
        """Extract closing-related features."""
        features = []
        last_lines = lines[-3:] if len(lines) >= 3 else lines  # Check last few lines

        for pattern in self.closing_patterns:
            # Check if pattern appears in last few lines
            found = any(re.search(pattern, line.strip().lower()) for line in last_lines)
            features.append(1.0 if found else 0.0)

        # Additional closing features
        has_closing = any(features)
        features.append(1.0 if has_closing else 0.0)

        return features

    def _extract_reply_features(self, text_lower: str, lines: list[str]) -> list[float]:
        """Extract reply formatting features."""
        features = []

        for pattern in self.reply_patterns:
            # Check if pattern appears anywhere in text
            found = any(re.search(pattern, line.lower()) for line in lines)
            features.append(1.0 if found else 0.0)

        # Count quoted lines (lines starting with >)
        quoted_lines = sum(1 for line in lines if line.strip().startswith(">"))
        quoted_ratio = quoted_lines / max(len(lines), 1)
        features.append(quoted_ratio)

        # Check for forward indicators
        has_forward = any(re.search(r"fwd?:", line.lower()) for line in lines)
        features.append(1.0 if has_forward else 0.0)

        return features

    def _extract_structure_features(self, text: str, lines: list[str]) -> list[float]:
        """Extract email structure features."""
        features = []

        # Paragraph features
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        features.append(len(paragraphs))  # Number of paragraphs

        if paragraphs:
            avg_para_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            features.append(avg_para_length)
        else:
            features.append(0.0)

        # Line features
        non_empty_lines = [line for line in lines if line.strip()]
        features.append(len(non_empty_lines))  # Number of non-empty lines

        if non_empty_lines:
            avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
            features.append(avg_line_length)
        else:
            features.append(0.0)

        # Empty line ratio (indicates spacing habits)
        empty_lines = len(lines) - len(non_empty_lines)
        empty_ratio = empty_lines / max(len(lines), 1)
        features.append(empty_ratio)

        # Capitalization features
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features.append(caps_ratio)

        # All caps words
        words = text.split()
        if words:
            all_caps_ratio = sum(1 for word in words if word.isupper() and len(word) > 1) / len(words)
            features.append(all_caps_ratio)
        else:
            features.append(0.0)

        return features
