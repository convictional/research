"""Extract training data from decide_development database."""

import asyncio
import re
from collections import defaultdict
import logging

import asyncpg
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ContentSample(BaseModel):
    """Represents a content sample for training."""

    author: str
    content: str
    content_type: str
    source: str
    title: str
    length: int


class AuthorshipDataExtractor:
    """Extract authorship training data from the database."""

    def __init__(self, db_url: str = "postgresql://decide@localhost/decide_development"):
        self.db_url = db_url
        self.min_content_length = 500
        self.max_content_length = 10000

        # Authors we want to focus on (contributors with substantial content).
        # The real roster was removed before open-sourcing — replace these with the
        # author names as they appear in your own database.
        self.target_authors = {
            "Person A",
            "Person B",
            "Person C",
        }

        # Maps platform usernames to the canonical author name used above, for
        # sources (e.g. GitHub) that record a handle rather than a full name.
        self.username_mapping: dict[str, str] = {}

        # Content types that are likely human-written (expanded set)
        self.human_content_types = {
            # Original types
            "comment",
            "discussion_comment",
            "update",
            "task",
            "decision",
            "discussion",
            "decision_process",
            # High-value GitHub content (8,291 + 2,195 + 178 = 10,664 samples!)
            "github_comment",
            "github_issue",
            "github_discussion_comment",
            # Google Docs content (347 samples)
            "google_doc",
            # Meeting content (1,147 samples)
            "meeting",
        }

    async def extract_training_samples(self) -> dict[str, list[ContentSample]]:
        """Extract training samples grouped by author."""
        conn = await asyncpg.connect(self.db_url)

        try:
            # Query for human-written content with substantial length
            query = """
            SELECT
                author,
                index_content,
                content_type,
                source,
                title,
                LENGTH(index_content) as content_length
            FROM content
            WHERE
                content_type = ANY($1)
                AND LENGTH(index_content) BETWEEN $2 AND $3
                AND author IS NOT NULL
                AND author != ''
                AND author ~ '[A-Za-z]'
            ORDER BY author, content_length DESC
            """

            rows = await conn.fetch(
                query, list(self.human_content_types), self.min_content_length, self.max_content_length
            )

            # Group samples by individual authors
            samples_by_author = defaultdict(list)

            for row in rows:
                # Parse author field (may contain multiple comma-separated authors)
                authors = self._parse_authors(row["author"])

                # Only include if single author and in our target list
                if len(authors) == 1 and authors[0] in self.target_authors:
                    author = authors[0]

                    # Clean content
                    cleaned_content = self._clean_content(row["index_content"])

                    if len(cleaned_content) >= self.min_content_length:
                        sample = ContentSample(
                            author=author,
                            content=cleaned_content,
                            content_type=row["content_type"],
                            source=row["source"],
                            title=row["title"],
                            length=len(cleaned_content),
                        )
                        samples_by_author[author].append(sample)

            # Filter authors with sufficient samples
            min_samples = 5
            filtered_samples = {
                author: samples for author, samples in samples_by_author.items() if len(samples) >= min_samples
            }

            logger.info(f"Extracted samples for {len(filtered_samples)} authors:")
            for author, samples in filtered_samples.items():
                logger.info(f"  {author}: {len(samples)} samples")

            return filtered_samples

        finally:
            await conn.close()

    def _parse_authors(self, author_field: str) -> list[str]:
        """Parse author field which may contain multiple comma-separated authors."""
        if not author_field:
            return []

        # Split by comma and clean up
        authors = [author.strip() for author in author_field.split(",")]
        # Remove empty strings
        authors = [author for author in authors if author]

        # Apply username mapping to convert GitHub usernames to full names
        mapped_authors = []
        for author in authors:
            mapped_author = self.username_mapping.get(author, author)
            mapped_authors.append(mapped_author)

        return mapped_authors

    def _clean_content(self, content: str) -> str:
        """Clean content text for training."""
        if not content:
            return ""

        # Remove markdown formatting
        content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)  # Bold
        content = re.sub(r"\*([^*]+)\*", r"\1", content)  # Italic
        content = re.sub(r"`([^`]+)`", r"\1", content)  # Code

        # Remove HTML tags
        content = re.sub(r"<[^>]+>", "", content)

        # Remove excessive whitespace
        content = re.sub(r"\s+", " ", content)
        content = re.sub(r"\n\s*\n", "\n\n", content)

        # Remove common platform-specific patterns
        content = re.sub(r"@\[[^\]]+\]", "", content)  # Mentions like @[Name]
        content = re.sub(r"\bhttps?://[^\s]+", "[URL]", content)  # URLs

        return content.strip()

    async def get_author_statistics(self) -> dict[str, dict]:
        """Get statistics about content by author."""
        conn = await asyncpg.connect(self.db_url)

        try:
            query = """
            SELECT
                author,
                content_type,
                COUNT(*) as count,
                AVG(LENGTH(index_content)) as avg_length,
                SUM(LENGTH(index_content)) as total_length
            FROM content
            WHERE
                LENGTH(index_content) > 100
                AND author IS NOT NULL
                AND author != ''
            GROUP BY author, content_type
            ORDER BY author, count DESC
            """

            rows = await conn.fetch(query)

            stats = defaultdict(lambda: defaultdict(dict))

            for row in rows:
                # Parse single authors only
                authors = self._parse_authors(row["author"])
                if len(authors) == 1:
                    author = authors[0]
                    content_type = row["content_type"]

                    stats[author][content_type] = {
                        "count": row["count"],
                        "avg_length": float(row["avg_length"]),
                        "total_length": row["total_length"],
                    }

            return dict(stats)

        finally:
            await conn.close()

    def filter_samples_for_training(
        self,
        samples_by_author: dict[str, list[ContentSample]],
        max_samples_per_author: int = 50,
        min_length_threshold: int = 250,
    ) -> dict[str, list[ContentSample]]:
        """Filter and balance samples for training."""
        filtered_samples = {}

        for author, samples in samples_by_author.items():
            # Filter by length and content quality
            quality_samples = []

            for sample in samples:
                # Skip very short samples
                if sample.length < min_length_threshold:
                    continue

                # Skip samples that are mostly structured data or lists
                # if self._is_structured_content(sample.content):
                #     continue

                # Skip samples with too many technical references
                # if self._has_excessive_technical_content(sample.content):
                #     continue

                quality_samples.append(sample)

            # Sort by length (longer samples first) and take top samples
            quality_samples.sort(key=lambda x: x.length, reverse=True)
            filtered_samples[author] = quality_samples[:max_samples_per_author]

        return filtered_samples

    def _is_structured_content(self, content: str) -> bool:
        """Check if content appears to be mostly structured data."""
        # Count lines that look like list items or structured data
        lines = content.split("\n")
        structured_lines = 0

        for line in lines:
            line = line.strip()
            if (
                line.startswith("*")
                or line.startswith("-")
                or line.startswith("•")
                or re.match(r"^\d+\.", line)
                or len(line) < 20
            ):  # Very short lines
                structured_lines += 1

        # If more than 60% of lines are structured, consider it structured content
        return structured_lines / max(len(lines), 1) > 0.6

    def _has_excessive_technical_content(self, content: str) -> bool:
        """Check if content has too much technical jargon (code, SQL, etc.)."""
        # Count technical indicators
        technical_patterns = [
            r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN)\b",  # SQL
            r"\b(function|class|import|return|if|else|for|while)\b",  # Code
            r"[{}()\[\];]",  # Code symbols
            r"\b[A-Z_]{3,}\b",  # Constants
        ]

        technical_matches = 0
        for pattern in technical_patterns:
            technical_matches += len(re.findall(pattern, content, re.IGNORECASE))

        # If more than 10% of words are technical, skip
        word_count = len(content.split())
        return technical_matches / max(word_count, 1) > 0.1


async def main():
    """Main function to extract and analyze training data."""
    extractor = AuthorshipDataExtractor()

    # Get author statistics
    logger.info("Extracting author statistics...")
    stats = await extractor.get_author_statistics()

    print("Author Statistics (single authors only):")
    for author in sorted(stats.keys()):
        if author in extractor.target_authors:
            print(f"\n{author}:")
            for content_type, metrics in stats[author].items():
                if content_type in extractor.human_content_types:
                    print(f"  {content_type}: {metrics['count']} items, avg {int(metrics['avg_length'])} chars")

    # Extract training samples
    logger.info("Extracting training samples...")
    samples = await extractor.extract_training_samples()

    # Filter for quality
    filtered_samples = extractor.filter_samples_for_training(samples)

    print(f"\nFiltered Training Data:")
    for author, author_samples in filtered_samples.items():
        print(f"{author}: {len(author_samples)} samples")
        if author_samples:
            lengths = [s.length for s in author_samples]
            print(f"  Length range: {min(lengths)}-{max(lengths)} chars")

    # Test the feature extraction pipeline
    print("\n" + "=" * 50)
    print("TESTING FEATURE EXTRACTION PIPELINE")
    print("=" * 50)

    await test_feature_pipeline(filtered_samples)

    return filtered_samples


async def test_feature_pipeline(filtered_samples):
    """Test the feature extraction pipeline with real data."""
    from features.extractors import StyleFeatureExtractor, SemanticFeatureExtractor
    from features.email_patterns import EmailPatternExtractor

    if not filtered_samples:
        print("No samples to test!")
        return

    print("Initializing feature extractors...")

    # Initialize extractors
    semantic_extractor = SemanticFeatureExtractor()
    style_extractor = StyleFeatureExtractor()
    email_extractor = EmailPatternExtractor()

    # Prepare training data for style extractor
    all_texts = []
    for author, samples in filtered_samples.items():
        texts = [sample.content for sample in samples[:5]]
        all_texts.extend(texts)

    print(f"Fitting style extractor on {len(all_texts)} samples...")
    style_extractor.fit(all_texts)

    # Test feature extraction on a sample text
    test_author = next(iter(filtered_samples.keys()))
    test_sample = filtered_samples[test_author][0]
    test_text = test_sample.content

    print(f"\nTesting with sample from {test_author} ({len(test_text)} chars)...")
    print(f"Sample text preview: {test_text[:200]}...")

    # Extract features
    semantic_features = semantic_extractor.extract_features(test_text)
    style_features = style_extractor.extract_features(test_text)
    email_features = email_extractor.extract_features(test_text)

    print(f"\nFeature dimensions:")
    print(f"  Semantic: {semantic_features.shape}")
    print(f"  Style: {style_features.shape}")
    print(f"  Email: {email_features.shape}")

    # Test combining features (like in dataset)
    import torch

    combined_style = torch.cat([style_features, email_features])
    print(f"  Combined style: {combined_style.shape}")

    print("\n✅ Feature extraction pipeline test completed successfully!")

    # Show sample distribution by author
    print("\nDetailed sample distribution:")
    for author, samples in filtered_samples.items():
        lengths = [s.length for s in samples]
        content_types = set(s.content_type for s in samples)
        print(f"  {author}: {len(samples)} samples")
        print(f"    Length range: {min(lengths)}-{max(lengths)} chars")
        print(f"    Content types: {', '.join(sorted(content_types))}")
        print(f"    Avg length: {sum(lengths) // len(lengths)} chars")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
