"""Reddit dataset loader using ConvoKit for authorship classification training."""

import logging
import pickle
from pathlib import Path
from typing import Tuple
from collections import defaultdict, Counter
import re

from convokit import Corpus, download
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RedditAuthor(BaseModel):
    """Reddit author with associated texts."""

    author_id: str
    username: str
    texts: list[str]
    subreddit: str
    comment_count: int
    total_chars: int
    avg_text_length: float


class RedditDatasetConfig(BaseModel):
    """Configuration for Reddit dataset loading."""

    dataset_name: str = "subreddit-Cornell"  # ConvoKit dataset name
    subreddits: list[str] | None = [
        # Business & Entrepreneurship
        "business",
        "Entrepreneur",
        "startups",
        "smallbusiness",
        "consulting",
        # Tech & Work
        "sysadmin",
        "ITCareerQuestions",
        "datascience",
        "MachineLearning",
        "programming",
        # Finance & Operations
        "finance",
        "FinancialCareers",
        "accounting",
        "AskHR",
        # Professional Communication
        "careerguidance",
    ]  # Target subreddits (None = all)
    min_comment_length: int = 100  # Minimum characters per comment
    max_comment_length: int = 5000  # Maximum characters per comment
    min_comments_per_author: int = 10  # Minimum comments per author
    max_authors: int | None = 5000  # Maximum authors to include
    exclude_deleted: bool = True  # Exclude [deleted] and [removed] content
    exclude_bots: bool = True  # Exclude known bot accounts
    cache_dir: str = "cache/reddit"
    force_regenerate: bool = False  # Force regeneration of cached data


class RedditDatasetLoader:
    """Load Reddit dataset from ConvoKit for authorship classification."""

    def __init__(self, config: RedditDatasetConfig):
        """
        Initialize Reddit dataset loader.

        Args:
            config: Reddit dataset configuration
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Known bot patterns to exclude
        self.bot_patterns = [
            r".*bot.*",
            r".*Bot.*",
            r"AutoModerator",
            r".*_bot$",
            r".*-bot$",
            r".*reply.*bot.*",
            r".*reminder.*",
            r".*archive.*",
        ]

        logger.info(f"RedditDatasetLoader initialized for {config.dataset_name}")

    def is_bot_username(self, username: str) -> bool:
        """Check if username matches known bot patterns."""
        if not username or username in ["[deleted]", "[removed]"]:
            return True

        for pattern in self.bot_patterns:
            if re.match(pattern, username, re.IGNORECASE):
                return True
        return False

    def clean_text(self, text: str) -> str:
        """Clean and preprocess Reddit comment text."""
        if not text:
            return ""

        # Skip deleted/removed content
        if text.strip().lower() in ["[deleted]", "[removed]", ""]:
            return ""

        # Remove Reddit formatting
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*(.*?)\*", r"\1", text)  # Italic
        text = re.sub(r"~~(.*?)~~", r"\1", text)  # Strikethrough
        text = re.sub(r"\^([^\s]+)", r"\1", text)  # Superscript

        # Remove Reddit-specific patterns
        text = re.sub(r"/u/\w+", "", text)  # User mentions
        text = re.sub(r"/r/\w+", "", text)  # Subreddit mentions
        text = re.sub(r"https?://\S+", "[URL]", text)  # URLs
        text = re.sub(r"www\.\S+", "[URL]", text)  # URLs without protocol

        # Remove quote blocks (lines starting with >)
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith(">")]
        text = "\n".join(lines)

        # Clean up whitespace
        text = re.sub(r"\n+", " ", text)  # Multiple newlines to space
        text = re.sub(r"\s+", " ", text)  # Multiple spaces to single space
        text = text.strip()

        return text

    def load_reddit_corpus(self) -> Tuple[dict[str, list[str]], dict[str, RedditAuthor]]:
        """
        Load Reddit corpus from ConvoKit.

        Returns:
            Tuple of (texts_by_author, author_metadata)
        """
        num_subreddits = len(self.config.subreddits) if self.config.subreddits else 0
        subreddits_identifier = f"{num_subreddits}subreddits" if num_subreddits > 0 else "all"
        cache_file = self.cache_dir / f"reddit_combined_{subreddits_identifier}_processed.pkl"

        # Load from cache if available (unless force_regenerate is True)
        if cache_file.exists() and not self.config.force_regenerate:
            logger.info(f"Loading cached Reddit data from {cache_file}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                return cached_data["texts_by_author"], cached_data["author_metadata"]
        elif cache_file.exists() and self.config.force_regenerate:
            logger.info(f"Force regenerating Reddit data (ignoring cache at {cache_file})")

        # For Reddit data, we need to load each subreddit individually if specific ones are requested
        utterances_by_author = defaultdict(list)

        if self.config.subreddits:
            logger.info(f"Loading {len(self.config.subreddits)} specific subreddits from ConvoKit")

            for subreddit in self.config.subreddits:
                subreddit_dataset_name = f"subreddit-{subreddit}"
                logger.info(f"Loading subreddit: {subreddit_dataset_name}")

                try:
                    # Download and load individual subreddit corpus
                    corpus = Corpus(filename=download(subreddit_dataset_name))
                    logger.info(f"Loaded {subreddit} with {len(corpus.utterances)} utterances")

                    # Process utterances from this subreddit
                    for utterance in corpus.iter_utterances():
                        author = utterance.speaker.id if utterance.speaker else None
                        text = utterance.text

                        # Skip if no author or text
                        if not author or not text:
                            continue

                        # Skip deleted users and bots
                        if self.config.exclude_deleted and author in ["[deleted]", "[removed]"]:
                            continue

                        if self.config.exclude_bots and self.is_bot_username(author):
                            continue

                        # Clean text
                        cleaned_text = self.clean_text(text)

                        # Skip if too short or too long after cleaning
                        if (
                            len(cleaned_text) < self.config.min_comment_length
                            or len(cleaned_text) > self.config.max_comment_length
                        ):
                            continue

                        utterances_by_author[author].append(
                            {
                                "text": cleaned_text,
                                "subreddit": subreddit,
                                "timestamp": utterance.timestamp,
                                "length": len(cleaned_text),
                            }
                        )

                except Exception as e:
                    logger.warning(f"Could not load subreddit {subreddit}: {e}")
                    continue

        else:
            # Load the full Reddit corpus (small version for testing)
            logger.info("Loading full Reddit corpus (small version)")
            try:
                corpus = Corpus(filename=download("reddit-corpus-small"))
                logger.info(f"Loaded corpus with {len(corpus.utterances)} utterances")

                # Process all utterances from the corpus
                for utterance in corpus.iter_utterances():
                    author = utterance.speaker.id if utterance.speaker else None
                    text = utterance.text

                    # Skip if no author or text
                    if not author or not text:
                        continue

                    # Skip deleted users and bots
                    if self.config.exclude_deleted and author in ["[deleted]", "[removed]"]:
                        continue

                    if self.config.exclude_bots and self.is_bot_username(author):
                        continue

                    # Clean text
                    cleaned_text = self.clean_text(text)

                    # Skip if too short or too long after cleaning
                    if (
                        len(cleaned_text) < self.config.min_comment_length
                        or len(cleaned_text) > self.config.max_comment_length
                    ):
                        continue

                    # Get subreddit from conversation metadata
                    conversation = corpus.get_conversation(utterance.conversation_id)
                    subreddit = (
                        conversation.meta.get("subreddit", "unknown")
                        if conversation and hasattr(conversation, "meta")
                        else "unknown"
                    )

                    utterances_by_author[author].append(
                        {
                            "text": cleaned_text,
                            "subreddit": subreddit,
                            "timestamp": utterance.timestamp,
                            "length": len(cleaned_text),
                        }
                    )
            except Exception as e:
                logger.error(f"Could not load reddit-corpus-small: {e}")

        logger.info(f"Found utterances from {len(utterances_by_author)} authors")

        # Filter authors by minimum comment count
        filtered_authors = {}
        author_metadata = {}

        for author, utterances in utterances_by_author.items():
            if len(utterances) >= self.config.min_comments_per_author:
                texts = [u["text"] for u in utterances]
                total_chars = sum(len(text) for text in texts)
                avg_length = total_chars / len(texts) if texts else 0

                # Get most common subreddit for this author
                subreddits = [u["subreddit"] for u in utterances if u["subreddit"]]
                common_subreddit = Counter(subreddits).most_common(1)[0][0] if subreddits else "unknown"

                filtered_authors[author] = texts
                author_metadata[author] = RedditAuthor(
                    author_id=author,
                    username=author,
                    texts=texts,
                    subreddit=common_subreddit,
                    comment_count=len(texts),
                    total_chars=total_chars,
                    avg_text_length=avg_length,
                )

        logger.info(
            f"Filtered to {len(filtered_authors)} authors with {self.config.min_comments_per_author}+ comments"
        )

        # Limit number of authors if specified
        if self.config.max_authors and len(filtered_authors) > self.config.max_authors:
            # Sort authors by comment count (descending) and take top N
            sorted_authors = sorted(author_metadata.items(), key=lambda x: x[1].comment_count, reverse=True)[
                : self.config.max_authors
            ]

            filtered_authors = {author: filtered_authors[author] for author, _ in sorted_authors}
            author_metadata = {author: metadata for author, metadata in sorted_authors}

            logger.info(f"Limited to top {self.config.max_authors} authors by comment count")

        # Cache processed data
        cache_data = {
            "texts_by_author": filtered_authors,
            "author_metadata": author_metadata,
            "config": self.config.dict(),
            "processing_stats": {
                "total_authors": len(utterances_by_author),
                "filtered_authors": len(filtered_authors),
                "total_comments": sum(len(texts) for texts in filtered_authors.values()),
                "avg_comments_per_author": sum(len(texts) for texts in filtered_authors.values())
                / len(filtered_authors)
                if filtered_authors
                else 0,
            },
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"Cached processed data to {cache_file}")

        # Log statistics
        total_comments = sum(len(texts) for texts in filtered_authors.values())
        avg_comments = total_comments / len(filtered_authors) if filtered_authors else 0
        logger.info(f"Final dataset: {len(filtered_authors)} authors, {total_comments} comments")
        logger.info(f"Average comments per author: {avg_comments:.1f}")

        return filtered_authors, author_metadata

    def create_author_splits(
        self, texts_by_author: dict[str, list[str]], train_ratio: float = 0.8
    ) -> Tuple[dict[str, list[str]], dict[str, list[str]]]:
        """
        Split texts by author into train/validation sets.

        Args:
            texts_by_author: dictionary of author -> texts
            train_ratio: Fraction of texts per author for training

        Returns:
            Tuple of (train_texts_by_author, val_texts_by_author)
        """
        train_texts = {}
        val_texts = {}

        for author, texts in texts_by_author.items():
            # Ensure each author has at least 1 text in validation
            if len(texts) < 2:
                # If author has only 1 text, put in training (they'll be filtered out anyway)
                train_texts[author] = texts
                continue

            split_idx = max(1, int(len(texts) * train_ratio))  # At least 1 for validation
            split_idx = min(split_idx, len(texts) - 1)  # At most len-1 for validation

            train_texts[author] = texts[:split_idx]
            val_texts[author] = texts[split_idx:]

        # Filter out authors with insufficient data in either split
        min_texts_per_split = 1

        train_filtered = {
            author: texts
            for author, texts in train_texts.items()
            if len(texts) >= min_texts_per_split
            and author in val_texts
            and len(val_texts[author]) >= min_texts_per_split
        }

        val_filtered = {author: texts for author, texts in val_texts.items() if author in train_filtered}

        logger.info(
            f"Train split: {len(train_filtered)} authors, {sum(len(texts) for texts in train_filtered.values())} texts"
        )
        logger.info(
            f"Val split: {len(val_filtered)} authors, {sum(len(texts) for texts in val_filtered.values())} texts"
        )

        return train_filtered, val_filtered


def load_reddit_dataset_for_training(
    config: RedditDatasetConfig,
) -> Tuple[dict[str, list[str]], dict[str, list[str]], dict[str, RedditAuthor]]:
    """
    Convenience function to load Reddit dataset with train/val splits.

    Args:
        config: Reddit dataset configuration

    Returns:
        Tuple of (train_texts, val_texts, author_metadata)
    """
    loader = RedditDatasetLoader(config)
    texts_by_author, author_metadata = loader.load_reddit_corpus()
    train_texts, val_texts = loader.create_author_splits(texts_by_author)

    return train_texts, val_texts, author_metadata
