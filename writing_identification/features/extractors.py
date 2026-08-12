"""Feature extraction modules for stylometric analysis."""

import re
from collections import Counter
from typing import Tuple
import string

import torch
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Semantic embeddings
from sentence_transformers import SentenceTransformer

# Local config for selecting the sentence-transformer backbone
from config.config import config as experiment_config


class TextPreprocessor:
    """Handles text preprocessing for feature extraction."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    def clean_text(self, text: str) -> str:
        """Basic text cleaning while preserving stylistic elements."""
        # Remove excessive whitespace but preserve single spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize_and_tag(self, text: str) -> list[Tuple[str, str]]:
        """Tokenize text and return (token, pos_tag) pairs."""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]


class StyleFeatureExtractor:
    """Extracts traditional stylometric features."""

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.char_vectorizer = None
        self.char_3gram_vectorizer = None
        self.word_vectorizer = None
        self.function_words = self._load_function_words()
        self.preprocessor = TextPreprocessor()

    def _load_function_words(self) -> set:
        """Load common function words."""
        function_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "under",
            "over",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "mine",
            "yours",
            "hers",
            "ours",
            "this",
            "that",
            "these",
            "those",
            "is",
            "am",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "shall",
        }
        return function_words

    def fit(self, texts: list[str]):
        """Fit vectorizers on training texts."""
        # Character n-grams (2, 4) - reduced allocation
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), max_features=self.max_features // 3, lowercase=True
        )
        self.char_vectorizer.fit(texts)

        # Character 3-grams specifically - higher allocation for focused features
        self.char_3gram_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 3), max_features=self.max_features // 3, lowercase=True
        )
        self.char_3gram_vectorizer.fit(texts)

        # Word n-grams
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=self.max_features // 3,
            lowercase=True,
            stop_words=None,  # Keep all words for stylometric analysis
        )
        self.word_vectorizer.fit(texts)

    def extract_features(self, text: str) -> torch.Tensor:
        """Extract all stylometric features from text."""
        features = {}

        # Character n-gram features
        if self.char_vectorizer:
            char_features = self.char_vectorizer.transform([text]).toarray()[0]
            features["char_ngrams"] = char_features

        # Character 3-gram features
        if self.char_3gram_vectorizer:
            char_3gram_features = self.char_3gram_vectorizer.transform([text]).toarray()[0]
            features["char_3grams"] = char_3gram_features

        # Word n-gram features
        if self.word_vectorizer:
            word_features = self.word_vectorizer.transform([text]).toarray()[0]
            features["word_ngrams"] = word_features

        # Statistical features
        features["statistical"] = self._extract_statistical_features(text)

        # POS tag features
        features["pos_tags"] = self._extract_pos_features(text)

        # Function word features
        features["function_words"] = self._extract_function_word_features(text)

        # Punctuation features
        features["punctuation"] = self._extract_punctuation_features(text)

        # Concatenate all features
        all_features = []
        for feature_type, feature_vector in features.items():
            if isinstance(feature_vector, list):
                all_features.extend(feature_vector)
            else:
                all_features.extend(feature_vector.tolist())

        return torch.tensor(all_features, dtype=torch.float32)

    def _extract_statistical_features(self, text: str) -> list[float]:
        """Extract statistical text features."""
        sentences = text.split(".")
        words = text.split()
        chars = list(text)

        features = [
            len(sentences),  # Sentence count
            len(words),  # Word count
            len(chars),  # Character count
            len(words) / max(len(sentences), 1),  # Avg words per sentence
            len(chars) / max(len(words), 1),  # Avg chars per word
            len([w for w in words if len(w) > 6]) / max(len(words), 1),  # Long word ratio
        ]

        return features

    def _extract_pos_features(self, text: str) -> list[float]:
        """Extract POS tag distribution features."""
        tokens_tags = self.preprocessor.tokenize_and_tag(text)

        if not tokens_tags:
            return [0.0] * 17  # Return zeros for 17 major POS categories

        pos_counts = Counter([tag for _, tag in tokens_tags])
        total_tokens = len(tokens_tags)

        # Major POS categories
        pos_categories = [
            "NOUN",
            "VERB",
            "ADJ",
            "ADV",
            "PRON",
            "DET",
            "ADP",
            "CONJ",
            "NUM",
            "PRT",
            "X",
            "PUNCT",
            "PROPN",
            "AUX",
            "CCONJ",
            "SCONJ",
            "INTJ",
        ]

        features = []
        for pos in pos_categories:
            ratio = pos_counts.get(pos, 0) / total_tokens
            features.append(ratio)

        return features

    def _extract_function_word_features(self, text: str) -> list[float]:
        """Extract function word frequency features."""
        words = text.lower().split()
        total_words = len(words)

        if total_words == 0:
            return [0.0] * len(self.function_words)

        word_counts = Counter(words)

        features = []
        for func_word in sorted(self.function_words):
            ratio = word_counts.get(func_word, 0) / total_words
            features.append(ratio)

        return features

    def _extract_punctuation_features(self, text: str) -> list[float]:
        """Extract punctuation usage features."""
        total_chars = len(text)

        if total_chars == 0:
            return [0.0] * len(string.punctuation)

        punct_counts = Counter([c for c in text if c in string.punctuation])

        features = []
        for punct in string.punctuation:
            ratio = punct_counts.get(punct, 0) / total_chars
            features.append(ratio)

        return features

    def __getstate__(self):
        """Custom pickling to avoid issues with spacy models."""
        state = self.__dict__.copy()
        # Remove the preprocessor with spacy model before pickling
        state['preprocessor'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling to recreate spacy model."""
        self.__dict__.update(state)
        # Recreate the preprocessor after unpickling
        self.preprocessor = TextPreprocessor()


class SemanticFeatureExtractor:
    """Extracts semantic embeddings using Sentence-Transformer models."""

    def __init__(self, model_name: str | None = None):
        """Create a new semantic feature extractor.

        Args:
            model_name: Identifier from `SentenceTransformer` hub.  If ``None``
                (default) the value defined in the experiment configuration
                (``experiment_config.model.sentence_bert_model``) is used.
        """
        # Resolve model name from config when not provided explicitly
        if model_name is None:
            # Handle both object and dict config formats
            if hasattr(experiment_config, "model") and hasattr(experiment_config.model, "sentence_bert_model"):
                model_name = experiment_config.model.sentence_bert_model
            elif hasattr(experiment_config, "model") and isinstance(experiment_config.model, dict):
                model_name = experiment_config.model.get("sentence_bert_model", "all-mpnet-base-v2")
            else:
                model_name = getattr(experiment_config, "sentence_bert_model", "all-mpnet-base-v2")

        self.model = SentenceTransformer(model_name)

        # Move to Apple-Silicon / Metal if available to speed-up inference
        if torch.backends.mps.is_available():
            self.model = self.model.to(torch.device("mps"))

    def extract_features(self, text: str) -> torch.Tensor:
        """Extract semantic embedding from text."""
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding.cpu()  # Move back to CPU for consistency
