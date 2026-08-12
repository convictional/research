import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vertexai
from google import genai
from openai import OpenAI
from vertexai.language_models import TextEmbeddingModel

CACHE_DIR = Path(__file__).parent / "cache"
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "${GCP_PROJECT}")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")


@dataclass
class EmbeddingResult:
    embeddings: list[list[float]]
    api_seconds: float
    texts_embedded: int

    @property
    def avg_ms_per_text(self) -> float:
        if self.texts_embedded == 0:
            return 0.0
        return (self.api_seconds / self.texts_embedded) * 1000


def _cache_key(model: str, text: str, dimensions: int) -> str:
    return hashlib.sha256(f"{model}:{dimensions}:{text}".encode()).hexdigest()


def _load_cached(model: str, text: str, dimensions: int) -> list[float] | None:
    path = CACHE_DIR / model / f"{_cache_key(model, text, dimensions)}.npy"
    if path.exists():
        return np.load(path).tolist()
    return None


def _save_cached(model: str, text: str, dimensions: int, embedding: list[float]) -> None:
    dir_path = CACHE_DIR / model
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{_cache_key(model, text, dimensions)}.npy"
    np.save(path, np.array(embedding, dtype=np.float32))


async def embed_with_text_embedding_005(
    texts: list[str],
    dimensions: int = 768,
    batch_size: int = 50,
) -> EmbeddingResult:
    """Embed texts using Vertex AI text-embedding-005."""
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    model_name = "text-embedding-005"

    results: list[list[float]] = []
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = _load_cached(model_name, text, dimensions)
        if cached is not None:
            results.append(cached)
        else:
            results.append([])
            uncached_indices.append(i)
            uncached_texts.append(text)

    api_seconds = 0.0
    if uncached_texts:
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start : batch_start + batch_size]
            t0 = time.monotonic()
            embeddings = model.get_embeddings(batch, output_dimensionality=dimensions)
            api_seconds += time.monotonic() - t0
            for j, emb in enumerate(embeddings):
                idx = uncached_indices[batch_start + j]
                results[idx] = emb.values
                _save_cached(model_name, uncached_texts[batch_start + j], dimensions, emb.values)
            if batch_start + batch_size < len(uncached_texts):
                time.sleep(0.5)

    return EmbeddingResult(embeddings=results, api_seconds=api_seconds, texts_embedded=len(uncached_texts))


def embed_with_openai(
    texts: list[str],
    dimensions: int = 1536,
    batch_size: int = 50,
) -> EmbeddingResult:
    """Embed texts using OpenAI text-embedding-3-small."""
    client = OpenAI()
    model_name = "text-embedding-3-small"

    results: list[list[float]] = []
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = _load_cached(model_name, text, dimensions)
        if cached is not None:
            results.append(cached)
        else:
            results.append([])
            uncached_indices.append(i)
            uncached_texts.append(text)

    api_seconds = 0.0
    if uncached_texts:
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start : batch_start + batch_size]
            t0 = time.monotonic()
            response = client.embeddings.create(
                input=batch,
                model=model_name,
                dimensions=dimensions,
            )
            api_seconds += time.monotonic() - t0
            for j, emb in enumerate(response.data):
                idx = uncached_indices[batch_start + j]
                results[idx] = emb.embedding
                _save_cached(model_name, uncached_texts[batch_start + j], dimensions, emb.embedding)
            if batch_start + batch_size < len(uncached_texts):
                time.sleep(0.5)

    return EmbeddingResult(embeddings=results, api_seconds=api_seconds, texts_embedded=len(uncached_texts))


async def embed_with_gemini_embedding_001(
    texts: list[str],
    dimensions: int = 1536,
    batch_size: int = 50,
) -> EmbeddingResult:
    """Embed texts using Google genai gemini-embedding-001."""
    client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    model_name = "gemini-embedding-001"

    results: list[list[float]] = []
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = _load_cached(model_name, text, dimensions)
        if cached is not None:
            results.append(cached)
        else:
            results.append([])
            uncached_indices.append(i)
            uncached_texts.append(text)

    api_seconds = 0.0
    if uncached_texts:
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start : batch_start + batch_size]
            t0 = time.monotonic()
            response = client.models.embed_content(
                model=model_name,
                contents=batch,
                config={"output_dimensionality": dimensions},
            )
            api_seconds += time.monotonic() - t0
            for j, emb in enumerate(response.embeddings):
                idx = uncached_indices[batch_start + j]
                results[idx] = list(emb.values)
                _save_cached(model_name, uncached_texts[batch_start + j], dimensions, list(emb.values))
            if batch_start + batch_size < len(uncached_texts):
                time.sleep(0.5)

    return EmbeddingResult(embeddings=results, api_seconds=api_seconds, texts_embedded=len(uncached_texts))
