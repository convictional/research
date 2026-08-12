import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

CACHE_DIR = Path(__file__).parent / "cache"


@dataclass
class EmbeddingResult:
    embeddings: list[list[float]]
    api_seconds: float
    texts_embedded: int


def _cache_key(text: str, dimensions: int) -> str:
    return hashlib.sha256(f"text-embedding-3-small:{dimensions}:{text}".encode()).hexdigest()


def _load_cached(text: str, dimensions: int) -> list[float] | None:
    path = CACHE_DIR / "text-embedding-3-small" / f"{_cache_key(text, dimensions)}.npy"
    if path.exists():
        return np.load(path).tolist()
    return None


def _save_cached(text: str, dimensions: int, embedding: list[float]) -> None:
    dir_path = CACHE_DIR / "text-embedding-3-small"
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{_cache_key(text, dimensions)}.npy"
    np.save(path, np.array(embedding, dtype=np.float32))


def embed_with_openai(
    texts: list[str],
    dimensions: int = 1536,
    batch_size: int = 50,
) -> EmbeddingResult:
    """Embed texts using OpenAI text-embedding-3-small with .npy file caching."""
    client = OpenAI()

    results: list[list[float]] = []
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = _load_cached(text, dimensions)
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
                model="text-embedding-3-small",
                dimensions=dimensions,
            )
            api_seconds += time.monotonic() - t0
            for j, emb in enumerate(response.data):
                idx = uncached_indices[batch_start + j]
                results[idx] = emb.embedding
                _save_cached(uncached_texts[batch_start + j], dimensions, emb.embedding)
            if batch_start + batch_size < len(uncached_texts):
                time.sleep(0.5)

    return EmbeddingResult(embeddings=results, api_seconds=api_seconds, texts_embedded=len(uncached_texts))
