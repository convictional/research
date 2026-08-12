import time
import random
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from src.data.extractor import ContentExtractor
from src.embedders.colbert import ColBERTEmbedder
from src.embedders.openai_embedder import OpenAIEmbedder
from src.evaluation.query_generator import GeneratedQuery, QueryGenerator
from src.models.content import SearchQuery
from src.search.multi_vector import ColBERTLocalSearch
from src.search.single_vector import OpenAIEmbeddingSearch
from src.search.production_hybrid import ProductionHybridSearch


class LatencyStats(BaseModel):
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    sample_size: int


class LatencyBenchmark(BaseModel):
    systems: dict[str, LatencyStats]


class LatencyBenchmarker:
    def __init__(
        self,
        extractor: ContentExtractor,
        colbert_search: ColBERTLocalSearch,
        openai_search: OpenAIEmbeddingSearch,
        production_search,
    ):
        self.extractor = extractor
        self.colbert_search = colbert_search
        self.openai_search = openai_search
        self.production_search = production_search

    async def benchmark(self, queries: list[GeneratedQuery], sample_size: int = 50) -> LatencyBenchmark:
        sample_queries = random.sample(queries, min(sample_size, len(queries)))

        print(f"Benchmarking ColBERT local search ({sample_size} queries)...")
        colbert_latencies = await self._measure_latencies(self.colbert_search, sample_queries)

        print(f"Benchmarking OpenAI embedding search ({sample_size} queries)...")
        openai_latencies = await self._measure_latencies(self.openai_search, sample_queries)

        print(f"Benchmarking production hybrid search ({sample_size} queries)...")
        production_latencies = await self._measure_latencies(self.production_search, sample_queries)

        return LatencyBenchmark(
            systems={
                "colbert_local": self._compute_stats(colbert_latencies),
                "openai_embedding": self._compute_stats(openai_latencies),
                "production_hybrid": self._compute_stats(production_latencies),
            }
        )

    async def _measure_latencies(self, search_engine, queries: list[GeneratedQuery]) -> list[float]:
        latencies = []

        for query in queries:
            search_query = SearchQuery(text=query.text, top_k=10)

            start = time.perf_counter()
            await search_engine.search(search_query)
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)

        return latencies

    def _compute_stats(self, latencies: list[float]) -> LatencyStats:
        arr = np.array(latencies)

        return LatencyStats(
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            sample_size=len(latencies),
        )
