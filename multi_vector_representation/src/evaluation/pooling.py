import json
import random
import time
from pathlib import Path

from pydantic import BaseModel

from src.data.extractor import ContentExtractor
from src.evaluation.query_generator import GeneratedQuery
from src.models.content import SearchQuery
from src.search.multi_vector import ColBERTLocalSearch
from src.search.single_vector import OpenAIEmbeddingSearch
from src.search.production_hybrid import ProductionHybridSearch
from src.search.production_reranked import ProductionRerankedSearch


class QueryLatency(BaseModel):
    query_id: str
    system_name: str
    latency_ms: float


class PooledDocument(BaseModel):
    doc_id: str
    title: str
    content_type: str
    preview: str
    retrieved_by: list[str]


class QueryPool(BaseModel):
    query_id: str
    query_text: str
    ground_truth_doc_ids: list[str]
    pooled_docs: list[PooledDocument]


class ResultPooler:
    def __init__(
        self,
        extractor: ContentExtractor,
        colbert_search: ColBERTLocalSearch,
        openai_search: OpenAIEmbeddingSearch,
        production_search: ProductionHybridSearch,
        production_reranked_search: ProductionRerankedSearch | None = None,
    ):
        self.extractor = extractor
        self.colbert_search = colbert_search
        self.openai_search = openai_search
        self.production_search = production_search
        self.production_reranked_search = production_reranked_search

    async def create_pools(
        self,
        queries: list[GeneratedQuery],
        top_k: int = 10,
        checkpoint_interval: int = 25,
        pools_path: Path | None = None,
        latencies_path: Path | None = None,
    ) -> tuple[list[QueryPool], list[QueryLatency]]:
        pools = []
        all_latencies = []
        completed_query_ids = set()

        if pools_path:
            partial_pools_path = pools_path.with_suffix(".partial.json")
            if partial_pools_path.exists():
                print(f"Resuming from {partial_pools_path}")
                pools = self.load_pools(partial_pools_path)
                completed_query_ids = {p.query_id for p in pools}
                print(f"Loaded {len(pools)} completed queries, resuming from query {len(pools) + 1}")

        if latencies_path:
            partial_latencies_path = latencies_path.with_suffix(".partial.json")
            if partial_latencies_path.exists():
                with open(partial_latencies_path) as f:
                    data = json.load(f)
                all_latencies = [QueryLatency(**item) for item in data]

        queries_to_process = [q for q in queries if q.id not in completed_query_ids]
        total_queries = len(queries)
        starting_index = len(pools)

        for idx, query in enumerate(queries_to_process, start=starting_index + 1):
            progress_pct = (idx / total_queries) * 100
            print(f"Pooling query {idx}/{total_queries} ({progress_pct:.1f}%): {query.id}")

            pool, latencies = await self._pool_for_query(query, top_k)
            pools.append(pool)
            all_latencies.extend(latencies)

            if checkpoint_interval and idx % checkpoint_interval == 0:
                if pools_path:
                    await self._save_checkpoint(pools, pools_path)
                if latencies_path:
                    await self._save_latencies_checkpoint(all_latencies, latencies_path)
                print(f"Checkpoint saved at query {idx}/{total_queries}")

        return pools, all_latencies

    async def _save_checkpoint(self, pools: list[QueryPool], pools_path: Path):
        partial_path = pools_path.with_suffix(".partial.json")
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(partial_path, "w") as f:
            json.dump([p.model_dump() for p in pools], f, indent=2)

    async def _save_latencies_checkpoint(self, latencies: list[QueryLatency], latencies_path: Path):
        partial_path = latencies_path.with_suffix(".partial.json")
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(partial_path, "w") as f:
            json.dump([lat.model_dump() for lat in latencies], f, indent=2)

    async def _pool_for_query(self, query: GeneratedQuery, top_k: int) -> tuple[QueryPool, list[QueryLatency]]:
        search_query = SearchQuery(text=query.text, top_k=top_k)
        latencies = []

        colbert_results = []
        try:
            start = time.perf_counter()
            colbert_results = await self.colbert_search.search(search_query)
            latencies.append(
                QueryLatency(
                    query_id=query.id, system_name="colbert_local", latency_ms=(time.perf_counter() - start) * 1000
                )
            )
        except Exception as e:
            print(f"  ⚠ colbert_local failed: {e}")

        openai_results = []
        try:
            start = time.perf_counter()
            openai_results = await self.openai_search.search(search_query)
            latencies.append(
                QueryLatency(
                    query_id=query.id, system_name="openai_embedding", latency_ms=(time.perf_counter() - start) * 1000
                )
            )
        except Exception as e:
            print(f"  ⚠ openai_embedding failed: {e}")

        production_results = []
        try:
            start = time.perf_counter()
            production_results = await self.production_search.search(search_query)
            latencies.append(
                QueryLatency(
                    query_id=query.id, system_name="production_hybrid", latency_ms=(time.perf_counter() - start) * 1000
                )
            )
        except Exception as e:
            print(f"  ⚠ production_hybrid failed: {e}")

        reranked_results = []
        if self.production_reranked_search:
            try:
                start = time.perf_counter()
                reranked_results = await self.production_reranked_search.search(search_query)
                latencies.append(
                    QueryLatency(
                        query_id=query.id,
                        system_name="production_reranked",
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
                )
            except Exception as e:
                print(f"  ⚠ production_reranked failed: {e}")

        doc_map = {}
        for result in colbert_results:
            doc_id = str(result.content_id)
            if doc_id not in doc_map:
                doc_map[doc_id] = PooledDocument(
                    doc_id=doc_id,
                    title=result.title,
                    content_type=result.content_type,
                    preview=result.preview,
                    retrieved_by=["colbert_local"],
                )
            else:
                doc_map[doc_id].retrieved_by.append("colbert_local")

        for result in openai_results:
            doc_id = str(result.content_id)
            if doc_id not in doc_map:
                doc_map[doc_id] = PooledDocument(
                    doc_id=doc_id,
                    title=result.title,
                    content_type=result.content_type,
                    preview=result.preview,
                    retrieved_by=["openai_embedding"],
                )
            else:
                if "openai_embedding" not in doc_map[doc_id].retrieved_by:
                    doc_map[doc_id].retrieved_by.append("openai_embedding")

        for result in production_results:
            doc_id = str(result.content_id)
            if doc_id not in doc_map:
                doc_map[doc_id] = PooledDocument(
                    doc_id=doc_id,
                    title=result.title,
                    content_type=result.content_type,
                    preview=result.preview,
                    retrieved_by=["production_hybrid"],
                )
            else:
                if "production_hybrid" not in doc_map[doc_id].retrieved_by:
                    doc_map[doc_id].retrieved_by.append("production_hybrid")

        for result in reranked_results:
            doc_id = str(result.content_id)
            if doc_id not in doc_map:
                doc_map[doc_id] = PooledDocument(
                    doc_id=doc_id,
                    title=result.title,
                    content_type=result.content_type,
                    preview=result.preview,
                    retrieved_by=["production_reranked"],
                )
            else:
                if "production_reranked" not in doc_map[doc_id].retrieved_by:
                    doc_map[doc_id].retrieved_by.append("production_reranked")

        pooled_docs = list(doc_map.values())
        random.shuffle(pooled_docs)

        pool = QueryPool(
            query_id=query.id,
            query_text=query.text,
            ground_truth_doc_ids=query.source_doc_ids,
            pooled_docs=pooled_docs,
        )

        return pool, latencies

    async def save_pools(self, pools: list[QueryPool], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([p.model_dump() for p in pools], f, indent=2)
        print(f"Saved {len(pools)} query pools to {output_path}")

        partial_path = output_path.with_suffix(".partial.json")
        if partial_path.exists():
            partial_path.unlink()

    async def save_latencies(self, latencies: list[QueryLatency], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([lat.model_dump() for lat in latencies], f, indent=2)
        print(f"Saved {len(latencies)} latency measurements to {output_path}")

        partial_path = output_path.with_suffix(".partial.json")
        if partial_path.exists():
            partial_path.unlink()

    @staticmethod
    def load_pools(input_path: Path) -> list[QueryPool]:
        with open(input_path) as f:
            data = json.load(f)
        return [QueryPool(**item) for item in data]
