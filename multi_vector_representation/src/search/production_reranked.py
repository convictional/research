import asyncio
import httpx
from uuid import UUID

from src.data.extractor import ContentExtractor
from src.models.content import SearchQuery, SearchResult
from src.search.production_hybrid import ProductionHybridSearch


class ProductionRerankedSearch:
    def __init__(self, production_search: ProductionHybridSearch, extractor: ContentExtractor, jina_api_key: str):
        self.production_search = production_search
        self.extractor = extractor
        self.jina_api_key = jina_api_key
        self.api_url = "https://api.jina.ai/v1/rerank"
        self.verbose = False

    async def _call_jina_api_with_retry(
        self, payload: dict, headers: dict, max_retries: int = 3, verbose: bool = False
    ) -> dict:
        for attempt in range(max_retries):
            try:
                if verbose:
                    print(f"  → Calling Jina API: {self.api_url}")
                    print(f"    Model: {payload['model']}")
                    print(f"    Documents: {len(payload['documents'])}")
                    print(f"    Top-N: {payload['top_n']}")

                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(self.api_url, json=payload, headers=headers)

                    if response.status_code != 200:
                        print(f"Jina API Error {response.status_code}: {response.text}")
                        response.raise_for_status()

                    result = response.json()
                    if verbose:
                        print(f"  ✓ Jina API returned {len(result.get('results', []))} results")

                    return result

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                wait_time = 2 ** attempt
                if attempt < max_retries - 1:
                    print(f"Jina API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Jina API failed after {max_retries} attempts: {e}")
                    raise

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        stage1_query = SearchQuery(
            text=query.text,
            top_k=50,
            content_types=query.content_types,
            organization_id=query.organization_id,
        )

        stage1_results = await self.production_search.search(stage1_query)

        if not stage1_results:
            return []

        content_map = {}
        for result in stage1_results:
            content_record = await self.extractor.get_content_by_id(result.content_id)
            if content_record:
                content_map[result.content_id] = content_record

        documents = []
        for result in stage1_results:
            record = content_map.get(result.content_id)
            if record:
                doc_text = f"{record.title}\n\n{record.index_content}"
                documents.append(doc_text)

        headers = {"Authorization": f"Bearer {self.jina_api_key}", "Content-Type": "application/json"}

        payload = {
            "model": "jina-colbert-v2",
            "query": query.text,
            "documents": documents,
            "top_n": query.top_k,
            "return_documents": False,
        }

        data = await self._call_jina_api_with_retry(payload, headers, verbose=self.verbose)

        reranked_results = []
        for rank, result_item in enumerate(data.get("results", []), 1):
            original_index = result_item["index"]
            score = result_item.get("relevance_score", 0.0)

            original_result = stage1_results[original_index]
            record = content_map.get(original_result.content_id)

            reranked_results.append(
                SearchResult(
                    content_id=original_result.content_id,
                    score=score,
                    rank=rank,
                    title=original_result.title,
                    preview=record.index_content[:500] if record else original_result.preview,
                    content_type=original_result.content_type,
                )
            )

        return reranked_results
