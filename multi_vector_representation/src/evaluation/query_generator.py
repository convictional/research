import json
import random
from pathlib import Path
from uuid import UUID

import instructor
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from src.data.extractor import ContentExtractor
from src.models.content import ContentRecord


class QueryWithRelevance(BaseModel):
    query: str
    relevant_indices: list[int]


class BatchQueryResponse(BaseModel):
    queries: list[QueryWithRelevance]


class GeneratedQuery(BaseModel):
    id: str
    text: str
    source_doc_ids: list[str]
    batch_id: int


class QueryGenerator:
    def __init__(self, extractor: ContentExtractor, anthropic_api_key: str):
        self.extractor = extractor
        anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
        self.client = instructor.from_anthropic(
            anthropic_client, mode=instructor.Mode.ANTHROPIC_REASONING_TOOLS
        )

    async def generate_queries(self, batch_size: int = 50, queries_per_batch: int = 25) -> list[GeneratedQuery]:
        content_ids = await self.extractor.get_content_ids_with_token_embeddings()

        shuffled_ids = random.sample(content_ids, len(content_ids))

        batch_ids = [shuffled_ids[i : i + batch_size] for i in range(0, len(shuffled_ids), batch_size)]

        all_queries = []
        for batch_idx, ids in enumerate(batch_ids):
            print(f"Generating queries for batch {batch_idx + 1}/{len(batch_ids)}...")
            batch_content = await self.extractor.get_content_by_ids(ids)
            batch_queries = await self._generate_for_batch(batch_content, queries_per_batch, batch_idx)
            all_queries.extend(batch_queries)

        return all_queries

    async def _generate_for_batch(self, docs: list[ContentRecord], count: int, batch_id: int) -> list[GeneratedQuery]:
        prompt = self._build_prompt(docs, count)

        response = await self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 5000},
            temperature=1.0,
            messages=[{"role": "user", "content": prompt}],
            response_model=BatchQueryResponse,
        )

        queries = []
        for idx, query_item in enumerate(response.queries):
            if not query_item.relevant_indices:
                continue

            relevant_doc_ids = [
                str(docs[i].id) for i in query_item.relevant_indices if i < len(docs)
            ]

            query_id = f"q{batch_id + 1}_{idx + 1}"
            queries.append(
                GeneratedQuery(
                    id=query_id,
                    text=query_item.query,
                    source_doc_ids=relevant_doc_ids,
                    batch_id=batch_id,
                )
            )

        return queries

    def _build_prompt(self, docs: list[ContentRecord], count: int) -> str:
        doc_descriptions = []
        for idx, doc in enumerate(docs):
            preview = doc.index_content[:300] if len(doc.index_content) > 300 else doc.index_content
            doc_descriptions.append(
                f'[Doc {idx}] Type: {doc.content_type}, Title: "{doc.title}", Content: {preview}...'
            )

        docs_text = "\n\n".join(doc_descriptions)

        return f"""You have {len(docs)} documents from a decision-making tool. Generate {count} diverse search queries where 1 or more documents would be relevant answers.

{docs_text}

Generate {count} diverse queries that users might realistically search for. Include:
- Single-doc queries (specific lookups)
- Multi-doc queries (comparative or aggregation questions)
- Broad exploratory queries
- Narrow targeted queries

For each query, indicate which documents (indices 0-{len(docs) - 1}) are relevant."""

    async def save_queries(self, queries: list[GeneratedQuery], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([q.model_dump() for q in queries], f, indent=2)
        print(f"Saved {len(queries)} queries to {output_path}")

    @staticmethod
    def load_queries(input_path: Path) -> list[GeneratedQuery]:
        with open(input_path) as f:
            data = json.load(f)
        return [GeneratedQuery(**item) for item in data]
