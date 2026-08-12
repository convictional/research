import json
from pathlib import Path

import instructor
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from src.evaluation.query_generator import GeneratedQuery


class QueryClassification(BaseModel):
    query_id: str
    query_text: str
    classification: str = Field(description="One of: factoid, navigational, exploratory")
    rationale: str = Field(description="Brief explanation for this classification")
    source_doc_count: int


class ClassificationBatch(BaseModel):
    classifications: list[QueryClassification]


class QueryClassifier:
    def __init__(self, anthropic_api_key: str):
        anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
        self.client = instructor.from_anthropic(anthropic_client, mode=instructor.Mode.ANTHROPIC_TOOLS)

    async def classify_queries(self, queries: list[GeneratedQuery], batch_size: int = 25) -> list[QueryClassification]:
        all_classifications = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            print(f"Classifying batch {i // batch_size + 1}/{(len(queries) + batch_size - 1) // batch_size}")

            batch_result = await self._classify_batch(batch)
            all_classifications.extend(batch_result.classifications)

        return all_classifications

    async def _classify_batch(self, queries: list[GeneratedQuery]) -> ClassificationBatch:
        query_list = "\n".join([f"{i + 1}. [{q.id}] {q.text}" for i, q in enumerate(queries)])

        prompt = f"""Classify each of the following queries into one of three categories:

**Factoid**: Query seeks a specific fact, answer, or piece of information. The user expects a precise answer.
Examples: "What did we name the predictions feature?", "Why did we switch from Superhuman to Gmail?"

**Navigational**: Query aims to find a specific document, decision, or resource. The user knows what they're looking for.
Examples: "Decisions about Avalara integration", "Find the document regarding customer onboarding"

**Exploratory**: Query explores a broad topic, seeks multiple perspectives or general guidance. The user is investigating an area.
Examples: "How do we handle customer integrations?", "Best practices for cross-functional collaboration"

Queries to classify:
{query_list}

For each query, provide:
1. The query_id (e.g., "q1_1")
2. The classification (factoid, navigational, or exploratory)
3. A brief rationale (1-2 sentences)

Consider:
- Question words: "What/Why/When" often indicate factoid, "How/Tell me about" suggest exploratory
- Specificity: Specific entities/names suggest factoid or navigational, general concepts suggest exploratory
- Intent: Does the user want a single answer, a specific document, or broad understanding?"""

        response = await self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            temperature=1.0,
            messages=[{"role": "user", "content": prompt}],
            response_model=ClassificationBatch,
        )

        classification_map = {c.query_id: c for c in response.classifications}
        for query in queries:
            if query.id in classification_map:
                classification_map[query.id].query_text = query.text
                classification_map[query.id].source_doc_count = len(query.source_doc_ids)

        return response

    async def save_classifications(self, classifications: list[QueryClassification], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([c.model_dump() for c in classifications], f, indent=2)
        print(f"Saved {len(classifications)} classifications to {output_path}")

    @staticmethod
    def load_classifications(input_path: Path) -> list[QueryClassification]:
        with open(input_path) as f:
            data = json.load(f)
        return [QueryClassification(**item) for item in data]
