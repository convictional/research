import asyncio
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from typing import List
import uuid
from tqdm import tqdm
import faiss

from ..helpers.async_helper import limited_task, execute_tasks_with_manual_pbar
from ..helpers.embeddings import aembed
from ..helpers.instruct_llm import ainstruct_llm
from ..prompts.engine import build_prompt
from ..settings import settings, logger
from ..helpers.io import save_checkpoint, load_checkpoint

BUSINESS_DESCRIPTION = """
### Company Overview
- Name: Convictional
- Mission: Convictional is the infrastructure that powers decisions at the world's most ambitious companies. We solve the context problems that prevent fast, effective decision making.
- Stage: Pre-revenue technology company
- Team Size: 17 people

### Core Values
1. Focused Intensity
   - Execute a small number of things exceptionally well
   - Maintain shortest possible iteration time

2. Caring Deeply
   - Strong commitment to mission, customers, and team members
   - Foster culture of mutual support and dedication

3. Extreme Simplicity
   - Eliminate non-essential complexity
   - Constantly question if elements are truly necessary

4. Disciplined Growth
   - Building for long-term independence and profitability
   - Maintain strong financial discipline

5. Craft Excellence
   - Technical expertise prioritized in decision-making
   - Technical leads given precedence over equivalent-level managers
"""


@dataclass
class Fact:
    content: str
    source_id: uuid.UUID
    confidence: float
    extracted_at: datetime
    created_at: datetime

    def __getstate__(self):
        return {
            "content": self.content,
            "source_id": self.source_id,
            "confidence": self.confidence,
            "extracted_at": self.extracted_at,
            "created_at": self.created_at,
        }

    def __setstate__(self, state):
        self.content = state["content"]
        self.source_id = state["source_id"]
        self.confidence = state["confidence"]
        self.extracted_at = state["extracted_at"]
        self.created_at = state.get("created_at", datetime.now())  # Fallback if missing


@dataclass
class NamedEntity:
    name: str
    entity_type: str
    facts: List[Fact]
    canonical_id: uuid.UUID  # For deduplication grouping
    embedding: np.ndarray
    description: str = ""
    summary: str = ""


class EntityDescriptionResponse(BaseModel):
    analysis: str = Field(..., description="The analysis of the entity facts and relevance.")
    detailed_description: str = Field(..., description="A detailed description of the entity.")
    summary: str = Field(
        ...,
        description="Your final summary of the entity, should be a one paragraph summary given the detailed description and analysis.",
    )


class FactWithConfidence(BaseModel):
    statement: str = Field(
        ..., description="The extracted fact statement. It should be specific, clear and factual from the context."
    )
    confidence: float = Field(
        ..., description="Confidence in the truth of the fact statement. Should be between 0.0 and 1.0."
    )
    created_at: SkipJsonSchema[datetime] = Field(
        default=None
    )  # This will be set manually and excluded from JSON serialization


class FactsResponse(BaseModel):
    facts: List[FactWithConfidence] = Field(..., default_factory=list)


class Entity(BaseModel):
    name: str = Field(..., description="The entity's unique name. Always use a Person's full name (first and last).")
    type: str = Field(..., description="The entity's type from the provided category list.")


class EntityExtraction(BaseModel):
    analysis: str = Field(..., description="Use this field to think through the potential entities to be extracted.")
    entities: List[Entity] = Field(..., default_factory=list)


class EntityMatchResponse(BaseModel):
    analysis: str = Field(..., description="Use this field to think through the decision.")
    are_same_entity: bool = Field(..., description="Whether the two entities refer to the same real-world entity.")
    confidence: float = Field(..., description="Confidence in the decision of whether the entities are the same.")
    explanation: str = Field(..., description="Explanation of the decision.")


class EntityExtractor:
    def __init__(self):
        self.business_entity_types = [
            "COMPANY",
            "DEPARTMENT",
            "TEAM",
            "ROLE",
            "PERSON",
            "PRODUCT",
            "INITIATIVE/PROJECT",
            "PROCESS",
            "SYSTEM/TOOL",
            "COMPETITOR",
            "PARTNER",
            "VENDOR",
            "LOCATION",
            "MARKET",
            "INTERNAL POLICY",
            "EXTERNAL REGULATION",
        ]
        self.semaphore = asyncio.Semaphore(25)

    async def extract_entities_and_facts(
        self, text: str, source_id: uuid.UUID, created_at: datetime
    ) -> List[NamedEntity]:
        entities_raw = await self._extract_entities_llm(text)

        async def process_entity(entity_info: Entity):
            facts_with_confidence = await limited_task(
                self._extract_facts_for_entity(text, entity_info.name, created_at), self.semaphore, 0.1
            )

            entity_text = f"{entity_info.name} - {entity_info.type}"
            embedding = await limited_task(
                aembed(entity_text, settings.embedding_model, settings.embedding_dimension), self.semaphore, 0.1
            )

            return NamedEntity(
                name=entity_info.name,
                entity_type=entity_info.type,
                facts=[
                    Fact(
                        content=fact.statement,
                        source_id=source_id,
                        confidence=fact.confidence,
                        extracted_at=datetime.now(),
                        created_at=fact.created_at,
                    )
                    for fact in facts_with_confidence
                ],
                canonical_id=uuid.uuid4(),
                embedding=embedding,
            )

        entity_tasks = [process_entity(ent) for ent in entities_raw.entities]
        entities = await execute_tasks_with_manual_pbar(entity_tasks)

        return entities

    async def _extract_entities_llm(self, text: str) -> EntityExtraction:
        entity_types = ", ".join(self.business_entity_types)
        user_prompt = build_prompt(
            "named_entity_knowledge_store/extract_entities_user.txt.jinja",
            business_description=BUSINESS_DESCRIPTION,
            text=text,
            entity_types=entity_types,
        )

        response = await ainstruct_llm(
            system_prompt=None, user_prompt=user_prompt, response_model=EntityExtraction, temperature=0.1
        )
        if not response:
            return EntityExtraction(analysis="Failed to extract", entities=[])
        return response

    async def _extract_facts_for_entity(
        self, context: str, entity: str, created_at: datetime
    ) -> List[FactWithConfidence]:
        user_prompt = build_prompt(
            "named_entity_knowledge_store/fact_extraction_user.txt.jinja",
            business_description=BUSINESS_DESCRIPTION,
            entity=entity,
            context=context,
        )

        response = await ainstruct_llm(
            system_prompt=None, user_prompt=user_prompt, response_model=FactsResponse, temperature=0.1
        )

        if response and hasattr(response, "facts"):
            # Set created_at for each fact
            for fact in response.facts:
                fact.created_at = created_at
            return response.facts
        return []

    async def _get_entity_embedding(self, entity: NamedEntity) -> np.ndarray:
        # Get embedding for entity name and type combined
        text = f"{entity.name} - {entity.entity_type}"
        return await aembed(text, settings.embedding_model, settings.embedding_dimension)


class EntityStore:
    def __init__(self, embedding_threshold: float = 0.90, llm_threshold: float = 0.8):
        self.entities: List[NamedEntity] = []
        self.embedding_threshold = embedding_threshold  # Similarity threshold for direct matching
        self.llm_threshold = llm_threshold
        self.semaphore = asyncio.Semaphore(25)
        self.collected_entities: List[NamedEntity] = []  # Temporary storage before de-duplication
        self.index = faiss.IndexFlatIP(settings.embedding_dimension)  # Using inner product
        self._checkpoint_prefix = "entity_store"
        self.merge_stats = {"embedding_merges": 0, "llm_merges": 0, "total_comparisons": 0}

    async def deduplicate_collected_entities(self, use_checkpoint: bool = True):
        """Perform de-duplication using matrix operations and greedy clustering"""
        logger.info(f"Starting deduplication of {len(self.collected_entities)} entities")

        if not self.collected_entities:
            return

        checkpoint_stages = {
            "exact_match": "exact_match_entities",
            "embeddings": "entity_embeddings",
            "clusters": "entity_clusters",
            "final": "final_entities",
        }

        # Step 1: Exact Match Merging
        logger.info("Step 1: Exact match merging")
        if use_checkpoint:
            merged_entities = load_checkpoint(f"{self._checkpoint_prefix}_{checkpoint_stages['exact_match']}")
            if merged_entities:
                logger.info(f"Loaded {len(merged_entities)} exactly matched entities from checkpoint")
            else:
                merged_entities = self._perform_exact_matching()
                save_checkpoint(merged_entities, f"{self._checkpoint_prefix}_{checkpoint_stages['exact_match']}")
        else:
            merged_entities = self._perform_exact_matching()

        # Step 2: Compute embeddings all at once
        logger.info("Step 2: Computing embeddings")
        if use_checkpoint:
            embeddings = load_checkpoint(f"{self._checkpoint_prefix}_{checkpoint_stages['embeddings']}")
            if embeddings is None:
                embeddings = await self._compute_embeddings(merged_entities)
                save_checkpoint(embeddings, f"{self._checkpoint_prefix}_{checkpoint_stages['embeddings']}")
        else:
            embeddings = await self._compute_embeddings(merged_entities)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        # Step 3: Compute all pairwise similarities at once
        logger.info("Step 3: Computing similarity matrix")
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        # Reset merge stats
        self.merge_stats = {"embedding_merges": 0, "llm_merges": 0, "total_comparisons": 0}

        # Step 4: Greedy clustering
        n_entities = len(merged_entities)
        logger.info(f"Step 4: Performing greedy clustering on {n_entities} entities")
        clusters = []
        used = set()

        # Sort entity indices by number of high-similarity neighbors (most connected first)
        connection_counts = np.sum(similarity_matrix >= self.llm_threshold, axis=1)
        entity_order = np.argsort(-connection_counts)

        for idx in tqdm(entity_order, desc="Clustering"):
            if idx in used:
                continue

            # Find all entities similar to the current one
            similar_indices = np.where(similarity_matrix[idx] >= self.embedding_threshold)[0]

            # Log embedding-based matches
            for similar_idx in similar_indices:
                if similar_idx != idx and similar_idx not in used:
                    logger.info(
                        f"Embedding merge: '{merged_entities[idx].name}' -> '{merged_entities[similar_idx].name}' "
                        f"(score: {similarity_matrix[idx][similar_idx]:.3f})"
                    )
                    self.merge_stats["embedding_merges"] += 1

            # For entities between llm_threshold and embedding_threshold, verify with LLM
            llm_check_indices = np.where(
                (similarity_matrix[idx] >= self.llm_threshold)
                & (similarity_matrix[idx] < self.embedding_threshold)
                & (np.arange(len(similarity_matrix)) != idx)  # Exclude self-comparison
            )[0]

            if llm_check_indices.size > 0:
                self.merge_stats["total_comparisons"] += len(llm_check_indices)

            llm_tasks = [
                limited_task(self._are_same_entity_llm(merged_entities[idx], merged_entities[i]), self.semaphore, 0.1)
                for i in llm_check_indices
                if i not in used
            ]

            if llm_tasks:
                llm_results = await execute_tasks_with_manual_pbar(
                    llm_tasks, desc=f"LLM checking {len(llm_tasks)} entities"
                )
                for i, is_same in enumerate(llm_results):
                    if is_same:
                        logger.info(
                            f"LLM merge: '{merged_entities[idx].name}' -> "
                            f"'{merged_entities[llm_check_indices[i]].name}' "
                            f"(score: {similarity_matrix[idx][llm_check_indices[i]]:.3f})"
                        )
                        self.merge_stats["llm_merges"] += 1
                confirmed_indices = llm_check_indices[[i for i, is_same in enumerate(llm_results) if is_same]]
                similar_indices = np.union1d(similar_indices, confirmed_indices)

            # Create cluster
            cluster = [idx]
            used.add(idx)

            for similar_idx in similar_indices:
                if similar_idx not in used:
                    cluster.append(similar_idx)
                    used.add(similar_idx)

            if cluster:
                clusters.append(cluster)

        # Step 5: Merge entities within clusters
        logger.info("Step 5: Merging clustered entities")
        final_entities = []

        for cluster in clusters:
            base_entity = merged_entities[cluster[0]]
            all_facts = base_entity.facts.copy()
            for idx in cluster[1:]:
                other_entity = merged_entities[idx]
                all_facts.extend(other_entity.facts)
            # Sort facts after merging cluster
            base_entity.facts = self._sort_facts_by_created_at(all_facts)
            final_entities.append(base_entity)

        self.entities = final_entities
        if use_checkpoint:
            save_checkpoint(self.entities, f"{self._checkpoint_prefix}_{checkpoint_stages['final']}")

        logger.info(f"After clustering: {len(self.entities)} entities")

        # Step 6: Generate descriptions
        logger.info("Step 6: Generating descriptions")
        desc_tasks = [
            limited_task(self._generate_description(entity), self.semaphore, 0.1) for entity in self.entities
        ]
        descriptions = await execute_tasks_with_manual_pbar(desc_tasks, desc="Generating descriptions")

        for entity, desc in zip(self.entities, descriptions):
            if desc:  # Only update if we got a valid description
                entity.description = desc.detailed_description
                entity.summary = desc.summary
            else:
                logger.warning(f"No description generated for {entity.name}")
                entity.description = "No description available"
                entity.summary = "No summary available"

        # Check and log entities without facts
        entities_without_facts = [e for e in self.entities if not e.facts]
        if entities_without_facts:
            logger.warning(f"Found {len(entities_without_facts)} entities without facts:")
            for e in entities_without_facts:
                logger.warning(f"- {e.name} ({e.entity_type})")

        # Clear temporary storage
        self.collected_entities = []
        logger.info(f"Deduplication complete. Final count: {len(self.entities)} unique entities")

        # After clustering, log summary stats
        logger.info("\nMerge Statistics:")
        logger.info(f"- Embedding-based merges: {self.merge_stats['embedding_merges']}")
        logger.info(f"- LLM-verified merges: {self.merge_stats['llm_merges']}")
        logger.info(f"- Total LLM comparisons: {self.merge_stats['total_comparisons']}")
        logger.info(f"- Final unique entities: {len(final_entities)}")

    def _sort_facts_by_created_at(self, facts: List[Fact]) -> List[Fact]:
        """Sort facts by created_at timestamp in descending order (newest first)"""

        # Add safety check for facts without created_at
        def get_created_at(fact):
            if not hasattr(fact, "created_at") or fact.created_at is None:
                return datetime.now()
            return fact.created_at

        return sorted(facts, key=get_created_at, reverse=True)

    def _normalize_name(self, name: str) -> str:
        """Normalize entity names for better matching."""
        # Convert to lowercase
        normalized = name.strip().lower()

        # Replace multiple spaces with single space
        normalized = " ".join(normalized.split())

        # Special handling for person names
        if any(char.isupper() for char in name):  # Potential proper noun
            # Capitalize each word for consistency
            normalized = " ".join(word.capitalize() for word in normalized.split())

        return normalized

    def _perform_exact_matching(self) -> List[NamedEntity]:
        """Extract exact matching logic with improved normalization"""
        exact_match_dict = {}
        for entity in tqdm(self.collected_entities, desc="Exact matching"):
            # Normalize both name and type
            normalized_name = self._normalize_name(entity.name)
            normalized_type = entity.entity_type.strip().upper()

            key = (normalized_name, normalized_type)

            if key not in exact_match_dict:
                # Store with normalized name for consistency
                entity.name = normalized_name
                exact_match_dict[key] = entity
            else:
                # Combine and sort facts when merging
                all_facts = exact_match_dict[key].facts + entity.facts
                exact_match_dict[key].facts = self._sort_facts_by_created_at(all_facts)

        return list(exact_match_dict.values())

    async def _compute_embeddings(self, entities: List[NamedEntity]) -> np.ndarray:
        """Compute embeddings for all entities"""
        embedding_tasks = []
        for entity in entities:
            entity.name = self._normalize_name(entity.name)
            task = limited_task(self._get_entity_embedding(entity), self.semaphore, 0.1)
            embedding_tasks.append(task)

        embeddings = await execute_tasks_with_manual_pbar(embedding_tasks, desc="Computing embeddings")
        return np.vstack([emb for emb in embeddings if emb is not None])

    async def _get_entity_embedding(self, entity: NamedEntity) -> np.ndarray:
        text = f"{entity.name} - {entity.entity_type}"
        try:
            embedding = await aembed(text, settings.embedding_model, settings.embedding_dimension)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for entity {entity.name}: {str(e)}")
            return np.zeros(settings.embedding_dimension)

    def _format_facts_for_llm(self, facts: List[Fact]) -> List[str]:
        """Format facts for LLM consumption with just essential info"""
        return [
            f"[{fact.created_at.strftime('%Y-%m-%d')}] {fact.content} (conf: {fact.confidence:.2f})" for fact in facts
        ]

    async def _generate_description(self, entity: NamedEntity) -> str:
        if not entity.facts:
            logger.warning(f"No facts found for entity {entity.name}, skipping description generation")
            return ""

        # Sort facts before truncating
        sorted_facts = self._sort_facts_by_created_at(entity.facts)
        formatted_facts = self._format_facts_for_llm(sorted_facts[0:400])

        if not formatted_facts:
            logger.warning(f"No formatted facts for entity {entity.name}, skipping description generation")
            return ""

        user_prompt = build_prompt(
            "named_entity_knowledge_store/generate_description_user.txt.jinja",
            business_description=BUSINESS_DESCRIPTION,
            named_entity=entity.name,
            named_entity_type=entity.entity_type,
            facts=formatted_facts,
        )

        try:
            response = await ainstruct_llm(
                system_prompt=None, user_prompt=user_prompt, response_model=EntityDescriptionResponse, temperature=0.3
            )

            return response

        except Exception as e:
            logger.error(f"Error generating description for {entity.name}: {str(e)}")
            return EntityDescriptionResponse(
                analysis="Error generating description",
                detailed_description="Error getting description",
                summary="Error getting summary",
            )

    async def _are_same_entity_llm(self, entity1: NamedEntity, entity2: NamedEntity) -> bool:
        # Sort facts for both entities before truncating
        sorted_facts1 = self._sort_facts_by_created_at(entity1.facts)
        sorted_facts2 = self._sort_facts_by_created_at(entity2.facts)

        formatted_facts1 = self._format_facts_for_llm(sorted_facts1[0:200])
        formatted_facts2 = self._format_facts_for_llm(sorted_facts2[0:200])

        facts1 = "\n".join([f"- {fact}" for fact in formatted_facts1])
        facts2 = "\n".join([f"- {fact}" for fact in formatted_facts2])

        user_prompt = build_prompt(
            "named_entity_knowledge_store/are_same_entity_user.txt.jinja",
            business_description=BUSINESS_DESCRIPTION,
            entity1=entity1,
            entity2=entity2,
            facts1=facts1,
            facts2=facts2,
        )

        response = await ainstruct_llm(
            system_prompt=None,
            user_prompt=user_prompt,
            response_model=EntityMatchResponse,
            temperature=0.1,  # Lower temperature for more consistent matching
        )

        return response.are_same_entity if response and response.confidence > 0.8 else False

    def export_to_csv(self, output_path: Path):
        import pandas as pd

        # Convert entities to rows
        rows = []
        for entity in self.entities:
            rows.append(
                {
                    "entity_name": entity.name,
                    "category": entity.entity_type,
                    "description": entity.description,
                    "summary": entity.summary,
                    "facts": entity.facts[0:100],
                }
            )

        # Create and save DataFrame
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(rows)} entities to {output_path}")
