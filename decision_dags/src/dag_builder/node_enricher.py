"""Node enrichment system for people impact and resource requirements using LLM assistance."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from common.instruct_llm import ainstruct_llm
from common.embeddings import aembed_query
from common.prompt_template_engine import build_prompt
from openai import AsyncOpenAI

from ..models import DecisionDAG, DecisionNode
from ..schemas import NodeEnrichmentSchema
from ..settings import settings
from .context import DAGBuildingContext

logger = logging.getLogger(__name__)


@dataclass
class NodeEnrichmentMetrics:
    """Metrics tracking for node enrichment."""

    total_nodes: int = 0
    enriched_nodes: int = 0
    failed_enrichments: int = 0
    skipped_nodes: int = 0
    people_impacted_added: int = 0
    resource_requirements_added: int = 0
    embeddings_added: int = 0
    processing_time_seconds: float = 0.0
    llm_calls_made: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate for enrichment."""
        return self.enriched_nodes / max(1, self.total_nodes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "total_nodes": self.total_nodes,
            "enriched_nodes": self.enriched_nodes,
            "failed_enrichments": self.failed_enrichments,
            "skipped_nodes": self.skipped_nodes,
            "people_impacted_added": self.people_impacted_added,
            "resource_requirements_added": self.resource_requirements_added,
            "embeddings_added": self.embeddings_added,
            "processing_time_seconds": self.processing_time_seconds,
            "llm_calls_made": self.llm_calls_made,
            "success_rate": self.success_rate,
        }


class NodeEnricher:
    """
    Node enrichment system that adds people impact and resource requirements to DAG nodes.

    Uses LLM assistance combined with organizational context to determine:
    - People and roles impacted by decisions/options
    - Resource requirements for implementation
    """

    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        """
        Initialize node enricher.

        Args:
            batch_size: Number of nodes to process in each batch
            max_concurrent: Maximum concurrent LLM calls
        """
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.metrics = NodeEnrichmentMetrics()
        self._openai_client = None

    async def enrich_nodes(
        self, dag: DecisionDAG, context: DAGBuildingContext
    ) -> Tuple[DecisionDAG, NodeEnrichmentMetrics]:
        """
        Enrich all nodes in the DAG with people impact and resource requirements.

        Args:
            dag: The DAG to enrich
            context: Building context with organizational information

        Returns:
            Tuple of enriched DAG and metrics
        """
        import time

        start_time = time.time()

        logger.info(f"Starting node enrichment for {len(dag.all_nodes)} nodes")

        # Reset metrics
        self.metrics = NodeEnrichmentMetrics()
        self.metrics.total_nodes = len(dag.all_nodes)

        # Process nodes in batches for efficiency
        nodes_list = list(dag.all_nodes.values())
        node_batches = self._create_node_batches(nodes_list)

        for batch_num, batch in enumerate(node_batches):
            logger.info(f"Processing node batch {batch_num + 1}/{len(node_batches)} ({len(batch)} nodes)")

            await self._process_node_batch(batch, dag, context)

            # Brief pause between batches to avoid rate limits
            if batch_num < len(node_batches) - 1:
                await asyncio.sleep(0.5)

        # Calculate final metrics
        self.metrics.processing_time_seconds = time.time() - start_time

        logger.info(f"Node enrichment complete. Results: {self.metrics.to_dict()}")

        return dag, self.metrics

    def _create_node_batches(self, nodes: List[DecisionNode]) -> List[List[DecisionNode]]:
        """Create batches of nodes for processing."""
        batches = []
        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i : i + self.batch_size]
            batches.append(batch)
        return batches

    async def _process_node_batch(
        self, nodes: List[DecisionNode], dag: DecisionDAG, context: DAGBuildingContext
    ) -> None:
        """Process a batch of nodes concurrently."""
        tasks = []

        for node in nodes:
            task = self._enrich_single_node(node, dag, context)
            tasks.append(task)

        # Execute batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and update metrics
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Node enrichment task failed for node {nodes[i].id}: {result}")
                self.metrics.failed_enrichments += 1
            elif result:  # Successfully enriched
                self.metrics.enriched_nodes += 1
                if nodes[i].people_impacted:
                    self.metrics.people_impacted_added += 1
                if nodes[i].resource_requirements:
                    self.metrics.resource_requirements_added += 1
                if nodes[i].embedding:
                    self.metrics.embeddings_added += 1

    async def _enrich_single_node(self, node: DecisionNode, dag: DecisionDAG, context: DAGBuildingContext) -> bool:
        """Enrich a single node with LLM assistance."""
        async with self.semaphore:
            try:
                # Check what needs to be enriched
                needs_people_resource = not (node.people_impacted and node.resource_requirements)
                needs_embedding = node.embedding is None

                # Skip if nothing needs enrichment
                if not needs_people_resource and not needs_embedding:
                    self.metrics.skipped_nodes += 1
                    return True

                # Generate embedding if needed
                if needs_embedding:
                    await self._generate_embedding(node)

                # Generate people/resource enrichment if needed
                if needs_people_resource:
                    enrichment_data = await self._generate_node_enrichment(node, dag, context)

                    # Update node with enrichment data
                    node.people_impacted = enrichment_data["people_impacted"]
                    node.resource_requirements = enrichment_data["resource_requirements"]

                    # Add enrichment metadata
                    node.metadata["enrichment_reasoning"] = enrichment_data["reasoning"]
                    node.metadata["enriched"] = True

                    self.metrics.llm_calls_made += 1

                return True

            except Exception as e:
                logger.error(f"Failed to enrich node {node.id}: {e}")
                return False

    async def _generate_node_enrichment(
        self, node: DecisionNode, dag: DecisionDAG, context: DAGBuildingContext
    ) -> Dict[str, Any]:
        """Generate enrichment data for a node using LLM."""
        # Get parent and child nodes for context
        parents = dag.get_parents(node)
        children = dag.get_children(node)

        # Build prompt data
        prompt_data = {
            "problem_statement": context.problem_statement,
            "node_type": node.type.value,
            "node_title": node.title,
            "node_description": node.description,
            "node_layer": node.layer,
            "decision_type": node.decision_type.value if node.decision_type else "N/A",
            "goal_impacts": node.goal_impacts,
            "parent_nodes": [{"title": p.title, "type": p.type.value} for p in parents],
            "child_nodes": [{"title": c.title, "type": c.type.value} for c in children],
            "organizational_goals": context.organizational_goals,
            "organizational_context": context.relevant_content,
        }

        # Build prompts
        system_prompt = build_prompt("node_enrichment_system.txt.jinja", **prompt_data)
        user_prompt = build_prompt("node_enrichment_user.txt.jinja", **prompt_data)

        # Call LLM with structured output
        response = await ainstruct_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=NodeEnrichmentSchema,
            llm_model=settings.llm_model,
            temperature=0.3,  # Lower temperature for consistency
            max_tokens=2048,
        )

        # Convert response to enrichment data
        enrichment_data = {
            "people_impacted": response.people_impacted,
            "resource_requirements": response.resource_requirements,
            "reasoning": response.reasoning,
        }

        logger.debug(f"Generated enrichment for node {node.id}: {node.title}")
        return enrichment_data

    async def _generate_embedding(self, node: DecisionNode) -> None:
        """Generate embedding for a node."""
        try:
            text_content = f"{node.title}. {node.description}"
            embedding = await aembed_query(
                async_openai_client=self._get_openai_client(),
                text=text_content,
                embedding_model=settings.embedding_model,
                embedding_dim=1536,
            )
            node.embedding = embedding
            logger.debug(f"Generated embedding for node {node.id}")
        except Exception as e:
            logger.warning(f"Failed to generate embedding for node {node.id}: {e}")

    def _get_openai_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI()
        return self._openai_client
