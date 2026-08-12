"""Edge enrichment system for cost/timeline/risk estimation using LLM assistance and database context."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt

from ..models import DecisionDAG, DecisionNode, DecisionEdge
from .context import DAGBuildingContext

logger = logging.getLogger(__name__)


@dataclass
class EdgeEnrichmentMetrics:
    """Comprehensive metrics tracking for edge enrichment."""

    total_edges: int = 0
    enriched_edges: int = 0
    failed_enrichments: int = 0
    skipped_edges: int = 0
    cost_estimates_added: int = 0
    timeline_estimates_added: int = 0
    risk_assessments_added: int = 0
    processing_time_seconds: float = 0.0
    llm_calls_made: int = 0
    average_enrichment_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate for enrichment."""
        return self.enriched_edges / max(1, self.total_edges)

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate for enrichment."""
        return self.failed_enrichments / max(1, self.total_edges)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "total_edges": self.total_edges,
            "enriched_edges": self.enriched_edges,
            "failed_enrichments": self.failed_enrichments,
            "skipped_edges": self.skipped_edges,
            "cost_estimates_added": self.cost_estimates_added,
            "timeline_estimates_added": self.timeline_estimates_added,
            "risk_assessments_added": self.risk_assessments_added,
            "processing_time_seconds": self.processing_time_seconds,
            "llm_calls_made": self.llm_calls_made,
            "average_enrichment_time": self.average_enrichment_time,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
        }


@dataclass
class EdgeEnrichmentResult:
    """Result of enriching a single edge."""

    edge_id: str
    source_id: str
    target_id: str
    success: bool
    cost_estimate: Optional[str] = None
    estimated_cost_dollars: Optional[float] = None
    timeline_estimate: Optional[str] = None
    implementation_risks: Optional[List[str]] = None
    conditions: Optional[List[str]] = None
    likelihood: Optional[str] = None
    reasoning: Optional[str] = None
    error_message: Optional[str] = None


class EdgeEnricher:
    """
    Edge enrichment system that adds detailed implementation information to DAG edges.

    Uses LLM assistance combined with organizational context to estimate:
    - Cost and resource requirements
    - Timeline estimates
    - Implementation risks
    - Success conditions
    - Likelihood assessments
    """

    def __init__(self, batch_size: int = 5, max_concurrent: int = 3):
        """
        Initialize edge enricher.

        Args:
            batch_size: Number of edges to process in each batch
            max_concurrent: Maximum concurrent LLM calls
        """
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.metrics = EdgeEnrichmentMetrics()

    async def enrich_edges(
        self, dag: DecisionDAG, context: DAGBuildingContext
    ) -> Tuple[DecisionDAG, EdgeEnrichmentMetrics]:
        """
        Enrich all edges in the DAG with detailed implementation information.

        Args:
            dag: The DAG to enrich
            context: Building context with organizational information

        Returns:
            Tuple of enriched DAG and metrics
        """
        import time

        start_time = time.time()

        logger.info(f"Starting edge enrichment for {len(dag.edges)} edges")

        # Reset metrics
        self.metrics = EdgeEnrichmentMetrics()
        self.metrics.total_edges = len(dag.edges)

        # Process edges in batches for efficiency
        edge_batches = self._create_edge_batches(dag.edges)
        logger.info(f"Processing {len(edge_batches)} edge batches concurrently")

        # Process all batches concurrently
        batch_tasks = []
        for batch_num, batch in enumerate(edge_batches):
            task = self._process_edge_batch(batch, dag, context)
            batch_tasks.append(task)

        # Wait for all batches to complete
        all_batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Collect results
        enrichment_results = []
        for batch_num, batch_results in enumerate(all_batch_results):
            if isinstance(batch_results, Exception):
                logger.error(f"Batch {batch_num + 1} failed: {batch_results}")
                # Continue with other batches
            else:
                enrichment_results.extend(batch_results)

        # Apply enrichment results to DAG
        self._apply_enrichment_results(dag, enrichment_results)

        # Calculate final metrics
        self.metrics.processing_time_seconds = time.time() - start_time
        self.metrics.average_enrichment_time = self.metrics.processing_time_seconds / max(
            1, self.metrics.enriched_edges
        )

        logger.info(f"Edge enrichment complete. Results: {self.metrics.to_dict()}")

        return dag, self.metrics

    def _create_edge_batches(self, edges: List[DecisionEdge]) -> List[List[DecisionEdge]]:
        """Create batches of edges for processing."""
        batches = []
        for i in range(0, len(edges), self.batch_size):
            batch = edges[i : i + self.batch_size]
            batches.append(batch)
        return batches

    async def _process_edge_batch(
        self, edges: List[DecisionEdge], dag: DecisionDAG, context: DAGBuildingContext
    ) -> List[EdgeEnrichmentResult]:
        """Process a batch of edges concurrently."""
        tasks = []

        for edge in edges:
            task = self._enrich_single_edge(edge, dag, context)
            tasks.append(task)

        # Execute batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and update metrics
        enrichment_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Edge enrichment task failed: {result}")
                self.metrics.failed_enrichments += 1
            else:
                enrichment_results.append(result)
                if result.success:
                    self.metrics.enriched_edges += 1
                    if result.cost_estimate:
                        self.metrics.cost_estimates_added += 1
                    if result.timeline_estimate:
                        self.metrics.timeline_estimates_added += 1
                    if result.implementation_risks:
                        self.metrics.risk_assessments_added += 1
                else:
                    self.metrics.failed_enrichments += 1

        return enrichment_results

    async def _enrich_single_edge(
        self, edge: DecisionEdge, dag: DecisionDAG, context: DAGBuildingContext
    ) -> EdgeEnrichmentResult:
        """Enrich a single edge with LLM assistance."""
        async with self.semaphore:
            try:
                # Get source and target nodes
                source_node = dag.get_node(edge.source_id)
                target_node = dag.get_node(edge.target_id)

                if not source_node or not target_node:
                    logger.warning(f"Missing nodes for edge {edge.source_id} -> {edge.target_id}")
                    return EdgeEnrichmentResult(
                        edge_id=f"{edge.source_id}->{edge.target_id}",
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        success=False,
                        error_message="Missing source or target node",
                    )

                # Skip enrichment if edge already has detailed information
                if self._is_edge_already_enriched(edge):
                    self.metrics.skipped_edges += 1
                    return EdgeEnrichmentResult(
                        edge_id=f"{edge.source_id}->{edge.target_id}",
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        success=True,
                        cost_estimate=edge.cost_estimate,
                        estimated_cost_dollars=edge.estimated_cost_dollars,
                        timeline_estimate=edge.timeline_estimate,
                        implementation_risks=edge.implementation_risks,
                    )

                # Generate enrichment using LLM
                enrichment_data = await self._generate_edge_enrichment(edge, source_node, target_node, context)

                self.metrics.llm_calls_made += 1

                return EdgeEnrichmentResult(
                    edge_id=f"{edge.source_id}->{edge.target_id}",
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    success=True,
                    **enrichment_data,
                )

            except Exception as e:
                logger.error(f"Failed to enrich edge {edge.source_id} -> {edge.target_id}: {e}")
                return EdgeEnrichmentResult(
                    edge_id=f"{edge.source_id}->{edge.target_id}",
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    success=False,
                    error_message=str(e),
                )

    def _is_edge_already_enriched(self, edge: DecisionEdge) -> bool:
        """Check if edge already has comprehensive enrichment data."""
        return (
            edge.cost_estimate is not None
            and edge.timeline_estimate is not None
            and edge.implementation_risks is not None
            and len(edge.implementation_risks) > 0
        )

    async def _generate_edge_enrichment(
        self, edge: DecisionEdge, source_node: DecisionNode, target_node: DecisionNode, context: DAGBuildingContext
    ) -> Dict[str, Any]:
        """Generate comprehensive enrichment data for an edge using LLM."""
        try:
            # Build prompt with comprehensive context
            prompt_data = self._build_enrichment_prompt_data(edge, source_node, target_node, context)

            # Build prompts for edge enrichment
            system_prompt = build_prompt("edge_enrichment_system.txt.jinja", **prompt_data)
            user_prompt = build_prompt("edge_enrichment_user.txt.jinja", **prompt_data)

            # Call LLM with structured output
            from ..schemas import EdgeEnrichmentSchema
            from ..settings import settings

            response = await ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=EdgeEnrichmentSchema,
                llm_model=settings.llm_model,
                temperature=0.3,  # Lower temperature for more consistent estimates
                max_tokens=2048,
            )

            # Convert response to enrichment data
            enrichment_data = {
                "cost_estimate": response.cost_estimate,
                "estimated_cost_dollars": response.estimated_cost_dollars,
                "timeline_estimate": response.timeline_estimate,
                "implementation_risks": response.implementation_risks,
                "conditions": response.success_conditions,
                "likelihood": response.likelihood,
                "reasoning": response.reasoning,
            }

            logger.debug(f"Generated enrichment for edge {edge.source_id} -> {edge.target_id}")
            return enrichment_data

        except Exception as e:
            logger.error(f"LLM enrichment failed for edge {edge.source_id} -> {edge.target_id}: {e}")
            # Return fallback enrichment
            return self._generate_fallback_enrichment(edge, source_node, target_node)

    def _build_enrichment_prompt_data(
        self, edge: DecisionEdge, source_node: DecisionNode, target_node: DecisionNode, context: DAGBuildingContext
    ) -> Dict[str, Any]:
        """Build comprehensive prompt data for edge enrichment."""
        return {
            "problem_statement": context.problem_statement,
            "source_node": {
                "title": source_node.title,
                "description": source_node.description,
                "type": source_node.type.value,
                "layer": source_node.layer,
                "tags": source_node.tags,
            },
            "target_node": {
                "title": target_node.title,
                "description": target_node.description,
                "type": target_node.type.value,
                "layer": target_node.layer,
                "tags": target_node.tags,
            },
            "edge": {
                "condition": edge.condition,
                "edge_type": edge.edge_type.value,
                "likelihood": edge.likelihood,
                "label": edge.label,
                "existing_cost_estimate": edge.cost_estimate,
                "existing_timeline_estimate": edge.timeline_estimate,
                "existing_risks": edge.implementation_risks or [],
            },
            "organizational_goals": getattr(context, "organizational_goals", []),
            "past_decisions": getattr(context, "past_decisions", []),
            "relevant_content": getattr(context, "relevant_content", []),
            "activity_insights": getattr(context, "activity_insights", {}),
            "strategic_paths": context.strategic_paths,
        }

    def _generate_fallback_enrichment(
        self, edge: DecisionEdge, source_node: DecisionNode, target_node: DecisionNode
    ) -> Dict[str, Any]:
        """Generate basic fallback enrichment when LLM fails."""
        # Simple heuristic-based enrichment
        layer_diff = target_node.layer - source_node.layer

        # Basic cost estimation based on complexity
        if "implementation" in target_node.description.lower() or "build" in target_node.description.lower():
            cost_estimate = "medium-high"
            estimated_cost_dollars = 50000.0
        elif "analysis" in target_node.description.lower() or "research" in target_node.description.lower():
            cost_estimate = "low-medium"
            estimated_cost_dollars = 15000.0
        else:
            cost_estimate = "medium"
            estimated_cost_dollars = 25000.0

        # Basic timeline estimation
        timeline_estimate = f"{layer_diff + 2}-{layer_diff + 4} weeks"

        # Basic risk assessment
        implementation_risks = [
            "Resource availability constraints",
            "Stakeholder alignment challenges",
            "Technical complexity risks",
        ]

        return {
            "cost_estimate": cost_estimate,
            "estimated_cost_dollars": estimated_cost_dollars,
            "timeline_estimate": timeline_estimate,
            "implementation_risks": implementation_risks,
            "conditions": ["Successful completion of previous step"],
            "likelihood": edge.likelihood or "medium",
            "reasoning": "Fallback heuristic-based estimation",
        }

    def _apply_enrichment_results(self, dag: DecisionDAG, enrichment_results: List[EdgeEnrichmentResult]) -> None:
        """Apply enrichment results to DAG edges."""
        # Create lookup for enrichment results
        enrichment_lookup = {
            (result.source_id, result.target_id): result for result in enrichment_results if result.success
        }

        # Update edges with enrichment data
        for edge in dag.edges:
            result = enrichment_lookup.get((edge.source_id, edge.target_id))
            if result:
                # Update edge with enrichment data
                if result.cost_estimate:
                    edge.cost_estimate = result.cost_estimate
                if result.estimated_cost_dollars:
                    edge.estimated_cost_dollars = result.estimated_cost_dollars
                if result.timeline_estimate:
                    edge.timeline_estimate = result.timeline_estimate
                if result.implementation_risks:
                    edge.implementation_risks = result.implementation_risks
                if result.conditions:
                    edge.conditions = result.conditions
                if result.likelihood:
                    edge.likelihood = result.likelihood

                # Add enrichment metadata
                edge.metadata.update(
                    {
                        "enriched": True,
                        "enrichment_method": "llm_assisted",
                        "enrichment_reasoning": result.reasoning,
                        "enrichment_id": str(uuid4()),
                    }
                )

                logger.debug(f"Applied enrichment to edge {edge.source_id} -> {edge.target_id}")
