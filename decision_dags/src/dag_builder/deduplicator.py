"""Node deduplication for maintaining DAG coherence."""

import asyncio
import logging
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass

from openai import AsyncOpenAI

from common.embeddings import aembed_query, cosine_similarity
from common.instruct_llm import ainstruct_llm

from ..models import DecisionNode
from ..settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationMetrics:
    """Metrics for deduplication operation."""

    original_count: int
    deduplicated_count: int
    clusters_created: int
    automatic_merges: int
    llm_assisted_merges: int
    similarity_scores: List[float]
    processing_time_seconds: float

    @property
    def reduction_percentage(self) -> float:
        """Calculate reduction percentage."""
        if self.original_count == 0:
            return 0.0
        return ((self.original_count - self.deduplicated_count) / self.original_count) * 100

    @property
    def average_similarity(self) -> float:
        """Calculate average similarity score."""
        return sum(self.similarity_scores) / len(self.similarity_scores) if self.similarity_scores else 0.0


class SimilarityPair(NamedTuple):
    """Represents similarity between two nodes."""

    node1_id: str
    node2_id: str
    similarity: float
    merge_decision: str  # "merged", "kept_separate", "weak_match"


class NodeDeduplicator:
    """Identifies and consolidates similar nodes to maintain DAG coherence."""

    def __init__(self, similarity_threshold: float | None = None):
        self.similarity_threshold = similarity_threshold or settings.similarity_threshold
        self.weak_similarity_threshold = settings.weak_similarity_threshold
        self._openai_client: AsyncOpenAI | None = None

        # Metrics tracking
        self.last_metrics: DeduplicationMetrics | None = None
        self.similarity_pairs: List[SimilarityPair] = []

    def _get_openai_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client for embeddings."""
        if not self._openai_client:
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        return self._openai_client

    async def deduplicate_layer(self, nodes: List[DecisionNode]) -> List[DecisionNode]:
        """
        Deduplicate nodes in a layer with comprehensive metrics tracking.

        Args:
            nodes: List of nodes to deduplicate

        Returns:
            Deduplicated list of nodes
        """
        import time

        start_time = time.time()

        # Reset metrics for this operation
        self.similarity_pairs = []
        automatic_merges = 0
        llm_assisted_merges = 0

        if len(nodes) <= 1:
            self.last_metrics = DeduplicationMetrics(
                original_count=len(nodes),
                deduplicated_count=len(nodes),
                clusters_created=len(nodes),
                automatic_merges=0,
                llm_assisted_merges=0,
                similarity_scores=[],
                processing_time_seconds=time.time() - start_time,
            )
            return nodes

        # Ensure all nodes have embeddings
        await self._generate_missing_embeddings(nodes)

        # Find similar node clusters and track similarities
        clusters = self._cluster_by_similarity_with_tracking(nodes)

        # Process each cluster
        deduplicated = []
        for cluster in clusters:
            if len(cluster) == 1:
                deduplicated.append(cluster[0])
            else:
                # Resolve duplicates in cluster
                merged_node, merge_type = await self._resolve_cluster_with_tracking(cluster)
                deduplicated.append(merged_node)

                if merge_type == "automatic":
                    automatic_merges += 1
                elif merge_type == "llm_assisted":
                    llm_assisted_merges += 1

        # Calculate and store metrics
        processing_time = time.time() - start_time
        all_similarities = [pair.similarity for pair in self.similarity_pairs]

        self.last_metrics = DeduplicationMetrics(
            original_count=len(nodes),
            deduplicated_count=len(deduplicated),
            clusters_created=len(clusters),
            automatic_merges=automatic_merges,
            llm_assisted_merges=llm_assisted_merges,
            similarity_scores=all_similarities,
            processing_time_seconds=processing_time,
        )

        logger.info(
            f"Deduplicated {len(nodes)} nodes to {len(deduplicated)} nodes "
            f"({self.last_metrics.reduction_percentage:.1f}% reduction) "
            f"in {processing_time:.2f}s"
        )

        if all_similarities:
            logger.debug(
                f"Similarity stats - avg: {self.last_metrics.average_similarity:.3f}, "
                f"min: {min(all_similarities):.3f}, max: {max(all_similarities):.3f}"
            )

        return deduplicated

    async def _generate_missing_embeddings(self, nodes: List[DecisionNode]) -> None:
        """Generate embeddings for nodes that don't have them."""
        tasks = []

        for node in nodes:
            if node.embedding is None:
                task = self._generate_node_embedding(node)
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _generate_node_embedding(self, node: DecisionNode) -> None:
        """Generate embedding for a single node."""
        try:
            text_content = f"{node.title}. {node.description}"
            embedding = await aembed_query(
                async_openai_client=self._get_openai_client(),
                text=text_content,
                embedding_model=settings.embedding_model,
                embedding_dim=1536,
            )
            node.embedding = embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding for node {node.id}: {e}")

    def _cluster_by_similarity(self, nodes: List[DecisionNode]) -> List[List[DecisionNode]]:
        """
        Cluster nodes by similarity.

        Args:
            nodes: Nodes to cluster

        Returns:
            List of clusters, each containing similar nodes
        """
        clusters = []
        assigned = set()

        for i, node in enumerate(nodes):
            if i in assigned:
                continue

            # Start new cluster
            cluster = [node]
            assigned.add(i)

            # Find similar nodes
            for j, other_node in enumerate(nodes):
                if j in assigned or j <= i:
                    continue

                similarity = self._calculate_node_similarity(node, other_node)
                if similarity >= self.similarity_threshold:
                    cluster.append(other_node)
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    def _cluster_by_similarity_with_tracking(self, nodes: List[DecisionNode]) -> List[List[DecisionNode]]:
        """
        Cluster nodes by similarity with comprehensive tracking.

        Args:
            nodes: Nodes to cluster

        Returns:
            List of clusters, each containing similar nodes
        """
        clusters = []
        assigned = set()

        for i, node in enumerate(nodes):
            if i in assigned:
                continue

            # Start new cluster
            cluster = [node]
            assigned.add(i)

            # Find similar nodes and track all comparisons
            for j, other_node in enumerate(nodes):
                if j in assigned or j <= i:
                    continue

                similarity = self._calculate_node_similarity(node, other_node)

                # Track similarity pair regardless of threshold
                if similarity >= self.similarity_threshold:
                    decision = "merged"
                    cluster.append(other_node)
                    assigned.add(j)
                elif similarity >= self.weak_similarity_threshold:
                    decision = "weak_match"
                else:
                    decision = "kept_separate"

                self.similarity_pairs.append(
                    SimilarityPair(
                        node1_id=node.id, node2_id=other_node.id, similarity=similarity, merge_decision=decision
                    )
                )

            clusters.append(cluster)

        return clusters

    def _calculate_node_similarity(self, node1: DecisionNode, node2: DecisionNode) -> float:
        """Calculate similarity between two nodes."""
        # Use embeddings if available
        if node1.embedding and node2.embedding:
            return cosine_similarity(node1.embedding, node2.embedding)

        # Fallback to text similarity
        return self._calculate_text_similarity(node1, node2)

    def _calculate_text_similarity(self, node1: DecisionNode, node2: DecisionNode) -> float:
        """Calculate text-based similarity between nodes."""
        # Simple text similarity based on title and description overlap
        title1_words = set(node1.title.lower().split())
        title2_words = set(node2.title.lower().split())

        desc1_words = set(node1.description.lower().split())
        desc2_words = set(node2.description.lower().split())

        # Calculate Jaccard similarity for titles and descriptions
        title_overlap = (
            len(title1_words & title2_words) / len(title1_words | title2_words) if title1_words | title2_words else 0
        )
        desc_overlap = (
            len(desc1_words & desc2_words) / len(desc1_words | desc2_words) if desc1_words | desc2_words else 0
        )

        # Weighted combination (titles are more important)
        return 0.7 * title_overlap + 0.3 * desc_overlap

    async def _resolve_cluster(self, cluster: List[DecisionNode]) -> DecisionNode:
        """
        Resolve a cluster of similar nodes into a single merged node.

        Args:
            cluster: List of similar nodes

        Returns:
            Merged node representing the cluster
        """
        if len(cluster) == 1:
            return cluster[0]

        # Calculate average similarity in cluster
        avg_similarity = self._calculate_cluster_similarity(cluster)

        if avg_similarity > self.similarity_threshold:
            # Strong match - automatic merge
            return self._merge_nodes_automatic(cluster)
        else:
            # Weak match - LLM-assisted merge
            return await self._merge_nodes_with_llm(cluster)

    async def _resolve_cluster_with_tracking(self, cluster: List[DecisionNode]) -> Tuple[DecisionNode, str]:
        """
        Resolve a cluster of similar nodes with tracking of merge type.

        Args:
            cluster: List of similar nodes

        Returns:
            Tuple of (merged node, merge type)
        """
        if len(cluster) == 1:
            return cluster[0], "no_merge"

        # Calculate average similarity in cluster
        avg_similarity = self._calculate_cluster_similarity(cluster)

        if avg_similarity > self.similarity_threshold:
            # Strong match - automatic merge
            merged_node = self._merge_nodes_automatic(cluster)
            return merged_node, "automatic"
        else:
            # Weak match - LLM-assisted merge
            merged_node = await self._merge_nodes_with_llm(cluster)
            return merged_node, "llm_assisted"

    def _calculate_cluster_similarity(self, cluster: List[DecisionNode]) -> float:
        """Calculate average similarity within a cluster."""
        if len(cluster) <= 1:
            return 1.0

        similarities = []
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                similarity = self._calculate_node_similarity(cluster[i], cluster[j])
                similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _merge_nodes_automatic(self, cluster: List[DecisionNode]) -> DecisionNode:
        """Automatically merge nodes with high similarity."""
        # Use the first node as base
        merged = cluster[0].copy()

        # Collect unique information from all nodes
        all_titles = [node.title for node in cluster]
        all_descriptions = [node.description for node in cluster]
        all_tags = []
        for node in cluster:
            all_tags.extend(node.tags)

        # Create merged title (use the shortest/most general one)
        merged.title = min(all_titles, key=len)

        # Create merged description (combine unique elements)
        unique_desc_parts = list(dict.fromkeys(all_descriptions))  # Preserve order, remove duplicates
        merged.description = ". ".join(unique_desc_parts)

        # Merge tags
        merged.tags = list(set(all_tags))

        # Update metadata
        merged.metadata["merged_from"] = [node.id for node in cluster]
        merged.metadata["merge_type"] = "automatic"
        merged.metadata["original_count"] = len(cluster)

        logger.info(f"Automatically merged {len(cluster)} nodes into: {merged.title}")
        return merged

    async def _merge_nodes_with_llm(self, cluster: List[DecisionNode]) -> DecisionNode:
        """Use LLM to intelligently merge similar nodes."""
        try:
            # Build prompt for LLM merge
            system_prompt = self._build_merge_system_prompt()
            user_prompt = self._build_merge_user_prompt(cluster)

            # Use LLM to create merged node
            response = await ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=DecisionNode,
                llm_model=settings.llm_model,
                temperature=settings.assessment_temperature,
                max_tokens=2000,
            )

            # Preserve original metadata and add merge info
            response.id = cluster[0].id  # Keep original ID
            response.layer = cluster[0].layer
            response.type = cluster[0].type
            response.metadata = cluster[0].metadata.copy()
            response.metadata["merged_from"] = [node.id for node in cluster]
            response.metadata["merge_type"] = "llm_assisted"
            response.metadata["original_count"] = len(cluster)

            logger.info(f"LLM-assisted merge of {len(cluster)} nodes into: {response.title}")
            return response

        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            # Fallback to automatic merge
            return self._merge_nodes_automatic(cluster)

    def _build_merge_system_prompt(self) -> str:
        """Build system prompt for LLM node merging."""
        return """You are an expert at consolidating similar strategic planning nodes. Your task is to merge multiple similar nodes into a single, coherent node that captures the essential meaning and value of all input nodes.

Guidelines:
1. Preserve the core strategic intent of all input nodes
2. Create a title that captures the common theme
3. Write a description that synthesizes the key elements
4. Ensure the merged node is actionable and specific
5. Maintain the same node type and layer as the inputs

Focus on creating a node that is more valuable than any individual input node while preserving their essential strategic insights."""

    def _build_merge_user_prompt(self, cluster: List[DecisionNode]) -> str:
        """Build user prompt for LLM node merging."""
        prompt_parts = [f"Please merge these {len(cluster)} similar nodes into a single, high-quality node:", ""]

        for i, node in enumerate(cluster, 1):
            prompt_parts.extend(
                [
                    f"Node {i}:",
                    f"Title: {node.title}",
                    f"Description: {node.description}",
                    f"Tags: {', '.join(node.tags)}",
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "Create a merged node that:",
                "1. Has a clear, concise title that captures the common theme",
                "2. Has a comprehensive description that synthesizes key elements",
                "3. Includes relevant tags from the input nodes",
                "4. Maintains strategic value and actionability",
            ]
        )

        return "\n".join(prompt_parts)

    def get_last_metrics(self) -> DeduplicationMetrics | None:
        """Get metrics from the last deduplication operation."""
        return self.last_metrics

    def get_similarity_pairs(self) -> List[SimilarityPair]:
        """Get all similarity pairs from the last deduplication operation."""
        return self.similarity_pairs.copy()

    def export_deduplication_report(self) -> Dict[str, any]:
        """Export a comprehensive deduplication report."""
        if not self.last_metrics:
            return {"error": "No deduplication metrics available"}

        # Group similarity pairs by decision
        decisions = {}
        for pair in self.similarity_pairs:
            decision = pair.merge_decision
            if decision not in decisions:
                decisions[decision] = []
            decisions[decision].append({"similarity": pair.similarity, "node_pair": (pair.node1_id, pair.node2_id)})

        return {
            "metrics": {
                "original_count": self.last_metrics.original_count,
                "deduplicated_count": self.last_metrics.deduplicated_count,
                "reduction_percentage": self.last_metrics.reduction_percentage,
                "clusters_created": self.last_metrics.clusters_created,
                "automatic_merges": self.last_metrics.automatic_merges,
                "llm_assisted_merges": self.last_metrics.llm_assisted_merges,
                "processing_time_seconds": self.last_metrics.processing_time_seconds,
                "average_similarity": self.last_metrics.average_similarity,
            },
            "similarity_distributions": {
                decision: {
                    "count": len(pairs),
                    "avg_similarity": sum(p["similarity"] for p in pairs) / len(pairs) if pairs else 0,
                    "similarity_range": [
                        min(p["similarity"] for p in pairs) if pairs else 0,
                        max(p["similarity"] for p in pairs) if pairs else 0,
                    ],
                }
                for decision, pairs in decisions.items()
            },
            "thresholds": {
                "similarity_threshold": self.similarity_threshold,
                "weak_similarity_threshold": self.weak_similarity_threshold,
            },
        }
