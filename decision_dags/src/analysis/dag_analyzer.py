"""DAG analysis using LLM for comparisons and insights."""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from common.instruct_llm import ainstruct_llm

from ..models import DecisionDAG
from ..settings import settings
from ..path_evolution.mutation_engine import JSONMutationEngine

logger = logging.getLogger(__name__)


class DAGComparisonAnalysis(BaseModel):
    """Schema for LLM-generated DAG comparison analysis."""

    structural_changes: str = Field(..., description="Summary of structural changes between DAGs")
    strategic_implications: str = Field(..., description="Strategic implications of the changes")
    improvement_assessment: str = Field(..., description="Assessment of whether changes represent improvement")
    risks_and_concerns: str = Field(..., description="Potential risks or concerns with the changes")
    recommendations: str = Field(..., description="Recommendations for further refinement")
    overall_analysis: str = Field(..., description="Complete markdown-formatted analysis")


class DAGAnalyzer:
    """Analyzes DAGs using LLM for insights and comparisons."""

    def __init__(self):
        """Initialize the DAG analyzer."""
        self.mutation_engine = JSONMutationEngine()

    async def compare_dags(
        self, dag1: DecisionDAG, dag2: DecisionDAG, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compare two DAGs and provide insights on their differences.

        Args:
            dag1: First DAG (usually original)
            dag2: Second DAG (usually evolved/modified)
            context: Additional context about the DAGs

        Returns:
            Markdown-formatted analysis of differences
        """
        try:
            # Convert DAGs to JSON for analysis
            dag1_json = await self.mutation_engine.dag_to_json(dag1)
            dag2_json = await self.mutation_engine.dag_to_json(dag2)

            # Prepare context information
            context_info = context or {}
            dag1_type = context_info.get("original_type", "first")
            dag2_type = context_info.get("evolved_type", "second")

            # Create the comparison prompt
            system_prompt = """You are an expert strategic analyst comparing two Decision DAGs (Directed Acyclic Graphs) that represent strategic planning pathways.

Your task is to analyze the differences between two DAGs and provide insights on:
1. Structural changes (nodes added/removed, edges modified)
2. Strategic improvements or regressions
3. Changes in decision confidence scores
4. Evolution of strategic focus and priorities
5. Risk profile changes
6. Implementation complexity changes

Provide a concise but insightful analysis that helps users understand the strategic implications of the changes."""

            user_prompt = f"""Please compare these two Decision DAGs and provide strategic insights:

**{dag1_type.title()} DAG:**
- Nodes: {len(dag1_json["nodes"])}
- Edges: {len(dag1_json["edges"])}
- Layers: {dag1_json["metadata"].get("max_layers", "Unknown")}

**{dag2_type.title()} DAG:**
- Nodes: {len(dag2_json["nodes"])}
- Edges: {len(dag2_json["edges"])}
- Layers: {dag2_json["metadata"].get("max_layers", "Unknown")}

Key nodes from {dag1_type} DAG:
{self._summarize_key_nodes(dag1_json["nodes"][:5])}

Key nodes from {dag2_type} DAG:
{self._summarize_key_nodes(dag2_json["nodes"][:5])}

Please provide:
1. A summary of the most significant structural changes
2. Strategic implications of these changes
3. Whether the evolution represents an improvement and why
4. Any potential risks or concerns with the changes
5. Recommendations for further refinement

Keep your analysis concise but insightful, focusing on strategic value rather than technical details.

Format your complete analysis as markdown in the overall_analysis field."""

            # Get LLM analysis using structured output
            response = await ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=DAGComparisonAnalysis,
                llm_model=settings.llm_model,
                temperature=0.7,
                max_tokens=1500,
            )

            return response.overall_analysis

        except Exception as e:
            logger.error(f"Failed to analyze DAG differences: {e}")
            return f"Analysis failed: {str(e)}"

    def _summarize_key_nodes(self, nodes: List[Dict[str, Any]]) -> str:
        """Summarize key nodes for the prompt."""
        if not nodes:
            return "No nodes available"

        summary = []
        for node in nodes:
            node_type = node.get("type", "unknown")
            title = node.get("title", "Untitled")
            confidence = node.get("confidence_score", 0)
            summary.append(f"- [{node_type}] {title} (confidence: {confidence:.2f})")

        return "\n".join(summary)

    async def summarize_dag(self, dag: DecisionDAG) -> str:
        """
        Provide a summary analysis of a single DAG.

        Args:
            dag: The DAG to analyze

        Returns:
            Markdown-formatted summary
        """
        try:
            # Convert DAG to JSON
            dag_json = await self.mutation_engine.dag_to_json(dag)

            # Create summary prompt
            system_prompt = """You are an expert strategic analyst reviewing a Decision DAG (Directed Acyclic Graph) that represents a strategic planning pathway.

Provide a concise summary that includes:
1. The overall strategic approach
2. Key decision points and their implications
3. Strengths of the current strategy
4. Potential weaknesses or gaps
5. Overall assessment of feasibility and impact"""

            user_prompt = f"""Please analyze this Decision DAG:

**DAG Overview:**
- Total Nodes: {len(dag_json["nodes"])} ({sum(1 for n in dag_json["nodes"] if n["type"] == "decision")} decisions, {sum(1 for n in dag_json["nodes"] if n["type"] == "option")} options)
- Total Edges: {len(dag_json["edges"])}
- Layers: {dag_json["metadata"].get("max_layers", "Unknown")}
- Average Confidence: {sum(n.get("confidence_score", 0) for n in dag_json["nodes"]) / len(dag_json["nodes"]):.2f}

**Key Nodes:**
{self._summarize_key_nodes(dag_json["nodes"][:10])}

Provide a strategic summary that would help a decision-maker understand the value and implications of this strategic pathway."""

            # Create a simple schema for single DAG analysis
            class DAGSummaryAnalysis(BaseModel):
                summary: str = Field(..., description="Complete markdown-formatted summary analysis")

            response = await ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=DAGSummaryAnalysis,
                llm_model=settings.llm_model,
                temperature=0.7,
                max_tokens=1000,
            )

            return response.summary

        except Exception as e:
            logger.error(f"Failed to summarize DAG: {e}")
            return f"Summary failed: {str(e)}"
