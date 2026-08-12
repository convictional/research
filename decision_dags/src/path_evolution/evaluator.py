"""Path fitness evaluation using organizational goals and ensemble LLM assessment."""

import asyncio
import logging
from typing import List, Dict, Any

from openai import AsyncOpenAI

from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt

from ..models import DecisionDAG, PathMetrics, ComprehensiveEvaluation, EvaluationContext, FitnessWeights
from ..settings import settings

logger = logging.getLogger(__name__)


class PathFitnessEvaluator:
    """Evaluate path fitness using organizational goals and ensemble LLM voting."""

    def __init__(self, fitness_weights: FitnessWeights | None = None):
        self.fitness_weights = fitness_weights or self._create_default_weights()
        self.ensemble_perspectives = self._create_ensemble_perspectives()
        self._openai_client: AsyncOpenAI | None = None
        self._context_cache: Dict[str, Any] = {}

    def _create_default_weights(self) -> FitnessWeights:
        """Create default fitness weights."""
        return FitnessWeights()

    def _create_ensemble_perspectives(self) -> List[Dict[str, Any]]:
        """Create ensemble of evaluation perspectives."""
        return [
            {
                "role": "Conservative strategic analyst focusing on proven, incremental approaches",
                "temperature": 0.5,
                "perspective": "conservative",
            },
            {
                "role": "Optimistic strategic analyst who values innovation and bold moves",
                "temperature": 0.8,
                "perspective": "optimistic",
            },
            {
                "role": "Balanced strategic analyst who weighs practicality equally",
                "temperature": 1.0,
                "perspective": "balanced",
            },
        ]

    def _get_openai_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client for embeddings."""
        if not self._openai_client:
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        return self._openai_client

    async def evaluate_path_batch(
        self,
        path_dags: List[DecisionDAG],
        context: EvaluationContext,
        organizational_goals: List[Dict[str, Any]] | None = None,
        max_concurrent: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate multiple paths concurrently for better performance.

        Args:
            path_dags: List of path DAGs to evaluate
            context: Evaluation context
            organizational_goals: List of organizational goals
            max_concurrent: Maximum concurrent evaluations

        Returns:
            Dictionary mapping path ID to fitness score
        """
        if not path_dags:
            return {}

        # Update context with all paths for uniqueness calculation
        context.all_paths = path_dags

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_single_with_limit(path_dag: DecisionDAG) -> tuple[str, float]:
            async with semaphore:
                try:
                    fitness = await self.evaluate_path(path_dag, context, organizational_goals)
                    return path_dag.id, fitness
                except Exception as e:
                    logger.error(f"Error evaluating path {path_dag.id}: {e}")
                    return path_dag.id, 0.0

        # Execute all evaluations concurrently
        tasks = [evaluate_single_with_limit(path) for path in path_dags]
        results = await asyncio.gather(*tasks)

        # Convert to dictionary
        fitness_scores = dict(results)

        logger.info(f"Batch evaluation complete: {len(fitness_scores)} paths evaluated")
        return fitness_scores

    async def evaluate_path(
        self,
        path_dag: DecisionDAG,
        context: EvaluationContext,
        organizational_goals: List[Dict[str, Any]] | None = None,
    ) -> float:
        """
        Evaluate path fitness using organizational goals and ensemble LLM voting.

        Args:
            path_dag: Path DAG to evaluate
            context: Evaluation context
            organizational_goals: List of organizational goals (fetched from database)

        Returns:
            Fitness score between 0 and 1
        """
        # Fetch actual organizational goals from database if organization_id provided
        if context.organization_id and not organizational_goals:
            organizational_goals = await self._fetch_organization_goals(context.organization_id)

        # Calculate quantitative metrics from path structure and edge data
        quantitative_metrics = self._calculate_quantitative_metrics(path_dag)

        # Ensemble LLM evaluation with multiple perspectives
        ensemble_scores = await self._evaluate_with_ensemble(
            path_dag, organizational_goals or [], quantitative_metrics, context
        )

        # Calculate path uniqueness score
        uniqueness_score = await self._evaluate_uniqueness(path_dag, context.all_paths)

        # Calculate weighted fitness using configured weights
        fitness = self._calculate_weighted_fitness(ensemble_scores, quantitative_metrics, uniqueness_score)

        logger.debug(f"Path {path_dag.id} fitness: {fitness:.3f}")
        return fitness

    async def _fetch_organization_goals(self, organization_id: str) -> List[Dict[str, Any]]:
        """
        Fetch active organizational goals with success conditions from database.

        This connects to the actual database to fetch goals following the README pattern.
        """
        # Import here to avoid circular imports
        from ..context.database_context import DatabaseContextProvider

        context_provider = DatabaseContextProvider()

        # Use the existing database context system to fetch goals
        # This will fail if database is not available - which is what we want
        context = await context_provider.get_context(organization_id)

        return context.organizational_goals

    def _calculate_quantitative_metrics(self, path_dag: DecisionDAG) -> PathMetrics:
        """Extract quantitative metrics from path structure and edge data."""
        nodes = list(path_dag.all_nodes.values())
        edges = path_dag.edges

        # Calculate total cost from edges
        total_cost = 0.0
        for edge in edges:
            if edge.estimated_cost_dollars:
                total_cost += edge.estimated_cost_dollars

        # Calculate timeline (sum of all edge timelines - could be improved with critical path)
        timeline_weeks = 0.0
        for edge in edges:
            if edge.timeline_estimate:
                # Simple parsing of timeline strings like "4 weeks", "2 months"
                timeline_weeks += self._parse_timeline_estimate(edge.timeline_estimate)

        # Count implementation risks
        total_risks = 0
        for edge in edges:
            if edge.implementation_risks:
                total_risks += len(edge.implementation_risks)

        # Calculate complexity score (inverted - lower complexity = higher score)
        node_count = len(nodes)
        complexity_score = max(0.0, 1.0 - (node_count / 20.0))  # Normalize to 0-1

        # Simple resource efficiency calculation
        resource_efficiency = 1.0 / (1.0 + total_cost / 100000.0) if total_cost > 0 else 1.0

        return PathMetrics(
            total_cost_dollars=total_cost,
            timeline_weeks=timeline_weeks,
            risk_count=total_risks,
            complexity_score=complexity_score,
            resource_efficiency=resource_efficiency,
            node_count=node_count,
            branching_factor=1.0,  # Paths are linear, so branching factor is 1
        )

    def _parse_timeline_estimate(self, timeline_str: str) -> float:
        """Parse timeline estimate string to weeks."""
        timeline_lower = timeline_str.lower()

        # Extract number
        import re

        numbers = re.findall(r"\d+(?:\.\d+)?", timeline_str)
        if not numbers:
            return 4.0  # Default to 4 weeks

        value = float(numbers[0])

        # Convert to weeks
        if "month" in timeline_lower:
            return value * 4.0
        elif "week" in timeline_lower:
            return value
        elif "day" in timeline_lower:
            return value / 7.0
        elif "year" in timeline_lower:
            return value * 52.0
        else:
            return value  # Assume weeks

    async def _evaluate_with_ensemble(
        self,
        path_dag: DecisionDAG,
        organizational_goals: List[Dict[str, Any]],
        metrics: PathMetrics,
        context: EvaluationContext,
    ) -> Dict[str, float]:
        """Evaluate using ensemble voting across multiple perspectives."""
        evaluation_tasks = []

        for perspective in self.ensemble_perspectives:
            task = self._evaluate_single_perspective(path_dag, organizational_goals, metrics, perspective, context)
            evaluation_tasks.append(task)

        # Execute all perspectives concurrently
        try:
            perspective_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in ensemble evaluation: {e}")
            return self._create_default_ensemble_scores()

        # Aggregate results
        aggregated_scores = self._aggregate_ensemble_results(perspective_results)

        return aggregated_scores

    async def _evaluate_single_perspective(
        self,
        path_dag: DecisionDAG,
        organizational_goals: List[Dict[str, Any]],
        metrics: PathMetrics,
        perspective: Dict[str, Any],
        context: EvaluationContext,
    ) -> ComprehensiveEvaluation:
        """Evaluate from a single perspective using structured output."""
        try:
            # Build evaluation prompts
            system_prompt = self._build_evaluation_system_prompt(perspective, organizational_goals)
            user_prompt = self._build_evaluation_user_prompt(path_dag, metrics, context)

            # Get structured evaluation
            response = await ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=ComprehensiveEvaluation,
                llm_model=settings.llm_model,
                temperature=perspective["temperature"],
                max_tokens=6000,
            )

            return response

        except Exception as e:
            logger.error(f"Error in single perspective evaluation ({perspective['perspective']}): {e}")
            return self._create_default_evaluation()

    def _build_evaluation_system_prompt(
        self, perspective: Dict[str, Any], organizational_goals: List[Dict[str, Any]]
    ) -> str:
        """Build system prompt for evaluation."""
        try:
            return build_prompt(
                "fitness_evaluation_system.txt.jinja",
                weights=self.fitness_weights,
                perspective_role=perspective["role"],
                organizational_goals=organizational_goals,
            )
        except Exception as e:
            logger.error(f"Error building evaluation system prompt: {e}")
            return self._build_fallback_system_prompt(perspective)

    def _build_evaluation_user_prompt(
        self, path_dag: DecisionDAG, metrics: PathMetrics, context: EvaluationContext
    ) -> str:
        """Build user prompt for evaluation."""
        try:
            # Prepare path data for template
            nodes = list(path_dag.all_nodes.values())
            nodes.sort(key=lambda n: n.layer)

            edges_data = []
            for edge in path_dag.edges:
                source_node = path_dag.get_node(edge.source_id)
                target_node = path_dag.get_node(edge.target_id)

                edge_data = {
                    "source_title": source_node.title if source_node else "Unknown",
                    "target_title": target_node.title if target_node else "Unknown",
                    "conditions": edge.conditions,
                    "cost_estimate": edge.cost_estimate,
                    "timeline_estimate": edge.timeline_estimate,
                    "implementation_risks": edge.implementation_risks,
                }
                edges_data.append(edge_data)

            return build_prompt(
                "fitness_evaluation_user.txt.jinja",
                path_title=f"Strategic Path {path_dag.id}",
                path_description=path_dag.metadata.get("description", "Strategic decision path"),
                path_nodes=nodes,
                path_edges=edges_data,
                quantitative_metrics=metrics,
                problem_statement=context.domain_context,
            )
        except Exception as e:
            logger.error(f"Error building evaluation user prompt: {e}")
            return self._build_fallback_user_prompt(path_dag, metrics)

    def _aggregate_ensemble_results(self, perspective_results: List[Any]) -> Dict[str, float]:
        """Aggregate results from multiple perspectives."""
        valid_results = []

        for result in perspective_results:
            if isinstance(result, ComprehensiveEvaluation):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Perspective evaluation failed: {result}")

        if not valid_results:
            logger.error("No valid perspective results")
            return self._create_default_ensemble_scores()

        # Calculate averages across perspectives
        goal_scores = [r.goal_alignment.score for r in valid_results]
        risk_scores = [r.risk_assessment.score for r in valid_results]
        cost_scores = [r.cost_assessment.score for r in valid_results]
        timeline_scores = [r.timeline_assessment.score for r in valid_results]

        return {
            "goal_alignment": sum(goal_scores) / len(goal_scores),
            "risk_mitigation": sum(risk_scores) / len(risk_scores),
            "cost_efficiency": sum(cost_scores) / len(cost_scores),
            "timeline_efficiency": sum(timeline_scores) / len(timeline_scores),
        }

    async def _evaluate_uniqueness(self, path_dag: DecisionDAG, all_paths: List[DecisionDAG]) -> float:
        """Evaluate path uniqueness using structural comparison."""
        if len(all_paths) <= 1:
            return 1.0

        try:
            # Extract path signature for comparison
            current_signature = self._extract_path_signature(path_dag)

            similarities = []
            for other_path in all_paths:
                if other_path.id != path_dag.id:
                    other_signature = self._extract_path_signature(other_path)

                    # Use Jaccard similarity on signatures
                    similarity = self._jaccard_similarity(current_signature, other_signature)
                    similarities.append(similarity)

            # Convert to uniqueness score
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            uniqueness = 1 - avg_similarity

            return max(0.0, min(1.0, uniqueness))

        except Exception as e:
            logger.error(f"Error calculating uniqueness: {e}")
            return 0.5  # Default neutral score

    def _extract_path_signature(self, path_dag: DecisionDAG) -> set:
        """Create a signature representing the path's key decisions and options."""
        signature = set()

        for node in path_dag.all_nodes.values():
            # Include node type and key terms from title
            key_terms = self._extract_key_terms(node.title)
            for term in key_terms:
                signature.add(f"{node.type.value}:{term}")

        return signature

    def _extract_key_terms(self, title: str) -> List[str]:
        """Extract key terms from a title."""
        # Simple key term extraction
        words = title.lower().split()
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms[:3]  # Limit to top 3 terms

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _calculate_weighted_fitness(
        self, ensemble_scores: Dict[str, float], metrics: PathMetrics, uniqueness: float
    ) -> float:
        """
        Calculate final fitness using configured weights as specified in README.

        Default weights per README:
        - Goals: 40% (primary importance)
        - Cost: 20% (cost efficiency)
        - Timeline: 20% (timeline efficiency)
        - Risk: 10% (risk mitigation)
        - Complexity: 10% (simplicity)
        """
        fitness = (
            self.fitness_weights.w_goals * ensemble_scores.get("goal_alignment", 0.5)
            + self.fitness_weights.w_cost * ensemble_scores.get("cost_efficiency", 0.5)
            + self.fitness_weights.w_timeline * ensemble_scores.get("timeline_efficiency", 0.5)
            + self.fitness_weights.w_risk * ensemble_scores.get("risk_mitigation", 0.5)
            + self.fitness_weights.w_complexity * metrics.complexity_score
        )

        # Add path-specific adjustments per README
        fitness += uniqueness * 0.05  # Small bonus for unique paths

        return max(0.0, min(1.0, fitness))

    def _create_default_ensemble_scores(self) -> Dict[str, float]:
        """Create default ensemble scores."""
        return {"goal_alignment": 0.5, "risk_mitigation": 0.5, "cost_efficiency": 0.5, "timeline_efficiency": 0.5}

    def _create_default_evaluation(self) -> ComprehensiveEvaluation:
        """Create default evaluation when LLM fails."""
        from ..models import GoalAlignmentScore, RiskAssessmentScore, CostAssessmentScore, TimelineAssessmentScore

        return ComprehensiveEvaluation(
            goal_alignment=GoalAlignmentScore(
                score=0.5, rationale="Default evaluation due to processing error", addressed_goals=[], goal_gaps=[]
            ),
            risk_assessment=RiskAssessmentScore(
                score=0.5,
                rationale="Default evaluation due to processing error",
                identified_risks=[],
                mitigation_strategies=[],
            ),
            cost_assessment=CostAssessmentScore(
                score=0.5,
                rationale="Default evaluation due to processing error",
                cost_breakdown={},
                cost_optimization_opportunities=[],
            ),
            timeline_assessment=TimelineAssessmentScore(
                score=0.5,
                rationale="Default evaluation due to processing error",
                critical_path_weeks=0.0,
                timeline_risks=[],
            ),
        )

    def _build_fallback_system_prompt(self, perspective: Dict[str, Any]) -> str:
        """Build fallback system prompt."""
        return f"""You are a {perspective["role"]}. Evaluate the strategic path across multiple dimensions:

1. Goal Alignment: How well does this path address organizational goals?
2. Risk Assessment: How well does this path identify and mitigate risks?
3. Cost Assessment: How cost-effective is this strategic path?
4. Timeline Assessment: How realistic and efficient is the timeline?

Provide structured evaluation with scores from 0.0 to 1.0 for each dimension."""

    def _build_fallback_user_prompt(self, path_dag: DecisionDAG, metrics: PathMetrics) -> str:
        """Build fallback user prompt."""
        nodes = list(path_dag.all_nodes.values())
        nodes.sort(key=lambda n: n.layer)

        path_summary = " -> ".join([n.title for n in nodes])

        return f"""Evaluate this strategic path:

Path: {path_summary}

Metrics:
- Total Cost: ${metrics.total_cost_dollars:,.0f}
- Timeline: {metrics.timeline_weeks} weeks
- Risk Count: {metrics.risk_count}
- Complexity Score: {metrics.complexity_score:.2f}

Provide comprehensive evaluation across all dimensions."""
