"""Path evolution engine for optimizing strategic paths using AlphaEvolve approach."""

import asyncio
import logging
import random
from typing import List, Tuple, Dict, Any, Optional

from ..models import AlphaEvolutionConfig, DecisionDAG, EvaluationContext, EvolutionVariant, MutationProposal
from ..settings import settings
from .evaluator import PathFitnessEvaluator
from .extractor import PathExtractionEngine
from .mutation_engine import JSONMutationEngine

logger = logging.getLogger(__name__)


class PathEvolutionEngine:
    """Evolution engine that optimizes paths using AlphaEvolve genetic algorithm approach."""

    def __init__(self, config: AlphaEvolutionConfig | None = None):
        self.config = config or self._create_default_config()
        self.path_evaluator = PathFitnessEvaluator()
        self.path_extractor = PathExtractionEngine()
        self.mutation_engine = JSONMutationEngine()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_evolutions)

        # AlphaEvolve-specific tracking
        self.evolution_history: List[EvolutionVariant] = []
        self.generation_stats: List[Dict[str, Any]] = []

    def _create_default_config(self) -> AlphaEvolutionConfig:
        """Create default evolution configuration."""
        return AlphaEvolutionConfig(
            max_concurrent_evolutions=getattr(settings, "max_concurrent_evolutions", 4),
            max_generations=getattr(settings, "max_generations", 10),
            min_improvement_threshold=getattr(settings, "min_improvement_threshold", 0.05),
        )

    async def evolve_paths(
        self,
        paths: List[DecisionDAG],
        context: EvaluationContext,
        organizational_goals: List[Dict[str, Any]] | None = None,
    ) -> List[DecisionDAG]:
        """
        Evolve multiple paths to improve their fitness.

        Args:
            paths: List of path DAGs to evolve
            context: Evolution context
            organizational_goals: Organizational goals for evaluation

        Returns:
            List of evolved paths with improved fitness
        """
        if not paths:
            logger.warning("No paths provided for evolution")
            return []

        logger.info(f"Starting evolution of {len(paths)} paths")

        # Evaluate baseline fitness for all paths using batch evaluation
        baseline_fitness = await self.path_evaluator.evaluate_path_batch(
            paths, context, organizational_goals, self.config.max_concurrent_evolutions
        )

        # Select paths for evolution based on fitness
        selected_paths = self._select_paths_for_evolution(paths, baseline_fitness)

        if not selected_paths:
            logger.info("No paths selected for evolution")
            return paths

        # Evolve selected paths
        evolved_paths = await self._evolve_paths_parallel(selected_paths, context, organizational_goals)

        # Filter evolved paths by improvement threshold
        improved_paths = self._filter_improved_paths(evolved_paths, baseline_fitness)

        # Combine improved paths with unchanged paths
        final_paths = self._combine_evolved_and_original_paths(paths, improved_paths, selected_paths)

        logger.info(f"Evolution complete: {len(improved_paths)} paths improved")
        return final_paths

    async def evolve_dag_paths(
        self,
        source_dag: DecisionDAG,
        context: EvaluationContext,
        organizational_goals: List[Dict[str, Any]] | None = None,
        filter_criteria: Dict[str, Any] | None = None,
    ) -> Tuple[List[DecisionDAG], Dict[str, Any]]:
        """
        Extract paths from a DAG and evolve them.

        Args:
            source_dag: Source DAG to extract paths from
            context: Evolution context
            organizational_goals: Organizational goals for evaluation
            filter_criteria: Optional criteria for path filtering

        Returns:
            Tuple of (evolved paths, evolution metrics)
        """
        logger.info(f"Starting DAG path evolution for {source_dag.id}")

        # Extract paths from source DAG
        if filter_criteria:
            extracted_paths, extraction_metrics = self.path_extractor.extract_paths_with_criteria(
                source_dag,
                min_length=filter_criteria.get("min_length", 2),
                max_length=filter_criteria.get("max_length", 20),
                include_incomplete=filter_criteria.get("include_incomplete", False),
            )
        else:
            extracted_paths, extraction_metrics = self.path_extractor.extract_paths(source_dag)

        if not extracted_paths:
            logger.warning("No paths extracted from DAG")
            return [], {
                "extraction_metrics": extraction_metrics,
                "evolution_metrics": {"evolved_paths": 0, "improved_paths": 0},
            }

        logger.info(f"Extracted {len(extracted_paths)} paths, evolving them")

        # Evolve the extracted paths
        evolved_paths = await self.evolve_paths(extracted_paths, context, organizational_goals)

        # Calculate evolution metrics
        evolution_metrics = {
            "original_dag_nodes": len(source_dag.all_nodes),
            "extracted_paths": len(extracted_paths),
            "evolved_paths": len(evolved_paths),
            "improved_paths": len([p for p in evolved_paths if "evolved_from" in p.metadata]),
        }

        return evolved_paths, {"extraction_metrics": extraction_metrics, "evolution_metrics": evolution_metrics}

    async def _evaluate_all_paths(
        self,
        paths: List[DecisionDAG],
        context: EvaluationContext,
        organizational_goals: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, float]:
        """Evaluate baseline fitness for all paths."""
        fitness_scores = {}

        # Update context with all paths for uniqueness calculation
        context.all_paths = paths

        evaluation_tasks = []
        for path in paths:
            task = self.path_evaluator.evaluate_path(path, context, organizational_goals)
            evaluation_tasks.append((path.id, task))

        # Execute evaluations concurrently
        for path_id, task in evaluation_tasks:
            try:
                fitness = await task
                fitness_scores[path_id] = fitness
            except Exception as e:
                logger.error(f"Error evaluating path {path_id}: {e}")
                fitness_scores[path_id] = 0.0

        return fitness_scores

    def _select_paths_for_evolution(
        self, paths: List[DecisionDAG], fitness_scores: Dict[str, float]
    ) -> List[DecisionDAG]:
        """Select paths for evolution based on fitness and strategy."""
        if self.config.selection_strategy == "top_k":
            # Select top K paths by fitness
            sorted_paths = sorted(paths, key=lambda p: fitness_scores.get(p.id, 0.0), reverse=True)
            return sorted_paths[: self.config.top_k_paths]

        elif self.config.selection_strategy == "fitness_proportional":
            # Select paths proportional to their fitness
            total_fitness = sum(fitness_scores.values())
            if total_fitness == 0:
                return paths[: self.config.top_k_paths]  # Fallback to top K

            selected = []
            for path in paths:
                fitness = fitness_scores.get(path.id, 0.0)
                probability = fitness / total_fitness

                # Simple selection based on probability threshold
                if probability > 0.1 and len(selected) < self.config.top_k_paths:
                    selected.append(path)

            return selected or paths[: min(3, len(paths))]  # Ensure at least some paths

        else:
            # Default: select paths with fitness below average for improvement
            avg_fitness = sum(fitness_scores.values()) / len(fitness_scores) if fitness_scores else 0.5
            below_avg = [p for p in paths if fitness_scores.get(p.id, 0.0) < avg_fitness]
            return below_avg[: self.config.top_k_paths]

    async def _evolve_paths_parallel(
        self,
        paths: List[DecisionDAG],
        context: EvaluationContext,
        organizational_goals: List[Dict[str, Any]] | None = None,
    ) -> List[Tuple[DecisionDAG, float]]:
        """Evolve paths concurrently with rate limiting."""
        tasks = []

        for path in paths:
            task = self._create_limited_evolution_task(path, context, organizational_goals)
            tasks.append(task)

        # Execute all evolution tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures
        evolved_paths = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Path evolution failed: {result}")
            else:
                evolved_paths.append(result)

        return evolved_paths

    async def _create_limited_evolution_task(
        self, path: DecisionDAG, context: EvaluationContext, organizational_goals: List[Dict[str, Any]] | None = None
    ):
        """Create rate-limited evolution task."""
        async with self.semaphore:
            return await self._evolve_single_path(path, context, organizational_goals)

    async def _evolve_single_path(
        self, path: DecisionDAG, context: EvaluationContext, organizational_goals: List[Dict[str, Any]] | None = None
    ) -> Tuple[DecisionDAG, float]:
        """
        Evolve a single path using AlphaEvolve genetic algorithm techniques.

        Implements LLM-guided mutations, inspiration from successful variants,
        and fitness-based selection over multiple generations.
        """
        try:
            # Get current fitness and detailed scorecard
            current_fitness = await self.path_evaluator.evaluate_path(path, context, organizational_goals)

            # Initialize population with the original path
            population = [path]
            fitness_scores = {path.id: current_fitness}

            best_path = path
            best_fitness = current_fitness

            # Track the original path as first variant
            original_variant = EvolutionVariant(
                generation=0,
                fitness_scores={"overall_fitness": current_fitness},
                evaluation_reasoning=f"Original path baseline: {current_fitness:.3f}",
            )
            self.evolution_history.append(original_variant)

            # Calculate warm-up period
            warmup_generations = int(self.config.max_generations * self.config.warmup_generations_ratio)
            logger.info(f"Evolution warm-up period: {warmup_generations} generations (of {self.config.max_generations} total)")

            # Genetic algorithm generations
            for generation in range(self.config.max_generations):
                is_warmup = generation < warmup_generations
                logger.debug(f"Evolution generation {generation + 1}/{self.config.max_generations} for path {path.id} ({'warm-up' if is_warmup else 'exploitation'})")

                # Generate new candidates through mutation and crossover
                new_candidates = []

                # Mutation: Create mutated versions using LLM guidance
                for existing_path in population:
                    if self._should_mutate():
                        logger.debug(f"Attempting mutation for path {existing_path.id} in generation {generation + 1}")
                        mutated_path = await self._mutate_path_llm_guided(
                            existing_path, context, organizational_goals, generation
                        )
                        if mutated_path:
                            new_candidates.append(mutated_path)
                            logger.debug(f"Successfully mutated path {existing_path.id}")
                        else:
                            logger.debug(f"Mutation failed for path {existing_path.id}")
                    else:
                        logger.debug(f"Skipping mutation for path {existing_path.id} (probability check)")

                # Crossover: Use inspiration from evolution history
                if len(self.evolution_history) > 1:
                    for existing_path in population:
                        if self._should_crossover():
                            crossover_path = await self._crossover_with_inspiration(
                                existing_path, context, organizational_goals, generation
                            )
                            if crossover_path:
                                new_candidates.append(crossover_path)

                # Evaluate new candidates
                for candidate in new_candidates:
                    try:
                        candidate_fitness = await self.path_evaluator.evaluate_path(
                            candidate, context, organizational_goals
                        )
                        fitness_scores[candidate.id] = candidate_fitness

                        # Track best path
                        if candidate_fitness > best_fitness:
                            best_path = candidate
                            best_fitness = candidate_fitness
                            logger.debug(f"New best path found: {candidate.id} with fitness {best_fitness:.3f}")

                        # Track variant in evolution history with enhanced details
                        variant = EvolutionVariant(
                            generation=generation + 1,
                            parent_variant_ids=[original_variant.variant_id],
                            fitness_scores={"overall_fitness": candidate_fitness},
                            evaluation_reasoning=f"Generated in generation {generation + 1}",
                            parent_fitness=current_fitness,
                            fitness_delta=candidate_fitness - current_fitness,
                            mutation_success=candidate_fitness > current_fitness,
                            mutation_type=candidate.metadata.get("evolution_method", "unknown"),
                            mutation_reasoning=candidate.metadata.get("mutation_reasoning", ""),
                            mutations=candidate.metadata.get("mutations", [])
                        )
                        self._update_variant_strength_areas(variant, candidate_fitness)
                        self.evolution_history.append(variant)

                    except Exception as e:
                        logger.warning(f"Error evaluating candidate path {candidate.id}: {e}")
                        continue

                # Selection: Keep best paths for next generation
                population = self._select_survivors(population + new_candidates, fitness_scores)

                # Early termination only after warm-up period
                if generation >= warmup_generations:
                    if best_fitness <= current_fitness + self.config.min_improvement_threshold:
                        logger.debug(f"No significant improvement after {generation + 1} generations (best: {best_fitness:.3f}, current: {current_fitness:.3f}, threshold: {self.config.min_improvement_threshold})")
                        # Allow at least one generation after warm-up before terminating
                        if generation > warmup_generations:
                            logger.info(f"Early termination after {generation + 1} generations (warm-up was {warmup_generations} generations)")
                            break
                else:
                    logger.debug(f"Generation {generation + 1} is in warm-up period, continuing regardless of fitness")

            # Return best path found
            improvement = best_fitness - current_fitness
            if improvement > 0:
                logger.debug(
                    f"Path {path.id} evolved successfully: {current_fitness:.3f} -> {best_fitness:.3f} (+{improvement:.3f})"
                )

            # Ensure evolved_from metadata is set even if no improvement
            if best_path.id == path.id and "evolved_from" not in best_path.metadata:
                # This is the original path with no successful mutations
                # Create a copy with evolved_from metadata
                evolved_path = DecisionDAG(
                    id=f"{path.id}_evolved",
                    root_nodes=path.root_nodes.copy(),
                    all_nodes=path.all_nodes.copy(),
                    edges=path.edges.copy(),
                    metadata=path.metadata.copy(),
                )
                evolved_path.metadata["evolved_from"] = path.id
                evolved_path.metadata["evolution_method"] = "no_improvement"
                evolved_path.metadata["generations_attempted"] = generation
                return (evolved_path, best_fitness)

            return (best_path, best_fitness)

        except Exception as e:
            logger.error(f"Error evolving path {path.id}: {e}")
            return (path, current_fitness if "current_fitness" in locals() else 0.0)

    def _create_minor_variation(self, path: DecisionDAG) -> DecisionDAG:
        """
        Create a minor variation of a path.

        This is a simplified implementation. A full implementation would use
        sophisticated mutation and crossover operations.
        """
        # Create a copy of the path
        varied_path = DecisionDAG(
            id=f"{path.id}_evolved",
            root_nodes=path.root_nodes.copy(),
            all_nodes=path.all_nodes.copy(),
            edges=path.edges.copy(),
            metadata=path.metadata.copy(),
        )

        # Add metadata to indicate this is an evolved path
        varied_path.metadata["evolved_from"] = path.id
        varied_path.metadata["evolution_method"] = "minor_variation"

        # For now, just return the same path with updated metadata
        # In a full implementation, this would make actual structural changes
        return varied_path

    def _filter_improved_paths(
        self, evolved_paths: List[Tuple[DecisionDAG, float]], baseline_fitness: Dict[str, float]
    ) -> List[DecisionDAG]:
        """Filter evolved paths by improvement threshold."""
        improved = []

        for evolved_path, new_fitness in evolved_paths:
            # Get original path ID (handle evolved path naming)
            original_id = evolved_path.metadata.get("evolved_from", evolved_path.id)
            baseline = baseline_fitness.get(original_id, 0.0)

            improvement = new_fitness - baseline
            if improvement >= self.config.min_improvement_threshold:
                improved.append(evolved_path)
                logger.debug(f"Path {original_id} meets improvement threshold: +{improvement:.3f}")

        return improved

    def _combine_evolved_and_original_paths(
        self, original_paths: List[DecisionDAG], improved_paths: List[DecisionDAG], selected_paths: List[DecisionDAG]
    ) -> List[DecisionDAG]:
        """Combine evolved paths with unchanged paths."""
        # Create mapping of evolved paths
        evolved_map = {}
        for evolved_path in improved_paths:
            original_id = evolved_path.metadata.get("evolved_from", evolved_path.id)
            evolved_map[original_id] = evolved_path

        # Build final list
        final_paths = []
        selected_ids = {p.id for p in selected_paths}

        for path in original_paths:
            if path.id in evolved_map:
                # Use evolved version
                final_paths.append(evolved_map[path.id])
            elif path.id not in selected_ids:
                # Keep original (wasn't selected for evolution)
                final_paths.append(path)
            # If path was selected but not improved, it's excluded

        return final_paths

    # AlphaEvolve-specific methods

    async def _mutate_path_llm_guided(
        self,
        path: DecisionDAG,
        context: EvaluationContext,
        organizational_goals: List[Dict[str, Any]] | None = None,
        generation: int = 0,
    ) -> Optional[DecisionDAG]:
        """Use LLM to propose and execute strategic mutations based on path weaknesses."""
        try:
            # Convert path to JSON for mutation
            path_json = await self.mutation_engine.dag_to_json(path)

            # Get current fitness scores for weakness identification
            current_fitness = await self.path_evaluator.evaluate_path(path, context, organizational_goals)
            fitness_scores = {"overall_fitness": current_fitness}

            # Identify weak and strong areas
            weak_areas = self._identify_weak_areas_from_scores(fitness_scores)
            strong_areas = [area for area in ["goal_alignment", "cost_efficiency", "timeline_efficiency", "risk_mitigation", "complexity"] if area not in weak_areas]

            # Get dynamic examples from evolution history
            dynamic_examples = self._select_dynamic_examples(weak_areas, strong_areas)
            dynamic_context = self._format_dynamic_learning_context(dynamic_examples, weak_areas, strong_areas)

            # Get mutation proposals from LLM based on weaknesses
            mutation_proposals = await self.mutation_engine.propose_mutations(
                dag_json=path_json,
                objectives=self.config.objectives,
                current_scores=fitness_scores,
                generation=generation,
                context=context.domain_context,
                dynamic_learning_context=dynamic_context
            )

            # Select best proposal based on confidence
            best_proposal = self._select_best_mutation_proposal(mutation_proposals)
            if not best_proposal:
                return None

            # Execute the mutation with self-correction
            mutated_json, mutations = await self.mutation_engine.execute_mutation_plan(
                path_json,
                best_proposal,
                context.domain_context,
                organization_id=context.organization_id,
                enable_self_correction=self.config.enable_self_correction,
            )

            # Create new path DAG from mutated JSON
            mutated_path = await self.mutation_engine.json_to_dag(mutated_json, f"{path.id}_mutated_{generation}")

            # Add metadata to track mutation
            mutated_path.metadata["evolved_from"] = path.id
            mutated_path.metadata["evolution_method"] = "llm_guided_mutation"
            mutated_path.metadata["generation"] = generation
            mutated_path.metadata["mutations"] = mutations
            mutated_path.metadata["mutation_reasoning"] = best_proposal.reasoning
            mutated_path.metadata["expected_improvements"] = best_proposal.expected_improvements

            # Propagate the original extracted path ID
            mutated_path.original_extracted_path_id = path.original_extracted_path_id

            return mutated_path

        except Exception as e:
            logger.warning(f"LLM-guided mutation failed for path {path.id}: {e}")
            return None

    async def _crossover_with_inspiration(
        self,
        path: DecisionDAG,
        context: EvaluationContext,
        organizational_goals: List[Dict[str, Any]] | None = None,
        generation: int = 0,
    ) -> Optional[DecisionDAG]:
        """Use inspiration from evolution history to address path weaknesses."""
        try:
            # Get current fitness to identify weaknesses
            current_fitness = await self.path_evaluator.evaluate_path(path, context, organizational_goals)
            fitness_scores = {"overall_fitness": current_fitness}

            # Identify weak areas (simplified - in full implementation would use detailed scorecard)
            weak_areas = self._identify_weak_areas_from_scores(fitness_scores)
            if not weak_areas:
                return None

            # Find inspiration from evolution history
            inspiration_candidates = []
            if self.evolution_history and weak_areas:
                # Find variants that excel in weak areas
                for variant in self.evolution_history:
                    if not variant.strong_areas:
                        continue
                    # Check if this variant is strong in any weak area
                    is_strong_in_weak_area = any(area in variant.strong_areas for area in weak_areas)
                    if is_strong_in_weak_area:
                        inspiration_candidates.append(variant)

            inspiration_context = self._select_inspiration_variants(weak_areas)
            if not inspiration_context:
                return None

            # Convert path to JSON
            path_json = await self.mutation_engine.dag_to_json(path)

            # Get crossover proposals using inspiration
            crossover_proposals = await self.mutation_engine.propose_mutations(
                dag_json=path_json,
                objectives=self.config.objectives,
                current_scores=fitness_scores,
                generation=generation,
                context=context.domain_context,
                organization_id=context.organization_id,
                inspiration_context=inspiration_context,
            )

            # Select and execute best crossover proposal
            best_proposal = self._select_best_mutation_proposal(crossover_proposals)
            if not best_proposal:
                return None

            # Execute crossover mutation
            crossed_json, mutations = await self.mutation_engine.execute_mutation_plan(
                path_json,
                best_proposal,
                context.domain_context,
                organization_id=context.organization_id,
                enable_self_correction=self.config.enable_self_correction,
            )

            # Create new path DAG
            crossed_path = await self.mutation_engine.json_to_dag(crossed_json, f"{path.id}_crossed_{generation}")

            # Add metadata to track crossover
            crossed_path.metadata["evolved_from"] = path.id
            crossed_path.metadata["evolution_method"] = "inspiration_crossover"
            crossed_path.metadata["generation"] = generation
            crossed_path.metadata["mutations"] = mutations
            crossed_path.metadata["mutation_reasoning"] = best_proposal.reasoning
            crossed_path.metadata["expected_improvements"] = best_proposal.expected_improvements
            crossed_path.metadata["inspiration_sources"] = [v.variant_id for v in inspiration_candidates[:3]] if 'inspiration_candidates' in locals() else []

            # Propagate the original extracted path ID
            crossed_path.original_extracted_path_id = path.original_extracted_path_id

            return crossed_path

        except Exception as e:
            logger.warning(f"Crossover with inspiration failed for path {path.id}: {e}")
            return None

    def _select_best_mutation_proposal(self, proposals: List[MutationProposal]) -> Optional[MutationProposal]:
        """Select the best mutation proposal based on confidence and relevance."""
        if not proposals:
            logger.debug("No mutation proposals generated")
            return None

        # Filter by confidence threshold
        logger.debug(f"Evaluating {len(proposals)} mutation proposals with confidence threshold {self.config.min_confidence_threshold}")
        for p in proposals:
            logger.debug(f"  Proposal: {p.mutation_type} - confidence: {p.confidence} - {p.description}")

        confident_proposals = [p for p in proposals if p.confidence >= self.config.min_confidence_threshold]

        if not confident_proposals:
            logger.debug(f"No proposals met confidence threshold of {self.config.min_confidence_threshold}")
            return None

        logger.debug(f"{len(confident_proposals)} proposals passed confidence threshold")

        # Sort by confidence and return the best
        confident_proposals.sort(key=lambda p: p.confidence, reverse=True)
        return confident_proposals[0]

    def _identify_weak_areas_from_scores(self, fitness_scores: Dict[str, float]) -> List[str]:
        """Identify areas where the path has low fitness scores."""
        weak_areas = []
        threshold = 0.6  # Below this is considered weak

        # Map fitness score keys to area names
        score_mapping = {
            "goal_alignment_score": "goal_alignment",
            "cost_score": "cost_efficiency",
            "timeline_score": "timeline_efficiency",
            "risk_score": "risk_mitigation",
            "complexity_score": "complexity",
        }

        for score_field, area_name in score_mapping.items():
            score = fitness_scores.get(score_field, 0.5)
            if score < threshold:
                weak_areas.append(area_name)

        return weak_areas

    def _select_inspiration_variants(self, weak_areas: List[str]) -> Optional[str]:
        """Select variants from evolution history that are strong in current weak areas."""
        if not self.evolution_history or not weak_areas:
            return None

        # Find variants that excel in weak areas
        inspiration_candidates = []
        for variant in self.evolution_history:
            if not variant.strong_areas:
                continue

            # Check if this variant is strong in any weak area
            is_strong_in_weak_area = any(area in variant.strong_areas for area in weak_areas)

            if is_strong_in_weak_area:
                inspiration_candidates.append(variant)

        if not inspiration_candidates:
            return None

        # Sort by overall fitness and select top 3
        inspiration_candidates.sort(key=lambda v: v.fitness_scores.get("overall_fitness", 0), reverse=True)
        selected = inspiration_candidates[:3]

        # Format inspiration context for LLM
        return self._format_inspiration_context(selected, weak_areas)

    def _select_dynamic_examples(self, weak_areas: List[str], strong_areas: List[str]) -> Dict[str, List[EvolutionVariant]]:
        """Select relevant examples from evolution history for dynamic few-shot learning."""
        examples = {
            "successful": [],
            "unsuccessful": [],
            "similar_weaknesses": []
        }

        if not self.evolution_history:
            return examples

        # Group variants by success/failure
        for variant in self.evolution_history:
            if variant.generation == 0:  # Skip original baseline
                continue

            # Track successful mutations
            if variant.mutation_success and variant.fitness_delta > 0:
                examples["successful"].append(variant)
            # Track unsuccessful mutations if configured
            elif self.config.track_failed_mutations and variant.fitness_delta < 0:
                examples["unsuccessful"].append(variant)

            # Find variants that addressed similar weaknesses
            if any(area in weak_areas for area in variant.strong_areas):
                examples["similar_weaknesses"].append(variant)

        # Limit examples per category
        max_examples = self.config.max_examples_per_category
        for category in examples:
            if len(examples[category]) > max_examples:
                # Sort by relevance (fitness delta for successful, recency for others)
                if category == "successful":
                    examples[category].sort(key=lambda v: v.fitness_delta, reverse=True)
                else:
                    examples[category].sort(key=lambda v: v.generation, reverse=True)
                examples[category] = examples[category][:max_examples]

        return examples

    def _format_dynamic_learning_context(self, examples: Dict[str, List[EvolutionVariant]], weak_areas: List[str], strong_areas: List[str]) -> str:
        """Format dynamic examples into rich context for LLM."""
        context_parts = [
            "📚 LEARNING FROM EVOLUTION HISTORY:",
            f"Current path needs improvement in: {', '.join(weak_areas)}",
            f"Current path is strong in: {', '.join(strong_areas)}",
            "",
        ]

        # Add successful mutations
        if examples["successful"]:
            context_parts.append("✅ SUCCESSFUL MUTATIONS (what worked):")
            for variant in examples["successful"]:
                context_parts.extend([
                    f"- Generation {variant.generation}: {variant.mutation_type}",
                    f"  Fitness improvement: +{variant.fitness_delta:.3f}",
                    f"  Reasoning: {variant.mutation_reasoning}",
                    f"  Strong areas after: {', '.join(variant.strong_areas)}",
                    ""
                ])

        # Add unsuccessful mutations
        if examples["unsuccessful"]:
            context_parts.append("❌ UNSUCCESSFUL MUTATIONS (what to avoid):")
            for variant in examples["unsuccessful"]:
                context_parts.extend([
                    f"- Generation {variant.generation}: {variant.mutation_type}",
                    f"  Fitness decline: {variant.fitness_delta:.3f}",
                    f"  What went wrong: {variant.mutation_reasoning}",
                    f"  Weakened areas: {', '.join(variant.weak_areas)}",
                    ""
                ])

        # Add examples that addressed similar weaknesses
        if examples["similar_weaknesses"]:
            context_parts.append("🎯 MUTATIONS THAT ADDRESSED SIMILAR WEAKNESSES:")
            for variant in examples["similar_weaknesses"]:
                if variant.mutation_success:
                    context_parts.extend([
                        f"- {variant.mutation_type} improved {', '.join(set(variant.strong_areas) & set(weak_areas))}",
                        f"  How: {variant.mutation_reasoning}",
                        f"  Result: +{variant.fitness_delta:.3f} fitness",
                        ""
                    ])

        # Add patterns and insights
        context_parts.extend([
            "📊 PATTERNS OBSERVED:",
            self._extract_evolution_patterns(examples),
            "",
            "💡 Apply these learnings while exploring new approaches for the current weaknesses."
        ])

        return "\n".join(context_parts)

    def _extract_evolution_patterns(self, examples: Dict[str, List[EvolutionVariant]]) -> str:
        """Extract high-level patterns from evolution history."""
        patterns = []

        # Analyze successful mutations
        if examples["successful"]:
            mutation_types = {}
            for v in examples["successful"]:
                mt = v.mutation_type
                if mt not in mutation_types:
                    mutation_types[mt] = 0
                mutation_types[mt] += v.fitness_delta

            best_type = max(mutation_types.items(), key=lambda x: x[1])[0] if mutation_types else "unknown"
            patterns.append(f"- {best_type} mutations tend to be most effective")

        # Analyze failure patterns
        if examples["unsuccessful"]:
            common_mistakes = set()
            for v in examples["unsuccessful"]:
                if v.weak_areas:
                    common_mistakes.update(v.weak_areas)
            if common_mistakes:
                patterns.append(f"- Avoid changes that weaken: {', '.join(list(common_mistakes)[:3])}")

        return "\n".join(patterns) if patterns else "- No clear patterns identified yet (early in evolution)"

    def _format_inspiration_context(self, inspirations: List[EvolutionVariant], weak_areas: List[str]) -> str:
        """Format inspiration variants into context for LLM."""
        context_parts = [
            "🎯 INSPIRATION FROM SUCCESSFUL VARIANTS:",
            f"Current path needs improvement in: {', '.join(weak_areas)}",
            "",
        ]

        for i, inspiration in enumerate(inspirations, 1):
            context_parts.extend(
                [
                    f"--- INSPIRATION {i}: Variant {inspiration.variant_id} ---",
                    f"Fitness: {inspiration.fitness_scores.get('overall_fitness', 0):.3f}",
                    f"Strong areas: {', '.join(inspiration.strong_areas)}",
                    f"Generation: {inspiration.generation}",
                    f"Reasoning: {inspiration.evaluation_reasoning}",
                    "",
                ]
            )

        context_parts.append("🔥 Use these successful patterns to improve the current path.")

        return "\n".join(context_parts)

    def _update_variant_strength_areas(self, variant: EvolutionVariant, fitness_score: float):
        """Update variant strength areas based on fitness score."""
        # Simplified - in full implementation would use detailed fitness breakdown
        if fitness_score > 0.7:
            variant.strong_areas = ["overall_performance"]
        elif fitness_score > 0.5:
            variant.strong_areas = ["adequate_performance"]
        else:
            variant.weak_areas = ["needs_improvement"]

    def _should_mutate(self) -> bool:
        """Determine if mutation should occur based on probability."""
        return random.random() < self.config.mutation_rate

    def _should_crossover(self) -> bool:
        """Determine if crossover should occur based on probability."""
        return random.random() < self.config.crossover_rate

    def _select_survivors(self, population: List[DecisionDAG], fitness_scores: Dict[str, float]) -> List[DecisionDAG]:
        """Select survivors for next generation using fitness-based selection."""
        if not population:
            return []

        # Sort by fitness (higher is better)
        sorted_population = sorted(population, key=lambda p: fitness_scores.get(p.id, 0), reverse=True)

        # Keep top performers (elitism)
        elite_count = max(1, int(len(population) * self.config.elite_preservation))
        survivors = sorted_population[:elite_count]

        # Fill remaining slots with fitness-proportional selection
        remaining_slots = min(self.config.population_size - elite_count, len(sorted_population) - elite_count)
        if remaining_slots > 0:
            remaining_population = sorted_population[elite_count:]
            additional_survivors = self._fitness_proportional_selection(
                remaining_population, fitness_scores, remaining_slots
            )
            survivors.extend(additional_survivors)

        return survivors

    def _fitness_proportional_selection(
        self, population: List[DecisionDAG], fitness_scores: Dict[str, float], count: int
    ) -> List[DecisionDAG]:
        """Select individuals proportional to their fitness."""
        if not population:
            return []

        # Calculate selection probabilities
        total_fitness = sum(fitness_scores.get(p.id, 0) for p in population)
        if total_fitness <= 0:
            return population[:count]  # Fallback to top-k if no positive fitness

        selected = []
        available_population = population.copy()

        for _ in range(count):
            if not available_population:
                break

            # Roulette wheel selection
            target = random.random() * total_fitness
            cumulative = 0

            for i, path in enumerate(available_population):
                cumulative += fitness_scores.get(path.id, 0)
                if cumulative >= target:
                    selected.append(path)
                    available_population.pop(i)
                    total_fitness -= fitness_scores.get(path.id, 0)
                    break

        return selected

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the evolution process."""
        if not self.evolution_history:
            return {"total_variants": 0}

        total_variants = len(self.evolution_history)
        fitness_scores = [v.fitness_scores.get("overall_fitness", 0) for v in self.evolution_history]

        # Calculate successful vs unsuccessful mutations
        successful_mutations = len([v for v in self.evolution_history if v.mutation_success])
        unsuccessful_mutations = len([v for v in self.evolution_history if not v.mutation_success and v.generation > 0])

        return {
            "total_variants": total_variants,
            "avg_fitness": sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0,
            "best_fitness": max(fitness_scores) if fitness_scores else 0,
            "generations_run": max(v.generation for v in self.evolution_history) if self.evolution_history else 0,
            "warmup_generations": int(self.config.max_generations * self.config.warmup_generations_ratio),
            "successful_mutations": successful_mutations,
            "unsuccessful_mutations": unsuccessful_mutations,
            "mutation_success_rate": successful_mutations / (successful_mutations + unsuccessful_mutations) if (successful_mutations + unsuccessful_mutations) > 0 else 0,
            "inspiration_usage": len([v for v in self.evolution_history if v.inspiration_sources]),
            "mutation_types": list(
                set(mut.get("mutation_type", "unknown") for v in self.evolution_history for mut in v.mutations)
            ),
        }
