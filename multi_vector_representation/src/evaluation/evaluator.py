import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from src.evaluation.annotations import Annotation, AnnotationManager
from src.evaluation.pooling import QueryPool, ResultPooler
from src.evaluation.metrics import mean_reciprocal_rank, recall_at_k, precision_at_k, ndcg_at_k


class SystemMetrics(BaseModel):
    mrr: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    ndcg_at_10: float
    queries_with_relevant: int
    total_queries: int
    coverage: float


class ComparisonReport(BaseModel):
    systems: dict[str, SystemMetrics]
    pairwise_comparisons: list[dict]


class SystemEvaluator:
    def __init__(self, pools_path: Path, annotations_path: Path, classifications_path: Path | None = None):
        self.pools = ResultPooler.load_pools(pools_path)
        self.annotation_manager = AnnotationManager(annotations_path)

        self.classifications = None
        if classifications_path and classifications_path.exists():
            from src.evaluation.query_classifier import QueryClassifier

            self.classifications = QueryClassifier.load_classifications(classifications_path)
            self.query_to_classification = {c.query_id: c.classification for c in self.classifications}
        else:
            self.query_to_classification = {}

        self.query_to_annotations = self._group_annotations_by_query()

    def _group_annotations_by_query(self) -> dict[str, dict[str, int]]:
        grouped = defaultdict(dict)
        for annotation in self.annotation_manager.annotations:
            grouped[annotation.query_id][annotation.doc_id] = annotation.relevance
        return dict(grouped)

    def evaluate(self, classification_filter: str | None = None) -> ComparisonReport:
        all_systems = {"colbert_local", "openai_embedding", "production_hybrid", "production_reranked"}

        found_systems = set()
        for pool in self.pools:
            for doc in pool.pooled_docs:
                found_systems.update(doc.retrieved_by)

        system_names = sorted(list(all_systems & found_systems))
        system_data = {name: {"rankings": [], "relevances": [], "ndcg_scores": []} for name in system_names}

        annotated_queries = set(self.query_to_annotations.keys())

        for pool in self.pools:
            if pool.query_id not in annotated_queries:
                continue

            if classification_filter and pool.query_id in self.query_to_classification:
                if self.query_to_classification[pool.query_id] != classification_filter:
                    continue

            annotations = self.query_to_annotations[pool.query_id]

            for system_name in system_names:
                system_docs = [doc for doc in pool.pooled_docs if system_name in doc.retrieved_by]

                relevant = set()
                for doc in system_docs:
                    if doc.doc_id in annotations and annotations[doc.doc_id] >= 2:
                        relevant.add(doc.doc_id)

                ranking = [doc.doc_id for doc in system_docs]
                graded_relevances = [annotations.get(doc.doc_id, 0) for doc in system_docs]
                ndcg = self._compute_ndcg(graded_relevances, 10)

                system_data[system_name]["rankings"].append(ranking)
                system_data[system_name]["relevances"].append(relevant)
                system_data[system_name]["ndcg_scores"].append(ndcg)

        system_metrics = {}
        for system_name in system_names:
            data = system_data[system_name]
            system_metrics[system_name] = self._calculate_system_metrics(
                data["rankings"], data["relevances"], data["ndcg_scores"], system_name
            )

        pairwise_comparisons = []
        for i, system1 in enumerate(system_names):
            for system2 in system_names[i + 1 :]:
                comparison = self._compare_systems(
                    system1,
                    system2,
                    system_data[system1]["ndcg_scores"],
                    system_data[system2]["ndcg_scores"],
                )
                pairwise_comparisons.append(comparison)

        return ComparisonReport(systems=system_metrics, pairwise_comparisons=pairwise_comparisons)

    def _calculate_system_metrics(
        self, rankings: list[list[str]], relevances: list[set[str]], ndcg_scores: list[float], system_name: str
    ) -> SystemMetrics:
        mrr = mean_reciprocal_rank(rankings, relevances)
        recall_1 = recall_at_k(rankings, relevances, k=1)
        recall_5 = recall_at_k(rankings, relevances, k=5)
        recall_10 = recall_at_k(rankings, relevances, k=10)
        precision_1 = precision_at_k(rankings, relevances, k=1)
        precision_5 = precision_at_k(rankings, relevances, k=5)
        precision_10 = precision_at_k(rankings, relevances, k=10)

        queries_with_relevant = sum(1 for rel_set in relevances if len(rel_set) > 0)
        total_queries = len(rankings)
        coverage = queries_with_relevant / total_queries if total_queries > 0 else 0.0

        avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

        return SystemMetrics(
            mrr=mrr,
            recall_at_1=recall_1,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            precision_at_1=precision_1,
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            ndcg_at_10=avg_ndcg,
            queries_with_relevant=queries_with_relevant,
            total_queries=total_queries,
            coverage=coverage,
        )

    def _compute_ndcg(self, relevances: list[int], k: int) -> float:
        relevances = relevances[:k]
        if not relevances:
            return 0.0

        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))

        ideal_relevances = sorted(relevances, reverse=True)
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))

        return float(dcg / idcg) if idcg > 0 else 0.0

    def _compare_systems(
        self, system1: str, system2: str, scores1: list[float], scores2: list[float]
    ) -> dict:
        from scipy import stats

        if len(scores1) != len(scores2) or len(scores1) < 2:
            return {"system1": system1, "system2": system2, "test": "insufficient_data"}

        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        mean1 = float(np.mean(scores1))
        mean2 = float(np.mean(scores2))
        improvement = ((mean1 - mean2) / mean2 * 100) if mean2 > 0 else 0.0

        return {
            "system1": system1,
            "system2": system2,
            "test": "paired_t_test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "system1_mean_ndcg": mean1,
            "system2_mean_ndcg": mean2,
            "improvement_pct": improvement,
        }
