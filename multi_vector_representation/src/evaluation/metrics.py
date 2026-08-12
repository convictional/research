import time
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from src.models.content import SearchResult


@dataclass
class EvaluationMetrics:
    mrr: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    avg_latency_ms: float
    total_queries: int


def mean_reciprocal_rank(results: list[list[UUID]], relevant: list[set[UUID]]) -> float:
    """
    Compute Mean Reciprocal Rank across multiple queries.

    Args:
        results: List of ranked result lists (by content_id) for each query
        relevant: List of sets of relevant content_ids for each query

    Returns:
        MRR score
    """
    reciprocal_ranks = []

    for result_list, relevant_set in zip(results, relevant):
        for rank, content_id in enumerate(result_list, 1):
            if content_id in relevant_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def recall_at_k(results: list[list[UUID]], relevant: list[set[UUID]], k: int) -> float:
    """
    Compute Recall@K across multiple queries.

    Args:
        results: List of ranked result lists (by content_id) for each query
        relevant: List of sets of relevant content_ids for each query
        k: Cutoff rank

    Returns:
        Average recall@k
    """
    recalls = []

    for result_list, relevant_set in zip(results, relevant):
        top_k = set(result_list[:k])
        if len(relevant_set) > 0:
            recall = len(top_k & relevant_set) / len(relevant_set)
            recalls.append(recall)

    return float(np.mean(recalls)) if recalls else 0.0


def precision_at_k(results: list[list[UUID]], relevant: list[set[UUID]], k: int) -> float:
    """
    Compute Precision@K across multiple queries.

    Args:
        results: List of ranked result lists (by content_id) for each query
        relevant: List of sets of relevant content_ids for each query
        k: Cutoff rank

    Returns:
        Average precision@k
    """
    precisions = []

    for result_list, relevant_set in zip(results, relevant):
        top_k = result_list[:k]
        if len(top_k) > 0:
            precision = len(set(top_k) & relevant_set) / len(top_k)
            precisions.append(precision)

    return float(np.mean(precisions)) if precisions else 0.0


def dcg_at_k(relevances: list[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at rank k.

    Args:
        relevances: Binary relevance scores (1 for relevant, 0 for not)
        k: Cutoff rank

    Returns:
        DCG@k score
    """
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(results: list[list[UUID]], relevant: list[set[UUID]], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank k.

    Args:
        results: List of ranked result lists (by content_id) for each query
        relevant: List of sets of relevant content_ids for each query
        k: Cutoff rank

    Returns:
        Average NDCG@k
    """
    ndcg_scores = []

    for result_list, relevant_set in zip(results, relevant):
        relevances = [1.0 if cid in relevant_set else 0.0 for cid in result_list[:k]]

        dcg = dcg_at_k(relevances, k)

        ideal_relevances = [1.0] * min(len(relevant_set), k)
        idcg = dcg_at_k(ideal_relevances, k)

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


class SearchEvaluator:
    def __init__(self):
        self.latencies: list[float] = []

    async def evaluate_search(
        self,
        search_engine,
        queries: list[str],
        relevant_docs: list[set[UUID]],
        top_k: int = 10,
    ) -> EvaluationMetrics:
        """
        Evaluate a search engine using a set of queries and relevance judgments.

        Args:
            search_engine: Search engine instance with a search() method
            queries: List of query strings
            relevant_docs: List of sets of relevant document IDs for each query
            top_k: Number of results to retrieve

        Returns:
            Evaluation metrics
        """
        all_results = []
        latencies = []

        for query_text in queries:
            start_time = time.perf_counter()

            from src.models.content import SearchQuery

            query = SearchQuery(text=query_text, top_k=top_k)
            results = await search_engine.search(query)

            latency = (time.perf_counter() - start_time) * 1000
            latencies.append(latency)

            result_ids = [r.content_id for r in results]
            all_results.append(result_ids)

        mrr = mean_reciprocal_rank(all_results, relevant_docs)
        recall_1 = recall_at_k(all_results, relevant_docs, k=1)
        recall_5 = recall_at_k(all_results, relevant_docs, k=5)
        recall_10 = recall_at_k(all_results, relevant_docs, k=10)
        ndcg_10 = ndcg_at_k(all_results, relevant_docs, k=10)
        avg_latency = float(np.mean(latencies))

        return EvaluationMetrics(
            mrr=mrr,
            recall_at_1=recall_1,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            ndcg_at_10=ndcg_10,
            avg_latency_ms=avg_latency,
            total_queries=len(queries),
        )
