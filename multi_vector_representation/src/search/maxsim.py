import torch
import numpy as np


def maxsim_score(query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> float:
    """
    Compute ColBERT MaxSim score between query and document token embeddings.

    MaxSim(Q, D) = Σ_q max_d (q · d) for all query tokens q

    Args:
        query_embeddings: Tensor of shape (num_query_tokens, embedding_dim)
        doc_embeddings: Tensor of shape (num_doc_tokens, embedding_dim)

    Returns:
        Relevance score (higher is better)
    """
    similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T)
    max_similarities = similarity_matrix.max(dim=1).values
    return max_similarities.sum().item()


def maxsim_scores_batch(
    query_embeddings: torch.Tensor, doc_embeddings_list: list[torch.Tensor]
) -> list[float]:
    """
    Compute MaxSim scores between a query and multiple documents.

    Args:
        query_embeddings: Tensor of shape (num_query_tokens, embedding_dim)
        doc_embeddings_list: List of document embedding tensors

    Returns:
        List of relevance scores
    """
    scores = []
    for doc_embeddings in doc_embeddings_list:
        score = maxsim_score(query_embeddings, doc_embeddings)
        scores.append(score)
    return scores


def maxsim_score_numpy(query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> float:
    """
    NumPy implementation of MaxSim for compatibility with non-torch code.

    Args:
        query_embeddings: Array of shape (num_query_tokens, embedding_dim)
        doc_embeddings: Array of shape (num_doc_tokens, embedding_dim)

    Returns:
        Relevance score (higher is better)
    """
    similarity_matrix = np.matmul(query_embeddings, doc_embeddings.T)
    max_similarities = similarity_matrix.max(axis=1)
    return float(max_similarities.sum())
