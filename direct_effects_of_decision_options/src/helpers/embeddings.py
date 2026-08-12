import faiss
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
import numpy as np

from common.async_helper import limited_task, wrap_task_progress_bar
from ..settings import settings
from common.embeddings import aembed, aembed_query


async_openai_client = AsyncOpenAI(
    api_key=settings.openai_api_key.get_secret_value(), organization=settings.openai_organization
)


async def aembed_to_faiss(
    texts_to_embed: list[str],
    index: faiss.IndexFlatL2,
    embedding_model: str,
    embedding_dim: int,
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> faiss.IndexFlatL2:
    """
    Embeds the texts to a FAISS index.
    Here we do the vector embedding asynchronously.
    """
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        limited_task(
            aembed(
                async_openai_client=async_openai_client,
                text=text_to_embed,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
            ),
            semaphore,
            delay_between_tasks,
        )
        for text_to_embed in texts_to_embed
    ]

    pbar = tqdm(total=len(tasks), desc="Getting embeddings for FAISS indexing...")
    wrapped_tasks = [wrap_task_progress_bar(task, pbar) for task in tasks]
    # Use asyncio.gather to preserve the order of the embeddings
    embedding_vectors = await asyncio.gather(*wrapped_tasks)
    pbar.close()

    for embedding_vector in embedding_vectors:
        index.add(np.array([embedding_vector]).astype("float32"))

    return index


async def aquery_faiss_index(
    query_text: str,
    index: faiss.IndexFlatL2,
    top_k: int,
    embedding_model: str,
    embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Queries a FAISS index with a query and returns the closest k vectors."""
    query_vector = await aembed_query(
        async_openai_client=async_openai_client,
        text=query_text,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    query_vector = query_vector.reshape(1, -1)  # Ensure the query vector has the correct shape
    distances, indices = index.search(query_vector.astype("float32"), top_k)
    return distances[0].tolist(), indices[0].tolist()
