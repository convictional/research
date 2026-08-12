import faiss
import asyncio
import openai
from openai import AsyncOpenAI
from tqdm import tqdm
import numpy as np

from .async_helper import limited_task, wrap_task_progress_bar
from ..settings import settings, logger


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
            aembed(text=text_to_embed, embedding_model=embedding_model, embedding_dim=embedding_dim),
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


async def aembed(
    text: str,
    embedding_model: str = settings.embedding_model,
    embedding_dim: int = settings.embedding_dimension,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> np.ndarray:
    for attempt in range(max_retries):
        try:
            response = await async_openai_client.embeddings.create(
                input=[text], model=embedding_model, dimensions=embedding_dim
            )

            data = response.data[0]
            if not data:
                raise ValueError("No embedding data returned from OpenAI")
            return np.array(data.embedding)
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"Failed to get embedding after {max_retries} attempts: {str(e)}")
                return np.zeros(embedding_dim)
            else:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff


async def aquery_faiss_index(
    query_text: str,
    index: faiss.IndexFlatL2,
    top_k: int,
    embedding_model: str,
    embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Queries a FAISS index with a query and returns the closest k vectors."""
    query_vector = await aembed_query(text=query_text, embedding_model=embedding_model, embedding_dim=embedding_dim)
    query_vector = query_vector.reshape(1, -1)  # Ensure the query vector has the correct shape
    distances, indices = index.search(query_vector.astype("float32"), top_k)
    return distances[0].tolist(), indices[0].tolist()


async def aembed_query(text: str, embedding_model: str, embedding_dim: int) -> list[float]:
    try:
        return await aembed(text, embedding_model, embedding_dim)
    except openai.BadRequestError:
        print("Input is too long to embed, recursively splitting and averaging embeddings...")

        # Split the input into smaller chunks recursively
        half_length = len(text) // 2
        first_half = text[:half_length]
        second_half = text[half_length:]

        embedding1 = await aembed_query(first_half, embedding_model, embedding_dim)
        embedding2 = await aembed_query(second_half, embedding_model, embedding_dim)

        # Combine the embeddings by averaging them
        combined_embedding = np.mean([embedding1, embedding2], axis=0)
        return combined_embedding.tolist()
