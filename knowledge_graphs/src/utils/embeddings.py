import asyncio
from itertools import batched
import tiktoken
import faiss
import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from typing import List

from ..config.experiment_settings import settings
from .async_helper import limited_task, wrap_task_progress_bar
from .tokens import get_tokens_from_text_batch

openai_client = OpenAI(api_key=settings.openai_api_key.get_secret_value(), organization=settings.openai_organization)
async_openai_client = AsyncOpenAI(
    api_key=settings.openai_api_key.get_secret_value(), organization=settings.openai_organization
)

tokenizer = tiktoken.encoding_for_model(settings.embedding_model)


def embed(
    text: str, embedding_model: str = settings.embedding_model, embedding_dim: int = settings.embedding_dimension
) -> np.ndarray:
    """Embeds a text using the OpenAI API."""
    response = openai_client.embeddings.create(input=[text], model=embedding_model, dimensions=embedding_dim)

    data = response.data[0]
    if not data:
        raise ValueError("No embedding data returned from OpenAI")
    return np.array(data.embedding)


def embed_to_faiss(
    text_to_embed: str,
    index: faiss.IndexFlatL2 = None,
    embedding_model: str = settings.faiss_embedding_model,
    embedding_dim: int = settings.faiss_embedding_dimension,
) -> faiss.IndexFlatL2:
    """Embeds the text to a FAISS index so we can quickly extend the index during inference loops.
    If no index is provided, a new one is created using the first embedding."""
    embedding_vector = embed(text=text_to_embed, embedding_model=embedding_model, embedding_dim=embedding_dim)

    if index is None:
        index = faiss.IndexFlatL2(len(embedding_vector))

    index.add(np.array([embedding_vector]).astype("float32"))

    return index


async def aembed_to_faiss(
    texts_to_embed: list[str],
    index: faiss.IndexFlatL2 = None,
    embedding_model: str = settings.faiss_embedding_model,
    embedding_dim: int = settings.faiss_embedding_dimension,
    max_concurrent_tasks: int = 100,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> faiss.IndexFlatL2:
    """
    Embeds the text to a FAISS index so we can quickly extend the index during inference loops.
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

    if index is None:
        index = faiss.IndexFlatL2(embedding_dim)

    for embedding_vector in embedding_vectors:
        index.add(np.array([embedding_vector]).astype("float32"))

    return index


async def query_faiss_index(
    index: faiss.IndexFlatL2,
    query_text: str,
    k: int = 10,
    embedding_model: str = settings.faiss_embedding_model,
    embedding_dim: int = settings.faiss_embedding_dimension,
) -> tuple[np.ndarray, np.ndarray]:
    """Queries a FAISS index with a query and returns the closest k vectors."""
    query_vector = await aembed_query(text=query_text, embedding_model=embedding_model, embedding_dim=embedding_dim)
    query_vector = query_vector.reshape(1, -1)  # Ensure the query vector has the correct shape
    distances, indices = index.search(query_vector.astype("float32"), k)
    return distances, indices[0].tolist()


async def aembed_query(text: str, embedding_model: str, embedding_dim: int) -> np.ndarray:
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
        return combined_embedding


async def aembed(
    text: str, embedding_model: str = settings.embedding_model, embedding_dim: int = settings.embedding_dimension
) -> np.ndarray:
    response = await async_openai_client.embeddings.create(
        input=[text], model=embedding_model, dimensions=embedding_dim
    )

    data = response.data[0]
    if not data:
        raise ValueError("No embedding data returned from OpenAI")
    return np.array(data.embedding)


async def aembed_chunk(chunk: list[int]):
    try:
        vector = await aembed(chunk)
    except Exception as e:
        print(f"Error embedding content chunk: {e}")
        print(f"Offending content chunk: {chunk}")
    return vector


def batched_by_max_tokens(tokenized_texts: list[list[int]], max_tokens=8192):
    token_count = 0
    current_batch = []
    for text in tokenized_texts:
        token_count += len(text)
        if token_count > max_tokens:
            yield current_batch
            current_batch = [text]
            token_count = len(text)
        else:
            current_batch.append(text)
    yield current_batch


def embed_batch(
    texts: list[str],
    embedding_model: str = settings.embedding_model,
    embedding_dim: int = settings.embedding_dimension,
):
    tokenized_texts: list[list[int]] = tokenizer.encode_batch(texts)

    all_embeddings = []

    for batch in batched_by_max_tokens(tokenized_texts):
        response = openai_client.embeddings.create(input=batch, model=embedding_model, dimensions=embedding_dim)

        embeddings = [data.embedding for data in response.data]
        if not embeddings:
            raise ValueError("No embedding data returned from OpenAI")

        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)


async def aembed_batch(
    texts: list[str],
    embedding_model: str = settings.embedding_model,
    embedding_dim: int = settings.embedding_dimension,
):
    tokenized_texts: list[list[int]] = tokenizer.encode_batch(texts)

    tasks = []
    for batch in batched_by_max_tokens(tokenized_texts):
        task = async_openai_client.embeddings.create(input=batch, model=embedding_model, dimensions=embedding_dim)
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    all_embeddings = np.vstack([data.embedding for result in results for data in result.data])

    return all_embeddings


async def split_and_embed_chunks(text_chunks: List[dict], max_tokens: int = 8192) -> List[np.ndarray]:
    text_chunks = [chunk["content"] for chunk in text_chunks]
    tokenized_chunks = get_tokens_from_text_batch(text_chunks)

    tasks = [aembed_chunk(batch) for chunk in tokenized_chunks for batch in batched(chunk, max_tokens)]

    pbar = tqdm(total=len(tasks), desc="Embedding text chunks...")
    wrapped_tasks = [wrap_task_progress_bar(task, pbar) for task in tasks]
    # Use asyncio.gather to preserve the order of the embeddings
    all_vectors = list(await asyncio.gather(*wrapped_tasks))
    pbar.close()

    return all_vectors
