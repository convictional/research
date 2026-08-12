from openai import AsyncOpenAI
import numpy as np
import openai


async def aembed(async_openai_client: AsyncOpenAI, text: str, embedding_model: str, embedding_dim: int) -> list[float]:
    """
    This function embeds the text using the OpenAI API.
    Note, this function needs an async OpenAI client.
    """
    response = await async_openai_client.embeddings.create(
        input=[text], model=embedding_model, dimensions=embedding_dim
    )

    data = response.data[0]
    if not data:
        raise ValueError("No embedding data returned from OpenAI")
    return data.embedding


async def aembed_query(
    async_openai_client: AsyncOpenAI, text: str, embedding_model: str, embedding_dim: int
) -> list[float]:
    """
    This function embeds the text using the OpenAI API.
    Note, this function needs an async OpenAI client.

    If the input is too long to embed, it will recursively split the input into smaller chunks and average the embeddings.
    """
    try:
        return await aembed(async_openai_client, text, embedding_model, embedding_dim)
    except openai.BadRequestError:
        print("Input is too long to embed, recursively splitting and averaging embeddings...")

        # Split the input into smaller chunks recursively
        half_length = len(text) // 2
        first_half = text[:half_length]
        second_half = text[half_length:]

        embedding1 = await aembed_query(async_openai_client, first_half, embedding_model, embedding_dim)
        embedding2 = await aembed_query(async_openai_client, second_half, embedding_model, embedding_dim)

        # Combine the embeddings by averaging them
        combined_embedding = np.mean([embedding1, embedding2], axis=0)
        return combined_embedding.tolist()


def cosine_similarity(first: list[float], second: list[float]) -> float:
    return np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
