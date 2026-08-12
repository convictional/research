from openai import AsyncOpenAI

EMBEDDING_DIMENSION = 1536  # OpenAI's ada-002 dimension
client = AsyncOpenAI()


async def aembed_query(query: str) -> list[float]:
    """Get embeddings for a query using OpenAI's embeddings API."""
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
        dimensions=EMBEDDING_DIMENSION,
    )
    return response.data[0].embedding
