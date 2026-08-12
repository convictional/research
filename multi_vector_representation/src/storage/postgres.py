import asyncpg
import torch
import numpy as np
from uuid import UUID


class PostgresStorage:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.connection_string, min_size=1, max_size=10)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def update_token_embeddings(
        self, content_id: UUID, token_embeddings: torch.Tensor | np.ndarray
    ) -> None:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        if isinstance(token_embeddings, torch.Tensor):
            token_embeddings = token_embeddings.cpu().numpy()

        embeddings_list = self._format_pgvector_array(token_embeddings)

        query = """
            UPDATE content
            SET token_embeddings = $1::vector(1024)[]
            WHERE id = $2
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, embeddings_list, content_id)

    async def bulk_update_token_embeddings(
        self, updates: list[tuple[UUID, torch.Tensor | np.ndarray]]
    ) -> None:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for content_id, token_embeddings in updates:
                    if isinstance(token_embeddings, torch.Tensor):
                        token_embeddings = token_embeddings.cpu().numpy()

                    embeddings_list = self._format_pgvector_array(token_embeddings)

                    await conn.execute(
                        "UPDATE content SET token_embeddings = $1::vector(1024)[] WHERE id = $2",
                        embeddings_list,
                        content_id,
                    )

    async def get_all_token_embeddings(self) -> dict[UUID, list[list[float]]]:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        query = """
            SELECT id, token_embeddings
            FROM content
            WHERE token_embeddings IS NOT NULL
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)

        result = {}
        for row in rows:
            content_id = row["id"]
            embeddings = [list(vec) for vec in row["token_embeddings"]]
            result[content_id] = embeddings

        return result

    def _format_pgvector_array(self, embeddings: np.ndarray) -> list[str]:
        """Format numpy array as list of pgvector strings for asyncpg."""
        vectors = []
        for vec in embeddings:
            vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
            vectors.append(vec_str)
        return vectors
