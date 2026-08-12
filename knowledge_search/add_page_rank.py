import json
from collections import defaultdict

import numpy as np
import tortoise.transactions
from experiments.knowledge_search.knowledge import TABLE_NAME, setup_db


async def fetch_data(connection) -> list[dict]:
    return await connection.execute_query_dict(f"SELECT id, keywords, named_entities FROM {TABLE_NAME}")


def build_adjacency_matrix(data: list[dict]) -> tuple[np.ndarray, dict[str, int]]:
    node_index: dict[str, int] = {entry["id"]: idx for idx, entry in enumerate(data)}
    size: int = len(data)
    M: np.ndarray = np.zeros((size, size))

    attribute_map: dict[str, set[str]] = defaultdict(set)
    for entry in data:
        id = entry["id"]
        for column in ["keywords", "named_entities"]:
            if entry[column]:
                for item in entry[column]:
                    attribute_map[f"{column}:{item}"].add(id)

    for ids in attribute_map.values():
        if len(ids) > 1:
            indices: list[int] = [node_index[id] for id in ids]
            for i in indices:
                for j in indices:
                    if i != j:
                        M[i][j] += 1

    column_sums = np.sum(M, axis=0)
    column_sums[column_sums == 0] = 1
    M /= column_sums

    return M, node_index


def compute_pagerank(M: np.ndarray, num_iterations: int = 100, d: float = 0.85) -> np.ndarray:
    N: int = M.shape[0]
    v: np.ndarray = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    teleport: float = (1 - d) / N
    for i in range(num_iterations):
        v = d * np.matmul(M, v) + teleport
    return v


async def update_page_rank(connection, node_index: dict[str, int], pageranks: np.ndarray) -> None:
    async with tortoise.transactions.in_transaction("default"):
        for node_id, idx in node_index.items():
            rank: float = pageranks[idx, 0]
            await connection.execute_query(f"UPDATE {TABLE_NAME} SET page_rank = $1 WHERE id = $2;", [rank, node_id])
    print("PageRank updated for all entries.")


async def add_page_rank(connection):
    data: list[dict] = await fetch_data(connection)

    for entry in data:
        for column in ["keywords", "named_entities"]:
            entry[column] = json.loads(entry[column])

    M, node_index = build_adjacency_matrix(data)
    pageranks: np.ndarray = compute_pagerank(M)

    await update_page_rank(connection, node_index, pageranks)


async def main() -> None:
    connection = await setup_db()
    await add_page_rank(connection)
