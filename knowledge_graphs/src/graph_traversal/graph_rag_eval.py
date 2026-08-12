import asyncio
from typing import List, Dict
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from ..config.experiment_settings import settings
from ..config.cypher_queries import PROJECT_CURRENT_GRAPH_FOR_GDS
from ..knowledge_graph import KnowledgeGraph
from ..graph_traversal.traverse_graph import traverse_graph, get_and_init_current_graph, calculate_pagerank
from ..utils.neo4j_graph_functions import project_gds_graph
from ..utils.instruct_llm import instruct_llm
from ..config.prompts import RANK_GRAPH_TRAVERSAL_RESULTS_SYSTEM_PROMPT


class GraphRagResults(BaseModel):
    results: List[str]


def rank_results(user_query: str, graph_traversal_results: Dict) -> List[str]:
    # Call instruct_llm to rank the results
    system_prompt = RANK_GRAPH_TRAVERSAL_RESULTS_SYSTEM_PROMPT
    user_prompt = f"User query: {user_query}\nResults: {graph_traversal_results}"

    response, _ = instruct_llm(
        system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0, response_model=GraphRagResults
    )

    # Assuming the response contains a ranked list of results
    return response.results


class GraphRAG:
    def __init__(self, name: str = "GraphRAG"):
        self.executor = ThreadPoolExecutor()
        self.name = name

    async def setup(self) -> None:
        # These project an in-neo4j-memory GDS graph from the current graph
        self.graph = await self.run_sync(get_and_init_current_graph)
        await self.run_sync(project_gds_graph, "testGraph", PROJECT_CURRENT_GRAPH_FOR_GDS)
        await self.run_sync(calculate_pagerank)

    async def hyperparameters(self) -> dict:
        # Example hyperparameters, modify as needed
        return {
            "model": settings.llm_model,
            "tools": "One hop neighbours, All shortest paths to target categories, Induced subgraph",
            "uses_llm_ranking": True,
            "uses_llm_keyword_for_vss": True,
        }

    async def find_knowledge(self, query: str) -> List[str]:
        graph = self.graph
        try:
            traversal_results = await self.run_sync(self._synchronous_traverse_graph, query, graph)
            ranked_results = await self.run_sync(rank_results, query, traversal_results)
            return ranked_results
        except Exception as e:
            # Handle or log the error appropriately
            print(f"Error in find_knowledge: {e}")
            raise

    async def run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)

    async def _synchronous_traverse_graph(self, query: str, graph: KnowledgeGraph) -> Dict:
        history = []  # Replace with actual history if available
        traversal_results = await traverse_graph(query, graph, history)
        return traversal_results
