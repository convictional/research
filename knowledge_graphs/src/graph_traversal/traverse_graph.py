import asyncio
from typing import Any, List, Dict

from .traversal_tools import (
    TraversalToolInputs,
    ReplyToUser,
)
from ..config.prompts import (
    CATEGORIES,
    GRAPH_TRAVERSAL_SYSTEM_PROMPT,
    KEYWORD_GENERATION_PROMPT,
    SUMMARIZE_RESULTS_SYSTEM_PROMPT,
)
from ..config.cypher_queries import CALC_PAGERANK
from ..knowledge_graph import KnowledgeGraph
from ..utils.neo4j_graph_functions import get_current_neo4j_graph, get_neo4j_session
from ..utils.instruct_llm import instruct_llm


async def aget_and_init_current_graph() -> KnowledgeGraph:
    """Get the current state of the knowledge graph."""
    print("Loading current graph from neo4j into memory...")
    current_graph = get_current_neo4j_graph()
    print(f"Loaded {len(current_graph.nodes)} nodes and {len(current_graph.edges)} edges.")
    print("Initializing the Faiss index...")
    await current_graph.ainit_or_update_faiss_index()
    print("Faiss index initialized.")
    return current_graph


def get_and_init_current_graph() -> KnowledgeGraph:
    """Get the current state of the knowledge graph."""
    print("Loading current graph from neo4j into memory...")
    current_graph = get_current_neo4j_graph()
    print(f"Loaded {len(current_graph.nodes)} nodes and {len(current_graph.edges)} edges.")
    print("Initializing the Faiss index...")
    asyncio.run(current_graph.ainit_or_update_faiss_index())
    print("Faiss index initialized.")
    return current_graph


def get_traversal_tool_inputs(
    user_query: str, induced_subgraph: KnowledgeGraph, graph_stats: Dict, history: List[Dict] | None = None
) -> TraversalToolInputs:
    """Get the LLM's selection of the traversal tool to use."""

    system_prompt = GRAPH_TRAVERSAL_SYSTEM_PROMPT.format(
        diameter=graph_stats["diameter"],
        total_nodes=graph_stats["num_nodes"],
        total_edges=graph_stats["num_edges"],
        num_connected_components=graph_stats["num_components"],
        node_categories=CATEGORIES,
        subgraph=induced_subgraph,
    )

    response, _ = instruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_query,
        temperature=0.0,
        response_model=TraversalToolInputs,
        few_shot=history,
    )

    return response


def get_traversal_tool_results(
    tool: TraversalToolInputs, current_graph: KnowledgeGraph, user_query: str, conversation_history: List[Dict]
) -> Any:
    """Execute all the available traversal tools and return the results along with LLM summary."""
    subgraph = tool.ISG.get_subgraph(current_graph)
    paths = tool.ASP.get_paths()
    neighbours = tool.OHN.get_neighbours()
    results = {
        "Induced Subgraph Tool": subgraph,
        "All Shortest Paths Tool": paths,
        "Single Node One-Hop Neighbors Tool": neighbours,
    }
    summary = summarize_results(results, user_query, conversation_history)
    return {
        "results": results,
        "summary": summary,
    }


def get_vss_keywords(user_query: str, conversation_history: List[Dict]) -> str:
    """Calls the LLM to use conversation history and the user query to generate keywords for VSS against nodes."""
    system_prompt = KEYWORD_GENERATION_PROMPT.format(conversation_history=str(conversation_history))
    response, _ = instruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_query,
        temperature=0.0,
        response_model=ReplyToUser,
    )
    return response.response


def summarize_results(tool_results: Any, user_query: str, conversation_history: List[Dict]) -> str:
    """Calls the LLM to summarize the results of the traversal tool."""
    system_prompt = SUMMARIZE_RESULTS_SYSTEM_PROMPT.format(
        user_query=user_query,
        conversation_history=str(conversation_history),
    )
    response, _ = instruct_llm(
        system_prompt=system_prompt,
        user_prompt=f"Results, or none if none are found: {tool_results}",
        temperature=0.0,
        response_model=ReplyToUser,
    )
    return response.response


def calculate_pagerank() -> Any:
    query = CALC_PAGERANK
    with get_neo4j_session() as session:
        session.run(query)
        print("PageRank calculated.")


async def traverse_graph(user_query: str, current_graph: KnowledgeGraph, history: List[Dict] | None = None) -> Any:
    """Traverse the knowledge graph based on the user query."""
    graph_stats = current_graph.get_graph_metrics()
    vss_keywords = get_vss_keywords(user_query, history)
    print(f"Searching graph with keywords: {vss_keywords}")
    similar_subgraph = await current_graph.get_similar_subgraph(vss_keywords)
    print(f"Found {len(similar_subgraph.nodes)} similar nodes in the graph")
    tool_inputs = get_traversal_tool_inputs(user_query, similar_subgraph, graph_stats, history)
    print(f"Tool inputs: {tool_inputs.dict()}")
    tool_results = get_traversal_tool_results(tool_inputs, similar_subgraph, user_query, history)
    return tool_results
