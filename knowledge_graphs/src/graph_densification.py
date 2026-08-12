import time
import pandas as pd
from typing import Tuple
import asyncio
from tqdm import tqdm
from pydantic import BaseModel, Field

from .utils.neo4j_graph_functions import (
    load_graph_from_cache_into_neo4j,
    print_neo4j_graph_properties,
    get_current_neo4j_graph,
    update_neo4j_graph,
)
from .config.experiment_settings import settings
from .utils.timing import time_function_calls, atime_coroutine_call
from .ontology.ontology import Edge
from .knowledge_graph import KnowledgeGraph
from .utils.instruct_llm import ainstruct_llm
from .config.prompts import GRAPH_DENSIFICATION_SYSTEM_PROMPT, GRAPH_DENSIFICATION_USER_PROMPT
from .utils.io import dump_to_pickle_file
from .utils.async_helper import limited_task


output_file_path = settings.output_path / "graph_densification"

NUM_NODES_SIMILAR_SUBGRAPH = 15


class ExtractedEdge(BaseModel):
    source: str = Field(..., description="Source name of the node that the edge begins from.")
    source_node_id: str = Field(..., description="The unique uuid of the source node.")
    target: str = Field(..., description="Destination name of the node that the edge terminates in.")
    target_node_id: str = Field(..., description="The unique uuid of the target node.")
    name: str = Field(..., description="The edge's name representing its meaning")

    __hash__ = object.__hash__


class ExtractedEdges(BaseModel):
    extracted_edges: list[ExtractedEdge] = Field(..., description="The extracted edges from the LLM response")


async def aextract_relationships_with_llm(
    content_chunks: list[dict],
    graph: KnowledgeGraph,
    record_measurements: bool,
    max_concurrent_tasks: int = 100,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> Tuple[list[ExtractedEdge], dict[str, float]]:
    """
    Extract relationships with llm using async methods..
    """
    print("Extracting relationships with LLM...")

    print(f"Initializing FAISS index with {len(graph.nodes)} nodes...")
    _, initialize_faiss_index_duration = await atime_coroutine_call(
        graph.ainit_or_update_faiss_index(load_from_pickle=False, dump_to_pickle=False)
    )

    extract_relationships_with_llm_start_time = time.perf_counter()

    print("Getting relationships...")
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        limited_task(
            get_relationships_from_llm(input_text=chunk["text_chunk"], input_num=i + 1, graph=graph),
            semaphore,
            delay_between_tasks,
        )
        for i, chunk in enumerate(content_chunks)
    ]

    edges = []
    for i, task in enumerate(
        tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Looping through tasks...",
        )
    ):
        try:
            extracted_edges = await task
            edges.extend(extracted_edges.extracted_edges)
        except Exception as e:
            print(f"Skipped task {i+1} due to an error: {e}")

    print(f"Extracted {len(edges)} relationships from LLM...")

    extract_relationships_with_llm_end_time = time.perf_counter()

    measurements_data = {}
    if record_measurements:
        measurements_data = {
            "initialize_faiss_index_duration_s": initialize_faiss_index_duration,
            "extract_relationships_with_llm_duration_s": extract_relationships_with_llm_end_time
            - extract_relationships_with_llm_start_time,
        }

    return edges, measurements_data


async def get_relationships_from_llm(
    input_text: str, input_num: int, graph: KnowledgeGraph, llm_model: str = settings.llm_model
) -> ExtractedEdges:
    print(f"Getting relationships from LLM for input text chunk number {input_num}...")

    similar_subgraph = await graph.get_similar_subgraph(input_text, num_nodes=NUM_NODES_SIMILAR_SUBGRAPH)
    system_prompt = GRAPH_DENSIFICATION_SYSTEM_PROMPT.format(current_graph=similar_subgraph.trimmed_graph_json())
    user_prompt = GRAPH_DENSIFICATION_USER_PROMPT.format(input_text=input_text)

    relationships, _ = await ainstruct_llm(
        system_prompt=system_prompt, user_prompt=user_prompt, response_model=ExtractedEdges, llm_model=llm_model
    )

    return relationships


def verify_extracted_relationships(extracted_edges: list[ExtractedEdge], graph: KnowledgeGraph) -> list[ExtractedEdge]:
    """
    Verify extracted relationships by checking if the nodes exist in the graph.
    """
    print(f"Verifying extracted relationships for {len(extracted_edges)} relationships...")

    node_ids = {node.node_id for node in graph.nodes}

    verified_relationships = []
    for edge in extracted_edges:
        if edge.source_node_id in node_ids and edge.target_node_id in node_ids:
            verified_relationships.append(edge)
        else:
            print("Skipped relationship because one or more nodes do not exist in the graph...")
            print(edge)

    print(f"Verified {len(verified_relationships)} relationships...")

    return verified_relationships


def transform_relationships_to_edges(relationships: list[ExtractedEdge]) -> KnowledgeGraph:
    """
    Transform relationships to Edges, and wrap those Edges in a KnowledgeGraph object.
    """
    print("Transforming relationships to Edges...")

    edges = [
        Edge(
            source_node_id=edge.source_node_id,
            source="",
            target_node_id=edge.target_node_id,
            target="",
            name=edge.name,
            other_fields={},
        )
        for edge in relationships
    ]

    return KnowledgeGraph(nodes=[], edges=edges)


def graph_densification(
    content_chunks: list[dict],
    load_graph_from_cache: bool = True,
    record_measurements: bool = False,
):
    """
    This function executes graph densification.

    More specifically, this function extracts relationships from text chunks given a current graph.
    Loop through text chunks. For each text chunk, send to LLM with similar subgraph to extract relationships.

    We then verify the extracted relationships by checking if the nodes exist in the graph,
    before updating the graph with the extracted relationships.

    This routine assumes there is an existing graph in Neo4j to run on.
    Optionally, you can load a graph from the cache into Neo4j.
    """
    print("Starting graph densification by extracting relationships from text chunks...")

    if load_graph_from_cache:
        load_graph_from_cache_into_neo4j(settings.output_path / "entity_resolution/final_graph_deduped.pkl")

    print_neo4j_graph_properties()

    routine_start_time = time.perf_counter()

    graph, get_neo4j_graph_duration = time_function_calls(lambda: get_current_neo4j_graph(min_node_relationships=0))

    # Content nodes don't need to be densified as they are already deterministically linked to other nodes
    content_node_ids = {node.node_id for node in graph.nodes if node.category in ["Content", "ContentChunk"]}
    filtered_nodes = [node for node in graph.nodes if node.node_id not in content_node_ids]
    filtered_edges = [
        edge
        for edge in graph.edges
        if edge.source_node_id not in content_node_ids and edge.target_node_id not in content_node_ids
    ]
    graph_sans_content = KnowledgeGraph(nodes=filtered_nodes, edges=filtered_edges)

    extracted_relationships, extraction_timing_data = asyncio.run(
        aextract_relationships_with_llm(
            content_chunks=content_chunks,
            graph=graph_sans_content,
            record_measurements=record_measurements,
        )
    )

    verified_relationships, verify_relationship_duration = time_function_calls(
        lambda: verify_extracted_relationships(extracted_relationships, graph)
    )

    transformed_relationships_graph, transformation_duration = time_function_calls(
        lambda: transform_relationships_to_edges(verified_relationships)
    )

    _, load_edges_into_neo4j_duration = time_function_calls(
        lambda: update_neo4j_graph(transformed_relationships_graph, update_with_merge=False)
    )

    routine_end_time = time.perf_counter()

    print_neo4j_graph_properties()

    # Dump current neo4j graph to cache
    print("Dumping current Neo4j graph to cache...")
    final_graph = get_current_neo4j_graph(min_node_relationships=0)
    dump_to_pickle_file(final_graph, output_file_path / "final_graph_densified.pkl")

    if record_measurements:
        file_path = output_file_path / "graph_densification_measurements.csv"
        print(f"Saving graph densification measurements data to file {file_path}...")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            [
                {
                    "total_runtime_s": routine_end_time - routine_start_time,
                    "get_neo4j_graph_duration_s": get_neo4j_graph_duration,
                    **extraction_timing_data,
                    "verify_relationship_duration_s": verify_relationship_duration,
                    "transformation_duration_s": transformation_duration,
                    "load_edges_into_neo4j_duration_s": load_edges_into_neo4j_duration,
                }
            ]
        )

        df.to_csv(file_path, index=False)
