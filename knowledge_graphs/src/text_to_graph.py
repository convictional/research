from datetime import datetime
from typing import List, Tuple
from pathlib import Path
import asyncio
from tqdm import tqdm
from itertools import batched
import time
import pandas as pd
import traceback


from .knowledge_graph import KnowledgeGraph

from .config.prompts import INSTRUCTOR_GENERATE_SYSTEM_PROMPT, CATEGORIES, INSTRUCTOR_GENERATE_FROM_DWH_SYSTEM_PROMPT

from .utils.instruct_llm import ainstruct_llm
from .utils.neo4j_graph_functions import update_neo4j_graph, get_current_neo4j_graph
from .utils.async_helper import limited_task
from .config.experiment_settings import settings
from .ontology.ontology import (
    Node,
    BaseEdge,
    CONTENT_NODE_CATEGORY,
    CONTENT_NODE_DESCRIPTION,
    CONTENT_NODE_EDGE_NAME,
    CONTENT_CHUNK_NODE_CATEGORY,
    CONTENT_CHUNK_NODE_DESCRIPTION,
    CONTENT_CHUNK_NODE_EDGE_NAME,
)

from .utils.timing import time_function_calls, atime_coroutine_call


async def aget_updates_from_llm_async_text_chunks(
    node_category: str,
    other_node_categories: str,
    cur_state: KnowledgeGraph,
    inp: str,
    content_chunk_lookup_id: str,
    content_chunk_node_dict: dict,
    **kwargs: dict,
) -> Tuple[KnowledgeGraph, dict]:
    print(
        f"""Working on row {kwargs['i']} of {kwargs['num_iterations']} rows for node group {kwargs['n']} of {kwargs['num_cat_groups']}..."""
    )

    system_prompt = INSTRUCTOR_GENERATE_SYSTEM_PROMPT.format(
        category=node_category,
        other_categories=other_node_categories,
        current_graph=cur_state.trimmed_graph_json(),
        current_datetime=str(datetime.now()),
    )

    user_prompt = f"""
                Below you will find a piece of text that you need to incorporate into the knowledge graph.
                Be creative! This is part {kwargs['i']}/{kwargs['num_iterations']} of the input

                Incorporate the following context into the knowledge graph:
                <context>
                {inp}
                </context>"""

    new_updates, completion_data = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=KnowledgeGraph,
        llm_model=settings.llm_model,
    )

    new_updates.convert_to_generic_nodes()
    new_updates.assign_node_ids_in_edges()

    # Get ContentChunkNode representing the input text with additional metadata
    content_chunk_node = content_chunk_node_dict[content_chunk_lookup_id]

    # Create edges between the ContentChunkNode and each node in new_updates
    for node in new_updates.nodes:
        new_edge = BaseEdge(
            source=content_chunk_node.name,
            source_node_id=content_chunk_node.node_id,
            target=node.name,
            target_node_id=node.node_id,
            name=CONTENT_CHUNK_NODE_EDGE_NAME,
        )
        new_updates.edges.append(new_edge)

    return new_updates, completion_data


async def few_shot_bottom_up_with_llm(
    node_category: str, other_node_categories: str, cur_state: KnowledgeGraph, inp: str, **kwargs: dict
):
    system_prompt = INSTRUCTOR_GENERATE_FROM_DWH_SYSTEM_PROMPT.format(
        category=node_category,
        other_categories=other_node_categories,
        current_graph=cur_state.trimmed_graph_json(),
        current_datetime=str(datetime.now()),
    )

    user_prompt = f"<context>\n{inp}\n</context>"
    few_shot = [
        {
            "role": "user",
            "content": "<context>\nexisting_node: ${GCP_PROJECT}.prod_challenge.challenge_gmv\n description: This  records transactional data for various orders, detailing aspects of each order such as item identification, buyer and seller information, platforms used, and financial details. It is designed to facilitate the analysis of sales performance, buyer and seller interactions, and order management.\n</context>",
        },
        {
            "role": "system",
            "content": """KnowledgeGraph(nodes=[
                BaseNode(name='Orders', category='BusinessData', description='Orders placed by customers', created_at='2024-05-23 10:24:13.840030', updated_at='2024-05-23 10:24:13.840030'),
                BaseNode(name='TransactionalData', category='BusinessData', description='Details aspects of each order such as items, buyer and seller information, platforms and finance details', created_at='2024-05-23 10:24:13.840030', updated_at='2024-05-23 10:24:13.840030'),
                BaseNode(name='SalesPerformance', category='BusinessData', description='Performance of sales', created_at='2024-05-23 10:24:13.840030', updated_at='2024-05-23 10:24:13.840030'),
                BaseNode(name='Buyers', category='Organizations', description='A Company that buys dropship products to sell to consumers', created_at='2024-05-23 10:24:13.840030', updated_at='2024-05-23 10:24:13.840030'),
                BaseNode(name='Sellers', category='Organizations', description='A company that sells dropship products', created_at='2024-05-23 10:24:13.840030', updated_at='2024-05-23 10:24:13.840030')
            ],
            edges=[
                BaseEdge(source='${GCP_PROJECT}.prod_challenge.challenge_gmv', target='TransactionalData', name='STORES_DATA_FOR'),
                BaseEdge(source='Orders', target='TransactionalData', name='CONTAINS'),
                BaseEdge(source='TransactionalData', target='SalesPerformance', name='AFFECTS'),
                BaseEdge(source='Buyers', target='Sellers', name='BUYS_FROM'),
                BaseEdge(source='Sellers', target='Buyers', name='SELLS_TO')
            ])""",
        },
    ]
    new_updates, completion_data = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=KnowledgeGraph,
        few_shot=few_shot,
    )
    return new_updates, completion_data


def get_iterations_measurements_dict(
    iteration_index: int,
    iteration_duration: int,
    llm_call_duration: int,
    update_neo4j_graph_duration: int,
    update_faiss_index_duration: int,
    update_graph_duration: int,
    graph_state: KnowledgeGraph,
    updates: KnowledgeGraph,
    llm_completion_data: dict,
    data_dicts: List[dict],
) -> dict:
    print(f"Recording measurements for iteration {iteration_index}...")

    completion_data = {
        "llm_model": llm_completion_data["model"],
        "num_input_tokens_from_completion": llm_completion_data["usage"]["input_tokens"],
        "num_output_tokens_from_completion": llm_completion_data["usage"]["output_tokens"],
    }

    return {
        "iteration_index": iteration_index,
        "created_at": datetime.now(),
        "iteration_duration_s": iteration_duration,
        "update_neo4j_graph_duration_s": update_neo4j_graph_duration,
        "llm_call_duration_s": llm_call_duration,
        "update_faiss_index_duration_s": update_faiss_index_duration,
        "update_graph_duration_s": update_graph_duration,
        "num_updated_graph_nodes": len(graph_state.nodes),
        "num_updated_graph_edges": len(graph_state.edges),
        "num_updates_nodes": len(updates.nodes),
        "num_updates_edges": len(updates.edges),
        **completion_data,
        **{k: v for d in data_dicts for k, v in d.items()},
        **graph_state.get_graph_metrics(),
    }


def dump_df_to_file(df: pd.DataFrame, file_path: Path):
    print(f"Saving data to file {file_path}...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


async def build_tasks(task_templates: list[dict], content_chunk_node_dict: dict):
    tasks = []
    for task_template in tqdm(task_templates, total=len(task_templates), desc="Building tasks..."):
        # Comment out if you don't care about the initial state, set similar_subgraph = KnowledgeGraph()
        # similar_subgraph = await task_template["current_graph_state"].get_similar_subgraph(task_template["inp"])
        similar_subgraph = KnowledgeGraph()

        task = limited_task(
            task_template["llm_function"](
                node_category="\n".join(task_template["node_category"]),
                other_node_categories="\n".join(
                    [cat for cat in CATEGORIES if cat not in task_template["node_category"]]
                ),
                cur_state=similar_subgraph,
                inp=task_template["inp"],
                content_chunk_lookup_id=task_template["content_chunk_lookup_id"],
                content_chunk_node_dict=content_chunk_node_dict,
                i=task_template["i"],
                num_iterations=task_template["num_iterations"],
                n=task_template["n"],
                num_cat_groups=task_template["num_cat_groups"],
            ),
            task_template["semaphore"],
            delay_between_tasks=task_template["delay_between_tasks"],
        )

        tasks.append(task)

    return tasks


def initialize_content_and_content_chunk_nodes_in_neo4j(
    content_chunks: List[dict], current_graph_state: KnowledgeGraph
) -> dict:
    """
    Initialize content and content chunk nodes in Neo4j and return a dictionary of content chunk nodes.
    """
    print("Initializing content and content chunk nodes in Neo4j...")
    content_node_dict = initialize_content_nodes_in_neo4j(content_chunks, current_graph_state)

    print("Initializing content chunk nodes and edges in Neo4j...")
    content_chunk_node_dict = {}
    content_chunk_nodes = []
    content_chunk_edges = []

    # create content chunk nodes
    for chunk in content_chunks:
        content_node = Node(
            name=chunk["title"],
            category=CONTENT_CHUNK_NODE_CATEGORY,
            description=CONTENT_CHUNK_NODE_DESCRIPTION,
            created_at=chunk["created_at"],
            updated_at=chunk["updated_at"],
            other_fields={
                "content_body": chunk["text_chunk"],
                "content_source": chunk["source"],
                "content_postgres_id": chunk["content_id"],
                "chunk_index": str(chunk["chunk_index"]),
            },
        )
        content_chunk_node_dict[f"{chunk["content_id"]} {chunk["chunk_index"]}"] = content_node
        content_chunk_nodes.append(content_node)

    # Create edges between the ContentNodes and ContentChunkNodes
    for node in content_chunk_nodes:
        new_edge = BaseEdge(
            source=content_node_dict[node.other_fields["content_postgres_id"]].name,
            source_node_id=content_node_dict[node.other_fields["content_postgres_id"]].node_id,
            target=node.name,
            target_node_id=node.node_id,
            name=CONTENT_NODE_EDGE_NAME,
        )
        content_chunk_edges.append(new_edge)

    current_graph_state.nodes.extend(content_chunk_nodes)
    current_graph_state.edges.extend(content_chunk_edges)
    update_neo4j_graph(KnowledgeGraph(nodes=content_chunk_nodes, edges=content_chunk_edges), update_with_merge=False)

    return content_chunk_node_dict


def initialize_content_nodes_in_neo4j(content_chunks: List[dict], current_graph_state: KnowledgeGraph) -> dict:
    """
    Initialize content nodes in Neo4j and return a dictionary of content nodes.
    """
    print("Initializing content nodes in Neo4j...")
    content_node_dict = {}
    content_nodes = []

    for chunk in content_chunks:
        # only want to create the content node once to avoid duplication
        if chunk["chunk_index"] == 1:
            content_node = Node(
                name=chunk["title"],
                category=CONTENT_NODE_CATEGORY,
                description=CONTENT_NODE_DESCRIPTION,
                created_at=chunk["created_at"],
                updated_at=chunk["updated_at"],
                other_fields={
                    "content_source": chunk["source"],
                    "content_postgres_id": chunk["content_id"],
                },
            )
            content_node_dict[chunk["content_id"]] = content_node
            content_nodes.append(content_node)

    current_graph_state.nodes.extend(content_nodes)
    update_neo4j_graph(KnowledgeGraph(nodes=content_nodes), update_with_merge=False)

    return content_node_dict


async def agenerate_graph_with_llm_async(
    content_chunks: List[dict],
    num_parallel_cats: int = 3,
    current_graph_state: KnowledgeGraph = KnowledgeGraph(),
    llm_function: callable = aget_updates_from_llm_async_text_chunks,
    record_measurements: bool = False,
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> Tuple[KnowledgeGraph, dict]:
    """
    Generate graph with llm using async methods.
    This function asyncs the loop over input text chunks and node category groups, compared to generate_graph_with_llm.
    """
    num_iterations = len(content_chunks)
    category_chunks = list(batched(CATEGORIES, num_parallel_cats))
    num_cat_groups = len(category_chunks)
    measurement_results = []
    num_errors = 0
    errors = []
    print(f"Number of category groups = {num_cat_groups}")

    # Comment out if you don't care about the initial state
    # print(f"Initializing FAISS index with {len(current_graph_state.nodes)} nodes...")
    # current_graph_state.init_or_update_faiss_index(load_from_pickle=False, dump_to_pickle=False)

    print(f"Graph number of nodes = {len(current_graph_state.nodes)}")
    print(f"Graph number of edges = {len(current_graph_state.edges)}")

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # Initialize the dictionary to track ContentChunkNodes
    content_chunk_node_dict = initialize_content_and_content_chunk_nodes_in_neo4j(content_chunks, current_graph_state)

    task_templates = [
        {
            "node_category": node_category,
            "inp": chunk["text_chunk"],
            "num_iterations": num_iterations,
            "n": n + 1,
            "num_cat_groups": num_cat_groups,
            "i": i + 1,
            "current_graph_state": current_graph_state,
            "llm_function": llm_function,
            "semaphore": semaphore,
            "delay_between_tasks": delay_between_tasks,
            "content_chunk_lookup_id": f"{chunk["content_id"]} {chunk["chunk_index"]}",
        }
        for n, node_category in enumerate(category_chunks)
        for i, chunk in enumerate(content_chunks)
    ]

    tasks, time_to_build_tasks = await atime_coroutine_call(build_tasks(task_templates, content_chunk_node_dict))

    for i, task in enumerate(
        tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Looping through tasks...",
        )
    ):
        try:
            iteration_start_time = time.perf_counter()

            (new_updates, llm_completion_data), llm_call_duration = await atime_coroutine_call(task)
            await new_updates.aaugment_all_nodes_with_embedding()

            if record_measurements:
                current_graph_metrics = current_graph_state.get_graph_metrics()
                current_graph_measurements = {
                    "num_current_graph_nodes": current_graph_metrics["num_nodes"],
                    "num_current_graph_edges": current_graph_metrics["num_edges"],
                }

            _, update_neo4j_graph_duration = time_function_calls(
                lambda: update_neo4j_graph(new_updates, update_with_merge=False)
            )

            # TODO: Parallelize this using batch embeddings?
            # deep within this method is a sync call to the embedding API, each call could take up to a couple of seconds
            # Comment out if you don't care about the initial state
            # _, update_faiss_index_duration = time_function_calls(
            #     lambda: current_graph_state.init_or_update_faiss_index(
            #         new_updates.nodes, load_from_pickle=False, dump_to_pickle=False
            #     )
            # )

            _, update_graph_duration = time_function_calls(
                lambda: current_graph_state.nodes.extend(new_updates.nodes),
                lambda: current_graph_state.edges.extend(new_updates.edges),
            )

            iteration_end_time = time.perf_counter()

            print(f"""Finished task {i+1}...""")

            if record_measurements:
                measurement_results.append(
                    get_iterations_measurements_dict(
                        iteration_index=i + 1,
                        iteration_duration=iteration_end_time - iteration_start_time,
                        llm_call_duration=llm_call_duration,
                        update_neo4j_graph_duration=update_neo4j_graph_duration,
                        # Comment out if you don't care about the initial state, set update_faiss_index_duration=0
                        # update_faiss_index_duration=update_faiss_index_duration,
                        update_faiss_index_duration=0,
                        update_graph_duration=update_graph_duration,
                        graph_state=current_graph_state,
                        updates=new_updates,
                        llm_completion_data=llm_completion_data,
                        data_dicts=[current_graph_measurements],
                    )
                )
        except Exception as ex:
            print(f"Skipped task {i+1} due to an error: {ex}")
            num_errors += 1
            errors.append(ex)
            traceback.print_exc()

    print(f"Number of errors encountered: {num_errors}")
    for error in errors:
        print(f"Error: {error}")

    graph_run_data = {}
    if record_measurements:
        dump_df_to_file(
            pd.DataFrame(measurement_results),
            settings.output_path / "kg_extraction_experiments" / "experiments_results" / "iterations_results.csv",
        )

        graph_run_data = {
            "num_input_chunks": num_iterations,
            "num_errors": num_errors,
            "num_node_categories": num_cat_groups,
            "time_to_build_tasks": time_to_build_tasks,
        }

    return current_graph_state, graph_run_data


def text_chunks_to_graph_async(
    content_chunks: List[dict],
    num_parallel_cats: int = 3,
    llm_function: callable = aget_updates_from_llm_async_text_chunks,
    record_measurements: bool = False,
) -> KnowledgeGraph:
    """
    Asynchronous text chunks to graph. Asyncs the loop over input text chunks and node category groups.
    """
    cur_state = get_current_neo4j_graph()

    run_start_time = time.perf_counter()
    new_state, graph_run_data = asyncio.run(
        agenerate_graph_with_llm_async(
            content_chunks=content_chunks,
            current_graph_state=cur_state,
            num_parallel_cats=num_parallel_cats,
            llm_function=llm_function,
            record_measurements=record_measurements,
        )
    )
    run_end_time = time.perf_counter()

    if record_measurements:
        dump_df_to_file(
            pd.DataFrame(
                [
                    {
                        **graph_run_data,
                        **{"total_run_duration_s": run_end_time - run_start_time},
                        **new_state.get_graph_metrics(),
                    }
                ]
            ),
            settings.output_path / "kg_extraction_experiments" / "experiments_results" / "graph_results.csv",
        )

    return new_state
