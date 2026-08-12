import time
import pandas as pd
import numpy as np
from typing import List, Set
import asyncio
from pydantic import BaseModel, Field
from tqdm import tqdm
import textwrap
from collections import defaultdict
from datetime import datetime

from .utils.neo4j_graph_functions import (
    update_neo4j_graph,
    query_neo4j,
    get_current_neo4j_graph,
    project_gds_graph,
    drop_gds_graph,
    print_neo4j_graph_properties,
    load_graph_from_cache_into_neo4j,
)
from .config.experiment_settings import settings
from .config.prompts import (
    ENTITY_RESOLUTION_DETECTION_SYSTEM_PROMPT,
    ENTITY_RESOLUTION_DETECTION_USER_PROMPT,
    ENTITY_RESOLUTION_MERGE_NODES_USER_PROMPT,
    ENTITY_RESOLUTION_MERGE_NODES_SYSTEM_PROMPT,
)
from .config.cypher_queries import (
    ENTITY_RESOLUTION_COUNT_ER_RELATIONSHIP_TYPES,
    ENTITY_RESOLUTION_CREATE_ER_EXACT_MATCHES,
    ENTITY_RESOLUTION_GET_EXACT_MATCH_PAIRS,
    ENTITY_RESOLUTION_CREATE_SIMILAR_MATCHES_BASED_ON_COSINE_SIMILARITY,
    ENTITY_RESOLUTION_GET_SIMILAR_PAIRS_THAT_ARE_NOT_SIMILAR_MATCH,
    ENTITY_RESOLUTION_CREATE_SIMILAR_MATCHES_BASED_ON_LLM_DECISION,
    ENTITY_RESOLUTION_PAIR_IMMEDIATE_NEIGHBOUR_COUNTS,
    ENTITY_RESOLUTION_CREATE_GDS_PROJECTION_QUERY,
    ENTITY_RESOLUTION_EXECUTE_WCC_ALGORITHM_QUERY,
    ENTITY_RESOLUTION_MERGE_RELATIONSHIPS_TO_MERGED_NODES_AND_DELETE_OLD_ENTITIES,
    ENTITY_RESOLUTION_DELETE_CUSTOM_RELATIONSHIPS,
)
from .utils.io import dump_to_pickle_file
from .utils.timing import time_function_calls
from .utils.math import safe_divide
from .utils.instruct_llm import ainstruct_llm
from .ontology.ontology import BaseNode, Node, Edge
from .knowledge_graph import KnowledgeGraph


output_file_path = settings.output_path / "entity_resolution"
ER_GDS_PROJECTION_NAME = "entity_resolution_wcc"


class LLMJudgement(BaseModel):
    """
    Class for a response model for when asking the LLM to check if two nodes are duplicates.
    """

    result: bool = Field(
        ...,
        description="Result of the duplicate check. True or False. True means that the nodes are duplicates of each other. False means they are not duplicates.",
    )
    reason: str = Field(
        ...,
        description="Reason for the result. If the result is True, this should contain the reason why the nodes are duplicates. If the result is False, this should contain the reason why the nodes are not duplicates.",
    )


def count_entity_resolution_relationship_types_in_neo4j():
    results = query_neo4j(ENTITY_RESOLUTION_COUNT_ER_RELATIONSHIP_TYPES)

    print("Number of entity resolution relationship types in neo4j:")
    if len(results) == 0:
        print("No custom relationships in neo4j graph.")
    else:
        for result in results:
            print(f"{result['RelationshipType']}: {result["Count"]}")
        total_count = sum([result["Count"] for result in results])
        print(f"Total number of entity resolution relationships in neo4j: {total_count}")


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    metadata: dict[str, str | float] | list[dict[str, str]],
    response_model: LLMJudgement | BaseNode,
) -> tuple[LLMJudgement | BaseNode, dict[str, str | dict[str, str | float] | list[dict[str, str]]]]:
    response, _ = await ainstruct_llm(
        system_prompt=system_prompt, user_prompt=user_prompt, response_model=response_model
    )

    return response, {"user_prompt": user_prompt, "metadata": metadata}


def create_exact_match_relationships_in_neo4j():
    """
    This function creates exact match relationships in Neo4j.
    An exact match relationship is when two nodes have the same name and category (lower cased).
    Exact match relationships have type = ER_EXACT_MATCH. (ER = entity resolution).
    This is done directly using Cypher.
    """
    print("Creating exact match relationships in Neo4j...")
    query_neo4j(ENTITY_RESOLUTION_CREATE_ER_EXACT_MATCHES)
    count_entity_resolution_relationship_types_in_neo4j()


def create_similar_relationships_in_neo4j(minimum_similarity: float):
    """
    This function creates similar relationships in Neo4j.
    A similar relationship is when two nodes have a cosine similarity above the minimum_similarity threshold,
    and their categories are the same.
    Similar relationships have type = SIMILAR.
    """
    print(f"Creating similar relationships in Neo4j with minimum cosine similarity {minimum_similarity}...")

    graph_nodes = get_current_neo4j_graph(min_node_relationships=0).nodes
    graph_nodes = [node for node in graph_nodes if node.category not in ["Content", "ContentChunk"]]
    similarity_matrix = build_cosine_similarity_matrix(graph_nodes)

    # get exact match pairs from neo4j and transform into a list of pair sets
    exact_match_pairs_from_neo4j = query_neo4j(ENTITY_RESOLUTION_GET_EXACT_MATCH_PAIRS)
    exact_match_pairs_sets = [
        set([pair["source_node_id"], pair["target_node_id"]]) for pair in exact_match_pairs_from_neo4j
    ]

    similar_edges = build_similar_relationships_in_memory(
        graph_nodes, similarity_matrix, minimum_similarity, exact_match_pairs_sets
    )

    # Load similar relationships into Neo4j
    similar_edges_graph = KnowledgeGraph(edges=similar_edges)
    update_neo4j_graph(similar_edges_graph, update_with_merge=False)

    count_entity_resolution_relationship_types_in_neo4j()


def build_cosine_similarity_matrix(nodes: List[Node]) -> np.ndarray:
    """
    Calculate cosine similarity of all pairs of nodes in a list of nodes.
    1. Construct a matrix of embeddings. Each column corresponds to the embeddings for a node.
    2. Normalize the matrix by dividing each column by its norm.
    3. Do matrix multiplication of the matrix with its transpose to get the cosine similarity matrix.
       The result is an n_nodes X n_nodes similarity matrix.
    """
    print("Building cosine similarity matrix...")

    embedding_matrix = np.array([node.embedding for node in nodes]).T
    embedding_norms = np.linalg.norm(embedding_matrix, axis=0)

    # handle division by zero in embedding norms
    # Initialize normalized matrix as zero, the same shape as the original
    normalized_embedding_matrix = np.zeros_like(embedding_matrix)
    non_zero_norms = embedding_norms != 0  # Find indices where norms are not zero
    # Perform normalization only where norms are non-zero, but preserve the original matrix shape
    normalized_embedding_matrix[:, non_zero_norms] = (
        embedding_matrix[:, non_zero_norms] / embedding_norms[non_zero_norms]
    )

    cosine_similarity_matrix = normalized_embedding_matrix.T @ normalized_embedding_matrix

    return cosine_similarity_matrix


def build_similar_relationships_in_memory(
    nodes: List[Node],
    similarity_matrix: np.ndarray,
    minimum_similarity: float,
    exact_match_pairs: List[Set[str]],
) -> List[Edge]:
    """
    Loop over pairs of nodes and create similar Edge objects based on embedding cosine similarity values and node categories.
    For pairs that are exact matches, don't create a similar relationship - they are already matched.
    """
    edges = []

    # Get indexes of pairs of nodes that have similarity above the minimum threshold, efficiently using numpy
    indexes_above_threshold = np.where(similarity_matrix > minimum_similarity)

    # Process indexes of pairs
    for i, j in zip(*indexes_above_threshold):
        node = nodes[i]
        other_node = nodes[j]

        is_pair_exact_match = set([node.node_id, other_node.node_id]) in exact_match_pairs
        is_pair_category_match = node.category.lower() == other_node.category.lower()
        cosine_similarity_value = similarity_matrix[i, j]

        # Only node pairs that are:
        # 1. unique (i.e. i < j) and don't count doubles (i.e. (i, j) and (j, i) are the same pair)
        # 2. not the same node (i.e. i != j, which is covered by i < j)
        # 3. not exact matches
        # 4. have the same category
        if i < j and not is_pair_exact_match and is_pair_category_match:
            edge = Edge(
                name="ER_SIMILAR",
                source=node.name,
                target=other_node.name,
                source_node_id=node.node_id,
                target_node_id=other_node.node_id,
                other_fields={"cosine_similarity": str(cosine_similarity_value)},
            )

            edges.append(edge)

    return edges


def create_similarity_match_edges_based_on_embedding_similarity_in_neo4j(embedding_match_minimum_similarity: float):
    """
    This function creates SIMILAR_MATCH edges in neo4j based on high similarity of the node pair's embeddings.
    This is done directly using Cypher.
    """
    print(
        f"Creating similarity match edges in Neo4j based on cosine similarity with minimum cosine similarity {embedding_match_minimum_similarity}..."
    )
    query_neo4j(
        ENTITY_RESOLUTION_CREATE_SIMILAR_MATCHES_BASED_ON_COSINE_SIMILARITY.format(
            threshold=embedding_match_minimum_similarity
        )
    )
    count_entity_resolution_relationship_types_in_neo4j()


def create_similarity_match_edges_based_on_llm_decision_in_neo4j(print_llm_results: bool):
    """
    This function creates SIMILAR_MATCH edges in neo4j using LLM-as-a-judge.
    1. Get pairs of nodes that are similar but not similar match.
    2. Send these pairs to LLM for judgement.
    3. Create SIMILAR_MATCH edges based on LLM's judgement and update the graph in Neo4j.
    """
    print("Creating similarity match edges using LLM-as-a-judge in neo4j...")

    pairs = query_neo4j(ENTITY_RESOLUTION_GET_SIMILAR_PAIRS_THAT_ARE_NOT_SIMILAR_MATCH)

    # send pairs to LLM for judgement
    if len(pairs) == 0:
        print("No similar pairs that are not similar match found in neo4j.")
    else:
        print(f"Found {len(pairs)} similar pairs that are not similar match in neo4j.")

        judgement_packets = send_node_pairs_to_llm_for_similarity_match_judgement(pairs)

        if print_llm_results:
            print("Printing LLM results to file...")
            with open(output_file_path / "llm_as_a_judge_results.txt", "w") as f:
                file_contents = []
                file_contents.append(f"There are {len(judgement_packets)} judgements in total.\n" + "-" * 40 + "\n\n")
                for i, packet in enumerate(judgement_packets):
                    file_contents.append(f"Judgement {i+1}:\n")
                    file_contents.append("Judgement details:\n")
                    file_contents.append(f"Judgement match result: {packet["judgement"].result}\n")
                    file_contents.append(
                        "Judgement match reason:\n" + textwrap.fill(f"{packet['judgement'].reason}", 110) + "\n\n"
                    )
                    file_contents.append(
                        "User prompt for LLM call:\n" + textwrap.fill(packet["user_prompt"], 110) + "\n\n"
                    )
                    file_contents.append("-" * 60 + "\n\n")

                f.write("".join(file_contents))

        # Create SIMILAR_MATCH edges based on LLM decision
        counter = 0
        for packet in judgement_packets:
            if packet["judgement"].result:
                query_neo4j(
                    ENTITY_RESOLUTION_CREATE_SIMILAR_MATCHES_BASED_ON_LLM_DECISION.format(
                        source_node_id=packet["metadata"]["source_node_id"],
                        target_node_id=packet["metadata"]["target_node_id"],
                        decision_reason=packet["judgement"].reason,
                    )
                )
                counter += 1
        print(f"Number of SIMILAR_MATCH edges created based on LLM decision: {counter}")

        count_entity_resolution_relationship_types_in_neo4j()


def send_node_pairs_to_llm_for_similarity_match_judgement(
    pairs: list[dict[str, str | float]],
) -> list[dict[str, LLMJudgement | str | dict[str, str]]]:
    """
    This function sends node pairs to LLM for similarity match judgement.

    Resulting dicts in list of dicts has keys: judgement, user_prompt, metadata (dict of pair properties, like node names, IDs, etc.)
    """
    print("Sending node pairs to LLM for judgement...")

    neighbour_counts = [
        get_immediate_neighbour_counts_from_neo4j(pair["source_node_id"], pair["target_node_id"]) for pair in pairs
    ]
    user_prompts = [
        ENTITY_RESOLUTION_DETECTION_USER_PROMPT.format(
            source_name=pair["source_name"],
            target_name=pair["target_name"],
            source_description=pair["source_description"],
            target_description=pair["target_description"],
            cosine_similarity=pair["cosine_similarity"],
            source_frac_num_common_neighbours=safe_divide(
                neighbour_count["num_common_neighbours"], neighbour_count["source_num_neighbours"]
            ),
            target_frac_num_common_neighbours=safe_divide(
                neighbour_count["num_common_neighbours"], neighbour_count["target_num_neighbours"]
            ),
        )
        for pair, neighbour_count in zip(pairs, neighbour_counts)
    ]

    # dicts in list of dicts has keys: judgement, user_prompt, metadata (dict of pair properties, like node names, IDs, etc.)
    judgement_packets = asyncio.run(get_llm_judgements(user_prompts, pairs))

    return judgement_packets


def get_immediate_neighbour_counts_from_neo4j(node1_id: str, node2_id: str) -> dict:
    results = query_neo4j(
        ENTITY_RESOLUTION_PAIR_IMMEDIATE_NEIGHBOUR_COUNTS.format(node1_id=node1_id, node2_id=node2_id)
    )

    return {
        "source_num_neighbours": results[0]["source_num_neighbours"],
        "target_num_neighbours": results[0]["target_num_neighbours"],
        "num_common_neighbours": results[0]["num_common_neighbours"],
    }


async def get_llm_judgements(
    user_prompts: list[str], pairs: list[dict[str, str | float]]
) -> list[dict[str, LLMJudgement | str | dict[str, str | float]]]:
    results = []
    tasks = [
        call_llm(ENTITY_RESOLUTION_DETECTION_SYSTEM_PROMPT, user_prompt, pair, LLMJudgement)
        for user_prompt, pair in zip(user_prompts, pairs)
    ]

    for task in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Looping through LLM calls...",
    ):
        result, data_packet = await task
        results.append({"judgement": result, **data_packet})

    return results


def get_groups_of_duplicates() -> dict[int, List[dict[str, str]]]:
    """
    This function gets groups of duplicates from Neo4j and returns them.
    1. Setup GDS projection for connected components algorithm via Cypher.
    2. Run connected components algorithm via Cypher.
    3. Process the results of the connected components algorithm to get groups of duplicates.

    The output is a dictionary with keys as connected component IDs and values as
    lists of nodes in that component represented as dictionaries.
    """
    print("Getting groups of duplicates...")

    # Setup GDS projection
    project_gds_graph(
        projection_name=ER_GDS_PROJECTION_NAME,
        projection_query=ENTITY_RESOLUTION_CREATE_GDS_PROJECTION_QUERY.format(projection_name=ER_GDS_PROJECTION_NAME),
    )

    # execute connected components algorithm
    wcc_results = query_neo4j(
        ENTITY_RESOLUTION_EXECUTE_WCC_ALGORITHM_QUERY.format(projection_name=ER_GDS_PROJECTION_NAME)
    )

    # group nodes by connected component
    grouped_data = defaultdict(list)
    for result in wcc_results:
        golden_id = result.pop("golden_id")
        grouped_data[golden_id].append(result)
    grouped_data = dict(grouped_data)

    # drop GDS projection to save memory
    drop_gds_graph(ER_GDS_PROJECTION_NAME)

    print(f"Number of duplicate nodes processed: {sum([len(value) for value in grouped_data.values()])}")
    print(f"Number of groups of duplicates: {len(grouped_data)}")

    return grouped_data


def create_merged_nodes_from_duplicates(
    grouped_duplicates: dict[int, List[dict[str, str]]], print_llm_results: bool
) -> list[dict[str, BaseNode | str | list[dict[str, str]]]]:
    """
    This function creates merged nodes from groups of duplicates using an LLM.
    For each group of duplicates, create a merged node using an LLM.

    Resulting dicts in list of dicts has keys: merged_node, user_prompt, metadata
    (i.e. list of duplicates)
    """
    print("Creating merged nodes from duplicates in Neo4j...")

    merged_nodes_with_duplicates = create_merged_nodes_using_llm(grouped_duplicates, print_llm_results)

    return merged_nodes_with_duplicates


def create_merged_nodes_using_llm(
    grouped_duplicates: dict[int, List[dict[str, str]]], print_llm_results: bool
) -> list[dict[str, BaseNode | str | list[dict[str, str]]]]:
    """
    Use the groups of duplicated nodes and create a merged node for each group using a LLM.

    The output is a list of dictionaries, where each dictionary contains the merged node and
    the list of original duplicates.

    Resulting dicts in list of dicts has keys: merged_node, user_prompt, metadata (i.e. list of duplicates)
    """
    print("Creating merged nodes using LLM...")

    lists_of_duplicates = [value for _, value in grouped_duplicates.items()]
    user_prompts = [
        ENTITY_RESOLUTION_MERGE_NODES_USER_PROMPT.format(
            num_nodes=len(duplicates),
            node_names="\n".join(
                [f"Duplicate node {i+1} name: {duplicate["name"]}" for i, duplicate in enumerate(duplicates)]
            ),
            node_descriptions="\n".join(
                [
                    f"Duplicate node {i+1} description: {duplicate["description"]}"
                    for i, duplicate in enumerate(duplicates)
                ]
            ),
        )
        for duplicates in lists_of_duplicates
    ]

    merged_node_data = asyncio.run(send_duplicates_to_llm(user_prompts, lists_of_duplicates))

    if print_llm_results:
        print("Printing LLM results to file...")
        with open(output_file_path / "node_merging_llm_results.txt", "w") as f:
            file_contents = []
            file_contents.append(f"There are {len(merged_node_data)} merged nodes in total.\n" + "-" * 40 + "\n\n")
            for i, merged_data in enumerate(merged_node_data):
                duplicates = merged_data["metadata"]

                file_contents.append(f"Merged node {i+1}:\n")
                file_contents.append("Merged node details:\n")
                file_contents.append(f"Name: '{merged_data["merged_node"].name}'\n")
                file_contents.append(f"Category: '{merged_data["merged_node"].category}'\n")
                file_contents.append(f"Description: '{merged_data["merged_node"].description}'\n")
                file_contents.append(f"Created at: {merged_data["merged_node"].created_at}\n")
                file_contents.append(f"Updated at: {merged_data["merged_node"].updated_at}\n\n")
                file_contents.append(f"There are {len(duplicates)} duplicate nodes.\n")
                file_contents.append("Children details:\n")
                for j, duplicate in enumerate(duplicates):
                    file_contents.append(f"Child {j+1}:\n")
                    file_contents.append(f"Name: '{duplicate['name']}'\n")
                    file_contents.append(f"Category: '{duplicate['category']}'\n")
                    file_contents.append(f"Description: '{duplicate['description']}'\n\n")
                file_contents.append("User prompt for LLM call:\n")
                file_contents.append(merged_data["user_prompt"] + "\n")
                file_contents.append("-" * 60 + "\n\n")

            f.write("".join(file_contents))

    print(f"Created {len(merged_node_data)} merged nodes using LLM.")

    return merged_node_data


async def send_duplicates_to_llm(
    user_prompts: list[str], lists_of_duplicates: list[list[dict[str, str]]]
) -> list[dict[str, BaseNode | str | list[dict[str, str]]]]:
    results = []
    tasks = [
        call_llm(ENTITY_RESOLUTION_MERGE_NODES_SYSTEM_PROMPT, user_prompt, duplicates, BaseNode)
        for user_prompt, duplicates in zip(user_prompts, lists_of_duplicates)
    ]

    for task in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Looping through LLM calls...",
    ):
        result, data_packet = await task
        results.append({"merged_node": result, **data_packet})

    return results


def create_merged_nodes_in_neo4j(merged_node_data: list[dict[str, BaseNode | str | list[dict[str, str]]]]):
    """
    This function creates merged nodes and relationships in Neo4j.
    The relationships are those between the merged nodes and their corresponding duplicates.
    1. Create merged node and relationship objects.
    2. Update Neo4j graph with the merged nodes and relationships.
    """
    print("Creating merged nodes and their relationships with duplicates in Neo4j...")

    merged_nodes = [
        Node(
            name=data["merged_node"].name,
            # to ensure the merged node has the same category as the duplicates
            category=data["metadata"][0]["category"],
            description=data["merged_node"].description,
            created_at=str(datetime.now()),
            updated_at=str(datetime.now()),
            node_id=data["merged_node"].node_id,
            embedding=[],
            other_fields={},
        )
        for data in merged_node_data
    ]
    merged_node_relationships = [
        Edge(
            name="ER_MERGED_NODE_OF",
            source=merged_node.name,
            source_node_id=merged_node.node_id,
            target=duplicate["name"],
            target_node_id=duplicate["node_id"],
            other_fields={},
        )
        for merged_node, data in zip(merged_nodes, merged_node_data)
        for duplicate in data["metadata"]
    ]

    merged_nodes_graph = KnowledgeGraph(nodes=merged_nodes, edges=merged_node_relationships)
    print(f"Number of merged nodes to insert into Neo4j: {len(merged_nodes_graph.nodes)}")
    print(f"Number of merged node relationships to insert into Neo4j: {len(merged_nodes_graph.edges)}")

    # augment merged nodes with embeddings
    asyncio.run(merged_nodes_graph.aaugment_all_nodes_with_embedding())

    # update Neo4j graph with merged nodes and relationships
    update_neo4j_graph(merged_nodes_graph, update_with_merge=False)
    count_entity_resolution_relationship_types_in_neo4j()
    print_neo4j_graph_properties()


def merge_duplicate_node_relationships_to_merged_nodes_and_clean_up_data_in_neo4j():
    """
    This function merges (redirects) relationships from duplicate nodes into the merged nodes, using Cypher.
    Furthermore, the old duplicate nodes, old duplicate node relationships, and custom relationships used for
    entity resoltion are deleted from the graph.

    This is all done directly using Cypher.

    Note, Cypher doesn't allow for a simple redirecting of relationships natively.
    Also, Cypher doesn't allow for creating relationships based on dynamic relationship types.
    That is, a simple "match relationships and CREATE new ones, but pointing to the merged nodes" query won't work,
    because the relationship type in a CREATE statement needs to be a (hardcoded) constant.
    Therefore, we use an `apoc` function to create the new relationships. This allows for dynamic relationship creation types.

    Furthermore, we actually use `apoc.merge.relationship` to not only create the merged node relationships,
    but deduplicate any of those relationships for a given merged node (i.e. merged node is for 2 duplicate
    nodes, and each has the same relationship to another node). This relationship deduplication could happen in some
    future graph pruning step but we do it here. Note, this won't work if we go with weighted relationships in the future.

    When deleting the duplicate nodes, we use `DETACH DELETE` to delete the nodes and all relationships connected to them.
    This helps clean up the custom relationships used for entity resolution. To be sure of cleanup, we also delete
    these custom relationships explicitly.
    """
    print("Merging relationships from duplicate nodes to merged nodes and cleaning up data in Neo4j...")

    results = query_neo4j(ENTITY_RESOLUTION_MERGE_RELATIONSHIPS_TO_MERGED_NODES_AND_DELETE_OLD_ENTITIES)[0]
    print(f"Number of duplicate nodes deleted: {results['num_deleted_duplicates']}")
    print(
        f"Net new relationships: {results["num_new_incoming_rels"] - results["num_deleted_incoming_relationships"] + results["num_new_outgoing_rels"] - results["num_deleted_outgoing_relationships"]}"
    )

    query_neo4j(ENTITY_RESOLUTION_DELETE_CUSTOM_RELATIONSHIPS)
    count_entity_resolution_relationship_types_in_neo4j()


def detect_duplicates(
    minimum_similarity: float, embedding_match_minimum_similarity: float, print_llm_results: bool
) -> dict[str, float]:
    print("Starting duplicate detection...")

    detection_start_time = time.perf_counter()
    _, exact_match_duration = time_function_calls(create_exact_match_relationships_in_neo4j)
    _, similar_relationship_duration = time_function_calls(
        lambda: create_similar_relationships_in_neo4j(minimum_similarity)
    )
    _, similar_match_embedding_duration = time_function_calls(
        lambda: create_similarity_match_edges_based_on_embedding_similarity_in_neo4j(
            embedding_match_minimum_similarity
        )
    )
    _, similar_match_llm_duration = time_function_calls(
        lambda: create_similarity_match_edges_based_on_llm_decision_in_neo4j(print_llm_results)
    )
    print_neo4j_graph_properties()
    detection_end_time = time.perf_counter()

    return {
        "duplicate_detection_runtime_s": detection_end_time - detection_start_time,
        "exact_match_duration_s": exact_match_duration,
        "similar_relationship_duration_s": similar_relationship_duration,
        "similar_match_embedding_duration_s": similar_match_embedding_duration,
        "similar_match_llm_duration_s": similar_match_llm_duration,
    }


def resolve_duplicates(print_llm_results: bool) -> dict[str, float]:
    print("Starting duplicate resolution...")

    resolution_start_time = time.perf_counter()
    grouped_duplicates, get_duplicate_groups_duration = time_function_calls(get_groups_of_duplicates)
    merged_node_data, create_merged_nodes_llm_duration = time_function_calls(
        lambda: create_merged_nodes_from_duplicates(grouped_duplicates, print_llm_results)
    )
    _, create_merged_nodes_neo4j_duration = time_function_calls(lambda: create_merged_nodes_in_neo4j(merged_node_data))
    _, merge_relationships_and_cleanup_duration = time_function_calls(
        merge_duplicate_node_relationships_to_merged_nodes_and_clean_up_data_in_neo4j
    )
    print_neo4j_graph_properties()
    resolution_end_time = time.perf_counter()

    return {
        "duplicate_resolution_runtime_s": resolution_end_time - resolution_start_time,
        "get_duplicate_groups_duration_s": get_duplicate_groups_duration,
        "create_merged_nodes_llm_duration_s": create_merged_nodes_llm_duration,
        "create_merged_nodes_neo4j_duration_s": create_merged_nodes_neo4j_duration,
        "merge_relationships_and_cleanup_duration_s": merge_relationships_and_cleanup_duration,
    }


def entity_resolution(
    load_from_cache: bool = False,
    record_measurements: bool = False,
    minimum_similarity: float = 0.9,
    embedding_match_minimum_similarity: float = 0.95,
    print_llm_results: bool = False,
):
    """
    This function executes the entity resolution routine.

    This routine assumes there is an existing graph in Neo4j to run on.
    Optionally, you can load a graph from the cache into Neo4j.

    This function requires that nodes have node_id and embedding properties.
    """
    if load_from_cache:
        load_graph_from_cache_into_neo4j()

    print_neo4j_graph_properties()

    routine_start_time = time.perf_counter()

    # Duplicate detection
    detection_timing_data = detect_duplicates(
        minimum_similarity, embedding_match_minimum_similarity, print_llm_results
    )

    # Duplicate resolution
    resolution_timing_data = resolve_duplicates(print_llm_results)

    routine_end_time = time.perf_counter()

    # Dump current neo4j graph to cache
    print("Dumping current Neo4j graph to cache...")
    final_graph = get_current_neo4j_graph(min_node_relationships=0)
    dump_to_pickle_file(final_graph, output_file_path / "final_graph_deduped.pkl")

    if record_measurements:
        file_path = output_file_path / "entity_resolution_measurements.csv"
        print(f"Saving entity resolution measurements data to file {file_path}...")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            [
                {
                    "total_runtime_s": routine_end_time - routine_start_time,
                    **detection_timing_data,
                    **resolution_timing_data,
                }
            ]
        )

        df.to_csv(file_path, index=False)
