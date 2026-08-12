from itertools import batched
from tqdm import tqdm
import json
import neo4j
from pydantic import BaseModel
from typing import List
from pathlib import Path

from ..config.cypher_queries import (
    ALL_PATHS_FROM_SOURCE_NODE_TO_LABELLED_NODES_N_OR_LESS_HOPS,
    ALL_SHORTEST_PATHS_BETWEEN_NODES,
    ONE_HOP_NEIGHBOURS,
    QUERY_NODES_ON_PAGERANK,
    QUERY_NODES_ON_PAGERANK_WITH_CAT,
    DROP_CURRENT_GDS_GRAPH,
)
from ..config.prompts import NEO4J_NODE_CATEGORIES
from ..ontology.ontology import Node, Edge
from ..knowledge_graph import KnowledgeGraph
from ..config.experiment_settings import settings
from .io import load_pickle_file

neo4j_driver: neo4j.Driver | None = None
async_neo4j_driver: neo4j.Driver | None = None


def get_neo4j_session() -> neo4j.Session:
    global neo4j_driver
    if neo4j_driver:
        neo4j_driver.verify_connectivity()
        return neo4j_driver.session()
    else:
        username = settings.neo4j_username
        password = settings.neo4j_password.get_secret_value()
        neo4j_driver = neo4j.GraphDatabase.driver(settings.neo4j_dsn, auth=(username, password))
        neo4j_driver.verify_connectivity()
        return neo4j_driver.session()


def query_neo4j(cypher_query: str) -> List[dict]:
    with get_neo4j_session() as session:
        with session.begin_transaction() as tx:
            result = tx.run(cypher_query).data()

    return result


def get_async_neo4j_session() -> neo4j.Session:
    global async_neo4j_driver
    if async_neo4j_driver:
        return async_neo4j_driver.session()
    else:
        username = settings.neo4j_username
        password = settings.neo4j_password.get_secret_value()
        async_neo4j_driver = neo4j.AsyncGraphDatabase.driver(settings.neo4j_dsn, auth=(username, password))
        return async_neo4j_driver.session()


async def aquery_neo4j(cypher_query: str) -> List[dict]:
    async with get_async_neo4j_session() as session:
        result = await session.run(cypher_query)
        results = await result.data()

    return results


class CypherBatchConfig(BaseModel):
    batchSize: int = 1000
    parallel: bool = True


def run_cypher_batch(
    cypher_query: str, cypher_action: str, batch_config: CypherBatchConfig = CypherBatchConfig()
) -> List[dict]:
    batch_query = f"""
    CALL apoc.periodic.iterate(
        "{cypher_query}",
        "{cypher_action}",
        {batch_config.model_dump()}
    )
    """
    with get_neo4j_session() as session:
        with session.begin_transaction() as tx:
            result = tx.run(batch_query).data()

    return result


def update_neo4j_graph(updates: KnowledgeGraph, update_with_merge: bool = True) -> None:
    print("Updating Neo4j graph...")
    with get_neo4j_session() as session:
        batch_size = 250
        # TODO: Once we have node deduplication, we can remove methods that merge nodes and relationships
        if update_with_merge:
            if updates.nodes:
                for query, params in tqdm(
                    updates.upsert_nodes_batches(updates.nodes, batch_size=batch_size),
                    desc="Upserting batches of nodes...",
                ):
                    session.run(query, params)

            if updates.edges:
                total_batches = (len(updates.edges) // batch_size) + (len(updates.edges) % batch_size > 0)
                for batch in tqdm(
                    batched(updates.edges, batch_size), total=total_batches, desc="Upserting batches of edges..."
                ):
                    with session.begin_transaction() as tx:
                        for edge in batch:
                            query, params = updates.create_relationship(edge)
                            tx.run(query, params)
        else:
            if updates.nodes:
                for query, params in tqdm(
                    updates.insert_nodes_batches(updates.nodes, batch_size=batch_size),
                    desc="Inserting batches of nodes...",
                ):
                    session.run(query, params)

            if updates.edges:
                total_batches = (len(updates.edges) // batch_size) + (len(updates.edges) % batch_size > 0)
                for batch in tqdm(
                    batched(updates.edges, batch_size), total=total_batches, desc="Upserting batches of edges..."
                ):
                    with session.begin_transaction() as tx:
                        for edge in batch:
                            query, params = updates.insert_relationship(edge)
                            try:
                                tx.run(query, params)
                            except Exception as e:
                                print(f"Skipping edge insertion due to error: {e}")


def clear_existing_graph() -> None:
    with get_neo4j_session() as session:
        query = "MATCH (n) DETACH DELETE n"
        with session.begin_transaction() as tx:
            tx.run(query)
            print("Cleared existing graph.")


def clean_data(data: dict) -> dict:
    for key, value in data.items():
        if value is None:
            if key == "other_fields":
                data[key] = {}
            else:
                data[key] = ""
        elif key == "embedding":
            data[key] = value
        elif isinstance(value, list):
            data[key] = ", ".join(map(str, value))
        elif key == "other_fields" and isinstance(value, str):
            try:
                data[key] = json.loads(value)
            except json.JSONDecodeError:
                data[key] = {}
    return data


def create_vector_index(node_category: str) -> None:
    query = f"""
    CREATE VECTOR INDEX { node_category.lower() }_embedding_index IF NOT EXISTS
    FOR (n:{ node_category })
    ON (n.embedding)
    OPTIONS {{indexConfig: {{
    `vector.dimensions`: { settings.embedding_dimension },
    `vector.similarity_function`: 'cosine'
    }}}}
    """
    with get_neo4j_session() as session:
        session.run(query)
        print(f"Created vector index for {node_category}")


def create_indexes():
    print("Creating vector index...")
    # create an index for each node category
    for category in NEO4J_NODE_CATEGORIES + ["All"]:
        create_vector_index(category)


def get_current_neo4j_graph(min_node_relationships: int = 1) -> KnowledgeGraph:
    """
    This function loads the graph from Neo4j into an in-memory KnowledgeGraph object.
    """
    print("Loading graph from Neo4j into KnowledgeGraph in-memory object...")

    graph = KnowledgeGraph()
    query = f"""
        MATCH (n)
        WHERE count{{(n)--()}} >= {min_node_relationships}
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN
            n.name as node_name,
            n.node_id as node_id,
            n.category as node_category,
            n.description as node_description,
            n.embedding as node_embedding,
            n.created_at as node_created_at,
            n.updated_at as node_updated_at,
            n.other_fields as node_other_fields,
            m.name as target_name,
            m.node_id as target_id,
            r.name as edge_name,
            r.other_fields as edge_other_fields
    """
    with get_neo4j_session() as session:
        result = session.run(query).data()

    for record in result:
        node_ids = [node.node_id for node in graph.nodes]
        if record["node_id"] and record["node_id"] not in node_ids:
            node_data = {
                "name": record["node_name"],
                "node_id": record["node_id"],
                "category": record["node_category"],
                "description": record["node_description"],
                "embedding": record["node_embedding"],
                "created_at": record["node_created_at"],
                "updated_at": record["node_updated_at"],
                "other_fields": record["node_other_fields"],
            }
            node = Node(**clean_data(node_data))
            graph.nodes.append(node)
        if record["target_id"]:
            edge_data = {
                "source": record["node_name"],
                "source_node_id": record["node_id"],
                "target": record["target_name"],
                "target_node_id": record["target_id"],
                "name": record["edge_name"],
                "other_fields": record["edge_other_fields"],
            }
            edge = Edge(**clean_data(edge_data))
            graph.edges.append(edge)

    print(f"Number of nodes in KnowledgeGraph object: {len(graph.nodes)}")
    print(f"Number of edges in KnowledgeGraph object: {len(graph.edges)}")

    return graph


def prune_orphan_nodes() -> None:
    cypher_query = """
    MATCH (n)
    WHERE NOT (n)--()
    DELETE n
    """
    return query_neo4j(cypher_query)


def project_gds_graph(projection_name: str, projection_query: str) -> None:
    try:
        drop_gds_graph(projection_name)
    except Exception as e:
        if "java.util.NoSuchElementException" in str(e):
            print(f"Graph {projection_name} does not exist, proceeding to project a new one.")
        else:
            print(f"An error occurred: {e}")
            raise e
    query_neo4j(projection_query)
    # Pro tip: To see projection details run CALL gds.graph.list('projection_name') in Neo4j Browser
    print(f"New {projection_name} projected.")


def drop_gds_graph(projection_name: str) -> None:
    query_neo4j(DROP_CURRENT_GDS_GRAPH.format(projection_name=projection_name))
    print(f"Existing {projection_name} dropped.")


def get_all_shortest_paths_to_category(start_node: str, target_node_category: str) -> List[dict]:
    """
    Find all shortest paths of less than 2 hops from the start node to nodes of the target category.

    Args:
    - start_node (str): The name of the start node.
    - target_node_category (str): The category of the target nodes.

    Returns:
    - Any: The query result containing the shortest paths.
    """
    path_query = ALL_PATHS_FROM_SOURCE_NODE_TO_LABELLED_NODES_N_OR_LESS_HOPS.format(
        start_node_name=start_node, target_node_category=target_node_category, N=3, top_k=10
    )
    paths = query_neo4j(path_query)
    return paths


def get_shortest_paths(start_node: str, target_node: str) -> List[dict]:
    """
    Find the shortest path between the start node and the target node.

    Args:
    - start_node (str): The name of the start node.
    - target_node (str): The name of the target node.

    Returns:
    - Any: The query result containing the shortest path.
    """
    path_query = ALL_SHORTEST_PATHS_BETWEEN_NODES.format(node_list=[start_node, target_node])
    paths = query_neo4j(path_query)
    return paths


def induced_shortest_path_graph(input_nodes: List[Node]) -> KnowledgeGraph:
    """
    Generate a subgraph induced by the nodes provided in the input list and find the shortest paths between them.

    Args:
    - input_nodes (List[Node]): List of node names to use as the vertices of the induced subgraph.

    Returns:
    - KnowledgeGraph: The subgraph induced by the input nodes and the shortest paths between them.
    """
    subgraph = KnowledgeGraph()
    subgraph.nodes = input_nodes
    subgraph.edges = []

    for i in range(len(input_nodes)):
        for j in range(i + 1, len(input_nodes)):
            paths = get_shortest_paths(input_nodes[i].name, input_nodes[j].name)
            for path in paths:
                for i in range(len(path) - 1):
                    source_node = Node(
                        name=path[i]["name"], category=path[i]["category"], description=path[i]["description"]
                    )
                    target_node = Node(
                        name=path[i + 1]["name"],
                        category=path[i + 1]["category"],
                        description=path[i + 1]["description"],
                    )
                    edge = Edge(source=source_node.name, target=target_node.name, name=path[i]["type"])
                    if source_node not in subgraph.nodes:
                        subgraph.nodes.append(source_node)
                    if target_node not in subgraph.nodes:
                        subgraph.nodes.append(target_node)
                    if edge not in subgraph.edges:
                        subgraph.edges.append(edge)

    return subgraph


def get_one_hop_neighbours(central_node: str) -> List[dict]:
    query = ONE_HOP_NEIGHBOURS.format(node_name=central_node)
    neighbours = query_neo4j(query)
    return neighbours


def query_pagerank(node_label: str = None) -> List[dict]:
    query = QUERY_NODES_ON_PAGERANK_WITH_CAT.format(category=node_label) if node_label else QUERY_NODES_ON_PAGERANK
    result = query_neo4j(query)
    return result


def get_neo4j_graph_properties():
    """
    This is a helper function that gets properties of the currently loaded graph in Neo4j.
    We can build on this over time as more properties become relevant.
    """
    query = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN count(distinct n) as num_nodes, count(distinct r) as num_edges
    """

    return query_neo4j(query)[0]


def print_neo4j_graph_properties():
    """
    This function gets the properties of the Neo4j graph and prints them.
    """
    results = get_neo4j_graph_properties()

    print("Neo4j graph properties:")
    print(f"Neo4j graph number of nodes: {results["num_nodes"]}")
    print(f"Neo4j graph number of edges: {results["num_edges"]}")


def load_graph_from_cache_into_neo4j(pickle_path: Path = settings.output_path / "graph.pkl"):
    """
    This function resets Neo4j, loads the graph from the cache, and loads it into Neo4j.
    """
    print("Loading graph from cache into Neo4j...")

    # Neo4j graph setup
    clear_existing_graph()
    create_indexes()

    # Load graph from cache
    graph = load_pickle_file(pickle_path)
    print(f"Number of nodes in KnowledgeGraph object: {len(graph.nodes)}")
    print(f"Number of edges in KnowledgeGraph object: {len(graph.edges)}")

    # Load graph into Neo4j
    update_neo4j_graph(graph, update_with_merge=False)
