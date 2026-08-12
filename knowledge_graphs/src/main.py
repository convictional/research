import re
import asyncio
from tqdm import tqdm
import json
from datetime import datetime, timezone


from .graph_traversal.traverse_graph import get_and_init_current_graph, traverse_graph, calculate_pagerank
from .knowledge_graph import KnowledgeGraph
from .bigquery_schema import BigQuerySchemaDetector, BigQueryHistory, DataWarehouseNode
from .build_decision_graph import build_decision_graph
from .text_to_graph import (
    text_chunks_to_graph_async,
    few_shot_bottom_up_with_llm,
)
from .utils.source_data import (
    get_source_content_from_bq,
    augment_content_chunks,
)
from .utils.neo4j_graph_functions import (
    create_indexes,
    clear_existing_graph,
    update_neo4j_graph,
    project_gds_graph,
)
from .utils.tokens import split_chunks_by_tokens
from .config.experiment_settings import settings
from .config.cypher_queries import PROJECT_CURRENT_GRAPH_FOR_GDS
from .utils.csv_upload import upload_csv_to_neo4j
from .utils.io import dump_to_pickle_file, load_pickle_file
from .entity_resolution import entity_resolution
from .graph_densification import graph_densification
from .rag_compare.decision_predictions import (
    get_app_only_predictions,
    get_vss_predictions,
    get_frag_predictions,
    get_graph_predictions,
    judge_predictions,
)


def import_csvs():
    # Left the loading here to make it obvious where to adjust your path in case you don't use the same names as me
    print("Uploading nodes and edges to Neo4j from CSVs...")

    # NOTE: The org chart CSVs were exported from an HR system and were removed before
    # open-sourcing. Supply your own node/edge CSVs at these paths to run this.
    print("Uploading Org Chart CSVs...")
    nodes_csv_path = settings.input_path / "org_chart_nodes.csv"
    edges_csv_path = settings.input_path / "org_chart_edges.csv"
    upload_csv_to_neo4j(nodes_csv_path, edges_csv_path, "Org Chart")


def bq_table_graph_expansion(table_nodes: list[DataWarehouseNode]):
    # TODO: Test whether we can batch table nodes to speed up the process
    text_chunks = [f"existing_node: {n.name}\n description: {n.description.replace('table', '')}" for n in table_nodes]
    text_chunks_to_graph_async(text_chunks, num_parallel_cats=5, llm_function=few_shot_bottom_up_with_llm)


async def import_bigquery_schema() -> KnowledgeGraph:
    print("Parsing BigQuery Schema...")
    bq_schema = BigQuerySchemaDetector().run()
    filtered_schema = [table for table in bq_schema if table.locator.startswith("${GCP_PROJECT}.prod_")]

    # to_graph_node is slow because it generates descriptions and embeddings, so we cache the results in development
    pickle_path_table_nodes = settings.output_path / "table_nodes.pkl"

    table_nodes = []
    if settings.is_env("development") and pickle_path_table_nodes.exists():
        print("Loading table nodes from cache")
        table_nodes = load_pickle_file(pickle_path_table_nodes)
    elif settings.is_env("development"):
        tasks = (table.to_graph_node() for table in filtered_schema)
        for task in tqdm(asyncio.as_completed(tasks), total=len(filtered_schema)):
            table_node = await task
            table_nodes.append(table_node)
        dump_to_pickle_file(table_nodes, pickle_path_table_nodes)
    else:
        tasks = (table.to_graph_node() for table in filtered_schema)
        for task in tqdm(asyncio.as_completed(tasks), total=len(filtered_schema)):
            table_node = await task
            table_nodes.append(table_node)

    print("Adding BigQuery Schema nodes...")
    new_graph = KnowledgeGraph(nodes=table_nodes)
    update_neo4j_graph(new_graph)
    print("BigQuery Schema nodes added to graph")

    return new_graph


async def import_bigquery_history():
    print("Parsing BigQuery History...")
    bq_history = BigQueryHistory()
    query_history = BigQueryHistory().run()
    history_nodes = await bq_history.get_nodes(query_history)
    history_edges = bq_history.get_edges(query_history)

    print("Adding BigQuery History nodes...")
    new_graph = KnowledgeGraph(nodes=history_nodes, edges=history_edges)
    update_neo4j_graph(new_graph)
    print("BigQuery History nodes added to graph")


def process_source_content_chunks(
    sources: list[str] | None = None, max_split_tokens: int = 50000, created_filter: str | None = None
) -> list[dict]:
    """
    This function fetches and processes source content chunks from local postgres database.
    More specifically, we fetch the list of dictionaries from postgres, split the content into chunks,
    and then augment the dictionaries with additional key-value pairs.
    By default, we import all sources. You can specify a set of sources by providing a list of source names as an argument,
    e.g. sources = ["guru", "google_drive"].
    Also, one can optionally filter for source content chunks created after a certain date by providing a string representation
    of the date in the format "%Y-%m-%d".
    """
    print("Processing source content chunks...")

    source_content_chunks = get_source_content_from_bq(sources)
    if created_filter:
        source_content_chunks = filter_content_chunks_by_created_date(source_content_chunks, created_filter)

    split_content_chunks = split_chunks_by_tokens(source_content_chunks, max_split_tokens)
    processed_content_chunks = augment_content_chunks(split_content_chunks)

    return processed_content_chunks


def filter_content_chunks_by_created_date(content_chunks: list[dict], created_filter: str) -> list[dict]:
    """
    This function filters content chunks by created date.
    """
    created_filter_datetime = datetime.strptime(created_filter, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    print(f"Filtering content chunks created after {created_filter_datetime}...")
    filtered_content_chunks = [chunk for chunk in content_chunks if chunk["created_at"] > created_filter_datetime]
    return filtered_content_chunks


def import_knowledge_from_sources(
    sources: list[str] | None = None,
    max_split_tokens: int = 50000,
    created_filter: str | None = None,
):
    """
    This function is the entrypoint for importing knowledge sources to use for graph generation.
    """
    print("Starting graph conversion for all sources...")
    content_chunks = process_source_content_chunks(sources, max_split_tokens, created_filter)
    graph = text_chunks_to_graph_async(content_chunks, num_parallel_cats=5)
    dump_to_pickle_file(graph, settings.output_path / "graph_from_sources.pkl")


def chat_with_graph_traversal():
    """Entrypoint for graph traversal Agent
    Run this using `make run_experiment ARGS="knowledge_graphs chat_with_graph_traversal"
    """
    print("Welcome to the Knowledge Graph Traversal Chatbot. Type 'exit' to quit.")
    current_graph = get_and_init_current_graph()
    project_gds_graph(projection_name="testGraph", projection_query=PROJECT_CURRENT_GRAPH_FOR_GDS)
    calculate_pagerank()
    while True:
        conversation_history = []
        user_query = input("Enter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting the chatbot. Goodbye!")
            break
        try:
            if conversation_history == []:
                results = asyncio.run(traverse_graph(user_query, current_graph))
            else:
                results = asyncio.run(traverse_graph(user_query, current_graph, conversation_history))
            conversation_history.append(
                [
                    {
                        "role": "user",
                        "content": user_query,
                    },
                    {
                        "role": "system",
                        "content": results,
                    },
                ]
            )
            print(json.dumps(results["summary"], indent=2))
        except Exception as e:
            print(f"Error processing query: {e}")
            raise e


def run_entity_resolution():
    """
    Run using `make run_experiment ARGS="knowledge_graphs run_entity_resolution"`
    """
    entity_resolution(load_from_cache=False, record_measurements=True, print_llm_results=True, minimum_similarity=0.8)


def run_graph_densification(
    sources: list[str] | None = None,
    max_split_tokens: int = 10000,
    created_filter: str | None = None,
):
    """
    Run using `make run_experiment ARGS="knowledge_graphs run_graph_densification"`
    """
    content_chunks = process_source_content_chunks(sources, max_split_tokens, created_filter)
    print(f"Running graph densification using {len(content_chunks)} text chunks...")

    graph_densification(content_chunks, load_graph_from_cache=False, record_measurements=True)


def run_decision_predictions():
    asyncio.run(get_graph_predictions())
    asyncio.run(get_app_only_predictions())
    asyncio.run(get_vss_predictions())
    asyncio.run(get_frag_predictions())
    asyncio.run(judge_predictions())


def bq_schmea_to_graph():
    """
    This method brings our Bigquery 'Hubspot' project schema in as table and query nodes.
    """
    schema_graph = asyncio.run(import_bigquery_schema())
    asyncio.run(import_bigquery_history())
    bq_table_graph_expansion(schema_graph.nodes)


def content_to_graph(clear_graph: bool = False, seed_with_org_chart: bool = False):
    """
    This method imports decisions as context which is then passed to an LLM to
    build a graph from. This executes the full pipeline of following the initial
    extraction up with entity resolution and graph densification.

    You can optionally start with a fresh graph and/or seed with our org chart.
    """
    if clear_graph:
        clear_existing_graph()
        create_indexes()

    if seed_with_org_chart:
        import_csvs()

    import_knowledge_from_sources()
    run_entity_resolution()
    run_graph_densification()


def main():
    """
    This is the main entrypoint for the knowledge_graphs experiment
    It should be used to create the graph. Other functions can and should be called via
    the command line interface using argparse.

    This will initiate a graph based on in-app decisions and our org chart. This
    process runs deterministically by using decision data stored in BigQuery.
    Run this using `make run_experiment ARGS="knowledge_graphs"`
    """
    # Graph setup
    clear_existing_graph()
    create_indexes()

    # Pre-formatted data
    import_csvs()

    # Decision graph
    build_decision_graph()

    # Bring in unstructured data
    import_knowledge_from_sources()

    # resolve entities
    run_entity_resolution()
