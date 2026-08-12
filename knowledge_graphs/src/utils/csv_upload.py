import asyncio

from tqdm import tqdm
import csv
from datetime import datetime
from typing import List, Tuple
from pathlib import Path
import faiss
import numpy as np
from pydantic import BaseModel, Field


from ..ontology.ontology import Node, Edge, BaseEdge
from ..knowledge_graph import KnowledgeGraph
from ..config.experiment_settings import settings
from ..config.prompts import GENERAL_DESCRIBE_SYSTEM_PROMPT, DESCRIBE_NODE_USER_PROMPT
from .neo4j_graph_functions import update_neo4j_graph
from .embeddings import aembed, split_and_embed_chunks
from .source_data import get_source_content_from_bq
from .instruct_llm import ainstruct_llm
from ..ontology.ontology import (
    CONTENT_NODE_CATEGORY,
    CONTENT_NODE_DESCRIPTION,
    CONTENT_NODE_EDGE_NAME,
    CONTENT_CHUNK_NODE_CATEGORY,
    CONTENT_CHUNK_NODE_DESCRIPTION,
    CONTENT_CHUNK_NODE_EDGE_NAME,
)


class DescribeGraphObject(BaseModel):
    description: str = Field(..., description="A description of the object given context.")


def load_csv_to_dict(filepath: str) -> List[dict]:
    with open(filepath, mode="r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


async def csv_to_knowledge_graph(
    node_csv: Path,
    edge_csv: Path,
    index: faiss.IndexFlatL2,
    guru_chunks: List[Tuple[str, str, str]],
    content_node_name: str,
) -> KnowledgeGraph:
    print("loading nodes")
    nodes_data = load_csv_to_dict(node_csv)
    node_field_data = []
    tasks = (ensure_node_fields(node, index, guru_chunks) for node in nodes_data)
    for task in tqdm(
        asyncio.as_completed(tasks), total=len(nodes_data), desc="Ensuring node fields and getting descriptions..."
    ):
        node_field = await task
        node_field_data.append(node_field)

    edges_data = load_csv_to_dict(edge_csv)
    nodes = [Node(**node_field) for node_field in node_field_data]
    edges = [Edge(**ensure_edge_fields(edge)) for edge in edges_data]
    graph = KnowledgeGraph(nodes=nodes, edges=edges)
    graph.assign_node_ids_in_edges()
    graph = initialize_content_and_content_chunk_nodes_and_edges(graph, content_node_name, node_csv, edge_csv)
    return graph


def initialize_content_and_content_chunk_nodes_and_edges(
    graph: KnowledgeGraph, content_node_name: str, node_csv: Path, edge_csv: Path
) -> KnowledgeGraph:
    print("Initializing content and content chunk nodes and edges...")

    # create nodes
    content_node = Node(
        name=content_node_name,
        category=CONTENT_NODE_CATEGORY,
        description=CONTENT_NODE_DESCRIPTION,
        created_at=str(datetime.now()),
        updated_at=str(datetime.now()),
        other_fields={
            "node_csv": str(node_csv.name),
            "edge_csv": str(edge_csv.name),
        },
    )
    content_chunk_node = Node(
        name=content_node_name,
        category=CONTENT_CHUNK_NODE_CATEGORY,
        description=CONTENT_CHUNK_NODE_DESCRIPTION,
        created_at=str(datetime.now()),
        updated_at=str(datetime.now()),
        other_fields={
            "node_csv": str(node_csv.name),
            "edge_csv": str(edge_csv.name),
        },
    )

    # create edges
    graph.edges.append(
        BaseEdge(
            source=content_node.name,
            source_node_id=content_node.node_id,
            target=content_chunk_node.name,
            target_node_id=content_chunk_node.node_id,
            name=CONTENT_NODE_EDGE_NAME,
        )
    )

    for node in graph.nodes:
        edge = BaseEdge(
            source=content_chunk_node.name,
            source_node_id=content_chunk_node.node_id,
            target=node.name,
            target_node_id=node.node_id,
            name=CONTENT_CHUNK_NODE_EDGE_NAME,
        )
        graph.edges.append(edge)

    graph.nodes.extend([content_node, content_chunk_node])

    return graph


async def ensure_node_fields(node: dict, index: faiss.IndexFlatL2, guru_chunks: List[dict]) -> dict:
    required_fields = {
        "description": "",
        "created_at": str(datetime.now()),
        "updated_at": str(datetime.now()),
        "other_fields": {},
    }
    filled_fields = {**required_fields, **node}

    query_vector = await aembed(node["name"] + " " + node["category"])
    D, I = index.search(np.array([query_vector]), k=settings.embedded_guru_top_k)
    if len(I) > 0 and I[0][0] < len(guru_chunks):
        print(f"Getting description for node {node['name']}")
        chunk = guru_chunks[I[0][0]]
        user_prompt = DESCRIBE_NODE_USER_PROMPT.format(
            name=node["name"], category=node["category"], context=chunk["content"]
        )
        try:
            llm_description, _ = await ainstruct_llm(
                system_prompt=GENERAL_DESCRIBE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=DescribeGraphObject,
            )
        except Exception as e:
            print(f"Error generating description for node {node['name']}: {e}")
            llm_description = DescribeGraphObject(description="")
        filled_fields["description"] += f"\nRetrieved Guru Context: {llm_description.description}"
        if llm_description.description:
            filled_fields["embedding"] = await aembed(llm_description.description)
        else:
            filled_fields["embedding"] = query_vector

    return filled_fields


def ensure_edge_fields(edge: dict) -> dict:
    required_fields = {
        "source": "",
        "target": "",
        "other_fields": {},
    }

    if "source" in edge:
        edge["source"] = edge["source"].replace(" ", "_")
    if "target" in edge:
        edge["target"] = edge["target"].replace(" ", "_")
    if "name" in edge:
        edge["name"] = edge["name"].replace(" ", "_")

    filled_fields = {**required_fields, **edge}
    return filled_fields


def upload_csv_to_neo4j(node_csv: Path, edge_csv: Path, content_node_name: str):
    if not Path(node_csv).exists():
        print(f"Error: The file '{node_csv}' does not exist.")
        return
    if not Path(edge_csv).exists():
        print(f"Error: The file '{edge_csv}' does not exist.")
        return

    source_chunks = get_source_content_from_bq()
    print(f"Got {len(source_chunks)} source chunks")
    source_vectors = asyncio.run(split_and_embed_chunks(source_chunks))
    dimension = len(source_vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(source_vectors).astype("float32"))

    knowledge_graph = asyncio.run(csv_to_knowledge_graph(node_csv, edge_csv, index, source_chunks, content_node_name))
    asyncio.run(knowledge_graph.aaugment_all_nodes_with_embedding())
    try:
        update_neo4j_graph(knowledge_graph, update_with_merge=False)
    except Exception as e:
        print(f"Error in attempting to upload CSV to Neo4j: {e}")
