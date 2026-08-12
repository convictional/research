from itertools import groupby, batched
import json
import faiss
from typing import List, Optional, Any
from datetime import datetime
import networkx as nx
import asyncio
from tqdm import tqdm

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from .config.experiment_settings import settings
from .utils.io import dump_to_pickle_file, load_pickle_file
from .utils.embeddings import embed_to_faiss, query_faiss_index, aembed, aembed_to_faiss
from .utils.math import safe_divide
from .ontology.ontology import (
    BaseNode,
    BaseEdge,
    Node,
    Edge,
    StructuredNode,
    CAT_TO_ATT,
)


class KnowledgeGraph(BaseModel):
    nodes: SkipJsonSchema[List[BaseNode]] = Field(..., default_factory=list)
    structured_nodes: Optional[List[StructuredNode]] = Field(..., default_factory=list)
    edges: Optional[List[BaseEdge]] = Field(..., default_factory=list)

    node_index: Optional[Any] = None

    def assign_node_ids_in_edges(self):
        """
        This method assigns node_ids to the edges based on the nodes in the KnowledgeGraph object.
        We don't return anything since we are referencing the KnowledgeGraph's edge objects directly.

        If there is an edge that references a node name that is not in the nodes list, we print an error message,
        and omit the edge from the graph.
        """
        node_name_to_id = {node.name: node.node_id for node in self.nodes}
        valid_edges = []

        for edge in self.edges:
            try:
                edge.source_node_id = node_name_to_id[edge.source]
                edge.target_node_id = node_name_to_id[edge.target]
                valid_edges.append(edge)
            except KeyError as e:
                print(f"Node name not found in nodes: {e}")
                print(f"Edge: {edge}")
                print(f"Nodes: {node_name_to_id}")
            except Exception as e:
                raise e

        self.edges = valid_edges

    def insert_node(self, node: Node) -> tuple[str, dict[str, any]]:
        """
        This method generates a Cypher query and parameters for creating a node in the graph.
        By "creating", we mean inserting with no merging of nodes.
        """
        props = node.model_dump()
        if "other_fields" in props:
            props["other_fields"] = json.dumps(props["other_fields"])

        create_props_text = ", ".join(f"n.{k} = {json.dumps(v)}" for k, v in props.items() if k != "node_id")

        query = f"""
        CREATE (n:All:{node.category.replace(' ', '')} {{node_id: '{node.node_id}'}})
        SET {create_props_text}
        """

        return query, props

    def upsert_node(self, node: Node) -> tuple[str, dict[str, any]]:
        props = node.model_dump()
        if "other_fields" in props:
            props["other_fields"] = json.dumps(props["other_fields"])
        merge_props_text = ", ".join(
            f"n.{k} = CASE WHEN n.{k} IS NULL THEN [{json.dumps(v)}] ELSE n.{k} + [{json.dumps(v)}] END"
            if k in ["description"]
            else f"n.{k} = COALESCE(n.{k}, {json.dumps(v)})"
            if k == "created_at"
            else f"n.{k} = {json.dumps(v)}"
            for k, v in props.items()
            if k != "name"
        )
        create_props_text = ", ".join(f"n.{k} = {json.dumps(v)}" for k, v in props.items() if k != "name")
        query = f"""
        MERGE (n:All:{node.category.replace(' ', '')} {{name: '{node.name}'}})
        ON CREATE SET {create_props_text}
        ON MATCH SET {merge_props_text}
        """
        return query, props

    def insert_nodes_batches(self, nodes: List[Node], batch_size=100) -> list[tuple[str, dict[str, any]]]:
        """
        This method generates a list of Cypher queries and parameters for creating nodes in batches.
        By "creating", we mean inserting with no merging of nodes.
        """
        nodes_props = [node.model_dump() for node in nodes]
        node_prop_keys = set().union(*(node.keys() for node in nodes_props))
        for node in nodes_props:
            if "other_fields" in node:
                node["other_fields"] = json.dumps(node["other_fields"])

        query = """
        WITH $nodes_props AS batch
        UNWIND batch AS props
        CREATE (n:All:{category} {{node_id: props.node_id}})
        SET {create_props_text}
        """

        category_groups = groupby(nodes_props, lambda node: node["category"].replace(" ", ""))
        create_props_text = ", ".join(f"n.{k} = props.{k}" for k in node_prop_keys if k != "node_id")

        return [
            (
                query.format(
                    category=category,
                    create_props_text=create_props_text,
                ),
                {"nodes_props": list(batched_props)},
            )
            for category, grouped_props in category_groups
            for batched_props in batched(grouped_props, batch_size)
        ]

    def upsert_nodes_batches(self, nodes: List[Node], batch_size=100) -> list[tuple[str, dict[str, any]]]:
        nodes_props = [node.model_dump() for node in nodes]
        node_prop_keys = set().union(*(node.keys() for node in nodes_props))
        for node in nodes_props:
            if "other_fields" in node:
                node["other_fields"] = json.dumps(node["other_fields"])

        query = """
        WITH $nodes_props AS batch
        UNWIND batch AS props
        MERGE (n:All:{category} {{name: props.name}})
        ON CREATE SET {create_props_text}
        ON MATCH SET {merge_props_text}
        """

        category_groups = groupby(nodes_props, lambda node: node["category"].replace(" ", ""))
        merge_props_text = ", ".join(
            f"n.{k} = CASE WHEN n.{k} IS NULL THEN [props.{k}] ELSE n.{k} + [props.{k}] END"
            if k in ["description"]
            else f"n.{k} = COALESCE(n.{k}, props.{k})"
            if k == "created_at"
            else f"n.{k} = props.{k}"
            for k in node_prop_keys
            if k != "name"
        )
        create_props_text = ", ".join(f"n.{k} = props.{k}" for k in node_prop_keys if k != "name")

        return [
            (
                query.format(
                    category=category,
                    create_props_text=create_props_text,
                    merge_props_text=merge_props_text,
                ),
                {"nodes_props": list(batched_props)},
            )
            for category, grouped_props in category_groups
            for batched_props in batched(grouped_props, batch_size)
        ]

    def insert_relationship(self, edge: Edge) -> tuple[str, dict[str, any]]:
        """
        This method generates a Cypher query and parameters for creating a relationship between two nodes.
        By "creating", we mean inserting with no merging of relationships.
        """
        props = edge.model_dump()
        if "other_fields" in props:
            props["other_fields"] = json.dumps(props["other_fields"])

        create_props_text = ", ".join(
            f"r.{k} = {json.dumps(v)}"
            for k, v in props.items()
            if k not in ["source", "target", "source_node_id", "target_node_id"]
        )

        query = f"""
        MATCH (a {{node_id: $source_node_id}}), (b {{node_id: $target_node_id}})
        CREATE (a)-[r:{edge.name.replace(' ','_').replace('-', '_')}]->(b)
        SET {create_props_text}
        """

        return query, {
            "source": edge.source,
            "target": edge.target,
            "source_node_id": edge.source_node_id,
            "target_node_id": edge.target_node_id,
            **{k: v for k, v in props.items() if k not in ["source", "target", "source_node_id", "target_node_id"]},
        }

    def create_relationship(self, edge: Edge) -> tuple[str, dict[str, any]]:
        props = edge.model_dump()
        if "other_fields" in props:
            props["other_fields"] = json.dumps(props["other_fields"])
        merge_props_text = ", ".join(
            f"r.{k} = {json.dumps(v)}" for k, v in props.items() if k not in ["source", "target"]
        )
        create_props_text = ", ".join(
            f"r.{k} = {json.dumps(v)}" for k, v in props.items() if k not in ["source", "target"]
        )
        query = f"""
        MATCH (a {{name: $source}}), (b {{name: $target}})
        MERGE (a)-[r:{edge.name.replace(' ','_').replace('-', '_')}]->(b)
        ON CREATE SET {create_props_text}
        ON MATCH SET {merge_props_text}
        """
        return query, {
            "source": edge.source,
            "target": edge.target,
            **{k: v for k, v in props.items() if k not in ["source", "target"]},
        }

    def trimmed_graph_json(self) -> str:
        """Generates a trimmed version of the graph's JSON representation for prompts."""
        trimmed_nodes = [
            {
                "name": node.name,
                "node_id": node.node_id,
                "category": node.category,
                "description": node.description,
            }
            for node in self.nodes
        ]
        trimmed_edges = [
            {
                "source": edge.source,
                "source_node_id": edge.source_node_id,
                "target": edge.target,
                "target_node_id": edge.target_node_id,
                "name": edge.name,
            }
            for edge in self.edges
        ]
        return json.dumps({"nodes": trimmed_nodes, "edges": trimmed_edges}, indent=2)

    def init_or_update_faiss_index(
        self, new_nodes: List[Node] = [], load_from_pickle: bool = True, dump_to_pickle: bool = True
    ) -> faiss.IndexFlatL2:
        """Create or update a FAISS index for the current nodes based on their names and descriptions."""
        pickle_path = settings.output_path / "graph_faiss.pkl"

        if load_from_pickle and pickle_path.exists():
            try:
                self.node_index = load_pickle_file(pickle_path)
                print("Loaded FAISS index from pickle file.")
            except Exception as e:
                print(f"Failed to load FAISS index from pickle file: {e}")
                self.node_index = None

        if self.node_index is None:
            self.node_index = faiss.IndexFlatL2(settings.faiss_embedding_dimension)
            for node in self.nodes:
                text_to_embed = f"{node.name} {node.description}"
                self.node_index = embed_to_faiss(
                    text_to_embed, self.node_index, settings.faiss_embedding_model, settings.faiss_embedding_dimension
                )

        if len(new_nodes) > 0:
            for node in new_nodes:
                text_to_embed = f"{node.name} {node.description}"
                self.node_index = embed_to_faiss(
                    text_to_embed, self.node_index, settings.faiss_embedding_model, settings.faiss_embedding_dimension
                )

        if dump_to_pickle:
            dump_to_pickle_file(self.node_index, pickle_path)

        return self.node_index

    async def ainit_or_update_faiss_index(
        self, new_nodes: List[Node] = [], load_from_pickle: bool = True, dump_to_pickle: bool = True
    ) -> faiss.IndexFlatL2:
        """Create or update a FAISS index for the current nodes based on their names and descriptions."""
        pickle_path = settings.output_path / "graph_faiss.pkl"

        if load_from_pickle and pickle_path.exists():
            try:
                self.node_index = load_pickle_file(pickle_path)
                print("Loaded FAISS index from pickle file.")
            except Exception as e:
                print(f"Failed to load FAISS index from pickle file: {e}")
                self.node_index = None

        if self.node_index is None:
            self.node_index = faiss.IndexFlatL2(settings.faiss_embedding_dimension)
            texts_to_embed = [f"{node.name} {node.description}" for node in self.nodes]
            self.node_index = await aembed_to_faiss(
                texts_to_embed, self.node_index, settings.faiss_embedding_model, settings.faiss_embedding_dimension
            )

        if len(new_nodes) > 0:
            texts_to_embed = [f"{node.name} {node.description}" for node in new_nodes]
            self.node_index = await aembed_to_faiss(
                texts_to_embed, self.node_index, settings.faiss_embedding_model, settings.faiss_embedding_dimension
            )

        if dump_to_pickle:
            dump_to_pickle_file(self.node_index, pickle_path)

        return self.node_index

    def get_induced_subgraph(self, spanning_nodes: list[str]) -> "KnowledgeGraph":
        """Generate the subgraph induced by the most similar nodes."""
        G = nx.MultiDiGraph()

        for node in self.nodes:
            G.add_node(node.node_id, name=node.name, category=node.category, description=node.description)

        for edge in self.edges:
            G.add_edge(
                edge.source_node_id, edge.target_node_id, source=edge.source, target=edge.target, name=edge.name
            )

        subgraph = G.subgraph(spanning_nodes)
        subgraph_knowledge_graph = KnowledgeGraph(
            nodes=[
                Node(
                    name=subgraph.nodes[n]["name"],
                    node_id=n,
                    category=subgraph.nodes[n]["category"],
                    description=subgraph.nodes[n]["description"],
                    embedding=[],
                    created_at=str(datetime.now()),
                    updated_at=str(datetime.now()),
                    other_fields={},
                )
                for n in subgraph.nodes
            ],
            edges=[
                Edge(
                    source_node_id=e[0],
                    source=subgraph.edges[e]["source"],
                    target_node_id=e[1],
                    target=subgraph.edges[e]["target"],
                    name=subgraph.edges[e]["name"],
                    other_fields={},
                )
                for e in subgraph.edges
            ],
        )

        return subgraph_knowledge_graph

    async def get_similar_subgraph(self, user_query: str, num_nodes: int = 100) -> "KnowledgeGraph":
        """Get the subgraph induced by the most similar nodes to the user query."""
        _, similar_nodes_indices = await query_faiss_index(query_text=user_query, index=self.node_index, k=num_nodes)
        if all(x == -1 for x in similar_nodes_indices):
            return KnowledgeGraph()
        similar_nodes_indices = list(map(int, similar_nodes_indices))
        similar_nodes = [self.nodes[idx].node_id for idx in similar_nodes_indices]
        similar_subgraph = self.get_induced_subgraph(similar_nodes)
        return similar_subgraph

    def get_graph_metrics(self) -> dict:
        # convert KnowledgeGraph to NetworkX graph
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from([node.name for node in self.nodes])
        nx_graph.add_edges_from([(edge.source, edge.target) for edge in self.edges])

        is_empty_graph = len(nx_graph) == 0

        return {
            "num_nodes": nx_graph.number_of_nodes(),
            "avg_edges_per_node": safe_divide(nx_graph.number_of_edges(), nx_graph.number_of_nodes()),
            "num_edges": nx_graph.number_of_edges(),
            "is_connected": nx.is_connected(nx_graph) if not is_empty_graph else False,
            "diameter": nx.diameter(nx_graph) if not is_empty_graph and nx.is_connected(nx_graph) else 0,
            "num_components": nx.number_connected_components(nx_graph),
        }

    async def aaugment_all_nodes_with_embedding(
        self, embedding_model: str = settings.embedding_model, embedding_dim: int = settings.embedding_dimension
    ):
        """
        This method augments all node objects with an embedding asynchronously.
        We don't return anything since we are referencing the KnowledgeGraph's node objects using aget_embeddings_for_node.
        """
        tasks = [self.aget_embedding_for_node(node, embedding_model, embedding_dim) for node in self.nodes]

        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Augmenting all nodes with embeddings...",
        ):
            await task

    async def aget_embedding_for_node(
        self,
        node: Node,
        embedding_model: str = settings.embedding_model,
        embedding_dim: int = settings.embedding_dimension,
    ) -> Node:
        """
        This method augments a single node object with an embedding asynchronously.
        This is done referencing the node object directly, but also returns the node object for convenience.
        """
        embedding = await aembed(
            node.name + node.description, embedding_model=embedding_model, embedding_dim=embedding_dim
        )

        node.embedding = embedding.tolist()

        return node

    def convert_to_generic_nodes(self) -> List[Node]:
        generic_node_list = []
        for structured_node in self.structured_nodes:
            attr = CAT_TO_ATT.get(structured_node.node_category)
            if attr:
                try:
                    specific_node = getattr(structured_node, attr)
                except AttributeError:
                    specific_node = None
                    print(f"AttributeError: {attr} not found in {structured_node}")
                if specific_node:
                    other_fields = {
                        k: v for k, v in specific_node.model_dump().items() if k not in Node.model_fields.keys()
                    }

                    # Create a generic Node instance
                    generic_node = Node(
                        name=specific_node.name,
                        category=specific_node.category,
                        description=specific_node.description,
                        created_at=specific_node.created_at,
                        updated_at=specific_node.updated_at,
                        node_id=specific_node.node_id,
                        embedding=specific_node.embedding,
                        other_fields=other_fields,
                    )

                    generic_node_list.append(generic_node)

        if not self.nodes:
            self.nodes = generic_node_list
        else:
            self.nodes.extend(generic_node_list)
