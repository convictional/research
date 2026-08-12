from typing import List, Any
from enum import Enum

from pydantic import BaseModel, Field

from ..knowledge_graph import KnowledgeGraph
from ..config.prompts import Neo4jNodeCategory
from ..utils.neo4j_graph_functions import (
    get_all_shortest_paths_to_category,
    induced_shortest_path_graph,
    get_one_hop_neighbours,
)


class InducedSubGraphTool(BaseModel):
    node_list: List[str] = Field(
        ...,
        description="List of node names to use as the vertices of the induced subgraph. YOU MUST INCLUDE AT LEAST 2 NODES.",
    )

    name: str = "Induced Subgraph Tool"
    description: str = "This tool generates a subgraph induced by the nodes provided in the input list and the shortest paths between them. The induced subgraph of the graph contains the nodes in node_list and the nodes + edges within the set of shortest paths between those nodes."

    def get_subgraph(self, parent_graph: KnowledgeGraph) -> KnowledgeGraph:
        _subgraph = KnowledgeGraph()
        _subgraph.nodes = [node for node in parent_graph.nodes if node.name in self.node_list]
        subgraph = induced_shortest_path_graph(_subgraph.nodes)

        return subgraph


class AllShortestPathsToLabelTool(BaseModel):
    start_node: str = Field(
        default=" ",
        description="The name of the start node from which all shortest paths to nodes with the target_node_label will be found.",
    )
    target_node_label: Neo4jNodeCategory = Field(
        default="All",
        description="The label of the target node(s) to find the shortest paths to. Do not use multiple labels.",
    )

    name: str = "All Shortest Paths Tool"
    description: str = "This tool finds all the shortest path from the start node to target nodes with the given label. Never provide multiple labels."

    def get_paths(self) -> Any:
        paths = get_all_shortest_paths_to_category(self.start_node, self.target_node_label.value)
        return paths


class SingleNodeOneHopNeighborsTool(BaseModel):
    central_node: str = Field(
        default=" ", description="The name of the central node from which all one-hop neighbors will be found."
    )

    name: str = "Single Node One-Hop Neighbors Tool"
    description: str = "This tool finds all the one-hop neighbors of the central node (both incoming and outgoing)."

    def get_neighbours(self) -> Any:
        neighbours = get_one_hop_neighbours(self.central_node)
        return neighbours


class ReplyToUser(BaseModel):
    response: str = Field(..., description="Your response to the User")


class TraversalToolInputs(BaseModel):
    ISG: InducedSubGraphTool = Field(
        ...,
        description="Induced Subgraph Tool to generate a subgraph from the current graph state based on a list of passed nodes and all paths of length 2 or less between them. Must provide at least two distinct nodes.",
    )

    ASP: AllShortestPathsToLabelTool = Field(
        ...,
        description="All Shortest Paths tool to find all the shortest paths from a start node to a target node label. Great for finding how nodes with a given label are related to a specific node",
    )

    OHN: SingleNodeOneHopNeighborsTool = Field(
        ...,
        description="Single Node One-Hop Neighbors tool to find all the one-hop neighbors of a central node. Great for exploring the immediate connections of a node (both ingoing and outgoing).",
    )
