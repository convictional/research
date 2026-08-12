# tree_visualizer.py

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import csv

logger = logging.getLogger(__name__)


class TreeNode:
    """
    Node in the research tree visualization.
    Retained primarily for metadata grouping, but we will build
    a truly hierarchical structure in a separate step for D3.
    """

    def __init__(self, id: str, type: str, label: str, depth: int, parent_id: Optional[str] = None):
        self.id = id
        self.type = type
        self.label = label
        self.depth = depth
        self.parent_id = parent_id
        self.metadata: Dict[str, Any] = {}

    def add_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }


class ResearchTree:
    """Research tree representation for producing a collapsible D3 tree HTML."""

    def __init__(self, research: Dict[str, Any]):
        """
        research is a dictionary that typically looks like:
        {
          "metadata": { ... },
          "iterations": {
            "iteration_id_1": { "id": ..., "title": ..., "depth": ..., etc. },
            ...
          },
          "query_results": {
            "query_id_1": { "id": ..., "iteration_id": ..., "query": ..., etc. },
            ...
          },
        }
        """
        self.research = research
        self.nodes: Dict[str, TreeNode] = {}
        # edges are still stored if needed, but we won't rely on them for layout
        self.edges: List[Tuple[str, str]] = []
        # iteration_to_queries organizes queries under each iteration
        self.iteration_to_queries: Dict[str, List[str]] = {}
        self.build_tree()

    def build_tree(self):
        """Build flat node structures and record iteration->query relationships."""
        for it_id, iteration in self.research["iterations"].items():
            node = TreeNode(
                id=it_id,
                type="iteration",
                label=iteration["title"],
                depth=iteration["depth"],
                parent_id=iteration["parent_iteration_id"],
            )
            node.add_metadata("directions", iteration["directions"])
            node.add_metadata("queries_count", iteration["queries_count"])
            self.nodes[it_id] = node
            self.iteration_to_queries[it_id] = []
            if iteration["parent_iteration_id"]:
                self.edges.append((iteration["parent_iteration_id"], it_id))

        for query_id, query in self.research["query_results"].items():
            iteration_id = query["iteration_id"]
            if iteration_id not in self.research["iterations"]:
                logger.warning(f"Query {query_id} refers to non-existent iteration {iteration_id}")
                continue

            # Associate query with iteration
            self.iteration_to_queries[iteration_id].append(query_id)

            # Optionally store search_results or other metadata in the iteration's node:
            # We won't create a separate TreeNode for queries since we will build them
            # in a hierarchical structure next, but let's store metadata in the iteration node
            query_metadata = {
                "id": query_id,
                "type": "query",
                "query_text": query.get("query", ""),
                "goals": query.get("goals", ""),
                "learnings": query.get("learnings", []),
                "urls": query.get("urls", []),
                "completed": query.get("completed", False),
                "title": query.get("title", ""),
                # We can also store search_results if present
                "search_results": query.get("search_results", {}),
            }
            # Prefix in case multiple queries exist
            self.nodes[iteration_id].add_metadata(f"query_{query_id}", query_metadata)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a simplified dictionary version of the entire structure:
         - metadata from the research
         - a list of node dicts
         - edges
         - iteration->queries map
        """
        return {
            "metadata": self.research.get("metadata", {}),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": self.edges,
            "iterations_to_queries": self.iteration_to_queries,
        }

    def save_json(self, output_path: Path) -> Path:
        """
        Save the 'flat' structure (nodes, edges, etc.) to JSON for debugging or other uses.
        """
        topic_slug = self.research["metadata"].get("topic", "untitled").replace(" ", "_")[:30]
        json_path = output_path / f"{topic_slug}_tree.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return json_path

    def export_research_json(self, output_path: Path) -> Path:
        """
        Save the raw research dictionary (including iterations, query_results, etc.)
        to JSON. This is typically the structure fed to the constructor.
        """
        topic_slug = self.research["metadata"].get("topic", "untitled").replace(" ", "_")[:30]
        json_path = output_path / f"{topic_slug}_research.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.research, f, indent=2)
        logger.info(f"Research data exported to JSON: {json_path}")
        return json_path

    def visualize(self, output_path: Path) -> Path:
        """
        Produce an HTML file that shows a collapsible D3.js tree of the entire
        research structure.

        Returns:
            Path to the .html file.
        """
        # 1) Convert the 'flat' iteration/query structure into a nested hierarchy for D3.
        hierarchy = self._build_hierarchy()

        # 2) Write out an HTML file with inline JS to render this `hierarchy` as a collapsible tree.
        topic_slug = self.research["metadata"].get("topic", "untitled").replace(" ", "_")[:30]
        html_path = output_path / f"{topic_slug}_collapsible_tree.html"

        html_content = self._generate_d3_html(hierarchy)
        html_path.write_text(html_content, encoding="utf-8")

        logger.info(f"Collapsible D3 tree visualization saved to: {html_path}")
        return html_path

    def _build_hierarchy(self) -> Dict[str, Any]:
        """
        Build a nested dict for d3.hierarchy:
        Iteration -> Queries -> Learning nodes
        """
        iteration_map = {}
        for it_id, node in self.nodes.items():
            iteration_info = self.research["iterations"][it_id]
            iteration_map[it_id] = {
                "id": it_id,
                "type": "iteration",
                # Include the iteration title and/or depth in the label
                "title": f"{iteration_info['title']} (Depth {iteration_info['depth']})",
                "children": [],
            }

        for it_id, query_ids in self.iteration_to_queries.items():
            for q_id in query_ids:
                query_md = self.nodes[it_id].metadata.get(f"query_{q_id}")
                if not query_md:
                    continue

                # Build a query node
                query_child = {
                    "id": q_id,
                    "type": "query",
                    "title": query_md["title"] or f"Query {q_id}",
                    "query_text": query_md["query_text"],
                    "goals": query_md["goals"],
                    "urls": query_md["urls"],
                    "completed": query_md["completed"],
                    # For top-down layout, the query can have learning children
                    "children": [],
                }

                # For each learning in the list, create a child node
                learning_nodes = []
                for i, learning_text in enumerate(query_md["learnings"]):
                    learning_child = {
                        "id": f"{q_id}-learning-{i}",
                        "type": "learning",
                        "title": f"Learning #{i + 1}",
                        "learning_text": learning_text,
                        # no further children for these leaves
                        "children": [],
                    }
                    learning_nodes.append(learning_child)

                # Attach them
                query_child["children"] = learning_nodes

                # Now attach the query to the iteration node
                iteration_map[it_id]["children"].append(query_child)

        # Identify root vs children
        roots = []
        for it_id, it_data in iteration_map.items():
            # If you have "parent_iteration_id" logic, do it here if you want multi-depth iterations.
            iteration = self.research["iterations"][it_id]
            parent_id = iteration.get("parent_iteration_id")
            if parent_id and parent_id in iteration_map:
                iteration_map[parent_id]["children"].append(it_data)
            else:
                roots.append(it_data)

        # If you have multiple root iterations, wrap them under an artificial root
        if len(roots) == 1:
            return self._to_d3_node(roots[0])
        else:
            return {"name": "Root of Research", "type": "root", "children": [self._to_d3_node(r) for r in roots]}

    def _to_d3_node(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an iteration/query/learning structure into the
        { name, type, attributes, children } shape for d3.hierarchy.
        """
        # Set the name based on node type
        if data["type"] == "iteration":
            name = data["title"]  # Use the iteration title directly
        elif data["type"] == "query":
            # Use the actual query text for the name
            query_text = data.get("query_text", "")
            name = (query_text[:100] + "..." if len(query_text) > 100 else query_text) or f"Query {data['id']}"
        elif data["type"] == "learning":
            # Use the actual learning text for the name
            learning_text = data.get("learning_text", "")
            name = (learning_text[:100] + "..." if len(learning_text) > 100 else learning_text) or data["title"]
        else:
            name = data["title"]

        node = {
            "name": name,
            "type": data["type"],
            # We'll store extra fields in 'attributes' for the hover info
            "attributes": {},
            "children": [],
        }

        if data["type"] == "iteration":
            node["attributes"]["id"] = data["id"]
            # Additional iteration info can go here
        elif data["type"] == "query":
            node["attributes"]["id"] = data["id"]
            # Store full query text in attributes for the hover info
            node["attributes"]["query"] = data["query_text"]
            node["attributes"]["goals"] = data["goals"]
            node["attributes"]["urls"] = ", ".join(data["urls"]) if isinstance(data["urls"], list) else data["urls"]
        elif data["type"] == "learning":
            node["attributes"]["id"] = data["id"]
            node["attributes"]["learning"] = data["learning_text"]

        # Recursively transform children
        for child_data in data.get("children", []):
            child_node = self._to_d3_node(child_data)
            node["children"].append(child_node)

        return node

    def _iteration_to_d3_node(self, iteration_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively convert iteration (and its child queries/iterations) into the format:
        {
          "name": "...some label...",
          "attributes": { ... metadata ... },
          "children": [... sub-iterations, queries ...]
        }
        """
        name = f"Iteration: {iteration_dict.get('title', '')}"
        if iteration_dict["type"] == "root":
            name = iteration_dict["title"]  # e.g. "Root of Research"

        node = {
            "name": name,
            "type": iteration_dict["type"],
            "attributes": {
                "id": iteration_dict["id"],
                "depth": iteration_dict["depth"],
            },
            "children": [],
        }

        for child in iteration_dict.get("children", []):
            if child["type"] == "iteration":
                node["children"].append(self._iteration_to_d3_node(child))
            else:
                # It's a query
                query_label = f"Query: {child.get('query_text', '')[:40]}..."
                query_node = {
                    "name": query_label,
                    "type": "query",
                    "attributes": {
                        "id": child["id"],
                        "title": child["title"],
                        "goals": child["goals"],
                        "learnings": child["learnings"],
                        "urls": child["urls"],
                        "completed": child["completed"],
                    },
                    "children": [],
                }
                node["children"].append(query_node)
        return node

    def _generate_d3_html(self, hierarchy: Dict[str, Any]) -> str:
        """Generate HTML with custom layout for the tree visualization."""
        data_json = json.dumps(hierarchy, indent=2)

        # Read the custom HTML template
        with open("deep_research_ability/src/deep_research/tree_viz.html", "r") as f:
            html_template = f.read()

        # Insert the JSON data
        return html_template.replace("DATA_JSON_PLACEHOLDER", data_json)


#
# Optional CSV-based utility function
#
def create_tree_visualization_from_csv(csv_path: Path, output_path: Path) -> Tuple[Path, Path, Path]:
    """
    Create a D3 collapsible tree visualization from a research CSV file.
    Returns (json_path, html_path, research_json_path).
    """
    research_data = {
        "metadata": {},
        "iterations": {},
        "query_results": {},
    }

    with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            section = row.get("section", "")
            if section == "metadata":
                research_data["metadata"] = {k: v for k, v in row.items() if k != "section"}
                research_data["metadata"]["timestamp"] = datetime.now().isoformat()
            elif section == "iteration":
                iteration_id = row["iteration_id"]
                research_data["iterations"][iteration_id] = {
                    "id": iteration_id,
                    "title": row.get("title", f"Iteration {iteration_id}"),
                    "directions": row.get("directions", ""),
                    "queries_count": int(row.get("queries_count", 0)),
                    "depth": int(row.get("depth", 0)),
                    "parent_iteration_id": row.get("parent_iteration_id") or None,
                    "parent_query_id": row.get("parent_query_id") or None,
                    "queries": [],
                }
            elif section == "query":
                query_id = row["query_id"]
                iteration_id = row["iteration_id"]
                research_data["query_results"][query_id] = {
                    "id": query_id,
                    "iteration_id": iteration_id,
                    "title": row.get("title", f"Query {query_id}"),
                    "query": row.get("query", ""),
                    "goals": row.get("goals", ""),
                    "urls": row.get("urls", "").split(", ") if row.get("urls") else [],
                    "learnings": row.get("learnings", "").split(", ") if row.get("learnings") else [],
                    "completed": row.get("completed", "False").lower() == "true",
                }
                if iteration_id in research_data["iterations"]:
                    research_data["iterations"][iteration_id]["queries"].append(query_id)

    tree = ResearchTree(research_data)
    json_path = tree.save_json(output_path)
    research_json_path = tree.export_research_json(output_path)
    html_path = tree.visualize(output_path)
    return json_path, html_path, research_json_path
