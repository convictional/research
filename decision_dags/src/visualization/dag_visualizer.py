"""DAG visualization using NetworkX and Plotly."""

import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any
import colorsys

from ..models import DecisionDAG, NodeType, DecisionType


class DAGVisualizer:
    """Creates interactive visualizations of Decision DAGs."""

    # Color schemes
    NODE_COLORS = {
        NodeType.DECISION: "#4A90E2",  # Blue
        NodeType.OPTION: "#7ED321",    # Green
    }

    DECISION_TYPE_COLORS = {
        DecisionType.STRATEGIC: "#E94B3C",      # Red
        DecisionType.IMPLEMENTATION: "#F0B27A",  # Orange
        DecisionType.RESOURCE: "#52BE80",       # Green
        DecisionType.TIMING: "#5DADE2",         # Light Blue
        DecisionType.RISK: "#AF7AC5",           # Purple
        DecisionType.MARKET: "#F4D03F",         # Yellow
        DecisionType.PRODUCT: "#EC7063",        # Light Red
    }

    def __init__(self, dag: DecisionDAG):
        """Initialize visualizer with a DAG."""
        self.dag = dag
        self.graph = self._build_networkx_graph()
        self.layout = self._compute_hierarchical_layout()

    def _build_networkx_graph(self) -> nx.DiGraph:
        """Convert DecisionDAG to NetworkX graph."""
        G = nx.DiGraph()

        # Add nodes with attributes
        for node_id, node in self.dag.all_nodes.items():
            G.add_node(
                node_id,
                layer=node.layer,
                type=node.type,
                title=node.title,
                description=node.description,
                decision_type=node.decision_type,
                confidence_score=node.confidence_score,
                people_impacted=node.people_impacted,
                resource_requirements=node.resource_requirements,
            )

        # Add edges with attributes
        for edge in self.dag.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type,
                condition=edge.condition,
                likelihood=edge.likelihood,
                decision_reasoning_type=edge.decision_reasoning_type,
                cost_estimate=edge.cost_estimate,
                timeline_estimate=edge.timeline_estimate,
            )

        return G

    def _compute_hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """Compute hierarchical layout with tree-based positioning."""
        import math

        pos = {}

        # Group nodes by layer
        layers = {}
        for node_id, data in self.graph.nodes(data=True):
            layer = data['layer']
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node_id)

        if not layers:
            return pos

        max_layer = max(layers.keys())

        # Use a tree-based layout algorithm
        # Start from root and propagate positions down
        vertical_spacing = 2.0

        # Initialize root positions
        if 0 in layers:
            root_nodes = layers[0]
            for i, node_id in enumerate(sorted(root_nodes)):
                pos[node_id] = (i * 4.0 - (len(root_nodes) - 1) * 2.0, max_layer * vertical_spacing)

        # Process each layer based on parent positions
        for layer in range(1, max_layer + 1):
            if layer not in layers:
                continue

            layer_nodes = layers[layer]

            # Group nodes by their parent
            nodes_by_parent = {}
            orphan_nodes = []

            for node_id in layer_nodes:
                parents = list(self.graph.predecessors(node_id))
                if parents:
                    # Use first parent for positioning
                    parent = parents[0]
                    if parent not in nodes_by_parent:
                        nodes_by_parent[parent] = []
                    nodes_by_parent[parent].append(node_id)
                else:
                    orphan_nodes.append(node_id)

            # Position nodes under their parents
            for parent_id, children in nodes_by_parent.items():
                if parent_id in pos:
                    parent_x, _ = pos[parent_id]
                    n_children = len(children)

                    if n_children == 1:
                        # Single child - place directly under parent
                        pos[children[0]] = (parent_x, (max_layer - layer) * vertical_spacing)
                    else:
                        # Multiple children - spread them around parent
                        # Reduce spread for many children to avoid excessive width
                        spread = min(1.5, 4.0 / math.sqrt(n_children))
                        for i, child_id in enumerate(sorted(children)):
                            offset = (i - (n_children - 1) / 2) * spread
                            pos[child_id] = (
                                parent_x + offset,
                                (max_layer - layer) * vertical_spacing
                            )

            # Position orphan nodes to the right
            if orphan_nodes:
                # Find rightmost position
                if pos:
                    max_x = max(x for x, _ in pos.values())
                    start_x = max_x + 3.0
                else:
                    start_x = 0

                for i, node_id in enumerate(sorted(orphan_nodes)):
                    pos[node_id] = (
                        start_x + i * 1.5,
                        (max_layer - layer) * vertical_spacing
                    )

        # Apply force-directed adjustments to reduce overlaps
        # This is a simple version - just push overlapping nodes apart
        for _ in range(3):  # A few iterations
            adjustments = {}

            for layer, nodes in layers.items():
                layer_positions = [(node_id, pos.get(node_id, (0, 0))[0]) for node_id in nodes if node_id in pos]
                layer_positions.sort(key=lambda x: x[1])

                for i in range(len(layer_positions) - 1):
                    node1_id, x1 = layer_positions[i]
                    node2_id, x2 = layer_positions[i + 1]

                    min_spacing = 1.2
                    if x2 - x1 < min_spacing:
                        # Nodes too close - push them apart
                        center = (x1 + x2) / 2
                        adjustments[node1_id] = center - min_spacing / 2
                        adjustments[node2_id] = center + min_spacing / 2

            # Apply adjustments
            for node_id, new_x in adjustments.items():
                if node_id in pos:
                    old_x, y = pos[node_id]
                    pos[node_id] = (new_x, y)

        return pos

    def create_plotly_figure(
        self,
        show_labels: bool = True,
        highlight_paths: Optional[List[List[str]]] = None,
        filter_nodes: Optional[List[str]] = None
    ) -> go.Figure:
        """Create an interactive Plotly figure of the DAG."""
        # Initialize edge trace
        edge_traces = []

        # Create edges
        for edge in self.dag.edges:
            if filter_nodes and (edge.source_id not in filter_nodes or edge.target_id not in filter_nodes):
                continue

            x0, y0 = self.layout[edge.source_id]
            x1, y1 = self.layout[edge.target_id]

            # Check if edge is in highlighted path
            is_highlighted = False
            if highlight_paths:
                for path in highlight_paths:
                    if edge.source_id in path and edge.target_id in path:
                        idx_source = path.index(edge.source_id)
                        if idx_source < len(path) - 1 and path[idx_source + 1] == edge.target_id:
                            is_highlighted = True
                            break

            edge_color = 'rgba(255, 165, 0, 0.8)' if is_highlighted else 'rgba(125, 125, 125, 0.5)'
            edge_width = 3 if is_highlighted else 1.5

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=edge_width, color=edge_color),
                hoverinfo='text',
                text=f"{edge.condition}<br>Likelihood: {edge.likelihood}<br>Cost: {edge.cost_estimate or 'N/A'}<br>Timeline: {edge.timeline_estimate or 'N/A'}",
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Create nodes
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_hover = []
        node_sizes = []

        for node_id, node in self.dag.all_nodes.items():
            if filter_nodes and node_id not in filter_nodes:
                continue

            x, y = self.layout[node_id]
            node_x.append(x)
            node_y.append(y)

            # Color based on node type and decision type
            if node.type == NodeType.DECISION and node.decision_type:
                color = self.DECISION_TYPE_COLORS.get(node.decision_type, self.NODE_COLORS[node.type])
            else:
                color = self.NODE_COLORS.get(node.type, '#999999')

            # Highlight if in path
            if highlight_paths:
                for path in highlight_paths:
                    if node_id in path:
                        # Make color brighter
                        color = self._brighten_color(color)
                        break

            node_colors.append(color)

            # Node text and hover info
            node_text.append(f"{node.title[:30]}..." if len(node.title) > 30 else node.title)

            hover_text = f"<b>{node.title}</b><br>"
            hover_text += f"Type: {node.type.value}<br>"
            hover_text += f"Layer: {node.layer}<br>"
            if node.decision_type:
                hover_text += f"Decision Type: {node.decision_type.value}<br>"
            hover_text += f"<br>{node.description[:200]}..."
            if node.confidence_score:
                hover_text += f"<br>Confidence: {node.confidence_score:.2f}"

            node_hover.append(hover_text)

            # Size based on connections
            in_degree = self.graph.in_degree(node_id)
            out_degree = self.graph.out_degree(node_id)
            node_sizes.append(20 + (in_degree + out_degree) * 3)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if show_labels else 'markers',
            text=node_text if show_labels else None,
            textposition="bottom center",
            hoverinfo='text',
            hovertext=node_hover,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color='white', width=2)
            ),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        # Update layout
        # Dynamically adjust height based on number of layers
        num_layers = len(set(data['layer'] for _, data in self.graph.nodes(data=True)))
        fig_height = max(800, num_layers * 150)  # At least 150px per layer

        # Calculate axis ranges with better padding
        if node_x and node_y:
            x_padding = (max(node_x) - min(node_x)) * 0.1
            y_padding = (max(node_y) - min(node_y)) * 0.1
            x_range = [min(node_x) - x_padding, max(node_x) + x_padding]
            y_range = [min(node_y) - y_padding, max(node_y) + y_padding]
        else:
            x_range = [-10, 10]
            y_range = [-1, 10]

        fig.update_layout(
            title=dict(
                text=f"Decision DAG: {self.dag.metadata.get('problem_statement', 'Unknown Problem')[:100]}...",
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=60),  # Increased margins
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=x_range,
                automargin=True
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=y_range,
                automargin=True
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=fig_height,
            clickmode='event+select',
            dragmode='pan'  # Default to pan mode for easier navigation
        )

        # Add legend for node types
        legend_items = [
            dict(name="Decision Node", marker=dict(color=self.NODE_COLORS[NodeType.DECISION], size=10)),
            dict(name="Option Node", marker=dict(color=self.NODE_COLORS[NodeType.OPTION], size=10)),
        ]

        for i, item in enumerate(legend_items):
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=item['marker'],
                showlegend=True,
                name=item['name']
            ))

        return fig

    def _brighten_color(self, hex_color: str) -> str:
        """Make a color brighter for highlighting."""
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Convert to HSL, increase lightness
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        l = min(1.0, l * 1.3)  # Increase lightness by 30%

        # Convert back to RGB
        r, g, b = colorsys.hls_to_rgb(h, l, s)

        # Convert to hex
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def get_critical_paths(self, top_k: int = 3) -> List[List[str]]:
        """Get the top k critical paths through the DAG."""
        # Find all paths from root to leaf nodes
        root_nodes = [n for n, d in self.graph.in_degree() if d == 0]
        leaf_nodes = [n for n, d in self.graph.out_degree() if d == 0]

        all_paths = []
        for root in root_nodes:
            for leaf in leaf_nodes:
                try:
                    paths = list(nx.all_simple_paths(self.graph, root, leaf))
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    continue

        # Score paths (could be based on confidence, cost, etc.)
        # For now, prefer longer paths with high-confidence nodes
        scored_paths = []
        for path in all_paths:
            score = len(path)  # Length bonus
            for node_id in path:
                node = self.dag.all_nodes[node_id]
                if node.confidence_score:
                    score += node.confidence_score
            scored_paths.append((score, path))

        # Sort by score and return top k
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        return [path for _, path in scored_paths[:top_k]]

    def get_dag_stats(self) -> Dict[str, Any]:
        """Get statistics about the DAG."""
        decision_nodes = [n for n in self.dag.all_nodes.values() if n.type == NodeType.DECISION]
        option_nodes = [n for n in self.dag.all_nodes.values() if n.type == NodeType.OPTION]

        # Count decision types
        decision_type_counts = {}
        for node in decision_nodes:
            if node.decision_type:
                dt = node.decision_type.value
                decision_type_counts[dt] = decision_type_counts.get(dt, 0) + 1

        return {
            "total_nodes": len(self.dag.all_nodes),
            "decision_nodes": len(decision_nodes),
            "option_nodes": len(option_nodes),
            "total_edges": len(self.dag.edges),
            "max_depth": self.dag.get_max_layer(),
            "decision_type_distribution": decision_type_counts,
            "avg_confidence": sum(n.confidence_score or 0 for n in self.dag.all_nodes.values()) / len(self.dag.all_nodes) if self.dag.all_nodes else 0,
        }
