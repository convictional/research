"""
Network Effect Multiplier: Group Explosion Data Generator

Generates data showing all groups that include YOU in a 16-person network.
Demonstrates combinatorial explosion from one person's perspective.

Author: Adam McCabe
"""

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple


# Configuration
NUM_NODES = 16
YOU_NODE = 0  # Node representing "you"
GROUP_SIZES = [2, 3, 4]


def generate_circular_layout(n: int) -> List[Dict[str, float]]:
    """
    Generate circular layout for n nodes.

    Args:
        n: Number of nodes

    Returns:
        List of {x, y} positions normalized to 0-1 range
    """
    positions = []
    for i in range(n):
        # Place nodes in circle, with YOU at top (90 degrees = pi/2 radians)
        angle = (math.pi / 2) - (2 * math.pi * i / n)
        # Normalize to 0-1 range with padding
        x = 0.5 + 0.4 * math.cos(angle)
        y = 0.5 + 0.4 * math.sin(angle)
        positions.append({"x": x, "y": y})
    return positions


def enumerate_groups_containing_node(node: int, all_nodes: List[int], group_size: int) -> List[List[int]]:
    """
    Enumerate all groups of given size that contain the specified node.

    Args:
        node: Node that must be in every group
        all_nodes: List of all node IDs
        group_size: Size of groups to enumerate

    Returns:
        List of groups (each group is a list of node IDs)
    """
    # Get other nodes (excluding the specified node)
    other_nodes = [n for n in all_nodes if n != node]

    # Choose (group_size - 1) nodes from others to form groups with specified node
    groups = []
    for combo in combinations(other_nodes, group_size - 1):
        group = [node] + list(combo)
        groups.append(group)

    return groups


def get_edges_in_group(group: List[int]) -> List[Tuple[int, int]]:
    """
    Get all edges connecting nodes in a group (complete subgraph).

    Args:
        group: List of node IDs

    Returns:
        List of edges as (u, v) tuples
    """
    edges = []
    for i, u in enumerate(group):
        for v in group[i + 1 :]:
            edges.append((min(u, v), max(u, v)))  # Normalize edge direction
    return edges


def generate_all_groups() -> Dict[int, List[Dict]]:
    """
    Generate all groups containing YOU_NODE for each group size.

    Returns:
        Dict mapping group_size -> list of group dicts
        Each group dict contains:
            - nodes: list of node IDs in group
            - edges: list of edges connecting those nodes
    """
    all_nodes = list(range(NUM_NODES))
    result = {}

    for group_size in GROUP_SIZES:
        groups_list = enumerate_groups_containing_node(YOU_NODE, all_nodes, group_size)

        # Convert to dict format with edges
        groups_data = []
        for group in groups_list:
            edges = get_edges_in_group(group)
            groups_data.append({"nodes": group, "edges": edges})

        result[group_size] = groups_data

        print(f"  Enumerated {len(groups_data)} groups of size {group_size}")

    return result


def generate_all_edges() -> List[Tuple[int, int]]:
    """
    Generate all edges in complete graph K_n.

    Returns:
        List of all edges as (u, v) tuples
    """
    edges = []
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            edges.append((i, j))
    return edges


def save_output(positions: List[Dict], all_edges: List[Tuple], groups: Dict, output_dir: Path) -> None:
    """
    Save generated data to JSON file.

    Args:
        positions: Node positions
        all_edges: All edges in complete graph
        groups: Enumerated groups by size
        output_dir: Directory to save output
    """
    output_dir.mkdir(exist_ok=True)

    data = {
        "num_nodes": NUM_NODES,
        "you_node": YOU_NODE,
        "node_positions": positions,
        "all_edges": all_edges,
        "group_sizes": GROUP_SIZES,
        "groups": {str(k): v for k, v in groups.items()},  # JSON keys must be strings
    }

    json_path = output_dir / "network_groups_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Saved network groups data: {json_path}")


def main() -> None:
    """Generate network groups animation data."""
    print("Generating network groups data...")
    print(f"  Network size: {NUM_NODES} people")
    print(f"  YOU are node {YOU_NODE}")
    print(f"  Group sizes: {GROUP_SIZES}")
    print()

    # Generate circular layout
    positions = generate_circular_layout(NUM_NODES)

    # Generate all edges
    all_edges = generate_all_edges()
    print(f"  Complete graph K_{NUM_NODES} has {len(all_edges)} edges")
    print()

    # Enumerate all groups containing YOU
    groups = generate_all_groups()

    # Calculate totals
    total_groups = sum(len(g) for g in groups.values())
    print(f"\nTotal groups containing YOU: {total_groups:,}")
    for k in GROUP_SIZES:
        print(f"  Groups of {k}: {len(groups[k]):,}")

    # Save output
    output_dir = Path(__file__).parent / "output"
    save_output(positions, all_edges, groups, output_dir)


if __name__ == "__main__":
    main()
