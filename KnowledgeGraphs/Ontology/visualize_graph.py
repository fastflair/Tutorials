#!/usr/bin/env python3
"""
visualize_graph.py

This script loads a knowledge graph saved as a pickle file ("knowledge_graph.pkl")
and visualizes it using NetworkX and matplotlib.

Each node is expected to have attributes "name" and "node_type".
This visualization displays the node's name along with its type, and the edges
are labeled with the "relation" attribute.
"""

import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt

KG_FILE = "knowledge_graph.pkl"

def load_graph(file_path: str):
    """Load a NetworkX graph from a pickle file."""
    if not os.path.exists(file_path):
        print(f"Knowledge graph file '{file_path}' not found.")
        return None
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph

def get_node_colors(graph: nx.Graph):
    """
    Returns a list of colors for the nodes.
    Nodes with a node_type other than "Other" are colored skyblue;
    nodes with "Other" are colored lightgray.
    """
    colors = []
    for n, data in graph.nodes(data=True):
        node_type = data.get("node_type", "Other")
        colors.append("skyblue" if node_type.lower() != "other" else "lightgray")
    return colors

def get_node_labels(graph: nx.Graph):
    """
    Constructs a dictionary for node labels.
    Each label is formatted as "name\n(node_type)".
    """
    labels = {}
    for n, data in graph.nodes(data=True):
        name = data.get("name", n)
        node_type = data.get("node_type", "Other")
        labels[n] = f"{name}\n({node_type})"
    return labels

def visualize_graph(graph: nx.Graph):
    """Visualize the graph with node and edge labels using a spring layout."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)  # spring layout

    # Generate node labels including node type info
    node_labels = get_node_labels(graph)
    
    # Get node colors
    node_colors = get_node_colors(graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=800, alpha=0.9)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(graph, pos, arrowstyle='->', arrowsize=15)
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10)
    
    # Draw edge labels using the relation attribute
    edge_labels = {(u, v): data.get("relation", "") for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red", font_size=9)
    
    plt.title("Knowledge Graph Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    graph = load_graph(KG_FILE)
    if graph is None:
        return
    visualize_graph(graph)

if __name__ == "__main__":
    main()
