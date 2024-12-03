import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

def is_graph_disconnected(graph):
    """
    Check if a graph is disconnected.

    Parameters:
        graph (nx.Graph): Input graph.

    Returns:
        bool: True if the graph is disconnected, False otherwise.
    """
    return nx.number_of_isolates(graph) > 0

def generate_erdos_renyi(n,p):
    """
    Generate many Erdős-Rényi graphs with given parameters.

    Parameters:
        n (int): Number of nodes in the graph
        p (float): Probability of edge existence between any two nodes.

    Returns:
        erdos_renyi_graph
    """
    return nx.erdos_renyi_graph(n,p)

def count_disconnected_graphs(n, p, num_graphs):
    """
    Count how many graphs are disconnected.

    Parameters:
        n (int): Number of nodes in the graph.
        p (float): Probability of edge existence between any two nodes.
        num_graphs (int): Number of graphs to generate.

    Returns:
        int: Number of disconnected graphs.
    """
    num_disconnected = 0
    for _ in tqdm(range(num_graphs)):
        graph = generate_erdos_renyi(n,p)
        if is_graph_disconnected(graph):
            num_disconnected += 1
    return num_disconnected

def plot_graph(graph):
    """
    Plots a single Erdős-Rényi graph with given parameters.

    Parameters:
        graph (nx.Graph): Input graph.
    """
    plt.figure()
    plt.title(f"Erdös-Rényi Graph (n={n}, p=1/(n-1))")
    nx.draw(graph, with_labels=False, node_color='red', node_size=5, width=0.1)
    plt.show()


# Parameters
n = 1000  # Number of nodes
eps = 0.005  # Small value for epsilon
p = (1 - eps) * np.log(n)/n - eps # Probability of edge existence
num_graphs = 10  # Number of graphs to generate

Erdos_Renyi = generate_erdos_renyi(n,p)
print(Erdos_Renyi)
#plot_graph(Erdos_Renyi)

# Generate and count disconnected graphs
num_disconnected = count_disconnected_graphs(n, p, num_graphs)

# Print results
print(f"Number of graphs with isolated component: {num_disconnected} out of {num_graphs}")
