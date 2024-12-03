import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial

def compute_clustering_coefficient(graph):
    """
    Compute the global clustering coefficient of a graph.

    Parameters:
        graph (nx.Graph): Input graph.

    Returns:
        float: Global clustering coefficient.
    """
    adjacency_matrix = nx.adjacency_matrix(graph)
    num=0
    for i in range(graph.number_of_nodes()):
        for j in range(graph.number_of_nodes()):
            for k in range(graph.number_of_nodes()):
                num += adjacency_matrix[i,j]*adjacency_matrix[j,k]*adjacency_matrix[k,i]

    denom = 0
    for row in adjacency_matrix:
        n_i = np.sum(row)
        denom += n_i*(n_i-1)
    if denom == 0:
        return 0
    return num/denom

def expected_clustering_coefficient(n, p, num_graphs):
    """
    Compute the expected clustering coefficient for G(n, p).

    Parameters:
        n (int): Number of nodes in the graph.
        p (float): Probability of edge existence between any two nodes.
        num_graphs (int): Number of graphs to generate.

    Returns:
        float: Expected clustering coefficient.
    """
    avg_clustering_coefficient = 0
    for _ in tqdm(range(num_graphs)):
        graph = nx.erdos_renyi_graph(n, p)
        avg_clustering_coefficient += compute_clustering_coefficient(graph)
    avg_clustering_coefficient /= num_graphs
    return avg_clustering_coefficient

def main():
    # Parameters
    n = 100  # Number of nodes
    num_graphs = 10  # Number of graphs to generate for each p
    p_values = np.linspace(0, 1, 10)  # Discretization of [0, 1] for p

    expected_C_values = []
    # Create a pool of workers
    for p in p_values:
        expected_C_values.append(expected_clustering_coefficient(n,p,num_graphs))

    # Plot expected C as a function of p
    plt.plot(p_values, expected_C_values)
    plt.xlabel('Probability (p)')
    plt.ylabel('Expected Clustering Coefficient (C)')
    plt.title('Expected Clustering Coefficient vs. Probability (p)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
