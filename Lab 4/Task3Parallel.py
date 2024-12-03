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
    number_of_nodes = graph.number_of_nodes()
    numerator = 0
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            for k in range(number_of_nodes):
                numerator += adjacency_matrix[i,j]*adjacency_matrix[j,k]*adjacency_matrix[k,i]

    denominator = 0
    for row in adjacency_matrix:
        n_i = np.sum(row)
        denominator += n_i*(n_i-1)
    if denominator == 0:
        return 0
    return numerator/denominator

def expected_clustering_coefficient_single(p, n, num_graphs):
    """
    Compute the expected clustering coefficient for G(n, p) in a single process.

    Parameters:
        p (float): Probability of edge existence between any two nodes.
        n (int): Number of nodes in the graph.
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

def expected_clustering_coefficient_parallel(p_values, n, num_graphs):
    """
    Compute the expected clustering coefficient for G(n, p) in parallel.

    Parameters:
        p_values (list): List of probabilities of edge existence between any two nodes.
        n (int): Number of nodes in the graph.
        num_graphs (int): Number of graphs to generate.

    Returns:
        list: List of expected clustering coefficients.
    """
    # Create a pool of workers
    pool = multiprocessing.Pool()
    func = partial(expected_clustering_coefficient_single, n=n, num_graphs=num_graphs)
    expected_C_values = list(pool.map(func, p_values))
    pool.close()
    pool.join()
    return expected_C_values

def main():
    # Parameters
    n = 100  # Number of nodes
    num_graphs = 10  # Number of graphs to generate for each p
    p_values = np.linspace(0, 1, 32)  # Discretization of [0, 1] for p

    # Compute expected clustering coefficient in parallel
    expected_C_values = expected_clustering_coefficient_parallel(p_values, n, num_graphs)

    # Plot expected C as a function of p
    plt.plot(p_values, expected_C_values)
    plt.xlabel('Probability (p)')
    plt.ylabel('Expected Clustering Coefficient (C)')
    plt.title('Expected Clustering Coefficient vs. Probability (p)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
