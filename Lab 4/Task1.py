import numpy as np
import urllib.request
import re
from igraph import *
import leidenalg as la
import networkx as nx

import igraph as ig

# Read the data from the URL
url = urllib.request.urlopen("http://vlado.fmf.uni-lj.si/pub/networks/data/sport/football.net")
data_file = url.readlines()


G = nx.read_pajek('Football.net')
countries = list(G.nodes)

# Create Adjacency matrix
adjacency_matrix = np.zeros((35,35))
weightlist = []

# G.edges did not give us the weight so we have to do this by hand
for arcs in data_file[37:-1]:
    arc = arcs.decode('utf-8')
    words_list = re.split(r'\s+', arc)
    row = int(words_list[1])
    col = int(words_list[2])
    weight = int(words_list[3])
    weightlist.append(weight)
    adjacency_matrix[row-1, col-1] = weight

g = Graph.Weighted_Adjacency(adjacency_matrix.tolist())

# Set vertex and enge attributes
g.vs["label"] = countries
g.es["width"] = [x/4 for x in weightlist]
g.es["arrow_size"] = 0.8

# Use Leiden algorithm to find clusters
partition = la.find_partition(g, la.ModularityVertexPartition)

# Plot the graph with community structure
plot(partition, vertex_size=16, vertex_label_size=12, edge_width=g.es["width"], edge_arrow_size=g.es["arrow_size"], layout=g.layout_circle(), target="football_network.png")

ig.plot(partition,target="football_network2.png")

# Create subgraphs for each cluster in partition
clusters = [g.subgraph(cluster) for cluster in partition]

# Plot each subgraph containing the clusters
for idx, cluster_graph in enumerate(clusters):
    plot(cluster_graph, vertex_size=22, vertex_label_size=16, edge_width=cluster_graph.es["width"], edge_arrow_size=cluster_graph.es["arrow_size"], layout=cluster_graph.layout_circle(), target=f"cluster_{idx+1}_network.png")

communities = [[2, 10, 13, 14, 18, 24, 25, 28, 29, 32], [1, 3, 5, 7, 15, 23, 26, 31, 34], [0, 4, 6, 8, 11, 17, 22, 27], [9, 12, 16, 19, 20, 21, 30, 33]]

# Initialize Q
Q = 0
m = np.sum(adjacency_matrix) / 2

# Iterate over each community
for i, community_i in enumerate(communities):
    for j, community_j in enumerate(communities):
        delta = 1 if i == j else 0
        Aij = 0
        sum_of_degrees_i = np.sum([np.sum(adjacency_matrix[node]) for node in community_i])
        sum_of_degrees_j = np.sum([np.sum(adjacency_matrix[node]) for node in community_j])
        for node_i in community_i:
            for node_j in community_j:
                Aij += adjacency_matrix[node_i][node_j]
        expected_edges = (sum_of_degrees_i * sum_of_degrees_j) / (2 * m)
        Q += (Aij - expected_edges) * delta

# Normalize Q by the total number of edges
Q /= (2 * m)

print("Modularity:", Q)


# Create an igraph Graph object from the adjacency matrix
g = ig.Graph.Adjacency((adjacency_matrix > 0).tolist())

# Assign community membership to each node
membership = [0] * len(adjacency_matrix)
for i, community in enumerate(communities):
    for node in community:
        membership[node] = i

# Compute Q using igraph's Q function
Q = g.modularity(membership)

print("Modularity (using igraph function):", Q)