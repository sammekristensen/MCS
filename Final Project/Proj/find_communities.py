import gravis as gv
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from colorthief import ColorThief
import tqdm as tqdm
import matplotlib as mpl

PLOT1 = 1

data1 = pd.read_csv('votes_1975_2015.csv')
# Group the DataFrame by 'Country' and 'Source Country', and calculate the mean of 'total_points'
data = data1.groupby(['Country', 'Source Country'])['total_points'].mean().reset_index()
data = data[data['total_points'] >= 5.5]


g = nx.from_pandas_edgelist(data, 
                            source='Source Country',
                            target='Country',
                            edge_attr='total_points',
                            create_using=nx.DiGraph())


# Convert networkx graph to igraph graph with names
import igraph as ig
import leidenalg as la
# Convert networkx graph to igraph graph
G_ig = ig.Graph.TupleList(g.edges(), directed=True)
edges = [(e[0], e[1], e[2]['total_points']) for e in g.edges(data=True)]
G_ig = ig.Graph.TupleList(edges, directed=True, edge_attrs=['weight'])
resolution = 1.0
communities = la.find_partition(G_ig, la.RBConfigurationVertexPartition, resolution_parameter=resolution)
# Get a mapping from igraph vertex indices to country names
index_to_name = {v.index: v['name'] for v in G_ig.vs}
# Convert communities to a list of country names
communities_list = [[index_to_name[node] for node in community] for community in communities]
communities = [frozenset(community) for community in communities_list]

# Compute modularity
modularity_value = nx.algorithms.community.quality.modularity(g, communities_list)
print(f'Modularity: {modularity_value}')

try:
    countries = pd.read_csv('countries.csv', index_col='Country', encoding='latin1')
except UnicodeDecodeError:
    countries = pd.read_csv('countries.csv', index_col='Country', encoding='ISO-8859-1')

pos = {}
for node in g.nodes():
    pos[node] = (
                    max(-10,min(countries.loc[node]['longitude'],55)), # fixing scale
                    -max(countries.loc[node]['latitude'],25) #fixing scale
    )

scale = 12
# Add coordinates as node annotations that are recognized by gravis
for name, (x, y) in pos.items():
    node = g.nodes[name]
    node['x'] = x * scale
    node['y'] = y * scale

if PLOT1:
    # Assign colors to nodes based on communities
    colors = ['red', 'blue', 'green', 'orange', 'pink', 'yellow']
    for i, community in enumerate(communities):
        for node in community:
            g.nodes[node]['color'] = colors[i]

    # Create the visualization using Gravis
    fig = gv.d3(g, layout_algorithm_active=False, use_node_size_normalization=False, node_size_normalization_max=30,
                        use_edge_size_normalization=True, edge_size_data_source='weight', edge_curvature=0.3)

    fig.display()
