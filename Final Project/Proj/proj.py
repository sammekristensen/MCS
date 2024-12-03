from __future__ import division
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import sys
import os
plt.rcParams["figure.figsize"] = (20,10)
from itertools import chain
import tqdm as tqdm
from colorthief import ColorThief
import leidenalg as la

warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

data1 = pd.read_csv('votes_1975_2015.csv')


# Group the DataFrame by 'Country' and 'Source Country', and calculate the mean of 'total_points'
data = data1.groupby(['Country', 'Source Country'])['total_points'].mean().reset_index()

G = nx.from_pandas_edgelist(data, 
                            source='Source Country',
                            target='Country',
                            edge_attr='total_points',
                            create_using=nx.DiGraph())

# Try reading the file with a different encoding
try:
    countries = pd.read_csv('countries.csv', index_col='Country', encoding='latin1')
except UnicodeDecodeError:
    countries = pd.read_csv('countries.csv', index_col='Country', encoding='ISO-8859-1')

pos_geo = {}
for node in G.nodes():
    pos_geo[node] = (
                    max(-10,min(countries.loc[node]['longitude'],55)), # fixing scale
                    max(countries.loc[node]['latitude'],25) #fixing scale
    )


flags = {}
flag_color = {}
for node in tqdm.tqdm(G.nodes()):
    flags[node] = 'flags/'+(countries.loc[node]['cc3']).lower().replace(' ','')+'.png'   
    flag_color[node] =  ColorThief(flags[node]).get_color(quality=1)

def RGB(red,green,blue): 
    return '#%02x%02x%02x' % (red,green,blue)

ax=plt.gca()
fig=plt.gcf()
plt.axis('off')
plt.title('Eurovision Final Votes 1975-2015',fontsize = 24)

trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform

pos = pos_geo

# draw edges
for e in G.edges(data=True):
    width = e[2]['total_points']/12 #normalize by max points
    if width > 0.4: #filter small votes
        nx.draw_networkx_edges(G,pos,edgelist=[e],width=width, edge_color = RGB(*flag_color[e[0]]) )
        # in networkx versions >2.1 arrowheads can be adjusted

#draw nodes    
for node in G.nodes():      
    imsize = max((0.3*G.in_degree(node,weight='total_points')
                  /max(dict(G.in_degree(weight='total_points')).values()))**2,0.03)
    
    # size is proportional to the votes
    flag = mpl.image.imread(flags[node])
    
    (x,y) = pos[node]
    xx,yy = trans((x,y)) # figure coordinates
    xa,ya = trans2((xx,yy)) # axes coordinates
    
    country = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
    country.imshow(flag)
    country.set_aspect('equal')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)

    
fig.savefig('images/eurovision_map.png')
plt.close(fig)  # Close the figure to free memory
print("Eurovision map done")

##########################################################################################################################################################
import igraph as ig

# Convert networkx graph to igraph graph
G_ig = ig.Graph.TupleList(G.edges(), directed=True)

# Community detection
resolution = 1.0
partition = la.find_partition(G_ig, la.RBConfigurationVertexPartition, resolution_parameter=resolution)
partition_list = [list(sub_partition) for sub_partition in partition]

# Print out communities
print("Detected communities:")
for i, community in enumerate(partition_list):
    print(f"Community {i+1}: {community}")

node_to_community = {}
for i, community in enumerate(partition_list):
    for node in community:
        node_to_community[G_ig.vs[node]["name"]] = i

community_colors = mpl.cm.tab20.colors  # Adjust the color scheme as needed

ax = plt.gca()
fig = plt.gcf()
plt.axis('off')
plt.title('Eurovision Final Votes 1975-2015 - Community Partitions', fontsize=24)

trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform

# Draw edges with community colors
for e in G.edges(data=True):
    width = e[2]['total_points'] /12 # normalize by max points
    if width > 0.4:  # filter small votes
        source_community = node_to_community.get(e[0], 0)
        edge_color = community_colors[source_community % len(community_colors)]
        nx.draw_networkx_edges(G, pos, edgelist=[e], width=width, edge_color=edge_color)

# Draw nodes with flags
for node in G.nodes():      
    imsize = max((0.3 * G.in_degree(node, weight='total_points') / max(dict(G.in_degree(weight='total_points')).values())) ** 2, 0.03)
    
    # size is proportional to the votes
    flag = mpl.image.imread(flags[node])
    
    (x, y) = pos[node]
    xx, yy = trans((x, y))  # figure coordinates
    xa, ya = trans2((xx, yy))  # axes coordinates
    
    country = plt.axes([xa - imsize / 2.0, ya - imsize / 2.0, imsize, imsize ])
    country.imshow(flag)
    country.set_aspect('equal')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)

fig.savefig('images/eurovision_communities.png')
plt.close(fig)  # Close the figure to free memory
print("Community partitions done")

##########################################################################################################################################################
# Centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

fig_centrality, ax_centrality = plt.subplots(figsize=(20, 10))
plt.axis('off')
plt.title('Eurovision Final Votes 1975-2015 - Centrality Measures', fontsize=24)

# Draw edges
nx.draw_networkx_edges(G, pos, ax=ax_centrality, alpha=0.5)

# Draw nodes with centrality measures
nodes = nx.draw_networkx_nodes(G, pos, ax=ax_centrality,
                               node_size=[v * 1000 for v in degree_centrality.values()],
                               node_color=list(betweenness_centrality.values()),
                               cmap=plt.cm.plasma,
                               alpha=0.8)

# Add color bar
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(betweenness_centrality.values()), vmax=max(betweenness_centrality.values())))
sm.set_array([])
fig_centrality.colorbar(sm, ax=ax_centrality, label='Betweenness Centrality')

# Draw labels for the nodes
nx.draw_networkx_labels(G, pos, ax=ax_centrality, font_size=12)

fig_centrality.savefig('images/eurovision_centrality_measures.png')
plt.close(fig_centrality)  # Close the figure to free memory

print("Centrally part done")

#########################################################################################################################################################