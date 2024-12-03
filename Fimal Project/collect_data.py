import networkx as nx
import pandas as pd
import pycountry
import country_converter as coco

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

data = pd.read_csv('votes.csv')
data.drop(['from_country_id', 'to_country_id', 'tele_points', 'jury_points'], axis=1, inplace=True)

# Only the final
data = data.loc[data['round'] == 'final']
data.drop(['round'], axis=1, inplace=True)

#Change to years 1975-2015, same voting system
# Filter out rows between 1975 and 2015
data = data.loc[(data['year'] >= 1975) & (data['year'] <= 2015)]
# Remove yu and cs since that is no longer countries
data = data.loc[~data['from_country'].isin(['cs', 'yu']) & ~data['to_country'].isin(['cs', 'yu'])]

european_country_codes = ['am','ge','az','al', 'ad', 'at', 'by', 'be', 'ba', 'bg', 'hr', 'cy', 'cz', 'dk', 'ee', 'fo', 'fi', 'fr', 'de', 'gi', 'gr', 'hu', 'is', 'ie', 'it', 'xk', 'lv', 'li', 'lt', 'lu', 'mt', 'md', 'mc', 'me', 'nl', 'mk', 'no', 'pl', 'pt', 'ro', 'ru', 'sm', 'rs', 'sk', 'si', 'es', 'se', 'ch', 'ua', 'gb', 'va', 'il']
data = data.loc[data['to_country'].isin(european_country_codes) & data['from_country'].isin(european_country_codes)]

# Convert country codes to country names
cc = coco.CountryConverter()
data['Country'] = data['to_country'].apply(cc.convert, to='name_short')
data['Source Country'] = data['from_country'].apply(cc.convert, to='name_short')


# Have to switch some country names
data['Country'].replace('Türkiye', 'Turkey', inplace=True)
data['Source Country'].replace('Türkiye', 'Turkey', inplace=True)
data['Country'].replace('North Macedonia', 'Macedonia', inplace=True)
data['Source Country'].replace('North Macedonia', 'Macedonia', inplace=True)
data['Country'].replace('Czechia', 'Czech Republic', inplace=True)
data['Source Country'].replace('Czechia', 'Czech Republic', inplace=True)

# Drop the original 'to_country' and 'from_country' columns
data.drop(['to_country', 'from_country'], axis=1, inplace=True)

print(data.head())
data.to_csv('votes_1975_2015.csv', index=False)

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
plt.title('Eurovision 2018 Final Votes',fontsize = 24)

trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform

tick_params = {'top':'off', 'bottom':'off', 'left':'off', 'right':'off',
              'labelleft':'off', 'labelbottom':'off'} #flag grid params

styles = ['dotted','dashdot','dashed','solid'] # line styles

pos = pos_geo  

# draw edges
for e in G.edges(data=True):
    width = e[2]['total_points']/24 #normalize by max points
    style=styles[int(width*3)]
    if width>0.3: #filter small votes
        nx.draw_networkx_edges(G,pos,edgelist=[e],width=width, style=style, edge_color = RGB(*flag_color[e[0]]) )
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