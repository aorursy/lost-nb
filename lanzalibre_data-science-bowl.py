#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
pd.set_option("display.max_rows", 999)


# In[3]:


trainLabelsDF=pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")


# In[4]:


traindf=pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv") 


# In[5]:


def extract_json(data){
    from pandas.io.json import json_normalize
    json=json_normalize(data, 'counties', ['state', 'shortname',['info', 'governor']])
    return json
}
#traindf.installation_id.value_counts()


# In[6]:


def extract_json(data):
    import json
    j1 = json.loads(data)
    return j1.keys()

traindf.head().event_data.apply(extract_json)


# In[7]:


def extract_json(data):
    import json
    from pandas.io.json import json_normalize
    j1 = json.loads(data)
    return json_normalize(j1)['event_code']

traindf.head().event_data.apply(extract_json)


# In[8]:


installDF=traindf[traindf['installation_id']=='f9296363']
#installDF.head()
#installDF.groupby(['game_session'])['event_id'].transform(lambda x: ','.join(x))
a=installDF[['event_count','event_id','game_session','game_time']]
a['event_count_plus']=a['event_count'].apply(lambda x: x+1)
edge_list=pd.merge(a,installDF[['event_count','event_id','game_session','game_time']],how='left',left_on=['event_count_plus','game_session'],right_on=['event_count','game_session'])
edge_list['duration']=edge_list['game_time_y']-edge_list['game_time_x']
edge_list.head()


# In[9]:


installDF.head()


# In[10]:


edge_list_count=edge_list.groupby(['event_id_x','event_id_y'])['event_count_x'].count().reset_index()
edge_list_count.event_count_x.describe()
#edge_list_count.head()


# In[11]:


node_list=edge_list.groupby(['event_id_x'])['duration','event_count_x'].sum().reset_index()


# In[12]:


import networkx as nx
import math
from IPython.display import FileLink
    
node_attr = node_list.set_index('event_id_x').to_dict('index')
G=nx.from_pandas_edgelist(edge_list_count, 'event_id_x', 'event_id_y', 'event_count_x')
nx.set_node_attributes(G, node_attr)

def drawG(G,file_name):
    fig = plt.figure()
    fig.clf()
    pos = nx.circular_layout(G)
    edges = G.edges()
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['event_count_x'] >= 5]
    emid = [(u, v) for (u, v, d) in G.edges(data=True) if 3 <= d['event_count_x'] < 5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['event_count_x'] < 3]

    pos = nx.spring_layout(G,k=1/math.sqrt(G.order()))  # positions for all nodes

    # nodes
    durations=[d['duration']/2000 for n,d in G.nodes(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=durations, font_size=0, label=None )
    total_duration=sum(durations)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1, edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=emid, width=1, edge_color='b')
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=1, edge_color='g')

    # labels
    custom_labels=dict((n,",".join([n,str(d['duration']),str(d['event_count_x'])])) for n,d in G.nodes(data=True))

    nx.draw_networkx_labels(G, pos, labels=custom_labels, font_size=3, font_family='sans-serif')

    
    fig.suptitle(file_name+":"+"{0:.2f}".format(total_duration), fontsize=20)
    #plt.figure(figsize=(1000,1000))
    plt.axis('off')
    #plt.show()
    #plt.savefig("graph.png", dpi=3000)
    #plt.savefig(file_name)
    fig.savefig(file_name)
    return FileLink(file_name)
drawG(G,"graph.pdf")


# In[13]:


subgraphs=[G.subgraph(c) for c in nx.connected_components(G)]


# In[14]:


[drawG(subgraphs[i],'graph'+str(i)+'.pdf') for i in range(0,20)]


# In[15]:


get_ipython().system('ls')


# In[16]:


from IPython.display import FileLink
FileLink("graph12.pdf")


# In[17]:



def CustomParser(data):
    import json
    j1 = json.loads(data)
    return j1
specsdf=pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv",converters={'args':CustomParser})
lookup=specsdf.reset_index().set_index("event_id")['info'].to_dict()
seq_data=pd.DataFrame([(n,d['duration'],d['event_count_x'],lookup[n]) for n,d in subgraphs[12].nodes(data=True)])
seq_data['avg_duration']=seq_data[1]/seq_data[2]
seq_data


# In[18]:


list(subgraphs[12].nodes())


# In[19]:


analysisDF=traindf[['event_id','event_code','type','title','world']]
labels=analysisDF.groupby('event_id')['event_code','type','title','world'].apply(lambda x: pd.unique(x.values.ravel()).tolist()).reset_index()
labels.head()


# In[20]:


labels.head()


# In[21]:


analysisDF=traindf[traindf['event_id'].isin(list(subgraphs[12].nodes()))][['event_id','event_code','type','title','world']]
analysisDF.groupby('event_id')['event_code','type','title','world'].apply(lambda x: pd.unique(x.values.ravel()).tolist())


# In[22]:


def CustomParser(data):
    import json
    j1 = json.loads(data)
    return j1
specsdf=pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv",converters={'args':CustomParser})
specsdf.reset_index().set_index("event_id")

