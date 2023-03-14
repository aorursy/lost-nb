#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import warnings

TRAIN_PATH = "../input/train.csv"
tr = pd.read_csv(TRAIN_PATH)
pos = tr[tr.is_duplicate==1]


# In[2]:


g = nx.Graph()
g.add_nodes_from(pos.question1)
g.add_nodes_from(pos.question2)
edges = list(pos[['question1', 'question2']].to_records(index=False))
g.add_edges_from(edges)


# In[3]:


len(set(pos.question1) | set(pos.question2)), g.number_of_nodes()


# In[4]:


len(pos), g.number_of_edges()


# In[5]:


d = g.degree()
np.mean([d[k] for k in d])


# In[6]:


cc = filter(lambda x : (len(x) > 3) and (len(x) < 10), 
            nx.connected_component_subgraphs(g))
g1 = next(cc)
g1.nodes()


# In[7]:


# with block handles a deprecation warning that occurs inside nx.draw_networkx
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nx.draw_circular(g1, with_labels=True, alpha=0.5, font_size=8)
    plt.show()


# In[8]:


g1 = next(cc)
g1.nodes()


# In[9]:


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nx.draw_circular(g1, with_labels=True, alpha=0.5, font_size=8)
    plt.show()


# In[10]:


g1 = next(cc)
g1.nodes()


# In[11]:


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nx.draw_circular(g1, with_labels=True, alpha=0.5, font_size=8)
    plt.show()


# In[12]:


cc = nx.connected_component_subgraphs(g)
node_cts = list(sub.number_of_nodes() for sub in cc)
cc = nx.connected_component_subgraphs(g)
edge_cts = list(sub.number_of_edges() for sub in cc)
cts = pd.DataFrame({'nodes': node_cts, 'edges': edge_cts})
cts['mean_deg'] = 2 * cts.edges / cts.nodes
cts.nodes.clip_upper(10).value_counts().sort_index()


# In[13]:


Most of the components have just 2 questions, the minimum possible. But there are also several thousand larger components.


# In[14]:


cts.plot.scatter('nodes', 'edges')
plt.show()


# In[15]:


cts.plot.scatter('nodes', 'mean_deg')
plt.show()


# In[16]:


cts[cts.nodes >= 5].edges.sum()

