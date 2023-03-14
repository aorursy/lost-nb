#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import seaborn as sns
import pandas as pd
import itertools 
import csv
import collections
import matplotlib.pyplot as plt

sns.set_context("paper")
get_ipython().run_line_magic('matplotlib', 'inline')

RES_DIR = "../input/"


# In[2]:


# Load train data (skips the content column)
def load_train_data():
    categories = ['cooking', 'robotics', 'travel', 'crypto', 'diy', 'biology']
    train_data = []
    for cat in categories:
        data = pd.read_csv("{}{}.csv".format(RES_DIR, cat), usecols=['id', 'title', 'tags'])
        data['category'] = cat
        train_data.append(data)
    
    return pd.concat(train_data)


# In[3]:


train_data = load_train_data()
train_data.head()


# In[4]:





# In[4]:


# Summary about tags
tag_lists = [t.strip().split() for t in train_data['tags'].values]
all_tags = list(itertools.chain(*tag_lists))
tag_list_size = np.array([len(x) for x in tag_lists])
print("""The corpus is composed by {} questions. Overall {} tags have been used, of which {} unique ones. 
Average number of tags per question {:.2f} (min={}, max={}, std={:.2f})""".format(
    len(train_data),
    len(all_tags), len(set(all_tags)),
    tag_list_size.mean(), 
    min(tag_list_size), max(tag_list_size),
    tag_list_size.std()))


# In[5]:


# Utility function to return top occuring tags in the passed df
def get_top_tags(df, n=None):
    tags = list(itertools.chain(*[t.strip().split() for t in df['tags'].values]))
    top_tags = collections.Counter(list(tags)).most_common(n)
    tags, count = zip(*top_tags)
    return tags, count


# In[6]:


# Created DataFrame indexed on tags
tags_df = pd.DataFrame(index=set(itertools.chain(*tag_lists)))
# For each category create a column and update the flag to tag count
for i, (name, group) in enumerate(train_data.groupby('category')):
    tags_df[name] = 0
    tmp_index, count = get_top_tags(group)
    tmp = pd.Series(count, index=tmp_index)
    tags_df[name].update(tmp)
# Number of categories for which a tag appeared at least 1 time
tags_df['categories_appears'] = tags_df.apply(lambda x: x.astype(bool).sum(), axis=1)
tags_df['categories_appears'].value_counts()


# In[7]:


# List of tags ordered by number of categories in which they appear, with total count for each
tags_df.sort_values('categories_appears', ascending=False)


# In[8]:


d = tags_df.unstack().reset_index()


# In[9]:


d.columns = ['source', 'target', 'weight']
import networkx as nx


# In[10]:


d = d[d.weight > 10]


# In[11]:


g = nx.from_pandas_dataframe(d, 'source', 'target', ['weight'])


# In[12]:


nx.node_connectivity(g)bnvhn


# In[13]:


import matplotlib.pyplot as plt
plt.figure(figsize=[50,50])
nx.draw_networkx(g,alpha=0.1)


# In[14]:



