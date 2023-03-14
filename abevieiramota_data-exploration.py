#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import regex as re


# In[2]:


train = pd.read_csv("../input/train.csv", na_values="-1", index_col='id')


# In[3]:


c_null = train.isnull().sum()
c_null[c_null > 0].sort_values()


# In[4]:


train.target.value_counts()


# In[5]:


col_pattern = re.compile("^ps_(?P<class>\w+)_(?P<number>\d+)(_(?P<type>\w+))?$")
ci = pd.DataFrame({column: col_pattern.search(column).groupdict() for column in train.columns[1:]}).T
ci.loc[ci.type.isnull(), 'type'] = 'num'
ci.sample(10)


# In[6]:


a = ci.groupby('class')


# In[7]:


a.


# In[8]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

X = pd.DataFrame()

im = Imputer()
ln = LinearRegression()
pipe = Pipeline([('imputer', im), ('model', ln)])

for class_name, group in ci.groupby('class'):
    
    class_cols = group.index.tolist()
    
    pipe.fit(train.drop('target', axis=1)[class_cols], train.target)
    
    X[class_name] = pipe.predict(train.drop('target', axis=1)[class_cols])


# In[9]:


from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier(max_depth=3)
dc.fit(X, train.target)


# In[10]:


import graphviz 
from sklearn.tree import export_graphviz
dot_data = export_graphviz(dc, out_file=None, 
                         feature_names=X.columns,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data)  
graph 


# In[11]:




