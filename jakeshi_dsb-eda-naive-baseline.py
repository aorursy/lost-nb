#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train_labels=pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
labels_map = dict(train_labels.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0])) # get the mode
labels_map


# In[3]:


import matplotlib.pylab as plt

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
test.groupby('installation_id')     .count()['event_id']     .sort_values()     .plot(kind='bar',
         figsize=(15, 5),
         title='Count of Events for each install')
plt.show()


# In[4]:



train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_event_cnt=train.groupby('installation_id')     .count()['event_id'] 
 
train_event_cnt.head()


# In[5]:


train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')


# In[6]:


train_labels.head()


# In[7]:


df1=train_labels.merge(train_event_cnt, how='inner', on=None,  left_on='installation_id', right_index=True )
df1.head()


# In[8]:


train_event_cnt.head()


# In[9]:


df1.groupby('accuracy_group').mean()


# In[10]:


train_game_cnt=train[train.type=='Game'].groupby('installation_id')     .count()['event_id'] 


# In[11]:


train_game_cnt.head()


# In[12]:


df2=train_labels.merge(train_game_cnt, how='inner', on=None,  left_on='installation_id', right_index=True )
df2.head()


# In[13]:


df2.groupby('accuracy_group').mean()


# In[14]:


ax = df2[df2.accuracy_group==3]['event_id'].plot.hist(bins=12, alpha=0.5)


# In[15]:


ax = df2[df2.accuracy_group==2]['event_id'].plot.hist(bins=12, alpha=0.5)


# In[16]:


ax = df2[df2.accuracy_group==1]['event_id'].plot.hist(bins=12, alpha=0.5)


# In[17]:


ax = df2[df2.accuracy_group==0]['event_id'].plot.hist(bins=12, alpha=0.5)


# In[18]:


df2[df2.event_id<500].accuracy_group.mean()


# In[19]:


df2[df2.event_id>500].accuracy_group.mean()


# In[20]:


df2[df2.event_id<100].accuracy_group.mean()


# In[21]:


df2[df2.event_id<10].accuracy_group.mean()


# In[22]:


df2[df2.event_id>1000].accuracy_group.mean()


# In[23]:


df2.accuracy_group.mean()


# In[24]:


submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

submission['accuracy_group'] = test.groupby('installation_id').last()['title'].map(labels_map).reset_index(drop=True)


# In[25]:


submission.head()


# In[26]:


submission.shape


# In[27]:


test_game_cnt=test[test.type=='Game'].groupby('installation_id')     .count()['event_id'] 


# In[28]:


submission2=submission.merge(test_game_cnt, how='left', on=None,  left_on='installation_id', right_index=True )
submission2.head()


# In[29]:


def f(row):
    if (row['event_id'] <10) & (row['accuracy_group']==3):
        val = 2
    else:
        val = row['accuracy_group']
    return val

submission2['accuracy_group2'] = submission2.apply(f, axis=1)


# In[30]:



df3 = submission2.drop(['event_id', 'accuracy_group'], axis=1)

df3.rename(columns={"accuracy_group2": "accuracy_group"})

df3.head()


# In[31]:


df3 = df3.rename(columns={"accuracy_group2": "accuracy_group"})


# In[32]:


df3.head()


# In[33]:


df3.shape


# In[34]:


df3.to_csv('submission.csv', index=None)

