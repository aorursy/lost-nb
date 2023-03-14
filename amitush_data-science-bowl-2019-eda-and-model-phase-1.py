#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import json

pd.set_option('max_rows', 500)


# In[2]:


train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
train.head()


# In[3]:


train.loc[:, 'timestamp'] = pd.to_datetime(train.timestamp)
train.sort_values('timestamp', inplace=True)


# In[4]:


new_order = ['timestamp', 'installation_id', 'game_session', 'world', 'type', 'title', 'game_time', 'event_count' , 'event_code', 'event_id', 'event_data']
train = train.loc[:, new_order]
train.head()


# In[5]:


d_world_type_title = {'TREETOPCITY': {'Activity': ['Fireworks', 'Flower Waterer', 'Bug Measurer'], 
                                      'Assessment': ['Mushroom Sorter', 'Bird Measurer'], 
                                      'Clip': ['Tree Top City - Level 1', 'Ordering Spheres', 'Costume Box', '12 Monkeys', 
                                               'Tree Top City - Level 2', "Pirate's Tale", 'Treasure Map', 'Tree Top City - Level 3', 'Rulers'], 
                                      'Game': ['All Star Sorting', 'Air Show', 'Crystals Rule']}, 
                      'MAGMAPEAK': {'Activity': ['Sandcastle Builder', 'Watering Hole', 'Bottle Filler'], 
                                    'Assessment': ['Cauldron Filler'], 
                                    'Clip': ['Magma Peak - Level 1', 'Slop Problem', 'Magma Peak - Level 2'], 
                                    'Game': ['Scrub-A-Dub', 'Dino Drink', 'Bubble Bath', 'Dino Dive']}, 
                      'CRYSTALCAVES': {'Activity': ['Chicken Balancer', 'Egg Dropper'], 
                                       'Assessment': ['Cart Balancer', 'Chest Sorter'], 
                                       'Clip': ['Crystal Caves - Level 1', 'Balancing Act', 'Crystal Caves - Level 2', 
                                                'Crystal Caves - Level 3', 'Lifting Heavy Things', 'Honey Cake', 'Heavy, Heavier, Heaviest'], 
                                       'Game': ['Chow Time', 'Pan Balance', 'Happy Camel', 'Leaf Leader']}}


# In[6]:


train.installation_id.nunique()


# In[7]:


train.groupby('installation_id')['type'].apply(lambda s: s.isin(['Assessment']).any()).value_counts()


# In[8]:


sessions = train.groupby('installation_id')['game_session'].nunique()


# In[9]:


ax = sessions.value_counts().iloc[:30].plot.bar(title='Sessions per installation (top 30)')


# In[10]:


my_inst = train.installation_id.iloc[12345]  # and others...
df_inst = train.loc[train.installation_id==my_inst]
print(len(df_inst))
df_inst.head()


# In[11]:


df_sessions = df_inst.groupby('game_session')    .apply(lambda df_session: {'timestamp': df_session.timestamp.min(),
                               'world': df_session.world.iloc[0],
                               'title': df_session.title.iloc[0],
                               'type': df_session.type.iloc[0], 
                               'length': df_session.game_time.max(), 
                               'events': df_session.event_count.max()})\
    .apply(pd.Series)


# In[12]:


df_sessions.head(30)


# In[13]:


my_session = df_sessions.loc[df_sessions.type=='Assessment'].index[0]
df_session = df_inst.loc[df_inst.game_session==my_session]
df_session.head()


# In[14]:


df_session.set_index('timestamp')['event_code']     .plot(style='.')


# In[15]:


for idx, row in df_inst.iterrows():
    if row.type=='Assessment':
        if row.event_code in [4100, 4110]:
            event_data = json.loads(row.event_data)
            print(f"{row.game_session:20}, {row.title:30}, {event_data['correct']}")


# In[16]:


train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
train_labels.head()


# In[17]:


my_train_labels = train_labels.loc[train_labels.installation_id==my_inst]
my_train_labels


# In[18]:


test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
test.loc[:, 'timestamp'] = pd.to_datetime(test.timestamp)
test.sort_values('timestamp', inplace=True)


# In[19]:


labels_map = dict(train_labels.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0])) # get the mode
labels_map


# In[20]:


test_predictions = test.groupby('installation_id').last()['title'].map(labels_map).rename("accuracy_group")
test_predictions


# In[21]:


submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
submission.head()


# In[22]:


submission = submission    .join(test_predictions, on='installation_id', lsuffix='_orig')    .drop('accuracy_group_orig', axis=1)


# In[23]:


submission.to_csv('submission.csv', index=None)


# In[24]:


my_inst = test.installation_id.iloc[123]
df_inst = test.loc[test.installation_id==my_inst]


# In[25]:


df_sessions = df_inst.groupby('game_session')    .apply(lambda df_session: {'timestamp': df_session.timestamp.min(),
                               'world': df_session.world.iloc[0],
                               'title': df_session.title.iloc[0],
                               'type': df_session.type.iloc[0], 
                               'length': df_session.game_time.max(), 
                               'events': df_session.event_count.max()})\
    .apply(pd.Series)


# In[26]:


df_sessions.sort_values('timestamp')

