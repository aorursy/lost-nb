#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = '../input/data-science-bowl-2019/'


# In[2]:


train_df = pd.read_csv(DATA_PATH + 'train.csv')
labels_df = pd.read_csv(DATA_PATH + 'train_labels.csv')


# In[3]:


train_df = train_df.rename(columns={"installation_id": "kid_id"})
labels_df = labels_df.rename(columns={"installation_id": "kid_id"})


# In[4]:


train_df = train_df[train_df.kid_id.isin(labels_df.kid_id.unique())]


# In[5]:


train_df.info()


# In[6]:


train_df.head()


# In[7]:


labels_df.info()


# In[8]:


labels_df.head()


# In[9]:


sessions_time = train_df.groupby('game_session').agg({'timestamp': ['min', 'max'],'type' : "unique",'world': "unique",'title': "unique"})
sessions_time.columns = ['Start Time', 'End Time','Type','World','Title']
sessions_time["Duration"]= pd.to_datetime(sessions_time["End Time"]) - pd.to_datetime(sessions_time["Start Time"])
sessions_time["Duration"] = sessions_time["Duration"].apply(lambda x: round(x.total_seconds()/60))
sessions_time["Type"] = sessions_time["Type"].apply(', '.join)
sessions_time["World"] = sessions_time["World"].apply(', '.join)
sessions_time["Title"] = sessions_time["Title"].apply(', '.join)
sessions_time = sessions_time.sort_values('Duration',ascending=False)


# In[10]:


sessions_time.head(20)


# In[11]:


sessions_time.groupby('Type')['Duration'].sum()     .plot(kind='bar', figsize=(15, 5), title='Time Spent on Session Type',colormap='winter')
plt.ylabel("Time (Minutes)")
plt.show()


# In[12]:


sessions_time.groupby('World')['Duration'].sum()     .plot(kind='bar', figsize=(15, 5), title='Time Spent on World',colormap='winter')
plt.ylabel("Time (Minutes)")
plt.show()


# In[13]:


sessions_time[sessions_time["Type"] == 'Activity'].groupby('Title')['Duration'].sum()     .plot(kind='bar', figsize=(15, 5), title='Time Spent on Activities',colormap='winter')
plt.ylabel("Time (Minutes)")
plt.xlabel("Activity")
plt.show()


# In[14]:


sessions_time[sessions_time["Type"] == 'Game'].groupby('Title')['Duration'].sum()     .plot(kind='bar', figsize=(15, 5), title='Time Spent on Games',colormap='winter')
plt.ylabel("Time (Minutes)")
plt.xlabel("Game")
plt.show()


# In[15]:


sessions_time[sessions_time["Type"] == 'Clip'].groupby('Title')['Duration'].count()     .plot(kind='bar', figsize=(15, 5), title='Clips Views',colormap='winter')
plt.ylabel("Views")
plt.xlabel("Clip")
plt.show()


# In[16]:


# Session duration is more than 10 hrs.
sessions_time[sessions_time["Duration"] > 10*60]


# In[17]:


sessions_time[sessions_time["Duration"] <= 0]


# In[18]:


valueable_sessions = sessions_time[(sessions_time["Duration"] > 0) & (sessions_time["Duration"] < 600)]
sessions_df = train_df[train_df.game_session.isin(valueable_sessions.index)]
labels_df = labels_df[labels_df.game_session.isin(valueable_sessions.index)]


# In[19]:


kids_performance = pd.DataFrame(labels_df.groupby(['kid_id','accuracy_group'])['num_correct'].count().sort_values()).reset_index().pivot(index='kid_id', columns='accuracy_group',values='num_correct').fillna(0).astype('int32')
kids_performance.columns = ['Failed','> 3rd Attempt','2nd Attempt','1st Attempt']
kids_performance = kids_performance[['1st Attempt','2nd Attempt','> 3rd Attempt','Failed']]
kids_performance


# In[20]:


sessions = pd.DataFrame(sessions_df.groupby('kid_id')['game_session'].unique())
sessions['game_session'] = sessions['game_session'].apply(lambda x: len(x))
kids_performance = pd.merge(right=kids_performance,left=sessions,left_index=True, right_index=True).sort_values(by='game_session',ascending=False)

sessions_type = pd.DataFrame(sessions_df.groupby(["kid_id","type"])['game_session'].unique())
sessions_type['game_session'] = sessions_type['game_session'].apply(lambda x: len(x))
sessions_type = sessions_type.reset_index().pivot(index='kid_id', columns='type',values='game_session').fillna(0).astype('int32')
kids_performance = pd.merge(right=kids_performance,left=sessions_type,left_index=True, right_index=True).sort_values(by='game_session',ascending=False)
kids_performance = kids_performance.rename(columns={"game_session": "Total Sessions"})


# In[21]:


kids_performance.head(50)


# In[22]:


kids_performance.tail(50)


# In[23]:


kids_performance = kids_performance[kids_performance["Assessment"] > 4]
kids_performance


# In[24]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
types = ['1st Attempt', '2nd Attempt', '>3rd Attemp', 'Failed']
types_sessions = [np.sum(kids_performance['1st Attempt']),np.sum(kids_performance['2nd Attempt']),np.sum(kids_performance['> 3rd Attempt']),np.sum(kids_performance['Failed'])]
ax.pie(types_sessions, labels = types,autopct='%1.2f%%')
plt.title('Assessments Performance')
plt.show()


# In[25]:


first_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 3].game_session)]
second_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 2].game_session)]
third_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 1].game_session)]
failed_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 0].game_session)]


# In[26]:


first_attempt.groupby('Title')['Duration'].count().sort_values(ascending=False)     .plot(kind='bar', figsize=(15, 5), title='Solved in 1st Attempt',colormap='winter')
plt.ylabel("Count")
plt.xlabel("Title")
plt.show()


# In[27]:


second_attempt.groupby('Title')['Duration'].count().sort_values(ascending=False)     .plot(kind='bar', figsize=(15, 5), title='Solved in 2nd Attempt',colormap='winter')
plt.ylabel("Count")
plt.xlabel("Title")
plt.show()


# In[28]:


third_attempt.groupby('Title')['Duration'].count().sort_values(ascending=False)     .plot(kind='bar', figsize=(15, 5), title='Solved in 3rd or more Attempts',colormap='winter')
plt.ylabel("Count")
plt.xlabel("Title")
plt.show()


# In[29]:


failed_attempt.groupby('Title')['Duration'].count().sort_values(ascending=False)     .plot(kind='bar', figsize=(15, 5), title='Never Solved',colormap='winter')
plt.ylabel("Count")
plt.xlabel("Title")
plt.show()


# In[30]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
types = ['1st Attempt', '2nd Attempt', '>3rd Attemp', 'Failed']
types_sessions = [np.sum(first_attempt['Duration']),np.sum(second_attempt['Duration']),np.sum(third_attempt['Duration']),np.sum(failed_attempt['Duration'])]
ax.pie(types_sessions, labels = types,autopct='%1.2f%%')
plt.title('Time Consumed in Assessments')
plt.show()


# In[31]:


first_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 3].game_session)]
second_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 2].game_session)]
third_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 1].game_session)]
failed_attempt = sessions_time[ sessions_time.index.isin(labels_df[labels_df.accuracy_group == 0].game_session)]


# In[32]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
types = list(sessions_time[sessions_time["Type"] == "Assessment"]["Title"].unique())
types_sessions = [np.sum(sessions_time[sessions_time["Title"] == types[0]]['Duration']),np.sum(sessions_time[sessions_time["Title"] == types[1]]['Duration']),np.sum(sessions_time[sessions_time["Title"] == types[2]]['Duration']),np.sum(sessions_time[sessions_time["Title"] == types[3]]['Duration']),np.sum(sessions_time[sessions_time["Title"] == types[4]]['Duration'])]
ax.pie(types_sessions, labels = types,autopct='%1.2f%%')
plt.title('Time Spent in Each Assessment')
plt.show()

