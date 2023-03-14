#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[2]:


lb = pd.read_csv('../input/imet-leaderboard/imet-2019-fgvc6-publicleaderboard.csv',parse_dates=['SubmissionDate'])


# In[3]:


lb.SubmissionDate.max()


# In[4]:


team655 = (lb[lb.Score>0.655].TeamId.unique())
print(len(team655))


# In[5]:


lb = lb[lb.Score>0.6]
lb['score_diff'] = lb['Score'] - lb.groupby('TeamId')['Score'].shift(1)
lb_team655 = lb[lb.TeamId.isin(team655)]
lb_team655 = lb_team655.sort_values(by='score_diff',ascending=False)


# In[6]:


lb_team655.groupby('TeamId').head(1)


# In[7]:


lb_top_50 = pd.read_csv('../input/imet-leaderboard/lb_score_top50.csv',index_col=[0])


# In[8]:


lb_top_50.sort_values(by='diff',ascending=False)

