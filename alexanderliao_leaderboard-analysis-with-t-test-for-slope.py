#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
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


a = lb_top_50.sort_values(by='diff',ascending=False)
a


# In[9]:


X = a['Score_pulblic_when download'][:8].values
Y = np.array([float(i) for i in a.Score_private[:8].values])
regression_results = sm.OLS(Y, X, missing = "drop").fit()
P_value = regression_results.pvalues [0]
R_squared = regression_results.rsquared
K_slope = regression_results.params [0]
conf_int = regression_results.conf_int ()
low_conf_int = conf_int [0][0]
high_conf_int = conf_int [0][1]
fig, ax = plt.subplots ()
ax.grid (True)
ax.scatter (X, Y, alpha = 1, color='orchid')
x_pred = np.linspace (min (X), max (X), 40)
y_pred = regression_results.predict (x_pred)
ax.plot (x_pred, y_pred, '-', color='darkorchid', linewidth=2)
print(low_conf_int, high_conf_int)


# In[10]:


X = a['Score_pulblic_when download'][8:30].values
Y = np.array([float(i) for i in a.Score_private[8:30].values])
regression_results = sm.OLS(Y, X, missing = "drop").fit()
P_value = regression_results.pvalues [0]
R_squared = regression_results.rsquared
K_slope = regression_results.params [0]
conf_int = regression_results.conf_int ()
low_conf_int = conf_int [0][0]
high_conf_int = conf_int [0][1]
fig, ax = plt.subplots ()
ax.grid (True)
ax.scatter (X, Y, alpha = 1, color='orchid')
x_pred = np.linspace (min (X), max (X), 40)
y_pred = regression_results.predict (x_pred)
ax.plot (x_pred, y_pred, '-', color='darkorchid', linewidth=2)
print(low_conf_int, high_conf_int)

