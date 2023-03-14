#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import HTML
HTML('<iframe width="1100" height="619" src="https://www.youtube.com/embed/45Da3eqQKXQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# In[2]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
pd.set_option('max_columns', 100)


py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from tqdm import tqdm_notebook

from IPython.display import HTML

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('figure', figsize=(15.0, 8.0))


# In[3]:


import os
print(os.listdir("../input/data-science-bowl-2019/"))


# In[4]:


get_ipython().run_cell_magic('time', '', "root = '../input/data-science-bowl-2019/'\n\n# Only load those columns in order to save space\nkeep_cols = ['event_id', 'game_session', 'installation_id', 'event_count', 'event_code', 'title', 'game_time', 'type', 'world']\ntrain = pd.read_csv(root + 'train.csv',usecols=keep_cols)\ntest = pd.read_csv(root + 'test.csv', usecols=keep_cols)\n\ntrain_labels = pd.read_csv(root + 'train_labels.csv')\nspecs = pd.read_csv(root + 'specs.csv')\nsample_submission = pd.read_csv(root + 'sample_submission.csv')")


# In[5]:


print('Size of train data', train.shape)
print('Size of train_labels data', train_labels.shape)
print('Size of specs data', specs.shape)
print('Size of test data', test.shape)


# In[6]:


train.head()


# In[7]:


train_labels.head()


# In[8]:


specs.head()


# In[9]:


train.dtypes.value_counts()


# In[10]:


train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[11]:


train_labels.dtypes.value_counts()


# In[12]:


train_labels.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[13]:


specs.dtypes.value_counts()


# In[14]:


specs.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[15]:


total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing__train_data.head(10)


# In[16]:


total = train_labels.isnull().sum().sort_values(ascending = False)
percent = (train_labels.isnull().sum()/train_labels.isnull().count()*100).sort_values(ascending = False)
missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing__train_data.head(10)


# In[17]:


total = specs.isnull().sum().sort_values(ascending = False)
percent = (specs.isnull().sum()/specs.isnull().count()*100).sort_values(ascending = False)
missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing__train_data.head(10)


# In[18]:


corrs = train.corr()
corrs


# In[19]:


plt.figure(figsize = (20, 8))

# Heatmap of correlations
sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# In[20]:


corrs2 = train_labels.corr()
corrs2


# In[21]:


plt.figure(figsize = (20, 8))

# Heatmap of correlations
sns.heatmap(corrs2, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# In[22]:


plt.figure(figsize=(8, 6))
sns.countplot(x="accuracy_group",data=train_labels, order = train_labels['accuracy_group'].value_counts().index)
plt.title('Accuracy Group Count Column')
plt.tight_layout()
plt.show()


# In[23]:


train_labels.groupby('accuracy_group')['game_session'].count()     .plot(kind='barh', figsize=(15, 5), title='Target (accuracy group)')
plt.show()


# In[24]:


train.head()


# In[25]:



palete = sns.color_palette(n_colors=10)


# In[26]:


train.groupby('installation_id')     .count()['event_id']     .apply(np.log1p)     .plot(kind='hist',
          bins=40,
          color=palete[1],
         figsize=(15, 5),
         title='Log(Count) of Observations by installation_id')
plt.show()


# In[27]:


train.groupby('title')['event_id']     .count()     .sort_values()     .plot(kind='barh',
          title='Count of Observation by Game/Video title',
         color=palete[1],
         figsize=(15, 15))
plt.show()


# In[28]:


train.groupby('world')['event_id']     .count()     .sort_values()     .plot(kind='bar',
          figsize=(15, 4),
          title='Count by World',
          color=palete[1])
plt.show()


# In[29]:


def group_and_reduce(df):
    # group1 and group2 are intermediary "game session" groups,
    # which are reduced to one record by game session. group1 takes
    # the max value of game_time (final game time in a session) and 
    # of event_count (total number of events happened in the session).
    # group2 takes the total number of event_code of each type
    group1 = df.drop(columns=['event_id', 'event_code']).groupby(
        ['game_session', 'installation_id', 'title', 'type', 'world']
    ).max().reset_index()

    group2 = pd.get_dummies(
        df[['installation_id', 'event_code']], 
        columns=['event_code']
    ).groupby(['installation_id']).sum()

    # group3, group4 and group5 are grouped by installation_id 
    # and reduced using summation and other summary stats
    group3 = pd.get_dummies(
        group1.drop(columns=['game_session', 'event_count', 'game_time']),
        columns=['title', 'type', 'world']
    ).groupby(['installation_id']).sum()

    group4 = group1[
        ['installation_id', 'event_count', 'game_time']
    ].groupby(
        ['installation_id']
    ).agg([np.sum, np.mean, np.std])

    return group2.join(group3).join(group4)


# In[30]:


get_ipython().run_cell_magic('time', '', 'train_small = group_and_reduce(train)\ntest_small = group_and_reduce(test)\n\nprint(train_small.shape)\ntrain_small.head()')


# In[31]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import KFold\nsmall_labels = train_labels[['installation_id', 'accuracy_group']].set_index('installation_id')\ntrain_joined = train_small.join(small_labels).dropna()\nkf = KFold(n_splits=5, random_state=2019)\nX = train_joined.drop(columns='accuracy_group').values\ny = train_joined['accuracy_group'].values.astype(np.int32)\ny_pred = np.zeros((len(test_small), 4))\nfor train, test in kf.split(X):\n    x_train, x_val, y_train, y_val = X[train], X[test], y[train], y[test]\n    train_set = lgb.Dataset(x_train, y_train)\n    val_set = lgb.Dataset(x_val, y_val)\n\n    params = {\n        'learning_rate': 0.01,\n        'bagging_fraction': 0.9,\n        'feature_fraction': 0.9,\n        'num_leaves': 50,\n        'lambda_l1': 0.1,\n        'lambda_l2': 1,\n        'metric': 'multiclass',\n        'objective': 'multiclass',\n        'num_classes': 4,\n        'random_state': 2019\n    }\n\n    model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50, valid_sets=[train_set, val_set], verbose_eval=50)\n    y_pred += model.predict(test_small)")


# In[32]:


get_ipython().run_cell_magic('time', '', "y_pred = y_pred.argmax(axis=1)\ntest_small['accuracy_group'] = y_pred\ntest_small[['accuracy_group']].to_csv('submission.csv')")


# In[33]:


get_ipython().run_cell_magic('time', '', 'val_pred = model.predict(x_val).argmax(axis=1)\nprint(classification_report(y_val, val_pred))')


# In[34]:



HTML('<iframe width="1106" height="622" src="https://www.youtube.com/embed/1ejHigxuR2Q" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')

