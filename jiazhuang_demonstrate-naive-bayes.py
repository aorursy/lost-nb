#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.DataFrame({
    'X1': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    'X2': ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L'],
    'y': [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
})


# In[3]:


data.y.value_counts()


# In[4]:


data[data.y == 1].X1.value_counts()


# In[5]:


data[data.y == 1].X2.value_counts()


# In[6]:


data[data.y == -1].X1.value_counts()


# In[7]:


data[data.y == -1].X2.value_counts()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import norm


# In[9]:


x = np.linspace(-5, 5)
y = norm.pdf(x)
plt.plot(x, y)
plt.vlines(ymin=0, ymax=0.4, x=1, colors=['red'])


# In[10]:


train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)
target = train.target.values
train.drop('target', axis=1, inplace=True)
train.shape, target.shape, test.shape


# In[11]:


pos_idx = (target == 1)
neg_idx = (target == 0)
stats = []
for col in train.columns:
    stats.append([
        train.loc[pos_idx, col].mean(),
        train.loc[pos_idx, col].std(),
        train.loc[neg_idx, col].mean(),
        train.loc[neg_idx, col].std()
    ])
    
stats_df = pd.DataFrame(stats, columns=['pos_mean', 'pos_sd', 'neg_mean', 'neg_sd'])
stats_df.head()


# In[12]:


# priori probability
ppos = pos_idx.sum() / len(pos_idx)
pneg = neg_idx.sum() / len(neg_idx)

def get_proba(x):
    # we use odds P(target=1|X=x)/P(target=0|X=x)
    return (ppos * norm.pdf(x, loc=stats_df.pos_mean, scale=stats_df.pos_sd).prod()) /           (pneg * norm.pdf(x, loc=stats_df.neg_mean, scale=stats_df.neg_sd).prod())


# In[13]:


tr_pred = train.apply(get_proba, axis=1)


# In[14]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target, tr_pred)


# In[15]:


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(train.loc[pos_idx, 'var_0'])
plt.plot(np.linspace(0, 20), norm.pdf(np.linspace(0, 20), loc=stats_df.loc[0, 'pos_mean'], scale=stats_df.loc[0, 'pos_sd']))
plt.title('target==1')
plt.subplot(1, 2, 2)
sns.distplot(train.loc[neg_idx, 'var_0'])
plt.plot(np.linspace(0, 20), norm.pdf(np.linspace(0, 20), loc=stats_df.loc[0, 'neg_mean'], scale=stats_df.loc[0, 'neg_sd']))
plt.title('target==0')


# In[16]:


from scipy.stats.kde import gaussian_kde


# In[17]:


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(train.loc[pos_idx, 'var_0'])
kde = gaussian_kde(train.loc[pos_idx, 'var_0'].values)
plt.plot(np.linspace(0, 20), kde(np.linspace(0, 20)))
plt.title('target==1')
plt.subplot(1, 2, 2)
sns.distplot(train.loc[neg_idx, 'var_0'])
kde = gaussian_kde(train.loc[neg_idx, 'var_0'].values)
plt.plot(np.linspace(0, 20), kde(np.linspace(0, 20)))
plt.title('target==0')


# In[18]:


stats_df['pos_kde'] = None
stats_df['neg_kde'] = None
for i, col in enumerate(train.columns):
    stats_df.loc[i, 'pos_kde'] = gaussian_kde(train.loc[pos_idx, col].values)
    stats_df.loc[i, 'neg_kde'] = gaussian_kde(train.loc[neg_idx, col].values)


# In[19]:


def get_proba2(x):
    proba = ppos / pneg
    for i in range(200):
        proba *= stats_df.loc[i, 'pos_kde'](x[i]) / stats_df.loc[i, 'neg_kde'](x[i])
    return proba


# In[20]:


get_ipython().run_cell_magic('time', '', 'get_proba2(train.iloc[0].values)')


# In[21]:


def get_col_prob(df, coli, bin_num=100):
    bins = pd.cut(df.iloc[:, coli].values, bins=bin_num)
    uniq = bins.unique()
    uniq_mid = uniq.map(lambda x: (x.left + x.right) / 2)
    dense = pd.DataFrame({
        'pos': stats_df.loc[coli, 'pos_kde'](uniq_mid),
        'neg': stats_df.loc[coli, 'neg_kde'](uniq_mid)
    }, index=uniq)
    return bins.map(dense.pos).astype(float) / bins.map(dense.neg).astype(float)


# In[22]:


tr_pred = ppos / pneg
for i in range(200):
    tr_pred *= get_col_prob(train, i)


# In[23]:


roc_auc_score(target, tr_pred)


# In[24]:


te_pred = ppos / pneg
for i in range(200):
    te_pred *= get_col_prob(test, i)


# In[25]:


pd.DataFrame({
    'ID_code': test.index,
    'target': te_pred
}).to_csv('sub.csv', index=False)

