#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('pip uninstall fastai -y')
get_ipython().system('pip install fastai==0.7.0')
get_ipython().system('pip list | grep fast')


# In[3]:


from fastai.imports import *
from fastai.structured import *

from sklearn.ensemble import RandomForestRegressor
from pandas_summary import DataFrameSummary
from IPython.display import display

from sklearn import metrics


# In[4]:


df_raw = pd.read_csv('../input/train/Train.csv', low_memory=False, parse_dates=['saledate'])


# In[5]:


def display_all(df):
    with pd.option_context('display.max_rows', 1000):
        with pd.option_context('display.max_columns', 1000):
            display(df)


# In[6]:


display_all(df_raw.tail().transpose())


# In[7]:


df_raw.SalePrice = np.log(df_raw.SalePrice)


# In[8]:


train_cats(df_raw)


# In[9]:


df_raw.UsageBand.cat.categories


# In[10]:


df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)


# In[11]:


os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/raw')


# In[12]:


add_datepart(df_raw, 'saledate')


# In[13]:


df, y, nas = proc_df(df_raw, 'SalePrice')


# In[14]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)


# In[15]:


m.score(df, y)


# In[16]:


def split_vals(a, n): return a[:n].copy(), a[n:].copy()

n_valid = 12000 # Same as kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# In[17]:


def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
              m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)


# In[18]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[19]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000)
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)


# In[20]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[21]:


n = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[22]:


# Draw tree here
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[23]:


# Bagging
# Training multiple trees with rows chosen at random
# So that each tree has a different insight on the data
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[24]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]


# In[25]:


# Our forest predicted 9.3, real value was 9.1
preds.shape


# In[26]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])


# In[27]:


# As we see adding more and more trees mean increasing the r_sqaured
# Let's try adding more trees
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[28]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[29]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[30]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)


# In[31]:


set_rf_samples(20000)


# In[32]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[33]:


reset_rf_samples()


# In[34]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[35]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[36]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[37]:


set_rf_samples(50000)


# In[38]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[39]:


get_ipython().run_line_magic('time', 'preds = np.stack([t.predict(X_valid) for t in m.estimators_])')
# mean, std. deviation
np.mean(preds[:, 0]), np.std(preds[:, 0])


# In[40]:


def get_preds(t): return t.predict(X_valid)
# parallel_trees is a fast ai function that get help you run trees in parallel
get_ipython().run_line_magic('time', 'preds = np.stack(parallel_trees(m, get_preds))')
np.mean(preds[:, 0]), np.std(preds[:, 0])


# In[41]:


x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x.Enclosure.value_counts().plot.barh();


# In[42]:


flds = ['Enclosure', 'SalePrice', 'pred', 'pred_std']
enc_summ = x[flds].groupby('Enclosure', as_index=False).mean()
enc_summ


# In[43]:


enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot('Enclosure', 'SalePrice', 'barh', xlim=(0, 11))


# In[44]:


enc_summ.plot('Enclosure', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(0, 11))


# In[45]:


raw_valid.ProductSize.value_counts().plot.barh()


# In[46]:


flds = ['ProductSize', 'SalePrice', 'pred', 'pred_std']
summ = x[flds].groupby('ProductSize').mean()
summ


# In[47]:


(summ.pred/summ.pred_std).sort_values(ascending=False)


# In[48]:


fi = rf_feat_importance(m, df_trn)
fi[:10]


# In[49]:


fi.plot('cols', 'imp', figsize=(10, 6), legend=False)


# In[50]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[51]:


plot_fi(fi[:30])


# In[52]:


to_keep = fi[fi.imp > 0.005].cols; len(to_keep)


# In[53]:


df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)


# In[54]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[55]:


fi = rf_feat_importance(m, df_keep)
plot_fi(fi)


# In[56]:


df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[57]:


fi = rf_feat_importance(m, df_trn2)
plot_fi(fi[:25])


# In[58]:


from scipy.cluster import hierarchy as hc


# In[59]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16, 12))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()


# In[60]:


def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


# In[61]:


get_oob(df_keep)


# In[62]:


for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):
    print(c, get_oob(df_keep.drop(c, axis=1)))


# In[63]:


to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop, axis=1))


# In[64]:


df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)


# In[65]:


np.save('tmp/keep_cols.npy', np.array(df_keep.columns))


# In[66]:


keep_cols = np.load('tmp/keep_cols.npy', allow_pickle=True)
df_keep = df_trn[keep_cols]


# In[67]:


reset_rf_samples()


# In[68]:


m = RandomForestRegressor(n_estimators=40, max_features=0.5, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[69]:


from pdpbox import pdp
from plotnine import *


# In[70]:


set_rf_samples(50000)


# In[71]:


df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[72]:


get_ipython().run_line_magic('pinfo2', 'plot_fi')


# In[73]:


plot_fi(rf_feat_importance(m, df_trn2)[:10])


# In[74]:


df_raw.plot('YearMade', 'saleElapsed', 'scatter', alpha=0.1, figsize=(10,8));


# In[75]:


x_all = get_sample(df_raw[df_raw.YearMade>1960], 300)


# In[76]:


ggplot(x_all, aes('YearMade', 'SalePrice')) + stat_smooth(se=True)


# In[77]:


x = get_sample(X_train[X_train.YearMade>1930], 500)


# In[78]:


get_ipython().run_line_magic('pinfo2', 'pdp.pdp_isolate')


# In[79]:


def plot_pdp(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, x.columns, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                       cluster=clusters is not None, n_cluster_centers=clusters)


# In[80]:


plot_pdp('YearMade')


# In[81]:


plot_pdp('YearMade', clusters=5)


# In[82]:


feats = ['saleElapsed', 'YearMade']
p = pdp.pdp_interact(m, x, x.columns, feats)
pdp.pdp_interact_plot(p, feats)


# In[83]:


plot_pdp(['Enclosure_EROPS w AC', 'Enclosure_EROPS', 'Enclosure_OROPS'], 5, 'Enclosure')


# In[84]:


df_raw.YearMade[df_raw.YearMade<1950] = 1950
df_keep['age'] = df_raw['age'] = df_raw.saleYear - df_raw.YearMade


# In[85]:


X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
plot_fi(rf_feat_importance(m, df_keep))


# In[86]:


get_ipython().system('pip install treeinterpreter')
from treeinterpreter import treeinterpreter as ti


# In[87]:


df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)


# In[88]:


row = X_valid.values[None, 0]; row


# In[89]:


prediction, bias, contributions = ti.predict(m, row)


# In[90]:


prediction[0], bias[0]


# In[91]:


[o for o in zip(df_keep.columns, df_valid.iloc[0], contributions[0])]


# In[92]:


contributions[0].sum()


# In[ ]:




