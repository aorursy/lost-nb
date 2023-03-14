#!/usr/bin/env python
# coding: utf-8



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




get_ipython().system('pip uninstall fastai -y')
get_ipython().system('pip install fastai==0.7.0')
get_ipython().system('pip list | grep fast')




from fastai.imports import *
from fastai.structured import *

from sklearn.ensemble import RandomForestRegressor
from pandas_summary import DataFrameSummary
from IPython.display import display

from sklearn import metrics




df_raw = pd.read_csv('../input/train/Train.csv', low_memory=False, parse_dates=['saledate'])




def display_all(df):
    with pd.option_context('display.max_rows', 1000):
        with pd.option_context('display.max_columns', 1000):
            display(df)




display_all(df_raw.tail().transpose())




df_raw.SalePrice = np.log(df_raw.SalePrice)




train_cats(df_raw)




df_raw.UsageBand.cat.categories




df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)




os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/raw')




add_datepart(df_raw, 'saledate')




df, y, nas = proc_df(df_raw, 'SalePrice')




m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)




m.score(df, y)




def split_vals(a, n): return a[:n].copy(), a[n:].copy()

n_valid = 12000 # Same as kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape




def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
              m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)




m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000)
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)




m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




n = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)




# Draw tree here
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)




# Bagging
# Training multiple trees with rows chosen at random
# So that each tree has a different insight on the data
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)




preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]




# Our forest predicted 9.3, real value was 9.1
preds.shape




plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])




# As we see adding more and more trees mean increasing the r_sqaured
# Let's try adding more trees
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)




m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)




m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)




df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)




set_rf_samples(20000)




m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)




reset_rf_samples()




m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)




m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)




m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)




set_rf_samples(50000)




m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)




get_ipython().run_line_magic('time', 'preds = np.stack([t.predict(X_valid) for t in m.estimators_])')
# mean, std. deviation
np.mean(preds[:, 0]), np.std(preds[:, 0])




def get_preds(t): return t.predict(X_valid)
# parallel_trees is a fast ai function that get help you run trees in parallel
get_ipython().run_line_magic('time', 'preds = np.stack(parallel_trees(m, get_preds))')
np.mean(preds[:, 0]), np.std(preds[:, 0])




x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x.Enclosure.value_counts().plot.barh();




flds = ['Enclosure', 'SalePrice', 'pred', 'pred_std']
enc_summ = x[flds].groupby('Enclosure', as_index=False).mean()
enc_summ




enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot('Enclosure', 'SalePrice', 'barh', xlim=(0, 11))




enc_summ.plot('Enclosure', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(0, 11))




raw_valid.ProductSize.value_counts().plot.barh()




flds = ['ProductSize', 'SalePrice', 'pred', 'pred_std']
summ = x[flds].groupby('ProductSize').mean()
summ




(summ.pred/summ.pred_std).sort_values(ascending=False)




fi = rf_feat_importance(m, df_trn)
fi[:10]




fi.plot('cols', 'imp', figsize=(10, 6), legend=False)




def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)




plot_fi(fi[:30])




to_keep = fi[fi.imp > 0.005].cols; len(to_keep)




df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)




m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)




fi = rf_feat_importance(m, df_keep)
plot_fi(fi)




df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)




fi = rf_feat_importance(m, df_trn2)
plot_fi(fi[:25])




from scipy.cluster import hierarchy as hc




corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16, 12))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()




def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_




get_oob(df_keep)




for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):
    print(c, get_oob(df_keep.drop(c, axis=1)))




to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop, axis=1))




df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)




np.save('tmp/keep_cols.npy', np.array(df_keep.columns))




keep_cols = np.load('tmp/keep_cols.npy', allow_pickle=True)
df_keep = df_trn[keep_cols]




reset_rf_samples()




m = RandomForestRegressor(n_estimators=40, max_features=0.5, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)




from pdpbox import pdp
from plotnine import *




set_rf_samples(50000)




df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)




get_ipython().run_line_magic('pinfo2', 'plot_fi')




plot_fi(rf_feat_importance(m, df_trn2)[:10])




df_raw.plot('YearMade', 'saleElapsed', 'scatter', alpha=0.1, figsize=(10,8));




x_all = get_sample(df_raw[df_raw.YearMade>1960], 300)




ggplot(x_all, aes('YearMade', 'SalePrice')) + stat_smooth(se=True)




x = get_sample(X_train[X_train.YearMade>1930], 500)




get_ipython().run_line_magic('pinfo2', 'pdp.pdp_isolate')




def plot_pdp(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, x.columns, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                       cluster=clusters is not None, n_cluster_centers=clusters)




plot_pdp('YearMade')




plot_pdp('YearMade', clusters=5)




feats = ['saleElapsed', 'YearMade']
p = pdp.pdp_interact(m, x, x.columns, feats)
pdp.pdp_interact_plot(p, feats)




plot_pdp(['Enclosure_EROPS w AC', 'Enclosure_EROPS', 'Enclosure_OROPS'], 5, 'Enclosure')




df_raw.YearMade[df_raw.YearMade<1950] = 1950
df_keep['age'] = df_raw['age'] = df_raw.saleYear - df_raw.YearMade




X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
plot_fi(rf_feat_importance(m, df_keep))




get_ipython().system('pip install treeinterpreter')
from treeinterpreter import treeinterpreter as ti




df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)




row = X_valid.values[None, 0]; row




prediction, bias, contributions = ti.predict(m, row)




prediction[0], bias[0]




[o for o in zip(df_keep.columns, df_valid.iloc[0], contributions[0])]




contributions[0].sum()






