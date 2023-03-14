#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
import feather


# In[ ]:


set_plot_sizes(12,14,16)


# In[ ]:


PATH = "C:/Users/jcat/fastai/data/bulldozers/"
# df_raw = pd.read_feather('tmp/bulldozers-raw')
df_raw = feather.read_dataframe('tmp/bulldozers-raw')
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')


# In[ ]:


# split data into training and validation parts
def split_vals(a,n): return a[:n], a[n:]
n_valid = 12000
n_trn = len(df_trn)-n_valid
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)


# In[ ]:


# functions to define and print scores
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


df_raw


# In[ ]:


# use a subset of examples for each tree, 
#     instead of the full bootstrap sample 
set_rf_samples(50000)


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'set_rf_samples')


# In[ ]:


# metric = 0.2509
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


# compare to full bootstrap sample
# score = 0.2268, so it's better by 0.024
reset_rf_samples()
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)

# return to subsampling
set_rf_samples(50000)


# In[ ]:


get_ipython().run_line_magic('time', 'preds = np.stack([t.predict(X_valid) for t in m.estimators_])')
np.mean(preds[:,0]), np.std(preds[:,0])


# In[ ]:


# problem with parallelization
def get_preds(t): return t.predict(X_valid)
get_ipython().run_line_magic('time', 'preds = np.stack(parallel_trees(m, get_preds))')
np.mean(preds[:,0]), np.std(preds[:,0])


# In[ ]:


preds.shape


# In[ ]:


x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x.Enclosure.value_counts().plot.barh();


# In[ ]:


flds = ['Enclosure', 'SalePrice', 'pred', 'pred_std']
enc_summ = x[flds].groupby('Enclosure', as_index=False).mean()
enc_summ


# In[ ]:


# plot sale price grouped by enclosure category
enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot('Enclosure', 'SalePrice', 'barh', xlim=(0,11));


# In[ ]:


# include error bars
enc_summ.plot('Enclosure', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(0,11));


# In[ ]:


raw_valid.ProductSize.value_counts().plot.barh();


# In[ ]:


flds = ['ProductSize', 'SalePrice', 'pred', 'pred_std']
summ = x[flds].groupby(flds[0]).mean()
summ


# In[ ]:


# fractional error in predicted price
(summ.pred_std/summ.pred).sort_values(ascending=False)


# In[ ]:


fi = rf_feat_importance(m, df_trn); fi[:10]


# In[ ]:


fi.plot('cols', 'imp', figsize=(10,6), legend=False);


# In[ ]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[ ]:


plot_fi(fi[:30]);


# In[ ]:


to_keep = fi[fi.imp>0.005].cols; len(to_keep)


# In[ ]:


# keep features with importance > 0.005
df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)


# In[ ]:


# eliminating unimportant features improved score by 0.007
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


# feature importances within reduced feature set 
#     vary a bit from previous order
fi = rf_feat_importance(m, df_keep)
plot_fi(fi);


# In[ ]:


# one-hot-encoding made metric worse by 0.01!
df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


# importance ordering is changed 
fi = rf_feat_importance(m, df_trn2)
plot_fi(fi[:25]);


# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


# spearman correlation
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()


# In[ ]:


def get_oob(df):
    # why vary parameters from original values?
    #     n_estimators = 40
    #     min_samples_leaf = 3
    #     max_features = 0.5
    # m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    # original parameter values improve metric by 0.005
    #     so let's keep them
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


# In[ ]:


# revert to df_keep
get_oob(df_keep)


# In[ ]:


# removing these features has little effect on oob score
for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):
    print(c, get_oob(df_keep.drop(c, axis=1)))


# In[ ]:


# metric is worse by 0.002
to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop, axis=1))


# In[ ]:


df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)


# In[ ]:


# save list of columns to keep
np.save('tmp/keep_cols.npy', np.array(df_keep.columns))


# In[ ]:


# retrieve list of columns to keep
keep_cols = np.load('tmp/keep_cols.npy')
df_keep = df_trn[keep_cols]


# In[ ]:


# revert to full bootstrap sample
reset_rf_samples()


# In[ ]:


# metric improved to 0.227 using full bootstrap, 
#     which is what we had before
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


# first, install pdpbox and plotnine
# pip install pdpbox
# conda install -c conda-forge plotnine
from pdpbox import pdp
from plotnine import *


# In[ ]:


set_rf_samples(50000)


# In[ ]:


# start with metric 0.253
df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train);
print_score(m)


# In[ ]:


plot_fi(rf_feat_importance(m, df_trn2)[:10]);


# In[ ]:


df_raw.plot('YearMade', 'saleElapsed', 'scatter', alpha=0.01, figsize=(10,8));


# In[ ]:


x_all = get_sample(df_raw[df_raw.YearMade>1930], 500)


# In[ ]:


# first install scikit-misc
# pip install scikit-misc
ggplot(x_all, aes('YearMade', 'SalePrice'))+stat_smooth(se=True, method='loess')


# In[ ]:


x = get_sample(X_train[X_train.YearMade>1930], 500)


# In[ ]:


def plot_pdp(feat_name, clusters=None):
    p = pdp.pdp_isolate(m, x, feature=feat_name, model_features=x.columns)
    return pdp.pdp_plot(p, feat_name, plot_lines=True, 
                        cluster=clusters is not None, n_cluster_centers=clusters)


# In[ ]:


plot_pdp('YearMade')


# In[ ]:


plot_pdp('YearMade', clusters=5)


# In[ ]:


feats = ['saleElapsed', 'YearMade']
p = pdp.pdp_interact(m, x, feats)
pdp.pdp_interact_plot(p, feats)


# In[ ]:


plot_pdp(['Enclosure_EROPS w AC', 'Enclosure_EROPS', 'Enclosure_OROPS'], 5)#, 'Enclosure')


# In[ ]:


# define engineered feature 'age'
df_raw.YearMade[df_raw.YearMade<1950] = 1950
df_keep['age'] = df_raw['age'] = df_raw.saleYear-df_raw.YearMade


# In[ ]:


# age becomes the most important feature!
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
plot_fi(rf_feat_importance(m, df_keep));


# In[ ]:


# install treeinterpreter
# pip install treeinterpreter
from treeinterpreter import treeinterpreter as ti


# In[ ]:


df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)


# In[ ]:


row = X_valid.values[None,0]; row


# In[ ]:


prediction, bias, contributions = ti.predict(m, row)


# In[ ]:


len(contributions[0])


# In[ ]:


prediction[0], bias[0]


# In[ ]:


idxs = np.argsort(contributions[0])


# In[ ]:


df_valid.iloc[0]


# In[ ]:


[o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]


# In[ ]:


contributions[0].sum()


# In[ ]:


df_ext = df_keep.copy()
df_ext['is_valid'] = 1
df_ext.is_valid[:n_trn] = 0
x, y, nas = proc_df(df_ext, 'is_valid')


# In[ ]:


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_


# In[ ]:


fi = rf_feat_importance(m, x); fi[:10]


# In[ ]:


# top 3 features
feats=['SalesID', 'saleElapsed', 'MachineID']


# In[ ]:


(X_train[feats]/1000).describe()


# In[ ]:


(X_valid[feats]/1000).describe()


# In[ ]:


# drop top three features
x.drop(feats, axis=1, inplace=True)


# In[ ]:


# score is a bit worse
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y)
m.oob_score_


# In[ ]:


fi = rf_feat_importance(m, x); fi[:10]


# In[ ]:


#speed up by subsampling
set_rf_samples(50000)


# In[ ]:


# return to original sample
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


# top six features
feats=['SalesID', 'saleElapsed', 'MachineID', 'age', 'YearMade', 'saleDayofyear']


# In[ ]:


# remove top six features, one at a time to see effect on metric
# metrics vary between 0.245 and 0.255
for f in feats:
    df_subs = df_keep.drop(f, axis=1)
    X_train, X_valid = split_vals(df_subs, n_trn)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print(f)
    print_score(m)


# In[ ]:


# revert to full bootstrap sample
reset_rf_samples()


# In[ ]:


# removing these features gave a significant score reduction
#     recall that previously with full bootstrap sample we
#     got a score of 0.2182, original score was 0.2268
# drop SalesID, MachineID, saleDayOfyear because dropping
#    them individually reduced the metric more than any of 
#    the other three.
df_subs = df_keep.drop(['SalesID', 'MachineID', 'saleDayofyear'], axis=1)
X_train, X_valid = split_vals(df_subs, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


plot_fi(rf_feat_importance(m, X_train));


# In[ ]:


np.save('tmp/subs_cols.npy', np.array(df_subs.columns))


# In[ ]:


# use more trees, and grow them completely
# our metric improved by 0.007 to 0.2114
m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:




