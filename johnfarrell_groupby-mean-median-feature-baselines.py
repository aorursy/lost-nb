#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path('../input/')
print(os.listdir(DATA_DIR))
air_visit_data = pd.read_csv(DATA_DIR / 'air_visit_data.csv', 
                             parse_dates=['visit_date'])
hpg_reserve = pd.read_csv(DATA_DIR / 'hpg_reserve.csv', 
                          parse_dates=['visit_datetime', 
                                       'reserve_datetime'])
air_store_info = pd.read_csv(DATA_DIR / 'air_store_info.csv')
sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')
air_reserve = pd.read_csv(DATA_DIR / 'air_reserve.csv', 
                          parse_dates=['visit_datetime', 
                                       'reserve_datetime'])
store_id_relation = pd.read_csv(DATA_DIR / 'store_id_relation.csv')
hpg_store_info = pd.read_csv(DATA_DIR / 'hpg_store_info.csv')
date_info = pd.read_csv(DATA_DIR / 'date_info.csv', 
                        parse_dates=['calendar_date']
                       ).rename(columns={'calendar_date':'visit_date'})


# In[2]:


## Build test set
def get_test_set():
    air_visit_test = sample_submission.copy()
    air_visit_test['air_store_id'] = sample_submission['id'].apply(
            lambda s:s.split('_2017')[0])
    air_visit_test['visit_date'] = pd.to_datetime(
            sample_submission['id'].apply(
                    lambda s:'2017'+s.split('_2017')[1]))
    test_id = air_visit_test[['id', 'air_store_id', 'visit_date']].copy()
    air_visit_test['visitors'] = 0
    del air_visit_test['id']
    return test_id, air_visit_test
test_id, air_visit_test = get_test_set()
train = pd.concat([air_visit_data, air_visit_test], axis=0).reset_index()
del train['index']
train['is_train'] = 0
train['is_train'][:len(air_visit_data)] = 1
# len(air_visit_data), sum(train['is_test'])


# In[3]:


## Build integer air_store_id
air_store_info['asid'] = list(range(len(air_store_info)))
asid = air_store_info[['air_store_id', 'asid']]

asid_dict = asid.set_index(['air_store_id']).to_dict()['asid']
del air_store_info['asid'], asid
def map_asid(df):
    assert 'air_store_id' in df.columns
    df['air_store_id'] = df['air_store_id'].map(asid_dict.get)
    return df
air_reserve = map_asid(air_reserve)
air_store_info = map_asid(air_store_info)
train = map_asid(train)
store_id_relation = map_asid(store_id_relation)
test_id = map_asid(test_id)


# In[4]:


get_ipython().run_cell_magic('time', '', "## Get visit date features\ndef get_date_feat(df, cache_size=512):\n    from functools import lru_cache\n    get_year  = lru_cache(cache_size)(lambda x: x.year - 2016)\n    get_month = lru_cache(cache_size)(lambda x: x.month)\n    get_week  = lru_cache(cache_size)(lambda x: x.week)\n    get_dow   = lru_cache(cache_size)(lambda x: x.dayofweek + 1)\n    df['visit_year']   = df['visit_date'].apply(get_year)\n    df['visit_month']  = df['visit_date'].apply(get_month)\n    df['visit_week']   = df['visit_date'].apply(get_week)\n    df['visit_dow']    = df['visit_date'].apply(get_dow)\n    return df\nget_date_feat(train, 0)\n# get_date_feat(train)")


# In[5]:


get_ipython().run_cell_magic('time', '', 'train = get_date_feat(train)')


# In[6]:


## Clear date info
date_clear = date_info.copy()
date_clear['dow'] = date_info['visit_date'].apply(lambda x:x.dayofweek+1)
del date_clear['day_of_week']
date_clear['Date_FriSatSun_flg'] = date_clear['dow'].apply(lambda x:int(5<=x<=7))
date_clear['Date_holiday_flg'] = date_info['holiday_flg'].copy()
date_clear['Date_FriSatSun_and_holiday_flg'] = date_clear['Date_FriSatSun_flg'] & date_clear['Date_holiday_flg']
date_clear['Date_FriSatSun_and_holiday_flg'] = date_clear['Date_FriSatSun_and_holiday_flg'].astype(int)
date_clear['Date_FriSatSun_or_holiday_flg'] = date_clear['Date_FriSatSun_flg'] | date_clear['Date_holiday_flg']
date_clear['Date_FriSatSun_or_holiday_flg'] = date_clear['Date_FriSatSun_or_holiday_flg'].astype(int)
del date_clear['holiday_flg'], date_clear['dow']
date_clear.head()


# In[7]:


## Calc nthday, nthday_last, continuous_span
def get_date_feat(date_df, flg_col):
    # really doesn't matter...
    if flg_col == 'Date_holiday_flg':
        diff0 = 6 #2015-12-23(Holiday)~2015-12-29(Holiday)
    elif flg_col == 'Date_FriSatSun_or_holiday_flg':
        diff0 = 2 #2015-12-27(Sun)~2015-12-29(Holiday)
    date_li = [pd.to_datetime('2015-12-29'), 
               pd.to_datetime('2015-12-30'), 
               pd.to_datetime('2015-12-31')]\
        +list(date_clear[date_clear[flg_col]==1]['visit_date']) 
    tmp_date_df = pd.DataFrame(date_li, columns=['visit_date'])
    tmp_date_df[flg_col+'_diff'] = tmp_date_df['visit_date'].diff().apply(lambda x:x.days)
    tmp_date_df[flg_col+'_diff'] = tmp_date_df[flg_col+'_diff'].fillna(diff0) 
    nthday_li = [1]
    span_li = []
    nthday, span = None, 1
    for diff in tmp_date_df[flg_col+'_diff'][1:]:
        if diff == 1:
            span = span + 1
            nthday = nthday_li[-1] + 1
            nthday_li.append(nthday)
        else:
            span_li = span_li + [nthday_li[-1]] * span
            span = 1
            nthday_li.append(1)
    if span > 1: 
        span_li = span_li + [nthday_li[-1]] * span
    nthday_li = np.array(nthday_li)
    span_li = np.array(span_li)
    tmp_date_df[flg_col+'_nthday'] = nthday_li
    tmp_date_df[flg_col+'_nthday_last'] = 1 + span_li - nthday_li
    tmp_date_df[flg_col+'_span'] = span_li
    return tmp_date_df
date_clear = date_clear.merge(get_date_feat(date_clear, flg_col='Date_holiday_flg'), 
                              how='left', on='visit_date').fillna(-1)
date_clear = date_clear.merge(get_date_feat(date_clear, flg_col='Date_FriSatSun_or_holiday_flg'), 
                              how='left', on='visit_date').fillna(-1)
date_clear.head()


# In[8]:


train = train.merge(date_clear, how='left', on='visit_date')
train.head()


# In[9]:


air_store_info.head()


# In[10]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
air_store_info['genre_id'] = le.fit_transform(air_store_info['air_genre_name'])
air_store_info.head()


# In[11]:


tmp = air_store_info.copy()
tmp['area_split'] = tmp['air_area_name'].apply(lambda x:tuple(x.split(' ')))
tmp['area_split_len'] = tmp['area_split'].apply(len)
tmp['area_split_len'].hist()


# In[12]:


for i in range(tmp['area_split_len'].max()):
    from functools import lru_cache
    tmp['area_split_'+str(i+1)] = tmp['area_split'].apply(
        lru_cache(512)(lambda x: x[:i+1]))
tmp.head()


# In[13]:


for i in range(tmp['area_split_len'].max()):
    le = LabelEncoder()
    tmp['area_id_'+str(i+1)] = le.fit_transform(tmp['area_split_'+str(i+1)])
tmp.head()


# In[14]:


tmp[list(filter(lambda c: '_id' in c, tmp.columns))].head()


# In[15]:


tmp = tmp[list(filter(lambda c: '_id' in c, tmp.columns))]
train = train.merge(tmp, how='left', on='air_store_id')
print(train.shape)
train.head()


# In[16]:


train['target'] = train['visitors'].map(np.log1p)
train[train['is_train']==1]['target'].hist()


# In[17]:


list(train.columns)


# In[18]:


# non_key_li = ['visitors', 'target', 'visit_date']
key_li = [('air_store_id'), 
          ('visit_dow'), # didn't try others yet
          #('Date_FriSatSun_flg'), #0.517990
          #('Date_holiday_flg'), #0.504020
          #('Date_FriSatSun_and_holiday_flg'), #0.514047
          #('Date_FriSatSun_or_holiday_flg'), #best 0.508041
          ('Date_holiday_flg_diff'), #0.495193 rmsle amoung all visit data not LB
          #('Date_holiday_flg_nthday'),  #0.49993
          #('Date_holiday_flg_nthday_last'), #0.499404
          #('Date_holiday_flg_span'), #0.498301
          #('Date_FriSatSun_or_holiday_flg_diff'), #0.503311
          #('Date_FriSatSun_or_holiday_flg_diff'), #0.503311
          #('Date_FriSatSun_or_holiday_flg_nthday'), #0.502385
          #('Date_FriSatSun_or_holiday_flg_nthday_last'), #0.499277
          #('Date_FriSatSun_or_holiday_flg_span'), #0.496434
          ('genre_id'), 
          ('area_id_5') # didn't try others yet
         ]

assert len(key_li) <=6

from itertools import combinations
for i in range(1, 1+len(key_li)):
    combos = list(combinations(key_li, i))
    print(i, 'keys', 'with', len(combos), 'combinations')
    print(combos)


# In[19]:


def get_grp_mean(df, by, y='target'):
    suffix = '|'.join(by)
    tmp = df.groupby(by).agg({y: [np.mean,np.median]})[y]
    tmp.columns = ['mean@', 'median@']
    tmp.columns = [c+suffix for c in tmp.columns]
    return tmp.reset_index()
res_d = {}
for i in range(1, 1+len(key_li)):
    for by in combinations(key_li, i):
        print(i, by)
        res_d[by] = get_grp_mean(train[train['is_train']==1], by)


# In[20]:


res_df = train[train['is_train']==1].copy()
by_li = []
for by, res in res_d.items():
    print(by)
    by_li.append(by)
    res_df = res_df.merge(res, how='left', on=by)
res_df.head()


# In[21]:


res_cols = list(filter(lambda c: '@' in c, res_df.columns))
res_cols


# In[22]:


from sklearn.metrics import mean_squared_error
rmsle_li = []
for col in res_cols:
    rmsle = mean_squared_error(res_df[col], res_df['target'])**.5
    rmsle_li.append(rmsle)
    print(col, rmsle)


# In[23]:


rmsle_df = pd.DataFrame(rmsle_li, columns=['rmsle'], index=res_cols)
rmsle_df = rmsle_df.reset_index().rename(columns={'index':'stat@by'})
rmsle_df['keys_len'] = rmsle_df['stat@by'].apply(lambda x:len(x.split('@')[-1].split('|')))
rmsle_df = rmsle_df.sort_values(by=['rmsle', 'keys_len']).reset_index()
del rmsle_df['index']
rmsle_df


# In[24]:


rmsle_df['rmsle'].hist()


# In[25]:


def keep_shortest_keys(x):
    return list(x.sort_values(by='keys_len')['stat@by'])[0]
rmsle_mini_df = rmsle_df.groupby(['rmsle'])['stat@by', 'keys_len'].apply(
    keep_shortest_keys).reset_index()
rmsle_mini_df = rmsle_mini_df.rename(columns={0: 'stat@by'})
rmsle_mini_df['keys_len'] = rmsle_mini_df['stat@by'].apply(lambda x:len(x.split('@')[-1].split('|')))
rmsle_mini_df


# In[26]:


print('rmsle<0.65 results')
rmsle_mini_df[rmsle_mini_df['rmsle']<0.65]


# In[27]:


selected = list(rmsle_mini_df[rmsle_mini_df['rmsle']<0.65]['stat@by'])
selected[0].split('@')[-1].split('|')


# In[28]:


test_df = train[train['is_train']==0].copy()
test_df = test_id.merge(test_df, how='right', on=['air_store_id', 'visit_date'])
print('test shape', test_df.shape)
test_df.head()


# In[29]:


for i in range(len(selected)):
    keys = selected[i].split('@')[-1].split('|')
    merge_params = dict(on=keys, how='left')
    pred = pd.merge(left=test_df, right=res_df[keys+[selected[i]]], 
                    **merge_params).drop_duplicates().reset_index()[selected[i]]
    test_df[f'pred_{i}'] = pred
test_df


# In[30]:


np.sum(np.isnan(test_df.iloc[:, -len(selected):]))


# In[31]:


nan_mask = np.isnan(test_df.iloc[:, -len(selected):])
nan_index_li = [test_df[nan_mask.iloc[:, i]].index for i in range(nan_mask.shape[1])]
nan_index_li[-5]


# In[32]:


tmp = test_df.iloc[:, -len(selected):].copy()
tmp.fillna(method='bfill', axis=1, inplace=True)
test_df.iloc[:, -len(selected):] = tmp.copy()
np.sum(np.isnan(test_df.iloc[:, -len(selected):]))


# In[33]:


test_df.iloc[:, -len(selected):] = np.expm1(test_df.iloc[:, -len(selected):])


# In[34]:


def get_sub(df, pred_col):
    sub = df[['id', pred_col]].copy()
    sub = sub.rename(columns={pred_col:'visitors'})
    return sub
sub_best_stat = get_sub(test_df, 'pred_0')
rmsle_0 = rmsle_mini_df[rmsle_mini_df['rmsle']<0.65]['rmsle'][0]
print('pred_0 rmsle', rmsle_0)
sub_best_stat.to_csv(f'sub_{rmsle_0}.csv', index=False)
sub_best_stat.head()


# In[35]:


selected_rmsle_mini_df = rmsle_mini_df[rmsle_mini_df['rmsle']<0.65]
selected_rmsle_mini_df


# In[36]:


pred_ratio = np.exp(len(selected) - np.arange(len(selected)))
pred_ratio = pred_ratio / np.sum(pred_ratio)
pred_ratio


# In[37]:


pred_avgw = (test_df.iloc[:, -len(selected):] * pred_ratio).sum(axis=1)
pred_avg = test_df.iloc[:, -len(selected):].mean(axis=1)
test_df['pred_avgw'] = pred_avgw
test_df['pred_avg'] = pred_avg


# In[38]:


sub_avgw = get_sub(test_df, 'pred_avgw')
sub_avgw.to_csv('sub_avgw.csv', index=False)
sub_avgw.head()


# In[39]:


sub_avg = get_sub(test_df, 'pred_avg')
sub_avg.to_csv('sub_avg.csv', index=False)
sub_avg.head()

