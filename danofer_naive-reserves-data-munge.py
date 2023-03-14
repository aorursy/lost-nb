#!/usr/bin/env python
# coding: utf-8

# In[1]:


# based on: https://www.kaggle.com/the1owl/surprise-me/code

import numpy as np
import pandas as pd
from sklearn import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'), #need to create reservation features
    'hr': pd.read_csv('../input/hpg_reserve.csv'), #need to create reservation features
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }


# In[3]:


data['tra'].head()


# In[4]:


data['id'].head()


# In[5]:


data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek


# In[6]:


unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)


# In[7]:


tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 


# In[8]:


stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 


# In[9]:


stores.head()


# In[10]:


stores.air_genre_name.value_counts()


# In[11]:


# stores["air_genre_name_0"] = stores.air_genre_name.str.split("/",expand=True)[0]


# In[12]:


lbl = preprocessing.LabelEncoder()
# stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
# stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])


# In[13]:


data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])


# In[14]:


train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(data['tra'], stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id','dow']) 


# In[15]:


train.head()


# In[16]:


data['hs'].head()


# In[17]:


data['hr'].head()


# In[18]:


data['ar'].head()


# In[19]:


data['id'].head()


# In[20]:


data['id'].nunique()


# In[21]:


train.shape


# In[22]:


# map IDs
## https://stackoverflow.com/questions/36971661/python-pandas-map-using-2-columns-as-reference?noredirect=1&lq=1
# train.set_index("air_store_id").join(data["id"]).tail()
# train.set_index("air_store_id").join(data["id"].set_index(["air_store_id","hpg_store_id"])).isnull().sum()

train = train.set_index("air_store_id").join(data["id"].set_index("air_store_id"))
# train.set_index("air_store_id").join(data["id"].set_index("air_store_id")).isnull().sum()
# train.set_index("air_store_id").join(data["id"].set_index("air_store_id")).nunique()
test = test.set_index("air_store_id").join(data["id"].set_index("air_store_id"))


# In[23]:


train.head(3)


# In[24]:


train.info()


# In[25]:


train.reset_index().select_dtypes(['number']).iloc[:,1:7]


# In[26]:


lr = linear_model.LinearRegression(normalize=True, n_jobs=-1)
# lr.fit(train[col], np.log1p(train['visitors'].values))
# lr.fit(train.reset_index().select_dtypes(['number']).iloc[:,1:7].fillna(-1), np.log1p(train['visitors'].values))


# In[27]:


train["pred_lr_naive"] = model_selection.cross_val_predict(lr,train.reset_index().select_dtypes(['number']).iloc[:,1:7].fillna(-1),np.log1p(train['visitors'].values  ))


# In[28]:


lr.fit(train.reset_index().select_dtypes(['number']).iloc[:,1:7].fillna(-1), np.log1p(train['visitors'].values))

test["pred_lr_naive"] = lr.predict(test.reset_index().select_dtypes(['number']).iloc[:,1:7].fillna(-1))


# In[29]:


train.head()


# In[30]:


test.head()


# In[31]:


train.to_csv("train_partmerged_v1.csv.gz",compression="gzip")
test.to_csv("test_partmerged_v1.csv.gz",compression="gzip")


# In[32]:




