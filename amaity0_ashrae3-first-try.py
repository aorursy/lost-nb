#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import sys, warnings, math
warnings.filterwarnings('ignore')
import gc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


with open("../input/ashrae-energy-prediction/train.csv") as f:
    head = [next(f) for x in range(10)]
print(head)


# In[4]:


#https://hackersandslackers.com/downcast-numerical-columns-python-pandas/
import feather

print('train data:')
train = pd.read_csv('../input/ashrae-energy-prediction/train.csv', 
                    dtype={'building_id':np.uint16, 'meter':np.uint8, 'meter_reading':np.float64})
train['timestamp'] = pd.to_datetime(train['timestamp'], format="%Y %m %d %H:%M:%S")
print(train.info(memory_usage='deep'))
train.to_feather('train.feather')

print('-'*20);print('test data:')
test = pd.read_csv('../input/ashrae-energy-prediction/test.csv', 
                   dtype={'row_id':np.uint16,'building_id':np.uint16,'meter':np.uint16})
test['timestamp'] = pd.to_datetime(test['timestamp'], format="%Y %m %d %H:%M:%S")
print(test.info(memory_usage='deep'))
test.to_feather('test.feather')

print('-'*20);print('weather_train data:')
weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv',dtype={'site_id':np.uint16})
weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'],infer_datetime_format=True)
weather_train[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = weather_train[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']].apply(pd.to_numeric,downcast='float')
print(weather_train.info(memory_usage='deep'))
weather_train.to_feather('weather_train.feather')

print('-'*20);print('weather_test data:')
weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv',dtype={'site_id':np.uint16})
weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'],infer_datetime_format=True)
weather_test[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = weather_test[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']].apply(pd.to_numeric,downcast='float')
print(weather_test.info(memory_usage='deep'))
weather_test.to_feather('weather_test.feather')

print('-'*20);print('building_metadata data:')
building_metadata = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
building_metadata['primary_use'] = building_metadata['primary_use'].astype('category')
print(building_metadata.info(memory_usage='deep'))
building_metadata.to_feather('building_metadata.feather')

sample_submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')
sample_submission.to_feather('sample_submission.feather')

del train, test, weather_train, weather_test, building_metadata, sample_submission
gc.collect()


# In[5]:


weather_train = pd.read_feather('weather_train.feather')
weather_train = weather_train.groupby('site_id')                              .apply(lambda group: group.interpolate(limit_direction='both'))
#weather_train = weather_train.set_index('timestamp').interpolate(method='time') 

print(weather_train.isnull().sum())
#weather_train.groupby('site_id').apply(lambda group: group.isna().sum())


# In[6]:


train = pd.read_feather('train.feather')
#weather_train = pd.read_feather('weather_train.feather')
#train['meter_reading'] = np.log1p(train['meter_reading'])
building_metadata = pd.read_feather('building_metadata.feather')
train = train.merge(building_metadata, on='building_id', how='left')
train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
del weather_train, building_metadata
gc.collect()

print(train.isnull().sum())
train.tail()


# In[7]:


#taken from here:https://www.kaggle.com/kaushal2896/ashrae-eda-fe-lightgbm-1-12
def mean_without_overflow_fast(col):
    col /= len(col)
    return col.mean() * len(col)

missing_values = (100-train.count() / len(train) * 100).sort_values(ascending=False)
missing_features = train.loc[:, missing_values > 0.0]
missing_features = missing_features.apply(mean_without_overflow_fast)

for key in train.loc[:, missing_values > 0.0].keys():
    if key == 'year_built' or key == 'floor_count':
        train[key].fillna(math.floor(missing_features[key]), inplace=True)
    else:
        train[key].fillna(missing_features[key], inplace=True)

train.isnull().sum()


# In[8]:


#Frequency of primary_use
train.groupby(['primary_use']).agg({'site_id':'nunique'}).rename(columns={'site_id':'N'}) 


# In[9]:


train['square_feet'].hist(bins=32) #is this is wrong?
plt.xlabel("square_feet")
plt.ylabel("Frequency")


# In[10]:


get_mean = train.groupby(['primary_use']).agg({'meter_reading':'mean'}).rename(columns={'meter_reading':'mr_mean'}) 
get_mean.sort_values(by='mr_mean')


# In[11]:


da = (train['timestamp'].iloc[-1] - train['timestamp'].iloc[0])
print("Number of hours between start and end dates: ", da.total_seconds()/3600 + 1)


# In[12]:


count_full = train.groupby('building_id')['timestamp'].nunique()
#Remember count_full is a Series object
count_full = count_full[count_full==count_full.max()]
#ids with whole length
print(count_full.index)


# In[13]:


trfull = train[train['building_id'].isin(count_full.index)]
trfull.head()


# In[14]:


del train
gc.collect()


# In[15]:


num_date = trfull[trfull['building_id']==0].groupby(trfull['timestamp'].dt.floor('d')).count()
num_date.tail()


# In[16]:


num_date['timestamp'].value_counts()


# In[17]:


trfull.groupby('site_id').apply(lambda x: x['meter'].nunique())


# In[18]:


trfull.groupby('site_id').apply(lambda x: x['building_id'].nunique())


# In[19]:


#sns.set(rc={"lines.linewidth": 0.5})
trfull[((trfull['site_id']==3) & (trfull['meter']==0))]              .plot(x='timestamp',y='meter_reading',figsize=(12,6))


# In[20]:


site14 = trfull[trfull['site_id']==14]
fig, axes = plt.subplots(4,1,figsize=(14, 18))
for i in range(4):
    site14[site14['meter']==i][['timestamp', 'meter_reading']].set_index('timestamp')             .resample('H').sum()['meter_reading']             .plot(ax=axes[i], alpha=0.8, label='By hour')             .set_ylabel('Summation meter reading')
    axes[i].legend();
    axes[i].set_title('Meter: ' + str(i));
plt.tight_layout()


# In[21]:


fig, axes = plt.subplots(4,1,figsize=(14, 18))
for i in range(4):
    trfull[trfull['meter']==i][['timestamp', 'meter_reading']].set_index('timestamp')             .resample('H').sum()['meter_reading']             .plot(ax=axes[i], alpha=0.8, label='By hour')             .set_ylabel('Summation meter reading')
    axes[i].legend();
    axes[i].set_title('Meter: ' + str(i));
plt.tight_layout()


# In[22]:


trfull = trfull[~((trfull['meter'] == 2) & (trfull['building_id'] == 1099))]


# In[23]:


trfull = trfull.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')


# In[24]:


fig, axes = plt.subplots(4,1,figsize=(14, 18))
for i in range(4):
    trfull[trfull['meter']==i].groupby('timestamp')['meter_reading'].sum().plot(ax=axes[i])
    axes[i].set_title('Meter: ' + str(i))
plt.tight_layout()


# In[25]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def getFrames(dftrn, dftst, idx):
    #train part
    dftrn = dftrn[dftrn['meter']==idx]
    dftrn.set_index('timestamp',inplace=True)
    dftrn['primary_use'] = le.fit_transform(dftrn['primary_use'])
    cols = list(dftrn.columns)
    cols.remove('meter_reading')
    #test part
    dftst = dftst[dftst['meter']==idx]
    dftst['index1'] = dftst.index
    dftst.set_index('timestamp',inplace=True)
    dftst['primary_use'] = le.fit_transform(dftst['primary_use'])
    return dftrn[cols], dftrn['meter_reading'], dftst[cols], dftst['index1']


# In[26]:


#error metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
my_scorer = make_scorer(rmse,greater_is_better=False)

#from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

#model = ElasticNet(
#        alpha=1.0,
#        l1_ratio=0.3,
#        fit_intercept=True,
#        normalize=False,
#        precompute=False,
#        max_iter=16,
#        copy_X=True,
#        tol=0.1,
#        warm_start=False,
#        positive=False,
#        random_state=None,
#        selection='random'
#    )
model = RandomForestRegressor(random_state = 1, n_jobs = -1)

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
tscv = TimeSeriesSplit(n_splits=5)


# In[27]:


weather_test = pd.read_feather('weather_test.feather')
cols = list(weather_test.columns)
weather_test = weather_test.groupby('site_id')                            .apply(lambda group: group.interpolate(limit_direction='both'))
#weather_test = weather_test.set_index('timestamp').interpolate(method='time') 
weather_test.groupby('site_id').apply(lambda group: group.isna().sum())


# In[28]:


print(weather_test.isnull().sum())
weather_test.tail()


# In[29]:


test = pd.read_feather('test.feather')
building_metadata = pd.read_feather('building_metadata.feather')
test = test.merge(building_metadata, on='building_id', how='left')
#weather_test = pd.read_feather('weather_test.feather')
test = test.merge(weather_test, on=['site_id', 'timestamp'], how='left')
del weather_test, building_metadata
gc.collect()

print(test.isnull().sum())
test.tail()


# In[30]:


missing_values = (100-test.count() / len(test) * 100).sort_values(ascending=False)
missing_features = test.loc[:, missing_values > 0.0]
missing_features = missing_features.apply(mean_without_overflow_fast)

for key in test.loc[:, missing_values > 0.0].keys():
    if key == 'year_built' or key == 'floor_count':
        test[key].fillna(math.floor(missing_features[key]), inplace=True)
    else:
        test[key].fillna(missing_features[key], inplace=True)
        
test.isnull().sum()


# In[31]:


for i in range(4):
    X, y, _, _ = getFrames(trfull, test, i)
    scores = cross_val_score(model, X, y, cv=tscv, scoring=my_scorer)
    print("Meter-{0:d} Loss: {1:.3f} (+/- {2:.3f})".format(i, scores.mean(), scores.std()))


# In[32]:


from sklearn.model_selection import GridSearchCV

#params = {
#    'alpha':(0.1, 0.3, 0.5, 0.7, 0.9),
#    'l1_ratio':(0.1, 0.3, 0.5, 0.7, 0.9) 
#}
params = {
    'max_features' : ["auto", "sqrt", "log2"],
    'min_samples_split' : np.linspace(0.1, 1.0, 10)
}
predict_list = []

for i in range(4):
    X, y, tst, ref = getFrames(trfull, test, i)
    print('-'*20);print("Meter-{0:d}".format(i))
    gs = GridSearchCV(model, param_grid=params, cv=tscv, scoring=my_scorer, verbose=1)
    gs.fit(X,y)
    yp = gs.predict(tst)
    predict_list.append(np.vstack((ref.values,yp)))


# In[33]:


p = np.hstack(predict_list)
p = p.T
p = p[p[:,0].argsort()]
#print(p)
#print(p.shape)

sub = pd.DataFrame(p, columns=['row_id','meter_reading'])
sub.loc[sub['meter_reading']<0, 'meter_reading'] = 0
sub['row_id'] = sub['row_id'].astype(int)
sub.tail()
#from collections import Counter
#print([item for item, count in Counter(p).items() if count > 1])


# In[34]:


sub.to_csv('submission.csv',index=False, float_format='%.4f')

