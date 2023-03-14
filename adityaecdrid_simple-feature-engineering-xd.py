#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries
import numpy as np 
import pandas as pd 
import os
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import lightgbm as lgb
#import xgboost as xgb
import time
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error,roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import gc
from catboost import CatBoostRegressor
from tqdm import tqdm

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings("ignore")

pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)
import os
print(os.listdir("../input"))


# In[2]:


#https://www.kaggle.com/theoviel/load-the-totality-of-the-data
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        #'RtpStateBitfield':                                     'float16',
        #'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
#        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int16',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        #'PuaMode':                                              'category',
        #'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        #'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        #'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    gc.collect()
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `fldname` of `df`."
    import re
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear']
    if time: attr = attr + ['Hour']
    for n in attr: 
        df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + n] = df[targ_pre + n]*1
    df = reduce_mem_usage(df,False)
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[3]:


cnt = np.load('../input/malware/ver_updated_defs_count.npy').item()
max(dict(cnt).values())


# In[4]:


# IMPORT TIMESTAMP DICTIONARY 
#from Public Datasets and Kernel by Chris
#fix link
datedict = np.load('../input/malware-timestamps/AvSigVersionTimestamps.npy')[()]
datedictOS = np.load('../input/malware-timestamps-2/OSVersionTimestamps.npy')[()]

from collections import OrderedDict
sorted_OS = OrderedDict(sorted(datedictOS.items(), key=lambda x: x[1]))
final_OS  = OrderedDict()
for idx,val in tqdm(enumerate(sorted_OS)):
    final_OS[val[0:]] = (val.split('.')[2], idx)
final_OS_ = OrderedDict(sorted(final_OS.items(), key=lambda x: x[1]))
datedictOS_idx = OrderedDict()
for idx,val in tqdm(enumerate(final_OS_)):
    try:
        datedictOS_idx[val.split('.')[2]+'.'+val.split('.')[3]] = idx
    except:
        datedictOS_idx[val.split('.')[1]+'.'+val.split('.')[2]] = idx
        #print(val)
del final_OS_, final_OS
gc.collect()


# In[5]:


all_dates = np.load('../input/malware/all_dates_v2.npy').item()
months={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
temp = pd.read_csv('../input/microsoft-malware-prediction/train.csv', usecols=['AvSigVersion', 'HasDetections'], dtype={'HasDetections':'uint8','AvSigVersion':'str'})


# In[6]:


temp['release_dates'] = temp['AvSigVersion'].map(all_dates)
temp['year'] = temp['release_dates'].apply(lambda x: x[7:11])
temp.drop('release_dates', axis=1, inplace=True)
gc.collect()


# In[7]:


sig_2018 = np.asarray(list(set(temp[(temp['year'] == '2017') | (temp['year'] == '2018')]['AvSigVersion'].values)))
del temp #haha mem hunger
gc.collect()
print(len(sig_2018)) #needs to be improved as we have many in which we couldn't find versions released..


# In[8]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/microsoft-malware-prediction/train.csv', dtype=dtypes, nrows= 1500000)\ntrain.loc[train[train['OsBuildLab'].isnull()]['OsBuildLab'].index, 'OsBuildLab'] = '17134.1.amd64fre.rs4_release.180410-1804'")


# In[9]:


numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = [c for c,v in dtypes.items() if v in numerics]
true_numerical_columns = [
    'Census_ProcessorCoreCount',
    'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges'
]
binary_variables = [c for c in train.columns if train[c].nunique() == 2]
categorical_columns = [c for c in train.columns 
                       if (c not in true_numerical_columns) & (c not in binary_variables)]


# In[10]:


# @cpmp https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
import numpy as np 
from numba import jit

@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

def eval_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'auc', fast_auc(labels, preds), True

def split_sig(sigs):
    """Split the signature given as string into a 4-tuple of integers."""
    return tuple(int(part) for part in sigs.replace('1.2&#x17;3.1144.0','1.273.1144.0').split('.'))

def my_key(item):
    return split_sig(item[0])

items = dict(sorted(datedict.items(), key=my_key))


# In[11]:


stats= []
for col in train.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0], train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))
    
stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
stats_df.sort_values(['Percentage of missing values', 'Unique_values'], ascending=False).head(15)
del stats_df, stats
gc.collect()


# In[12]:


good_cols = list(train.columns)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.8 and col not in ['ProductName', 'DefaultBrowsersIdentifier','Firewall', 'IsProtected', 'AVProductsEnabled', 'Census_GenuineStateName']:
        good_cols.remove(col)

train = train[good_cols]
train = reduce_mem_usage(train)
gc.collect()
len(good_cols)


# In[13]:


get_ipython().run_cell_magic('time', '', "test_dtypes = {k: v for k, v in dtypes.items() if k in good_cols}\ntest = pd.read_csv('../input/microsoft-malware-prediction/test.csv', dtype=test_dtypes, usecols=good_cols[:-1])\ndel test_dtypes , good_cols\ntest.loc[6529507, 'OsBuildLab'] = '17134.1.amd64fre.rs4_release.180410-1804' #17134.1*amd64fre.rs4_release.180410-1804\ntest.loc[test[test['OsBuildLab'].isnull()]['OsBuildLab'].index, 'OsBuildLab'] = '17134.1.amd64fre.rs4_release.180410-1804'\ngc.collect()")


# In[14]:


new_map, cnt = {}, 0
for key in tqdm(items.keys()):
    new_map[key] = cnt
    cnt +=1

#Time Sereis EDA Kernel by Chris
#https://www.kaggle.com/cdeotte/time-series-eda-malware-0-64
# FEATURE ENGINEER - WEEK
first = datetime.datetime(2016,1,1); datedict2 = {}
for x in datedict: datedict2[x] = (datedict[x]-first).days//7
train['Week_2016'] = train['AvSigVersion'].map(datedict2)
test['Week_2016'] = test['AvSigVersion'].map(datedict2)

# FEATURE ENGINEER - WEEK
first = datetime.datetime(2018,1,1); datedict2 = {}
for x in datedict: datedict2[x] = (datedict[x]-first).days//7
train['Week_2018'] = train['AvSigVersion'].map(datedict2)
test['Week_2018'] = test['AvSigVersion'].map(datedict2)

train['sort'] = train['AvSigVersion'].map(new_map)
test['sort'] = test['AvSigVersion'].map(new_map)


train['x1'] = train['OsBuildLab'].apply(lambda x: x.split('.')[0] +'.'+ x.split('.')[1])
train['x2'] = train['Census_OSVersion'].apply(lambda x: x.split('.')[2] +'.'+ x.split('.')[3])
train['temp_x1'] = train['x1'].map(datedictOS_idx)
train['temp_x2'] = train['x2'].map(datedictOS_idx)
train['temp_diffs'] = abs(train['temp_x2'] - train['temp_x1'])

test['x1'] = test['OsBuildLab'].apply(lambda x: x.split('.')[0] +'.'+ x.split('.')[1])
test['x2'] = test['Census_OSVersion'].apply(lambda x: x.split('.')[2] +'.'+ x.split('.')[3])
test['temp_x1'] = test['x1'].map(datedictOS_idx)
test['temp_x2'] = test['x2'].map(datedictOS_idx)
test['temp_diffs'] = abs(test['temp_x2'] - test['temp_x1'])

del train['temp_x1'], train['temp_x2'], train['x1'], train['x2'], test['temp_x1'], test['temp_x2'], test['x1'], test['x2']
del first, datedict, datedict2
gc.collect()


# In[15]:


train.head()


# In[16]:


frequency_encoded_variables = [
    'Census_OEMModelIdentifier',
    'CityIdentifier',
    'OrganizationIdentifier',
    'Census_FirmwareVersionIdentifier',
    'AvSigVersion',
    'Census_OSInstallTypeName',
    'Census_OEMNameIdentifier',
    'DefaultBrowsersIdentifier'
]


# In[17]:


#public kernel (will fix the link)
train.SmartScreen=train.SmartScreen.str.lower()
train.SmartScreen.replace({"promt":"prompt",
                        "promprt":"prompt",
                        "00000000":"0",
                        "enabled":"on",
                        "of":"off" ,
                        "deny":"0" , # just one
                        "requiredadmin":"requireadmin"
                       },inplace=True)
train.SmartScreen = train.SmartScreen.astype("category")

test.SmartScreen = test.SmartScreen.str.lower()
test.SmartScreen.replace({"promt":"prompt",
                        "promprt":"prompt",
                        "00000000":"0",
                        "enabled":"on",
                        "of":"off" ,
                        "deny":"0" , # just one
                        "requiredadmin":"requireadmin"
                       },inplace=True)
test.SmartScreen = test.SmartScreen.astype("category")


# In[18]:


# print('grouping combination...')
# gp = train[['CountryIdentifier','OrganizationIdentifier', 'Census_OSInstallTypeName']].groupby(by=['CountryIdentifier','OrganizationIdentifier'], sort=False)[['Census_OSInstallTypeName']].count().reset_index().rename(columns={'Census_OSInstallTypeName':'cnt_cnt_org_os'})
# train = train.merge(gp, on=['CountryIdentifier','OrganizationIdentifier'], how='left')
# del gp
# gc.collect()
# print('grouping combination...')
# gp = test[['CountryIdentifier','OrganizationIdentifier', 'Census_OSInstallTypeName']].groupby(by=['CountryIdentifier','OrganizationIdentifier'], sort=False)[['Census_OSInstallTypeName']].count().reset_index().rename(columns={'Census_OSInstallTypeName':'cnt_cnt_org_os'})
# test = test.merge(gp, on=['CountryIdentifier','OrganizationIdentifier'], how='left')
# del gp
# train = reduce_mem_usage(train)
# test = reduce_mem_usage(test)
# gc.collect()

# print('grouping combination...')
# gp = train[['CountryIdentifier','OrganizationIdentifier','CityIdentifier', 'Census_OSInstallTypeName']].groupby(['CountryIdentifier','OrganizationIdentifier', 'CityIdentifier'])[['Census_OSInstallTypeName']].count().reset_index().rename(columns={'Census_OSInstallTypeName':'cnt_cnt_org_city_os'})
# train = train.merge(gp, on=['CountryIdentifier','OrganizationIdentifier','CityIdentifier'], how='left')
# del gp
# gc.collect()
# print('grouping combination...')
# gp = test[['CountryIdentifier','OrganizationIdentifier','CityIdentifier', 'Census_OSInstallTypeName']].groupby(['CountryIdentifier','OrganizationIdentifier', 'CityIdentifier'])[['Census_OSInstallTypeName']].count().reset_index().rename(columns={'Census_OSInstallTypeName':'cnt_cnt_org_city_os'})
# test = test.merge(gp, on=['CountryIdentifier','OrganizationIdentifier', 'CityIdentifier'], how='left')
# del gp
# train = reduce_mem_usage(train)
# test = reduce_mem_usage(test)
# gc.collect()

# print('grouping combination...')
# gp = train[['CountryIdentifier','OrganizationIdentifier','Census_OSBuildNumber', 'Census_OSInstallTypeName']].groupby(['CountryIdentifier','OrganizationIdentifier', 'Census_OSBuildNumber'], sort=False)[['Census_OSInstallTypeName']].count().reset_index().rename(columns={'Census_OSInstallTypeName':'cnt_cnt_org_build_type'})
# train = train.merge(gp, on=['CountryIdentifier','OrganizationIdentifier', 'Census_OSBuildNumber'], how='left')
# del gp
# gc.collect()
# print('grouping combination...')
# gp = test[['CountryIdentifier','OrganizationIdentifier','Census_OSBuildNumber', 'Census_OSInstallTypeName']].groupby(['CountryIdentifier','OrganizationIdentifier', 'Census_OSBuildNumber'], sort=False)[['Census_OSInstallTypeName']].count().reset_index().rename(columns={'Census_OSInstallTypeName':'cnt_cnt_org_build_type'})
# test = test.merge(gp, on=['CountryIdentifier','OrganizationIdentifier', 'Census_OSBuildNumber'], how='left')
# del gp
# train = reduce_mem_usage(train)
# test = reduce_mem_usage(test)
# gc.collect()


train['fe_avsig_org_freq']        = train[['AvSigVersion','OrganizationIdentifier','OsBuild']].groupby(['AvSigVersion','OrganizationIdentifier'])['OsBuild'].transform('count') / train.shape[0]
train['fe_avsig_cty_freq']        = train[['AvSigVersion','CityIdentifier','OsBuild']].groupby(['AvSigVersion','CityIdentifier'])['OsBuild'].transform('count') / train.shape[0]
train['fe_avsig_gamer_freq']      = train[['AvSigVersion','Wdft_IsGamer', 'OsBuild']].groupby(['AvSigVersion','Wdft_IsGamer'])['OsBuild'].transform('count') / train.shape[0]
train['fe_cpucores_region_freq']  = train[['Census_ProcessorCoreCount','Wdft_RegionIdentifier','OsBuild']].groupby(['Census_ProcessorCoreCount','Wdft_RegionIdentifier'])['OsBuild'].transform('count') / train.shape[0]

test['fe_avsig_org_freq']        = test[['AvSigVersion','OrganizationIdentifier','OsBuild']].groupby(['AvSigVersion','OrganizationIdentifier'])['OsBuild'].transform('count') / test.shape[0]
test['fe_avsig_cty_freq']        = test[['AvSigVersion','CityIdentifier','OsBuild']].groupby(['AvSigVersion','CityIdentifier'])['OsBuild'].transform('count') / test.shape[0]
test['fe_avsig_gamer_freq']      = test[['AvSigVersion','Wdft_IsGamer', 'OsBuild']].groupby(['AvSigVersion','Wdft_IsGamer'])['OsBuild'].transform('count') / test.shape[0]
test['fe_cpucores_region_freq']  = test[['Census_ProcessorCoreCount','Wdft_RegionIdentifier','OsBuild']].groupby(['Census_ProcessorCoreCount','Wdft_RegionIdentifier'])['OsBuild'].transform('count') / test.shape[0]


# In[19]:


print(gc.collect())
train['AvSigVersion'] = train['AvSigVersion'].replace(r'[^\.|0-9]','1.273.1826.0')
train['EngineVersion_2'] = train['EngineVersion'].apply(lambda x: x.split('.')[2]).astype('category')

train['OsBuild_exact'] = train['OsBuildLab'].apply(lambda x: x.split('.')[0] +'.'+ x.split('.')[1])
train['OsBuild_exact'] = train['OsBuild_exact'].astype('category')

train['AppVersion_1'] = train['AppVersion'].apply(lambda x: x.split('.')[1]).astype('category')
train['AppVersion_2'] = train['AppVersion'].apply(lambda x: x.split('.')[2]).astype('category')
train['AppVersion_3'] = train['AppVersion'].apply(lambda x: x.split('.')[3]).astype('category')

train['AvSigVersion_minor'] = train['AvSigVersion'].apply(lambda x: x.split('.')[1]).astype('category')
train['AvSigVersion_build'] = train['AvSigVersion'].apply(lambda x: x.split('.')[2]).astype('category')
train['AvSigVersion_minor_build'] = train['AvSigVersion'].apply(lambda x: float((x.split('.')[1]) +'.'+(x.split('.')[2]))).astype('float32')


# In[20]:


test['AvSigVersion'] = test['AvSigVersion'].replace(r'[^\.|0-9]','1.273.1826.0')
test['EngineVersion_2'] = test['EngineVersion'].apply(lambda x: x.split('.')[2]).astype('category')

test['OsBuild_exact']  =  test['OsBuildLab'].apply(lambda x: x.split('.')[0] +'.'+ x.split('.')[1])
test['OsBuild_exact']  = test['OsBuild_exact'].astype('category')

test['AppVersion_1'] = test['AppVersion'].apply(lambda x: x.split('.')[1]).astype('category')
test['AppVersion_2'] = test['AppVersion'].apply(lambda x: x.split('.')[2]).astype('category')
test['AppVersion_3'] = test['AppVersion'].apply(lambda x: x.split('.')[3]).astype('category')

test['AvSigVersion_minor'] = test['AvSigVersion'].apply(lambda x: x.split('.')[1]).astype('category')
test['AvSigVersion_build'] = test['AvSigVersion'].apply(lambda x: x.split('.')[2]).astype('category')
test['AvSigVersion_minor_build'] = test['AvSigVersion'].apply(lambda x: float((x.split('.')[1]) +'.'+(x.split('.')[2]))).astype('float32')


# In[21]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

'''IsProtected - 
This is a calculated field derived from the Spynet Report's AV Products field. Returns: 
a. TRUE if there is at least one active and up-to-date antivirus product running on this machine. 
b. FALSE if there is no active AV product on this machine, or if the AV is active, but is not receiving the latest updates. 
c. null if there are no Anti Virus Products in the report. Returns: Whether a machine is protected.''';

train['no_av_at_risk'] = 0
train.loc[train['AVProductsEnabled'].isin([0]) == True, 'no_av_at_risk'] = 1

test['no_av_at_risk'] = 0
test.loc[test['AVProductsEnabled'].isin([0]) == True, 'no_av_at_risk'] = 1

train['not_genuine_user'] = 0
train.loc[train['Census_GenuineStateName'].isin(['IS_GENUINE']) == False, 'not_genuine_user'] = 1

test['not_genuine_user'] = 0
test.loc[test['Census_GenuineStateName'].isin(['IS_GENUINE']) == False, 'not_genuine_user'] = 1

train['AvSigVersion_sum'] = train['AvSigVersion'].apply(lambda x: float(x.split('.')[1]) + float(x.split('.')[2])).astype(int).values
test['AvSigVersion_sum'] = test['AvSigVersion'].apply(lambda x: float(x.split('.')[1]) + float(x.split('.')[2])).astype(int).values

train['AvSigVersion'] = train['AvSigVersion'].astype('category')
test['AvSigVersion'] = test['AvSigVersion'].astype('category')

train['OsBuild_exact'] = train['OsBuildLab'].apply(lambda x: x.split('.')[0] +'.'+ x.split('.')[1])
test['OsBuild_exact']  =  test['OsBuildLab'].apply(lambda x: x.split('.')[0] +'.'+ x.split('.')[1])

train['OsBuild_exact'] = train['OsBuild_exact'].astype('category')
test['OsBuild_exact']  = test['OsBuild_exact'].astype('category')

top_20 = train['AVProductStatesIdentifier'].value_counts(dropna=False, normalize=True).cumsum().index[:20]
train['magic_4'] = 0
test['magic_4']  = 0
train.loc[train['AVProductStatesIdentifier'].isin(top_20) == True, 'magic_4'] = 1
test.loc[test['AVProductStatesIdentifier'].isin(top_20) == True, 'magic_4']   = 1
del top_20
gc.collect()


# In[22]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

train['aspect_ratio'] = train['Census_InternalPrimaryDisplayResolutionHorizontal']/ train['Census_InternalPrimaryDisplayResolutionVertical']
test['aspect_ratio']  = test['Census_InternalPrimaryDisplayResolutionHorizontal']/ train['Census_InternalPrimaryDisplayResolutionVertical']

train['primary_drive_c_ratio'] = train['Census_SystemVolumeTotalCapacity']/ train['Census_PrimaryDiskTotalCapacity']
test['primary_drive_c_ratio'] = test['Census_SystemVolumeTotalCapacity']/ test['Census_PrimaryDiskTotalCapacity']

train['non_primary_drive_MB'] = train['Census_PrimaryDiskTotalCapacity'] - train['Census_SystemVolumeTotalCapacity']
test['non_primary_drive_MB']  = test['Census_PrimaryDiskTotalCapacity']  - test['Census_SystemVolumeTotalCapacity']

train['ram_per_processor'] = train['Census_TotalPhysicalRAM']/ train['Census_ProcessorCoreCount']
test['ram_per_processor']  = test['Census_TotalPhysicalRAM']/ test['Census_ProcessorCoreCount']

train['physical_cores'] = train['Census_ProcessorCoreCount'] / 2
test['physical_cores']  = test['Census_ProcessorCoreCount'] / 2

train['hghdec_cnt'] = 0
test['hghdec_cnt'] = 0
train.loc[train['CountryIdentifier'].isin([104,95,214,89,94,59,21,100,85,195,159,57,155,188,33,44,18,88,81,205,141]) == True, 'hghdec_cnt'] = 1
test.loc[test['CountryIdentifier'].isin([104,95,214,89,94,59,21,100,85,195,159,57,155,188,33,44,18,88,81,205,141]) == True, 'hghdec_cnt'] = 1;

train['SmartScreen_dummy'] = 0
test['SmartScreen_dummy'] = 0
train.loc[train['SmartScreen'].isin(['ExistsNotSet', 'RequireAdmin', 'Warn']) == True, 'SmartScreen_dummy'] = 1
test.loc[test['SmartScreen'].isin(['ExistsNotSet', 'RequireAdmin', 'Warn']) == True, 'SmartScreen_dummy'] = 1;

train['one_less_AVproductInstalled'] = train['AVProductsInstalled'] - 1
test['one_less_AVproductInstalled'] = test['AVProductsInstalled'] - 1


# In[23]:


def frequency_encoding(variable):
    t = train[variable].value_counts().reset_index()
    t = t.reset_index()
    t.loc[t[variable] == 1, 'level_0'] = np.nan
    t.set_index('index', inplace=True)
    max_label = t['level_0'].max() + 1
    t.fillna(max_label, inplace=True)
    return t.to_dict()['level_0']

from tqdm import tqdm_notebook as tqdm
for variable in tqdm(frequency_encoded_variables):
    freq_enc_dict = frequency_encoding(variable)
    train[variable] = train[variable].map(lambda x: freq_enc_dict.get(x, -1)).astype(int)
    test[variable] = test[variable].map(lambda x: freq_enc_dict.get(x, -1)).astype(int)
    categorical_columns.remove(variable)

gc.collect()

train.drop(['ProductName', 'DefaultBrowsersIdentifier','Firewall', 'IsProtected', 'AVProductsEnabled', 'Census_GenuineStateName'], inplace=True, axis=1)
test.drop(['ProductName', 'DefaultBrowsersIdentifier','Firewall', 'IsProtected', 'AVProductsEnabled', 'Census_GenuineStateName'], inplace=True, axis=1)
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
gc.collect()


# In[24]:


'''
'Microsoft.Windows.Appraiser.General.SystemWimAdd' -> 'Census_IsWIMBootEnabled'
'Census.Hardware'    ->  'Census_ChassisTypeName', 'Census_OEMNameIdentifier', 'Census_OEMModelIdentifier',
'Census.Firmware'    ->  'Census_FirmwareManufacturerIdentifier', 'Census_FirmwareVersionIdentifier',
'Census.Flighting'   ->  'Census_IsFlightingInternal'
'Census.UserDisplay' ->  'Census_InternalPrimaryDiagonalDisplaySizeInInches','Census_InternalPrimaryDisplayResolutionHorizontal',
                         'Census_InternalPrimaryDisplayResolutionVertical',
'Census.Storage'     ->  'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName', 'Census_SystemVolumeTotalCapacity',
'Census.Processor'   ->  'Census_ProcessorCoreCount','Census_ProcessorManufacturerIdentifier', 'Census_ProcessorModelIdentifier',
'Census.Battery'     ->  'Census_InternalBatteryType', 'Census_InternalBatteryNumberOfCharges',
'Census.OS'          ->  'Census_OSVersion','Census_OSBranch', 'Census_ActivationChannel'
                         'Census_OSBuildNumber','Census_OSBuildRevision','Census_OSEdition', 'Census_IsSecureBootEnabled'
                         'Census_OSUILocaleIdentifier','Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName'
'''
'''Win10Version	Arch	Year	Critical Updates Last Date	Bypass something	Gain Information	Gain Privileges Total Vul Cnt
1511	x86	2015	08-06-18	4	2	9	38
1511	x86	2016	08-06-18	3	3	7	38
1511	x86	2017	08-06-18	1	3	0	38
1511	x64	2015	08-06-18	4	2	9	40
1511	x64	2016	08-06-18	3	3	7	40
1511	x64	2017	08-06-18	1	3	0	40
1607	x86	2016	08-06-18	1	0	2	9
1607	x86	2017	08-06-18	1	3	0	9
1607	x86	2018	08-06-18	0	0	0	9
1607	x64	2016	08-06-18	1	0	2	13
1607	x64	2017	08-06-18	1	3	0	13
1607	x64	2018	08-06-18	1	1	0	13
1703	x64	2018	15-05-18	1	1	0	419
1709	x64	2018	15-05-18	1	1	0	236
1803	x64	2018	-	1	1	0	131
1809	x64	2018	-	1	0	0	41
''';


# In[25]:


train['Census_ProcessorModelIdentifier'] = train['Census_ProcessorModelIdentifier'].astype('category')
test['Census_ProcessorModelIdentifier']  = test['Census_ProcessorModelIdentifier'].astype('category')


# In[26]:


top_10 = train['Census_TotalPhysicalRAM'].value_counts(dropna=False, normalize=True).cumsum().index[:10]
train.loc[train['Census_TotalPhysicalRAM'].isin(top_10) == False, 'Census_TotalPhysicalRAM'] = 1024
test.loc[test['Census_TotalPhysicalRAM'].isin(top_10) == False, 'Census_TotalPhysicalRAM']   = 1024
del top_10


# In[27]:


train['Census_InternalPrimaryDiagonalDisplaySizeInInches'] = train['Census_InternalPrimaryDiagonalDisplaySizeInInches'].astype('category')
test['Census_InternalPrimaryDiagonalDisplaySizeInInches'] = test['Census_InternalPrimaryDiagonalDisplaySizeInInches'].astype('category')
train['Census_InternalPrimaryDisplayResolutionHorizontal'] = train['Census_InternalPrimaryDisplayResolutionHorizontal'].astype('category')
test['Census_InternalPrimaryDisplayResolutionHorizontal'] = test['Census_InternalPrimaryDisplayResolutionHorizontal'].astype('category')
train['Census_InternalPrimaryDisplayResolutionVertical'] = train['Census_InternalPrimaryDisplayResolutionVertical'].astype('category')
test['Census_InternalPrimaryDisplayResolutionVertical'] = test['Census_InternalPrimaryDisplayResolutionVertical'].astype('category')


# In[28]:


train['OsBuild'] = train['OsBuild'].astype('category')
test['OsBuild'] = test['OsBuild'].astype('category')


# In[29]:


# https://www.kaggle.com/youhanlee/my-eda-i-want-to-see-all
# grouping battary types by name
def group_battery(x):
    x = x.lower()
    if 'li' in x:
        return 1
    else:
        return 0
    
train['Census_InternalBatteryType'] = train['Census_InternalBatteryType'].apply(group_battery)
test['Census_InternalBatteryType'] = test['Census_InternalBatteryType'].apply(group_battery)


# In[30]:


def rename_edition(x):
    x = x.lower()
    if 'core' in x:
        return 'Core'
    elif 'pro' in x:
        return 'pro'
    elif 'enterprise' in x:
        return 'Enterprise'
    elif 'server' in x:
        return 'Server'
    elif 'home' in x:
        return 'Home'
    elif 'education' in x:
        return 'Education'
    elif 'cloud' in x:
        return 'Cloud'
    else:
        return x


# In[31]:


train['Census_OSEdition'] = train['Census_OSEdition'].astype(str)
test['Census_OSEdition'] = test['Census_OSEdition'].astype(str)
train['Census_OSEdition'] = train['Census_OSEdition'].apply(rename_edition)
test['Census_OSEdition'] = test['Census_OSEdition'].apply(rename_edition)
train['Census_OSEdition'] = train['Census_OSEdition'].astype('category')
test['Census_OSEdition'] = test['Census_OSEdition'].astype('category')


# In[32]:


train['Census_OSSkuName'] = train['Census_OSSkuName'].astype(str)
test['Census_OSSkuName'] = test['Census_OSSkuName'].astype(str)
train['Census_OSSkuName'] = train['Census_OSSkuName'].apply(rename_edition)
test['Census_OSSkuName'] = test['Census_OSSkuName'].apply(rename_edition)
train['Census_OSSkuName'] = train['Census_OSSkuName'].astype('category')
test['Census_OSSkuName'] = test['Census_OSSkuName'].astype('category')


# In[33]:


train['Census_OSInstallLanguageIdentifier'] = train['Census_OSInstallLanguageIdentifier'].astype('category')
test['Census_OSInstallLanguageIdentifier'] = test['Census_OSInstallLanguageIdentifier'].astype('category')


# In[34]:


train['Census_OSUILocaleIdentifier'] = train['Census_OSUILocaleIdentifier'].astype('category')
test['Census_OSUILocaleIdentifier'] = test['Census_OSUILocaleIdentifier'].astype('category')


# In[35]:


train['OsSuite'] = train['OsSuite'].astype('category')
test['OsSuite'] = test['OsSuite'].astype('category')


# In[36]:


train.head()


# In[37]:


cat_cols = [col for col in train.columns if col not in (['MachineIdentifier', 'Census_SystemVolumeTotalCapacity', 'HasDetections'] + frequency_encoded_variables)  and str(train[col].dtype) == 'category']
len(cat_cols)


# In[38]:


print(train.shape, test.shape)
assert(train.shape[1] == test.shape[1]+1)


# In[39]:


train = reduce_mem_usage(train, True)
test  = reduce_mem_usage(test, True)


# In[40]:


for col in cat_cols:
    if train[col].nunique() > 2800:
        print(col, train[col].nunique())
        train.drop([col], axis=1, inplace=True)
        test.drop([col], axis=1, inplace=True)
        cat_cols.remove(col)


# In[41]:


train = reduce_mem_usage(train, True)
test  = reduce_mem_usage(test, True)
gc.collect()


# In[42]:


get_ipython().run_cell_magic('time', '', 'indexer = {}\nfor col in cat_cols:\n    # print(col)\n    _, indexer[col] = pd.factorize(train[col])\n    \nfor col in tqdm(cat_cols):\n    \n    gc.collect()\n    train[col] = indexer[col].get_indexer(train[col])\n    test[col] = indexer[col].get_indexer(test[col])\n    \n    train = reduce_mem_usage(train, False)\n    test  = reduce_mem_usage(test, False)')


# In[43]:


y = train['HasDetections']
train = train.drop(['HasDetections', 'MachineIdentifier', 'OsPlatformSubRelease'], axis=1)
test = test.drop(['MachineIdentifier', 'OsPlatformSubRelease'], axis=1)
del cnt, sig_2018, all_dates
gc.collect()


# In[44]:


#no neeed though
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# In[45]:


n_fold = 3
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=17)
gc.collect()


# In[46]:


cat_cols.pop(4)


# In[47]:


def train_model(X=train, X_test=test, y=y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False):

    _ = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    
    for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(X, y))):
        
        print('Fold', fold_n, 'started at', time.ctime())
        
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        print('Shapes Are', X_train.shape, X_valid.shape)
        
        if model_type == 'lgb':
            
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature = cat_cols)
            valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature = cat_cols)    

            model = lgb.train(params,
                    train_data,
                    num_boost_round= 2500,
                    valid_sets = [train_data, valid_data],
                    verbose_eval= 100,
                    early_stopping_rounds = 200, 
                    feval=eval_auc)
            
            del train_data, valid_data, X_train, train_index, valid_index
            print(gc.collect())
            
            y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
            del X_valid
            
            gc.collect()
            
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        scores.append(roc_auc_score(y_valid, y_pred_valid))
        print('Fold AUC:', roc_auc_score(y_valid, y_pred_valid))
        prediction += y_pred    
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance(importance_type='gain')
            fold_importance["importance_split"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        
        if plot_feature_importance:
            feature_importance["importance"] /= n_fold
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features importance_type=\'gain\' (avg over folds)');
        
            return _, prediction, feature_importance, model
        return _, prediction
    else:
        return _, prediction


# In[48]:


params = {'num_leaves': 40,
         'min_data_in_leaf': 70, 
         'objective':'binary',
         'max_depth': 10,
         'learning_rate': 0.05,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 5,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 17,
         "lambda_l1": 0.1,
         "random_state": 17,
         "verbosity": -1,
         "subsample": 0.7,
         "drop_rate": 0.01}


# In[49]:


_ , prediction_lgb_1, feats_imp, model_lgb = train_model(params=params, model_type='lgb', plot_feature_importance=True)


# In[50]:


feats_imp = feats_imp[feats_imp['fold'] == 3]
cm = sns.light_palette("red", as_cmap=True)
feats_imp[['feature','importance','importance_split']].sort_values(by='importance', ascending=False).head(30).rename(columns={'importance':'imp_gain', 'importance_split':'imp_split'}).style.background_gradient(cmap = cm)
#this makes a lot more sense now (Imp by Gain) (Thanks tp cpmp and satian)


# In[51]:


import joblib
joblib.dump(model_lgb, 'model_lgb.model');


# In[52]:


feats_imp.to_csv('feats_imp_max_depth.csv', index=None)
del feats_imp, model_lgb
submission = pd.read_csv('../input/microsoft-malware-prediction/sample_submission.csv')
submission['HasDetections'] = prediction_lgb_1
submission.to_csv('lgb_max_depth.csv', index=False, compression='zip')
del prediction_lgb_1, submission
gc.collect()


# In[53]:


train['HasDetections'] = y
train.to_csv('new_train.csv.gz', index=False, compression='zip')
test.to_csv('new_test.csv.gz', index=False, compression='zip')

