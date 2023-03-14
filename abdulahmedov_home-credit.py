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


import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ShuffleSplit, learning_curve
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import zipfile

from IPython.display import Image


# In[3]:


path = '/kaggle/input/home-credit-default-risk/'

POS_CASH_balance = pd.read_csv(path+'POS_CASH_balance.csv')
bureau_balance = pd.read_csv(path+'bureau_balance.csv')
application_train = pd.read_csv(path+'application_train.csv')
previous_application = pd.read_csv(path+'previous_application.csv')
installments_payments = pd.read_csv(path+'installments_payments.csv')
credit_card_balance = pd.read_csv(path+'credit_card_balance.csv')
application_test = pd.read_csv(path+'application_test.csv')
bureau = pd.read_csv(path+'bureau.csv')


# In[4]:


# Выведем изображение с моделью данных
Image(url = "https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png")


# In[5]:


# Выведем shape'ы тренировочных и тестовых данных
print('application_train shape: {} rows, {} columns'.format(*application_train.shape))
print('application_test shape: {} rows, {} columns'.format(*application_test.shape))


# In[6]:


application_train.set_index('SK_ID_CURR', inplace=True)
application_test.set_index('SK_ID_CURR', inplace=True)

y = application_train['TARGET']


# In[7]:


# Определим категориальные и вещественные признаки
categorical_features = [col for col in application_test.columns if application_test[col].dtype == 'object']
numerical_features = [col for col in application_test.columns if application_test[col].dtype != 'object']
        
print('Data has {} categorical features, and {} numerical features'.format(
    len(categorical_features), len(numerical_features)))


# In[8]:


# Запилим функцию для визуализации распределения вещественных признаков
def plot_features_hist(df, features, cols=3, bins=200, window_width=7.5, window_height=5):
    cols = cols
    rows = (len(features) + cols - 1) // cols
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(cols * window_width, rows * window_height))
    for feature, grd in zip(features,
                            range(len(features))):
        ax = plt.subplot(gs[grd // cols, grd % cols])
        fig = plt.hist(df[feature].dropna(), bins=bins)
        plt.title(str(feature)
                  +' (min:'+str(round(min(df[feature].dropna())))
                  +', mean:'+str(round(np.mean(df[feature].dropna())))
                  +', max:'+str(round(max(df[feature].dropna())))+')')
    plt.show()


# In[9]:


plot_features_hist(application_train, numerical_features)


# In[10]:


plot_features_hist(application_test, numerical_features)


# In[11]:


print('application_test "DAYS_EMPLOYED" anomalies {}, {}%'.format(
    len(application_test[application_test['DAYS_EMPLOYED']==365243]),
    len(application_test[application_test['DAYS_EMPLOYED']==365243]) / len(application_test) * 100))
print('')
print('application_train "DAYS_EMPLOYED" anomalies {}, {}%'.format(
    len(application_train[application_train['DAYS_EMPLOYED']==365243]),
    len(application_train[application_train['DAYS_EMPLOYED']==365243]) / len(application_train) * 100))


# In[12]:


application_train['DAYS_EMPLOYED_ANOM'] = application_train["DAYS_EMPLOYED"] == 365243
application_train["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
application_test['DAYS_EMPLOYED_ANOM'] = application_test["DAYS_EMPLOYED"] == 365243
application_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)


# In[13]:


application_train = pd.get_dummies(data=application_train, columns=categorical_features, dummy_na=True)
application_test = pd.get_dummies(data=application_test, columns=categorical_features, dummy_na=True)


# In[14]:


print('application_train shape: {} rows {} columns'.format(*application_train.shape))
print('application_test shape: {} rows {} columns'.format(*application_test.shape))


# In[15]:


application_train, application_test = application_train.align(application_test, join='inner', axis = 1)

print('application_train shape: ', application_train.shape)
print('application_test shape: ', application_test.shape)


# In[16]:


missing_df = (application_train.isna().sum() / len(application_train)).reset_index()
missing_df.sort_values(ascending=False, by=0)


# In[17]:


def missing_indicator(df, features=None, inplace=False):
    if not features:
        features = df.columns
    if not inplace:
        df = df.copy()
    for feature in df[features].columns:
        if df[feature].isna().sum() > 0:
            df['missing_'+feature] = df[feature].isna().astype(int)
    return df


# In[18]:


application_train = missing_indicator(application_train)
application_test = missing_indicator(application_test)

print('application_train shape: ', application_train.shape)
print('application_test shape: ', application_test.shape)


# In[19]:


application_train, application_test = application_train.align(application_test, join='inner', axis = 1)

print('application_train shape: ', application_train.shape)
print('application_test shape: ', application_test.shape)


# In[20]:


binary_features_train = application_train[numerical_features].nunique()
binary_features_train = binary_features_train[binary_features_train<=2]
binary_features_train = binary_features_train.index

binary_features_test = application_test[numerical_features].nunique()
binary_features_test = binary_features_test[binary_features_test<=2]
binary_features_test = binary_features_test.index

min(binary_features_train == binary_features_test)


# In[21]:


binary_features_train


# In[22]:


application_train[binary_features_train] = application_train[binary_features_train].fillna(0)
application_test[binary_features_test] = application_test[binary_features_test].fillna(0)


# In[23]:


mean_imputer = Imputer(missing_values='NaN', strategy='mean')
application_train[numerical_features] = mean_imputer.fit_transform(application_train[numerical_features])
application_test[numerical_features] = mean_imputer.transform(application_test[numerical_features])


# In[24]:


application_train.corrwith(y).sort_values(ascending=False)


# In[25]:


application_train.corrwith(y).sort_values()


# In[26]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')
get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')


# In[27]:


get_ipython().system('apt-get install -y -qq libboost-all-dev')


# In[28]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[29]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')


# In[30]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# In[31]:


get_ipython().system('nvidia-smi')


# In[32]:


param = {
        'num_leaves': 10,
        'max_bin': 127,
        'min_data_in_leaf': 11,
        'learning_rate': 0.02,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0, 
        'bagging_freq': 5, 
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }


# In[33]:


import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


# In[34]:


get_ipython().run_cell_magic('time', '', 'nfold = 2\n\ntarget = \'target\'\npredictors = application_train.columns.values.tolist()\n\nskf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)\n\noof = np.zeros(len(application_train))\npredictions = np.zeros(len(application_test))\n\ni = 1\nfor train_index, valid_index in skf.split(application_train, y.values):\n    print("\\nfold {}".format(i))\n    xg_train = lgb.Dataset(application_train.iloc[train_index][predictors].values,\n                           label=y.iloc[train_index].values,\n                           feature_name=predictors,\n                           free_raw_data = False\n                           )\n    xg_valid = lgb.Dataset(application_train.iloc[valid_index][predictors].values,\n                           label=y.iloc[valid_index].values,\n                           feature_name=predictors,\n                           free_raw_data = False\n                           )   \n\n    \n    clf = lgb.train(param, xg_train, 5000, valid_sets = [xg_valid], verbose_eval=50, early_stopping_rounds = 50)\n    oof[valid_index] = clf.predict(application_train.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) \n    \n    predictions += clf.predict(application_test[predictors], num_iteration=clf.best_iteration) / nfold\n    i = i + 1\n\nprint("\\n\\nCV AUC: {:<0.2f}".format(metrics.roc_auc_score(y.values, oof)))')


# In[35]:


sample_submission = pd.read_csv(path+'sample_submission.csv')
sample_submission


# In[36]:


my_submission = pd.DataFrame({'SK_ID_CURR': application_test.index, 'TARGET': predictions})
my_submission


# In[37]:


my_submission.to_csv('submission.csv', index=False)


# In[38]:


my_submission


# In[39]:


from IPython.display import FileLink
FileLink('submission.csv')

