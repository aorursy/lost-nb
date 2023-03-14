#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


from sklearn.linear_model import SGDRegressor, LinearRegression, Lasso, Ridge, LogisticRegression

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from datetime import timedelta

import pickle
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

import warnings
warnings.filterwarnings(action='once')

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 1000)
plt.rcParams['figure.figsize'] = [8, 4]  # 12, 8  width


# In[3]:


train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
train['Date'] = pd.to_datetime(train['Date'])
train['County']=train['County'].fillna("")
train['Province_State']=train['Province_State'].fillna("")
print(min(train.Date),max(train.Date))
train.head()


# In[4]:


# negative TargetValue?
train.sort_values(by='TargetValue').head(20)


# In[5]:


a=train.groupby(['County','Province_State','Country_Region','Target'])['TargetValue'].quantile(q=0.05).reset_index()
b=train.groupby(['County','Province_State','Country_Region','Target'])['TargetValue'].quantile(q=0.5).reset_index()
c=train.groupby(['County','Province_State','Country_Region','Target'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['County','Province_State','Country_Region','Target','q0.05']
b.columns=['County','Province_State','Country_Region','Target','q0.5']
c.columns=['County','Province_State','Country_Region','Target','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05'].clip(0,10000)
a['q0.5']=a['q0.5'].clip(0,10000)
a['q0.95']=a['q0.95'].clip(0,10000)
a.head()


# In[6]:


test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
test['Date'] = pd.to_datetime(test['Date'])
test['County']=test['County'].fillna("")
test['Province_State']=test['Province_State'].fillna("")
test.head()


# In[7]:


print('train', min(train.Date), max(train.Date))
print('test', min(test.Date), max(test.Date))


# In[8]:


get_ipython().run_cell_magic('time', '', '\ndt_pred = pd.date_range(pd.to_datetime(max(train.Date)) + timedelta(days=1), max(test.Date))\nx_cols = [\'Weight\',\'x\', \'x2\']\ny_col = [\'TargetValue\']\ni = 0\na = pd.DataFrame()\nmodels = dict()\n\nfor key,grp in tqdm(train.groupby([\'County\',\'Province_State\',\'Country_Region\',\'Target\'])):\n    print(f\'key={key}\')\n    \n    grp = grp.sort_values(by=[\'Date\'])\n\n    n_train = grp.shape[0]\n    n_test  = dt_pred.shape[0]\n    n_all   = n_train + n_test\n\n    df_test = grp.head(len(dt_pred)).copy()\n    df_test[\'Id\'] = -1\n    df_test[\'Date\'] = dt_pred\n    df_test[\'TargetValue\'] = 0.\n    df = grp.append(df_test).copy().reset_index(drop=True)\n    test_filter = df.Date >= min(dt_pred)\n    df_test = df[test_filter]\n\n    # features\n    df[\'q0.05\'] = 0.\n    df[\'q0.5\'] = 0.\n    df[\'q0.95\'] = 0.\n    df[\'x\'] = list(range(df.shape[0]))\n    df[\'x2\'] = df.x**2\n\n\n    # train / fit\n    df_train, df_test = df[:n_train], df[n_train:]\n\n    # start with first non-zero\n    try:\n        start_x = min(df_train.query(\'TargetValue > 0\')[\'x\'])\n    except:\n        start_x = min(df_train.x)\n    df_train = df_train.query(f\'x >= {start_x}\')\n\n    X_train, y_train = df_train[x_cols], df_train[y_col]\n    X_test, y_test = df_test[x_cols], df_train[y_col]\n\n\n    # median / mean\n    f = Ridge(alpha=10.0).fit(X_train,y_train)\n    models[key] = f\n    fitted = f.predict(X_train)\n    y_test = f.predict(X_test)   \n    fitted_resid = fitted - y_train\n\n    quant = dict()\n    for q in [0.05, 0.5, 0.95]:\n        quant[q] = np.quantile(fitted_resid, q=q)\n\n\n    # 0.05\n    fitted_05 = fitted + (quant[0.05])\n    y_test_05 = y_test + (quant[0.05])\n\n    # 0.95  \n    fitted_95 = fitted + (quant[0.95])\n    y_test_95 = y_test + (quant[0.95])\n\n\n#         df_train[\'q0.05\'], df_train[\'q0.5\'], df_train[\'q0.95\'] = fitted_05.clip(0,10000), fitted.clip(0,10000), fitted_95.clip(0,10000)\n#         df_test.loc[:,\'q0.05\'], df_test.loc[:,\'q0.5\'], df_test.loc[:,\'q0.95\'] = y_test_05.clip(0,10000), y_test.clip(0,10000), y_test_95.clip(0,10000)\n#         df = df_train.append(df_test)\n    try:\n        start_x_dt = min(df.query(\'TargetValue > 0\')[\'Date\'])\n    except:\n        start_x_dt = min(df.Date)\n    df.loc[df.Date >= start_x_dt, \'q0.05\'] = np.concatenate([fitted_05,y_test_05]).clip(0, 10000)\n    df.loc[df.Date >= start_x_dt, \'q0.5\'] = np.concatenate([fitted,y_test]).clip(0, 10000)\n    df.loc[df.Date >= start_x_dt, \'q0.95\'] = np.concatenate([fitted_95,y_test_95]).clip(0, 10000)\n\n    a_cols = [\'County\',\'Province_State\',\'Country_Region\',\'Target\',\'Date\',\'q0.05\',\'q0.5\',\'q0.95\']\n    a = a.append(df.query(f\'Date>="{min(test.Date)}"\')[a_cols])\n\n    country = key[2]\n    prov    = key[1]\n    # Note: US has a LOT of counties\n    if i <= 4 or country in [\'Spain\',\'Italy\',\'Monaco\',\'China\',\'UK\',\'Canada\',\'Mexico\',\'Brazil\',\'France\', \'Japan\', \'Taiwan*\'] \\\n        or prov in [\'British Columbia\',\'Hong Kong\',\'New York\']:\n        fig, ax = plt.subplots()\n        ax.plot(X_train.x, y_train, label=\'data\')\n        ax.plot(X_train.x, fitted, label=\'fitted\')\n        ax.plot(X_test.x, y_test, label=\'pred\')\n        ax.plot(X_train.x, fitted_95, label=\'fitted_95\')\n        ax.plot(X_test.x, y_test_95, label=\'pred_95\')\n        ax.plot(X_train.x, fitted_05, label=\'fitted_05\')\n        ax.plot(X_test.x, y_test_05, label=\'pred_05\')\n        ax.legend()\n        title = f"{key}"\n        ax.set_title(title)\n\n#         if i >= 4:\n#             break\n\n    i += 1\n        \nm_fn = \'models.pickle\'\nwith open(m_fn, \'wb\') as f:\n    pickle.dump(models, f)\nprint(f\'Saved {len(models)} to {m_fn}\')')


# In[9]:


a


# In[10]:


test2 = test.merge(a,on=['Country_Region','County','Province_State','Target', 'Date'],how='left')
test2.head()


# In[11]:


# test=test.merge(a,on=['Country_Region','County','Province_State','Target'],how='left')
# test.head()


# In[12]:


sub = pd.melt(test2[['ForecastId','q0.05','q0.5','q0.95']], id_vars=['ForecastId'], value_vars=['q0.05','q0.5','q0.95'])
sub


# In[13]:


# sub=pd.melt(test2, id_vars=['ForecastId'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['ForecastId'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head()

