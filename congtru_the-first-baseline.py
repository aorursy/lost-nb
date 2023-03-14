#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer 
import re
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


app_train=pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
app_test=pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')


# In[3]:


install=pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
des=pd.read_csv('/kaggle/input/home-credit-default-risk/HomeCredit_columns_description.csv',encoding= 'unicode_escape')
pos_cash=pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
sample=pd.read_csv('/kaggle/input/home-credit-default-risk/sample_submission.csv')
bureau=pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
credit_card=pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
prev=pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
default=pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')


# In[4]:


print(app_train.shape, app_test.shape)


# In[5]:


sample=pd.read_csv('/kaggle/input/home-credit-default-risk/sample_submission.csv')
sample.head()


# In[6]:


app_train.describe()


# In[7]:


app_test.describe()


# In[8]:


def mis_data(data):
    Total=data.isnull().sum()
    Percent=Total/data.shape[0]
    missing=pd.DataFrame({'Total': Total, 'Percent':Percent}).sort_values(by='Total', ascending=False)
    return missing
mis_train=mis_data(app_train)
print(mis_train.head(10))
mis_test=mis_data(app_train)
print(mis_test.head(10))


# In[9]:


num_cols=app_train.drop(['SK_ID_CURR', 'TARGET'], axis=1).select_dtypes(include='number').columns
print(num_cols)
print(len(num_cols))


# In[10]:


num_n_val=pd.DataFrame({'n_values':app_train[num_cols].nunique().sort_values(ascending=False)})
num_n_val[:10]


# In[11]:


# Put the negative values to positive
app_train[num_cols]=np.sign(app_train[num_cols])*app_train[num_cols]
app_test[num_cols]=np.sign(app_test[num_cols])*app_test[num_cols]


# In[12]:


# Outliers
def outlier(data, cols):
    out=data[cols].quantile([0.25, 0.75])
    data[cols]=data[cols].clip(2.5*out.loc[0.25]-1.5*out.loc[0.75], 2.5*out.loc[0.75]-1.5*out.loc[0.25], axis=1)
    return data


# In[13]:


#processing outliers
outlier_col=num_n_val[num_n_val['n_values']>50].index
app_train=outlier(app_train, outlier_col)
app_test=outlier(app_test, outlier_col)


# In[14]:


def hist_plot(data, cols, m,n, n_bins):
    j=0
    plt.figure(figsize=(20,20))
    for i in cols:
        j=j+1
        plt.subplot(m,n,j)
        sns.distplot(data[i], bins=n_bins)


# In[15]:


def box_plot(data, cols, m,n):
    j=0
    plt.figure(figsize=(20,20))
    for i in cols:
        j=j+1
        plt.subplot(m,n,j)
        sns.boxplot(y=data[i])


# In[16]:


#Logarit transform for the amt_cols
amt_cols=['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_INCOME_TOTAL','AMT_GOODS_PRICE']
app_train[amt_cols]=np.log(app_train[amt_cols])
app_test[amt_cols]=np.log(app_test[amt_cols])


# In[17]:


#transform the day to year for the day_cols
day_cols=['DAYS_REGISTRATION',  'DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE','DAYS_EMPLOYED', 'DAYS_BIRTH']
for col in day_cols:
    app_train[col]=app_train[col]//365
    app_test[col]=app_test[col]//365


# In[18]:


"""
#discretzation a variable continue
def discretization(data, cols, n_bins):
    for col in cols:
        val_map=data[col].value_counts(bins=n_bins, normalize=True)
        data[col]=data[col].map(val_map)
    return data
    """


# In[19]:


def discretization(train, test, n_bins, cols):
    total=pd.concat([app_train,app_test], ignore_index=True)
    for col in cols:
        total_cut, Bins=pd.cut(total[col], bins=n_bins, retbins=True)
        train_cut=pd.cut(app_train[col], Bins)
        test_cut=pd.cut(app_train[col], Bins)
        val=total_cut.value_counts(normalize=True)
        train[col+'_cut']=train_cut.map(val)
        test[col+'_cut']=test_cut.map(val)
    return train, test


# In[20]:


discret_cols=amt_cols+['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
app_train, app_test=discretization(app_train, app_test, 20, discret_cols)
app_train, app_test=discretization(app_train, app_test, 10, day_cols)


# In[21]:


cut_cols=discret_cols+day_cols
cut_cols=[s+'_cut' for s in cut_cols]
cut_cols


# In[22]:


"""
app_train=discretization(app_train, ['DAYS_BIRTH'], range(10,80,10))
app_test=discretization(app_test, ['DAYS_BIRTH'], range(10,80,10))
app_train=discretization(app_train, ['DAYS_EMPLOYED'], range(0, 40, 5))
app_test=discretization(app_test, ['DAYS_EMPLOYED'], range(0, 40, 5))
"""


# In[23]:


from sklearn.preprocessing import MinMaxScaler
#scale max min
def scale(train, test, cols):
    scl=MinMaxScaler()
    total=pd.concat([train[cols], test[cols]], ignore_index=True)
    scl.fit(total)
    train_arr=scl.transform(train[cols])
    test_arr=scl.transform(test[cols])
    train[cols]=pd.DataFrame(data=train_arr, columns=cols)
    test[cols]=pd.DataFrame(data=test_arr, columns=cols)
    return train, test


# In[24]:


#scale_cols=list(set(num_cols).difference({'DAYS_EMPLOYED', 'DAYS_BIRTH'}))
app_train, app_test=scale(app_train, app_test, num_cols)


# In[25]:


app_train[num_cols].head()


# In[26]:


app_test[num_cols].head()


# In[27]:


"""
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(np.log(app_train['AMT_ANNUITY']), bins=20)
plt.subplot(1,2,2)
sns.distplot(np.log(app_train['AMT_CREDIT']), bins=20)
##########
j=0
plt.figure(figsize=(15,5))
for i in ['DAYS_REGISTRATION',  'DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']:
    j+=1
    plt.subplot(1,3,j)
    sns.distplot(app_train[i], bins=10)
##############
j=0
plt.figure(figsize=(15,5))
for i in ['DAYS_REGISTRATION',  'DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']:
    j+=1
    plt.subplot(1,3,j)
    sns.distplot(app_train[i]//365, bins=20)
############
plt.figure(figsize=(20,5))
j=0
for i in amt_cols:
    j+=1
    plt.subplot(1,4,j)
    sns.distplot(np.log(app_train[i]), bins=10)
###############
plt.figure(figsize=(20,5))
j=0
for i in amt_cols:
    j+=1
    plt.subplot(1,4,j)
    sns.distplot((app_train[i]), bins=10)
##########
plt.figure(figsize=(20,10))
j=0
for i in day_cols:
    j+=1
    plt.subplot(2,3,j)
    sns.distplot((app_train[i]), bins=10)
###########
hist_plot(app_train, num_n_val.index[40:60], 5,4)
##########
hist_plot(app_train, num_n_val.index[60:70], 5,4)
##########
hist_plot(app_train, num_n_val.index[:20], 5,4)
#########
box_plot(app_train, num_n_val.index[:20], 5,4)
#########
box_plot(app_train, num_n_val.index[20:40], 5,4)
#########
box_plot(app_train, num_n_val.index[60:80], 5,4)


# In[28]:


object_cols=app_train.select_dtypes(include='object').columns
print(object_cols)
print(len(object_cols))


# In[29]:


obj_val=pd.DataFrame({'n_values':app_train[object_cols].nunique().sort_values(ascending=False)})
obj_val


# In[30]:


"""#ORGANIZATION_TYPE
per_type=pd.pivot_table(app_train[['ORGANIZATION_TYPE', 'TARGET']], index='TARGET', columns='ORGANIZATION_TYPE', aggfunc=len)
#print(per_type)
ratio_label=(per_type.loc[0]/per_type.loc[1]).sort_values(ascending=True)
print(ratio_label)
"""


# In[31]:


"""for col in object_cols:
    print('The {} has {} values'.format(col, app_train[col].nunique()))
    print(app_train[col].value_counts(), '\n')
    """


# In[32]:


"""
plt.figure(figsize=(20,7))
ax=sns.countplot(app_train['ORGANIZATION_TYPE'])
plt.xticks(rotation=45)
for p in ax.patches:
    total=app_train.shape[0]
    height=p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height+3, '{:1.2f}'.format(height/total), ha='center')    
#############
fig=plt.figure(figsize=(20,15))
#fig, axes=plt.subplots(4,4)
#fig.tight_layout()
for i in range(9):
    plt.subplot(3,3,i+1)
    ax=sns.countplot(app_train[object_cols[i]])
    plt.xticks(rotation=45, horizontalalignment='right')
    total=len(app_train)
    for p in ax.patches:
        height=p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height+3, '{:1.2f}'.format(height/total), ha='center')
##################
sns.set(style="darkgrid")
fig=plt.figure(figsize=(20,15))
#fig, axes=plt.subplots(4,4)
#fig.tight_layout()
for i in range(7):
    plt.subplot(3,3,1+i)
    ax=sns.countplot(app_train[object_cols[i+9]])
    plt.xticks(rotation=45, horizontalalignment='right')
    total=len(app_train)
    for p in ax.patches:
        height=p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height+3, '{:1.2f}'.format(height/total), ha='center')
"""


# In[33]:


app_train[object_cols].replace(['XNA', 'unknown'], np.nan, inplace=True)
app_test[object_cols].replace(['XNA', 'unknown'], np.nan, inplace=True)


# In[34]:


"""
#check the difference values
for col in object_cols:
    dif=set(app_test[col].unique()).difference(set(app_train[col].unique()))
    if len(dif)>0:
        print(col, dif)
"""


# In[35]:


#frequence encoding the object values
def obj_enc(train, test, cols):
    total=pd.concat([train, test], ignore_index=True)
    for col in cols:
        val_map=total[col].value_counts(normalize=True)#.round(2)
        train[col]=train[col].map(val_map)
        test[col]=test[col].map(val_map)
    return train, test


# In[36]:


app_train, app_test=obj_enc(app_train, app_test, object_cols)
print(app_train['NAME_CONTRACT_TYPE'].value_counts())
print('-----------')
print(app_test['NAME_CONTRACT_TYPE'].value_counts())


# In[37]:


y_train=app_train['TARGET']
X_train=app_train.drop(['SK_ID_CURR','TARGET'], axis=1)
X_test=app_test.drop(['SK_ID_CURR'], axis=1)


# In[38]:


X_train.columns


# In[39]:


#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
def impute_iterative(data):
    impute=IterativeImputer(n_nearest_features=20, max_iter=100)
    imp_arr=impute.fit_transform(data)
    imp_data=pd.DataFrame(data=imp_arr, columns=data.columns)
    return imp_data


# In[40]:


def impute_knn(data):
    impute=KNNImputer()
    imp_arr=impute.fit_transform(data)
    imp_data=pd.DataFrame(data=imp_arr, columns=data.columns)
    return imp_data


# In[41]:


from sklearn.impute import SimpleImputer
def impute_simple(data):
    impute=SimpleImputer()
    impute.fit(data)
    arr_data=impute.transform(data)
    imp_data=pd.DataFrame(data=arr_data, columns=data.columns)
    return imp_data


# In[42]:


"""
# simple_impute 
X_train=impute_simple(X_train)
X_test=impute_simple(X_test)
"""


# In[43]:


"""
#check the difference values
for col in object_cols:
    dif=set(X_test[col].unique()).difference(set(X_train[col].unique()))
    if len(dif)>0:
        print(col, dif)
        """


# In[44]:


#calulate the woe
def woe_feature(train, test, cols, label):
    for col in cols:
        woe=train.groupby(col)[label].agg(['sum', 'count'])
        woe['%good']=(woe['sum']+0.5)/woe['sum'].sum()
        woe['bad']=woe['count']-woe['sum']
        woe['%bad']=(woe['bad']+0.5)/(woe['bad'].sum())
        woe['woe']=np.log(woe['%good']/woe['%bad'])
        #woe['iv']=(woe['%good']-woe['%bad'])*woe['woe']
        train[col+'_woe']=train[col].map(woe['woe'])
        test[col+'_woe']=test[col].map(woe['woe'])
        #data[col+'_iv']=data[col].map(woe['iv'])
    return train, test


# In[45]:


#target encoding
def smooth(trn_set, val_set, label, cols):
    for i in cols:
        mean=trn_set[label].mean()
        agg=trn_set.groupby([i])[label].agg(['mean','count'])
        means=agg['mean']
        count=agg['count']
        smooth=(count*means+300*mean)/(count+300)
        val_set[i+'_target']=val_set[i].map(smooth)
    return val_set
############################
def reg(trn_set, label, cols):
    for i in cols:
        sum_tr=trn_set.groupby([i])[label].transform('sum')
        count_tr=trn_set.groupby([i])[label].transform('count')
        trn_set[i+'_target']=(sum_tr-trn_set[label])/count_tr
    return trn_set    


# In[46]:


for i in object_cols:
    print(i, X_train[i].value_counts())


# In[47]:


import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve


# In[48]:


print(X_train.shape, X_test.shape, y_train.shape)


# In[49]:


#GridSearchCV
'''
params={'n_estimators':[400, 500, 600],
       'max_depth':[10,15, 20]}
rdf=RandomForestClassifier(random_state=100,
                      n_jobs=-1)
rdf_grid=GridSearchCV(rdf, params, cv=3, 
                      scoring='roc_auc',
                     verbose=1,
                     n_jobs=-1)
rdf_grid.fit(X_train, y_train)
############
rdf_grid.best_params_
rdf_grid.score(X_train, y_train)
rdf_grid.cv_results_
print(rdf_grid.cv_results_['mean_test_score'])
'''


# In[50]:


"""
#KFold
woe_cols=list(object_cols)+cut_cols
kf=KFold(n_splits=5, shuffle=True, random_state=30)
plt.figure(figsize=(6,6))
for idx_tr, idx_te in kf.split(X_train):
    train_set, test_set=X_train.loc[idx_tr], X_train.loc[idx_te]
    target_train, target_test=y_train[idx_tr], y_train[idx_te]
    #Target encoding
    train_set['label']=target_train
    train_set, test_set=woe_feature(train_set, test_set, woe_cols, 'label')
    ###################
    train_set=train_set.drop(['label'], axis=1)
    train_set=train_set.drop(cut_cols, axis=1)
    test_set=test_set.drop(cut_cols, axis=1)
    #Missing data
    train_set=impute_simple(train_set)
    test_set=impute_simple(test_set)
    print(train_set.shape, test_set.shape)
    #########################
    rdf=RandomForestClassifier(n_estimators=700,
                               max_depth=10,
                              random_state=100,
                              verbose=1,
                              n_jobs=-1)
    rdf.fit(train_set, target_train)
    print(rdf.score(train_set, target_train))
    ################
    y_predict_trn=rdf.predict_proba(train_set)[:,1]
    fal_pos, tru_pos, thres=roc_curve(target_train, y_predict_trn)
    #######################
    y_predict_te=rdf.predict_proba(test_set)[:,1]
    fal_pos_te, tru_pos_te, thres=roc_curve(target_test, y_predict_te)
    print('Train set: ', auc(fal_pos, tru_pos), ': Test set: ', auc(fal_pos_te, tru_pos_te))
    """


# In[51]:


"""
imp_feature=pd.DataFrame({'imp_values':rdf.feature_importances_}, index=train_set.columns).sort_values(by='imp_values', ascending=False)
imp_feature[:20].plot.barh(figsize=(8,8))
"""


# In[52]:


#woe columns
woe_cols=list(object_cols)+cut_cols
X_train['label']=y_train
X_train, X_test=woe_feature(X_train, X_test, woe_cols, 'label')
X_train=X_train.drop(['label'], axis=1)
X_train=X_train.drop(cut_cols, axis=1)
X_test=X_test.drop(cut_cols, axis=1)
#Missing data
X_train=impute_simple(X_train)
X_test=impute_simple(X_test)


# In[53]:


#model
rdf_woe=RandomForestClassifier(n_estimators=700,
                               max_depth=10,
                              random_state=100,
                              verbose=1,
                              n_jobs=-1
                               )
rdf_woe.fit(X_train, y_train)
y_predict_trn=rdf_woe.predict_proba(X_train)[:,1]
fal_pos, tru_pos, thres=roc_curve(y_train, y_predict_trn)
print(auc(fal_pos, tru_pos))
############
##########
y_test=rdf_woe.predict_proba(X_test)[:,1]
submission=pd.DataFrame({'SK_ID_CURR':app_test['SK_ID_CURR'], 'TARGET':y_test})
submission.to_csv('rdf_submission.csv', index=False)


# In[54]:


imp_feat_woe=pd.DataFrame({'imp_values':rdf_woe.feature_importances_},
                          index=X_train.columns).sort_values(by='imp_values', ascending=False)
imp_feat_woe[:30]


# In[55]:


"""
train_lgb = lgb.Dataset(X_train_set,label=y_train)
params = {
    'learning_rate': 0.01,
    'num_leaves' : 500,
    'max_depth' : 40,
    'min_data_in_leaf': 10
    'objective' : 'binary',
    'metric' : 'auc',
    'scale_pos_weight':0.1
}
lgb.cv(params, train_lgb, nfold=5)
"""

