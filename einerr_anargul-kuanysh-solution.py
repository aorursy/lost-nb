#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os
print(os.listdir("../input"))


# In[2]:


import numpy as np
import pandas as pd
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

max_bin = 20
force_bin = 3

def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)


# In[3]:


X = pd.read_csv('../input/train.csv')
y =X['target']
Z = pd.read_csv('../input/test.csv')


# In[4]:


X.head()


# In[5]:


X['education'].value_counts()


# In[6]:


X['martial'].value_counts()


# In[7]:


X['repay_sep'].value_counts()


# In[8]:


X.sort_values(by=['credit_bal'], ascending=True).head(20)


# In[9]:


X.info()


# In[10]:


X.columns


# In[11]:


Z.columns


# In[12]:


X.drop('id_code', axis=1, inplace=True)
Z.drop('id_code', axis=1, inplace=True)


# In[13]:


X['repay_sum'] = X[["repay_sep", "repay_aug", "repay_jul", "repay_jun","repay_may", "repay_april"]].sum(axis = 1)/5
X["pay_sum"] = X[["pay_sep", "pay_aug", "pay_jul", "pay_jun","pay_may", "pay_apr"]].sum(axis = 1)/5


# In[14]:


Z['repay_sum'] = Z[["repay_sep", "repay_aug", "repay_jul", "repay_jun","repay_may", "repay_april"]].sum(axis = 1)/5
Z["pay_sum"] = Z[["pay_sep", "pay_aug", "pay_jul", "pay_jun","pay_may", "pay_apr"]].sum(axis = 1)/5


# In[15]:


def credit_bal_q(x):
    if x==10000.000000:
        return 'Credit bal Q1'
    elif x>10000.000000 and x<=50000.000000:
        return 'Credit bal Q2'
    elif x>50000.000000 and x<=140000.000000:
        return 'Credit bal Q3'
    elif x>140000.000000 and x<=240000.000000:
        return 'Credit bal Q4'
    else:
        return 'Credit bal Q5'    
X['credit_bal_q'] = X['credit_bal'].apply(lambda x: credit_bal_q(x))


# In[16]:


def credit_bal_q(x):
    if x==10000.000000:
        return 'Credit bal Q1'
    elif x>10000.000000 and x<=50000.000000:
        return 'Credit bal Q2'
    elif x>50000.000000 and x<=140000.000000:
        return 'Credit bal Q3'
    elif x>140000.000000 and x<=240000.000000:
        return 'Credit bal Q4'
    else:
        return 'Credit bal Q5'    
Z['credit_bal_q'] = Z['credit_bal'].apply(lambda x: credit_bal_q(x))


# In[17]:


dfiv,iv=data_vars(X, y)


# In[18]:


dfiv


# In[19]:


iv.sort_values(by='IV', ascending=False)


# In[20]:


profile_report = ProfileReport(X)
profile_report


# In[21]:


profile_report.get_rejected_variables()


# In[22]:


X=X.drop(['bill_apr', 'bill_aug', 'bill_jul', 'bill_jun', 'bill_may', 'gender', 'target'], axis=1) 
Z=Z.drop(['bill_apr', 'bill_aug', 'bill_jul', 'bill_jun', 'bill_may', 'gender'], axis=1)


# In[23]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=101)


# In[24]:


X_dummy = pd.get_dummies(X)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.2, random_state=101)


# In[26]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[27]:


rf=RandomForestClassifier(random_state=101, n_estimators = 20)
rf.fit(X_train, y_train)
y_pred_proba_rf=rf.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_proba_rf)


# In[28]:


#params_grid={
    'n_estimators':[150, 500],
    'max_features':['auto','sqrt'],
    'max_depth':[15, 25],
    'min_samples_split':[5, 10],
    'min_samples_leaf':[ 2, 4],
    'bootstrap':[True, False]
}
#rf=RandomForestClassifier(random_state=101, n_estimators = 100, n_jobs=-1)
#gsrt=GridSearchCV(rf, params_grid, scoring='roc_auc', cv=skf, n_jobs=-1, verbose=1)
#gsrt.fit(X_train, y_train)
#gsrt.best_params_


# In[29]:


gsrt={'bootstrap': True,
 'max_depth': 15,
 'max_features': 'auto',
 'min_samples_leaf': 4,
 'min_samples_split': 10,
 'n_estimators': 500}


# In[30]:


rff=RandomForestClassifier(random_state=101, **gsrt)
rff.fit(X_train, y_train)
y_pred_proba_rff=rff.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_proba_rff)


# In[31]:


feature_importances = pd.DataFrame(rff.feature_importances_,
                                   index = X_dummy.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances.head(20)


# In[32]:


feature_importances.tail(20)


# In[33]:


X_dummy_test = pd.get_dummies(Z)


# In[34]:


y_pred_test=rff.predict_proba(X_dummy_test)


# In[35]:


submission = pd.read_csv('../input/sample_submission.csv')


# In[36]:


submission['target'] = y_pred_test


# In[37]:


submission['target'].value_counts()


# In[38]:


submission.to_csv('submission10.csv', index=False)

