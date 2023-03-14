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


import seaborn as sns

import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import xgboost as xgb

import warnings
warnings.simplefilter('ignore')

from scipy import stats
from scipy.stats import norm, skew #for some statistics

import os


# In[3]:


import pandas as pd
sample_submission = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv")
test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")
train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")


# In[4]:


train.info()


# In[5]:


train.describe()


# In[6]:


sns.countplot(x='target', data=train)


# In[7]:


print(train.shape, test.shape)


# In[8]:


print (train.isna().sum())
print (train.isnull().sum())


# In[9]:


X, y = train.iloc[:,2:], train.iloc[:,1]


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 123, stratify = y)


# In[11]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


# In[12]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# In[13]:


logreg=LogisticRegression()
logreg.fit(X_train,y_train)


# In[14]:


from sklearn.metrics import confusion_matrix
y_pred = logreg.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[15]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[16]:


test_inp = test.drop(columns = ['ID_code'])
y_pred = logreg.predict_proba(test_inp)[:,1]
submit = test[['ID_code']]
submit['target'] = y_pred
submit.head()


# In[17]:


submit.to_csv('log_reg_baseline.csv', index = False)


# In[18]:


from sklearn.svm import SVC
#svclassifier = SVC(kernel='linear')
#svclassifier.fit(X_train, y_train)


# In[19]:


#y_pred_svm = svclassifier.predict(X_test)


# In[20]:


#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, y_pred_svm))
#print(classification_report(y_test, y_pred_svm))


# In[21]:


#submission_svm = pd.DataFrame({
 #   "ID_code": test["ID_code"],
#    "target": y_pred_svm
##})
#submission_svm.to_csv('svm_submission.csv', index=False)


# In[22]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()


# In[23]:


gnb.fit(X_train, y_train)


# In[24]:


y_predit_svc = gnb.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

accuracy_score(y_test, y_predit_svc)


# In[26]:


confusion_matrix(y_test,y_predit_svc)


# In[27]:


print(classification_report(y_test, y_predit_svc))


# In[28]:


y_preds_res = gnb.predict(X)


# In[29]:


accuracy_score(y, y_preds_res)


# In[30]:


confusion_matrix(y, y_preds_res)


# In[31]:


print(classification_report(y, y_preds_res))


# In[32]:


submission_nb = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": y_preds_res
})
submission_nb.to_csv('naive_baise_submission.csv', index=False)


# In[33]:


#from sklearn.tree import DecisionTreeClassifier
#
#tree = DecisionTreeClassifier(random_state=0)
#tree.fit(X, y)


# In[34]:


#y_predit_tree = tree.predict(X_test)


# In[35]:


#print(classification_report(y_test, y_predit_tree))
#print(confusion_matrix(y_test,y_predit_tree))


# In[36]:


#from sklearn import metrics
#from sklearn.model_selection import StratifiedKFold
#from imblearn.ensemble import *
#clf = BalancedRandomForestClassifier(n_estimators=500, 
                                     criterion='entropy', 
                                     n_jobs=-1)


# In[37]:


#clf.fit(x_train, y_train)


# In[38]:


#y_pred = clf.predict(x_train)


# In[39]:


#print('Score:', clf.score(x_train, y_train))


# In[40]:



#print('AUC Score:', metrics.roc_auc_score(y_train, y_pred))


# In[41]:


#print(classification_report(y_train, y_pred))
#print(confusion_matrix(y_train, y_pred))


# In[42]:


#submission_res = pd.DataFrame({
#    "ID_code": test["ID_code"],
#    "target": y_preds_res
#})
#submission_res.to_csv('naive_baise_submission.csv', index=False)


# In[43]:


#model_xgb = xgb.XGBRegressor(n_estimators=5, max_depth=4, learning_rate=0.5) 
#model_xgb.fit(X_train, y_train)


# In[44]:


#xgb_preds = model_xgb.predict(X_test)
#submission_nb = pd.DataFrame({
#    "ID_code": test["ID_code"],
#    "target": xgb_preds
#})
#submission_xgb.to_csv('naive_baise_submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




