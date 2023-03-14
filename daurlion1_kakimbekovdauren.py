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


dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
testset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[3]:


dataset.sample(10)


# In[4]:


dataset.info()


# In[5]:


dataset.isna().sum()


# In[6]:


dataset.nunique()


# In[7]:


dataset.dropna(inplace=True)


# In[8]:


X = dataset.iloc[:, 2:].values
y = np.floor(dataset.iloc[:, 1].values)


# In[9]:


X[1]


# In[10]:


y[1]


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 80,stratify =y)


# In[13]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


# In[14]:


target_names = ['0 chance', '1 chance']


# In[15]:


print(classification_report(y_test, y_preds, target_names=target_names))


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[17]:


logreg = LogisticRegression(max_iter=100000)
logreg.fit(X_train, y_train)


# In[18]:


# test = testset.iloc[:, 1:].values
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
result = pd.DataFrame({"target" : np.array(y_pred).T})
result['target'].value_counts()


# In[19]:


print(classification_report(y_test, y_pred, target_names=target_names))


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


# In[21]:


logreg = LogisticRegression(max_iter=100000)
logreg.fit(X_train, y_train)


# In[22]:


test = testset.iloc[:, 1:].values
y_pred = logreg.predict(test)


# In[23]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('logregFi.csv', index=False)


# In[ ]:





# In[24]:


from sklearn.naive_bayes import GaussianNB


# In[25]:


gnb = GaussianNB()


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 80,stratify =y)


# In[27]:


gnb.fit(X_train, y_train)


# In[28]:


y_pred = gnb.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[29]:


print(classification_report(y_test, y_pred, target_names=target_names))


# In[30]:


y_pred = gnb.predict(X_test)
print('Accuracy of gnb classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)))
result = pd.DataFrame({"target" : np.array(y_pred).T})
result['target'].value_counts()


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


# In[32]:


test = testset.iloc[:, 1:].values

y_pred = gnb.predict(test)


# In[33]:



sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('gaussianF.csv', index=False)


# In[34]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 80,stratify =y)


# In[36]:


bnb = BernoulliNB(binarize=0.0)


# In[37]:


bnb.fit(X_train, y_train)


# In[38]:


bnb.score(X_test, y_test)


# In[39]:


y_pred = bnb.predict(X_test)


# In[40]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[41]:



print('Accuracy of bnb classifier on test set: {:.2f}'.format(bnb.score(X_test, y_test)))
result = pd.DataFrame({"target" : np.array(y_pred).T})
result['target'].value_counts()


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


# In[43]:


bnb.fit(X_train, y_train)


# In[44]:


y_pred = bnb.predict(test)


# In[45]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
# train_ds['target'].value_counts()
result['target'].value_counts()


# In[46]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('bernulliF.csv', index=False)


# In[47]:


from sklearn.tree import DecisionTreeClassifier


# In[48]:


from sklearn import metrics 


# In[49]:


from sklearn.tree import export_graphviz 


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 80,stratify =y)


# In[51]:


clf = DecisionTreeClassifier()


# In[52]:


clf = clf.fit(X_train,y_train)


# In[53]:


y_pred = clf.predict(X_test)


# In[54]:


print('Accuracy of clf classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
result = pd.DataFrame({"target" : np.array(y_pred).T})
result['target'].value_counts()


# In[55]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


# In[57]:


clf = clf.fit(X_train,y_train)


# In[58]:


y_pred = clf.predict(test)


# In[59]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
# train_ds['target'].value_counts()
result['target'].value_counts()


# In[60]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('decissionTreeFi.csv', index=False)


# In[61]:


import os  
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[62]:


import xgboost as xgb
import pandas as pd


# In[63]:


xg_cl = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=1,
              learning_rate=0.1, max_delta_step=0, max_depth=8,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 10,stratify =y)


# In[65]:


xg_cl.fit(X_train, y_train)


# In[66]:


y_pred = xg_cl.predict(X_test)


# In[67]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[68]:



print('Accuracy of XGBOOST classifier on test set: {:.2f}'.format(xg_cl.score(X_test, y_test)))
result = pd.DataFrame({"target" : np.array(y_pred).T})
result['target'].value_counts()


# In[69]:


print(classification_report(y_test, y_pred, target_names=target_names))


# In[70]:


dataset_dmatrix = xgb.DMatrix(data = X,label = y)
dataset_dmatrix


# In[71]:


params = {"objective":"reg:logistic", "max_depth":3}
params


# In[72]:


# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "rmse", as_pandas = True, seed = 123)


# In[73]:


print(cv_results)


# In[74]:


print(1-cv_results["test-rmse-mean"].tail(1))


# In[75]:


cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)


# In[76]:


print(cv_results["test-auc-mean"].tail(1))


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


# In[78]:


xg_cl.fit(X_train, y_train)


# In[79]:


y_pred = xg_cl.predict(test)


# In[80]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
# train_ds['target'].value_counts()
result['target'].value_counts()


# In[81]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('XGBOOSTv2.csv', index=False)

