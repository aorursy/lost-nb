#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read in Train Data
train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")


# In[3]:


# Read in Test Data
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")


# In[4]:


# Number of rows and columns of training and test data
train.shape, test.shape


# In[5]:


# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score


# In[6]:


# Standardize the training dataset
from sklearn.preprocessing import StandardScaler
standardized_train = StandardScaler().fit_transform(train.set_index(['ID_code','target']))


# In[7]:


standardized_train = pd.DataFrame(standardized_train, columns=train.set_index(['ID_code','target']).columns)
standardized_train = standardized_train.join(train[['ID_code','target']])


# In[8]:


# Standardize the test data as well
standardized_test = StandardScaler().fit_transform(test.set_index(['ID_code']))
standardized_test = pd.DataFrame(standardized_test, columns=test.set_index(['ID_code']).columns)
standardized_test = standardized_test.join(test[['ID_code']])


# In[9]:


# Split Train Dataset into Predictor variables Matrix and Target variable Matrix
X_train = standardized_train.set_index(['ID_code','target']).values.astype('float64')
y_train = standardized_train['target'].values


# In[10]:


from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)


# In[11]:


plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, nb_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


# In[12]:


cross_val_score(nb_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# In[13]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42).fit(X_train,y_train)


# In[14]:


plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, rf_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


# In[15]:


cross_val_score(rf_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# In[16]:


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')

nb_pred = nb_clf.predict_proba(X_test)[:,1]
rf_pred = rf_clf.predict_proba(X_test)[:,1]


# In[17]:


submission = submission.join(pd.DataFrame(nb_pred, columns=['target1'])).join(pd.DataFrame(rf_pred, columns=['target2']))


# In[18]:


submission['target'] = (submission.target1 + submission.target2) / 2


# In[19]:


del submission['target1']
del submission['target2']


# In[20]:


submission.head()


# In[21]:


submission.to_csv('nb_rf_mean_ensemble.csv', index=False)


# In[22]:


submission = submission.rename(columns={'target':'nb_rf_mean_target'})

logit_lda_qda_mean_ensemble = pd.read_csv('../input/logit-lda-qda-mean-ensemblecsv/logit_lda_qda_mean_ensemble.csv').drop('ID_code', axis=1)

submission = submission.join(logit_lda_qda_mean_ensemble)


# In[23]:


submission.head()


# In[24]:


submission['final_target'] = (submission.nb_rf_mean_target * 2 + submission.target * 3) / 2


# In[25]:


submission = submission.rename(columns={'final_target':'target'})


# In[26]:


submission.head()


# In[27]:


submission = submission.iloc[:,[0,3]]


# In[28]:


submission.to_csv('logit_lda_qda_nb_rf_mean_ensemble.csv', index=False)

