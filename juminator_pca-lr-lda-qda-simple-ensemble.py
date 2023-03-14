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
train = pd.read_csv("../input/train.csv")


# In[3]:


# Read in Test Data
test = pd.read_csv("../input/test.csv")


# In[4]:


# Number of rows and columns of training and test data
train.shape, test.shape


# In[5]:


train.head()


# In[6]:


train.info()


# In[7]:


train.select_dtypes(include="object").columns


# In[8]:


# Checking if ID_code is unique
train.ID_code.nunique() == train.shape[0]


# In[9]:


sns.countplot(train.target)


# In[10]:


train.target.value_counts() *100 / train.target.count()


# In[11]:


train.groupby("target").mean()


# In[12]:


train.groupby("target").median()


# In[13]:


np.mean(train.groupby("target").mean().iloc[1] >= train.groupby("target").mean().iloc[0])


# In[14]:


np.mean(train.groupby("target").median().iloc[1] >= train.groupby("target").mean().iloc[0])


# In[15]:


features = train.columns.values[2:203]


# In[16]:


from scipy.stats import normaltest


# In[17]:


# # D’Agostino’s K^2 Test on TRAIN DATA
# non_normal_features = []
# for feature in features:
#     stat, p = normaltest(train[feature])
#     if p <= 0.01:
#         print(feature,"not normal")
#         non_normal_features.append(feature)


# In[18]:


# # D’Agostino’s K^2 Test on TEST DATA
# non_normal_features_test_data = []
# for feature in test.columns.values[1:202]:
#     stat, p = normaltest(test[feature])
#     if p <= 0.05:
#         print(feature,"not normal")
#         non_normal_features_test_data.append(feature)


# In[19]:


train.isnull().sum().sum()


# In[20]:


correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.tail(10)


# In[21]:


from sklearn.preprocessing import StandardScaler
standardized_train = StandardScaler().fit_transform(train.set_index(['ID_code','target']))


# In[22]:


standardized_train = pd.DataFrame(standardized_train, columns=train.set_index(['ID_code','target']).columns)
standardized_train = standardized_train.join(train[['ID_code','target']])


# In[23]:


from sklearn.decomposition import PCA
k=80
pca = PCA(n_components=k, random_state=42, whiten=True)
pca.fit(standardized_train.set_index(['ID_code','target']))


# In[24]:


plt.figure(figsize=(25,5))
plt.plot(pca.explained_variance_ratio_)
plt.xticks(range(k))
plt.xlabel("Number of Features")
plt.ylabel("Proportion of variance explained by additional feature")


# In[25]:


sum(pca.explained_variance_ratio_)


# In[26]:


sum(PCA(n_components=120, random_state=42, whiten=True).fit(standardized_train.set_index(['ID_code','target'])).explained_variance_ratio_)


# In[27]:


sum(PCA(n_components=160, random_state=42, whiten=True).fit(standardized_train.set_index(['ID_code','target'])).explained_variance_ratio_)


# In[28]:


pca = PCA(n_components=160).fit_transform(standardized_train.set_index(['ID_code','target']))


# In[29]:


pca_col_names = []
for i in range(160):
    pca_col_names.append("pca_var_" + str(i))
pca_col_names


# In[30]:


# Save PCA transformed train dataset just in case
pca_train = pd.DataFrame(pca, columns=pca_col_names).join(train[['ID_code','target']])
pca_train.to_csv("pca_train.csv")


# In[31]:


# Standardize the test data as well
standardized_test = StandardScaler().fit_transform(test.set_index(['ID_code']))
standardized_test = pd.DataFrame(standardized_test, columns=test.set_index(['ID_code']).columns)
standardized_test = standardized_test.join(test[['ID_code']])


# In[32]:


pca = PCA(n_components=160).fit_transform(standardized_test.set_index(['ID_code']))


# In[33]:


pca_col_name_for_test = []
for i in range(160):
    pca_col_name_for_test.append("pca_var_" + str(i))


# In[34]:


# Save PCA transformed test dataset just in case
pca_test = pd.DataFrame(pca, columns=pca_col_name_for_test).join(train[['ID_code']])
pca_test.to_csv("pca_test.csv")


# In[35]:


# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score


# In[36]:


# X = standardized_train.drop('target',axis=1).set_index('ID_code')
# y = standardized_train[['target']]


# In[37]:


# # Split training dataset to train and validation set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[38]:


# Split Train Dataset into Predictor variables Matrix and Target variable Matrix
X_train = standardized_train.set_index(['ID_code','target']).values.astype('float64')
y_train = standardized_train['target'].values


# In[39]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
logit_clf = LogisticRegression(random_state=42).fit(X_train,y_train)


# In[40]:


plt.figure(figsize=(10, 10))
fpr, tpr, thr = roc_curve(y_train, logit_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


# In[41]:


cross_val_score(logit_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# In[42]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)


# In[43]:


plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, lda_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


# In[44]:


cross_val_score(lda_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# In[45]:


qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)


# In[46]:


plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, qda_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


# In[47]:


cross_val_score(qda_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# In[48]:


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = logit_clf.predict_proba(X_test)[:,1]
submission.to_csv('LR.csv', index=False)


# In[49]:


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = lda_clf.predict_proba(X_test)[:,1]
submission.to_csv('lda.csv', index=False)


# In[50]:


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = qda_clf.predict_proba(X_test)[:,1]
submission.to_csv('lda.csv', index=False)


# In[51]:


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')

logit_pred = logit_clf.predict_proba(X_test)[:,1]
lda_pred = lda_clf.predict_proba(X_test)[:,1]
qda_pred = qda_clf.predict_proba(X_test)[:,1]


# In[52]:


submission = submission.join(pd.DataFrame(qda_pred, columns=['target1'])).join(pd.DataFrame(logit_pred, columns=['target2'])).join(pd.DataFrame(lda_pred, columns=['target3']))


# In[53]:


submission['target'] = (submission.target1 + submission.target2 + submission.target3) / 3


# In[54]:


submission.head()


# In[55]:


del submission['target1']
del submission['target2']
del submission['target3']


# In[56]:


submission.head()


# In[57]:


submission.to_csv('logit_lda_qda_mean_ensemble.csv', index=False)

