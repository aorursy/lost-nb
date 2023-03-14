#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')




# Read in Train Data
train = pd.read_csv("../input/train.csv")




# Read in Test Data
test = pd.read_csv("../input/test.csv")




# Number of rows and columns of training and test data
train.shape, test.shape




train.head()




train.info()




train.select_dtypes(include="object").columns




# Checking if ID_code is unique
train.ID_code.nunique() == train.shape[0]




sns.countplot(train.target)




train.target.value_counts() *100 / train.target.count()




train.groupby("target").mean()




train.groupby("target").median()




np.mean(train.groupby("target").mean().iloc[1] >= train.groupby("target").mean().iloc[0])




np.mean(train.groupby("target").median().iloc[1] >= train.groupby("target").mean().iloc[0])




features = train.columns.values[2:203]




from scipy.stats import normaltest




# # D’Agostino’s K^2 Test on TRAIN DATA
# non_normal_features = []
# for feature in features:
#     stat, p = normaltest(train[feature])
#     if p <= 0.01:
#         print(feature,"not normal")
#         non_normal_features.append(feature)




# # D’Agostino’s K^2 Test on TEST DATA
# non_normal_features_test_data = []
# for feature in test.columns.values[1:202]:
#     stat, p = normaltest(test[feature])
#     if p <= 0.05:
#         print(feature,"not normal")
#         non_normal_features_test_data.append(feature)




train.isnull().sum().sum()




correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.tail(10)




from sklearn.preprocessing import StandardScaler
standardized_train = StandardScaler().fit_transform(train.set_index(['ID_code','target']))




standardized_train = pd.DataFrame(standardized_train, columns=train.set_index(['ID_code','target']).columns)
standardized_train = standardized_train.join(train[['ID_code','target']])




from sklearn.decomposition import PCA
k=80
pca = PCA(n_components=k, random_state=42, whiten=True)
pca.fit(standardized_train.set_index(['ID_code','target']))




plt.figure(figsize=(25,5))
plt.plot(pca.explained_variance_ratio_)
plt.xticks(range(k))
plt.xlabel("Number of Features")
plt.ylabel("Proportion of variance explained by additional feature")




sum(pca.explained_variance_ratio_)




sum(PCA(n_components=120, random_state=42, whiten=True).fit(standardized_train.set_index(['ID_code','target'])).explained_variance_ratio_)




sum(PCA(n_components=160, random_state=42, whiten=True).fit(standardized_train.set_index(['ID_code','target'])).explained_variance_ratio_)




pca = PCA(n_components=160).fit_transform(standardized_train.set_index(['ID_code','target']))




pca_col_names = []
for i in range(160):
    pca_col_names.append("pca_var_" + str(i))
pca_col_names




# Save PCA transformed train dataset just in case
pca_train = pd.DataFrame(pca, columns=pca_col_names).join(train[['ID_code','target']])
pca_train.to_csv("pca_train.csv")




# Standardize the test data as well
standardized_test = StandardScaler().fit_transform(test.set_index(['ID_code']))
standardized_test = pd.DataFrame(standardized_test, columns=test.set_index(['ID_code']).columns)
standardized_test = standardized_test.join(test[['ID_code']])




pca = PCA(n_components=160).fit_transform(standardized_test.set_index(['ID_code']))




pca_col_name_for_test = []
for i in range(160):
    pca_col_name_for_test.append("pca_var_" + str(i))




# Save PCA transformed test dataset just in case
pca_test = pd.DataFrame(pca, columns=pca_col_name_for_test).join(train[['ID_code']])
pca_test.to_csv("pca_test.csv")




# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score




# X = standardized_train.drop('target',axis=1).set_index('ID_code')
# y = standardized_train[['target']]




# # Split training dataset to train and validation set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




# Split Train Dataset into Predictor variables Matrix and Target variable Matrix
X_train = standardized_train.set_index(['ID_code','target']).values.astype('float64')
y_train = standardized_train['target'].values




# Logistic Regression
from sklearn.linear_model import LogisticRegression
logit_clf = LogisticRegression(random_state=42).fit(X_train,y_train)




plt.figure(figsize=(10, 10))
fpr, tpr, thr = roc_curve(y_train, logit_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)




cross_val_score(logit_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()




from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)




plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, lda_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)




cross_val_score(lda_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()




qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)




plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, qda_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)




cross_val_score(qda_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()




X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = logit_clf.predict_proba(X_test)[:,1]
submission.to_csv('LR.csv', index=False)




X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = lda_clf.predict_proba(X_test)[:,1]
submission.to_csv('lda.csv', index=False)




X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = qda_clf.predict_proba(X_test)[:,1]
submission.to_csv('lda.csv', index=False)




X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')

logit_pred = logit_clf.predict_proba(X_test)[:,1]
lda_pred = lda_clf.predict_proba(X_test)[:,1]
qda_pred = qda_clf.predict_proba(X_test)[:,1]




submission = submission.join(pd.DataFrame(qda_pred, columns=['target1'])).join(pd.DataFrame(logit_pred, columns=['target2'])).join(pd.DataFrame(lda_pred, columns=['target3']))




submission['target'] = (submission.target1 + submission.target2 + submission.target3) / 3




submission.head()




del submission['target1']
del submission['target2']
del submission['target3']




submission.head()




submission.to_csv('logit_lda_qda_mean_ensemble.csv', index=False)

