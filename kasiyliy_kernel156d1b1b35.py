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
dataset.sample(5)


# In[3]:


dataset.info()


# In[4]:


dataset.isna().sum()


# In[5]:


X = dataset.iloc[:, 2:].values
Y = dataset.iloc[:, 1].values


# In[6]:


X


# In[7]:


Y


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.99, random_state = 1,stratify =Y)


# In[10]:


from sklearn.neighbors import KNeighborsClassifier


# In[11]:


knn = KNeighborsClassifier(21)


# In[12]:


X_test[1:2]


# In[13]:


knn.fit(X_train, y_train)


# In[14]:


y_preds = knn.predict(X_test)


# In[15]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


# In[16]:


accuracy_score(y_test, y_preds)


# In[17]:


target_names = ['good rate', 'Not bad rate']


# In[18]:


print(classification_report(y_test, y_preds, target_names=target_names))


# In[19]:


test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[20]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename.csv')


# In[ ]:





# In[21]:


cm = confusion_matrix(y_test, y_preds)


# In[22]:


cm


# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[24]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[25]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[26]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[27]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[28]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[29]:


test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[30]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename1.csv')


# In[31]:


from sklearn.svm import SVC


# In[32]:


svclassifier = SVC()


# In[33]:


plt.scatter(X_train[:, 0], X_train[:, 4], c=y_train, cmap = 'spring')


# In[34]:


svclassifier.fit(X_train, y_train)


# In[35]:


y_pred = svclassifier.predict(X_test)


# In[36]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[37]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[38]:


print(classification_report(y_test,y_pred))


# In[39]:


svclassifier1 = SVC(kernel='sigmoid')


# In[40]:


svclassifier1.fit(X_train, y_train)


# In[41]:


y_pred = svclassifier1.predict(X_test)


# In[42]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[43]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[44]:


print(classification_report(y_test, y_pred))


# In[45]:


test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[46]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename2.csv')


# In[47]:


from sklearn.naive_bayes import GaussianNB


# In[48]:


gnb = GaussianNB()


# In[49]:


gnb.fit(X_train, y_train)


# In[50]:


y_pred = gnb.predict(X_test)


# In[51]:


from sklearn import metrics


# In[52]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[53]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[54]:


print("Accuracy:",metrics.classification_report(y_test, y_pred))


# In[55]:


test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[56]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename3.csv')


# In[57]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split


# In[58]:


bnb = BernoulliNB(binarize=0.0)


# In[59]:


bnb.fit(X_train, y_train)


# In[60]:


bnb.score(X_test, y_test)


# In[61]:


y_pred = bnb.predict(X_test)


# In[62]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[63]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[64]:


print("Accuracy:",metrics.classification_report(y_test, y_pred))


# In[65]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename4.csv')


# In[66]:


from sklearn.naive_bayes import MultinomialNB


# In[67]:


from sklearn.model_selection import train_test_split


# In[68]:


mnb = MultinomialNB(alpha=0.01)


# In[69]:


from sklearn.preprocessing import Normalizer


# In[70]:


normalizer = Normalizer(norm='l2', copy=True)


# In[71]:


X_train = Normalizer(copy=False).fit_transform(X_train)


# In[72]:


X_train


# In[73]:


from sklearn.tree import DecisionTreeClassifier


# In[74]:


from sklearn import metrics 


# In[75]:


from sklearn.tree import export_graphviz 


# In[76]:


clf = DecisionTreeClassifier()


# In[77]:


clf = clf.fit(X_train,y_train)


# In[78]:


y_pred = clf.predict(X_test)


# In[79]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[80]:


pip install pydotplus


# In[81]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  


# In[82]:


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)


# In[83]:


clf = clf.fit(X_train,y_train)


# In[84]:


y_pred = clf.predict(X_test)


# In[85]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[86]:


from sklearn.ensemble import RandomForestRegressor


# In[87]:


regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[88]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[89]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[90]:


from sklearn.ensemble import RandomForestClassifier


# In[91]:


clf=RandomForestClassifier(n_estimators=100)


# In[92]:


clf.fit(X_train,y_train)


# In[93]:


y_pred=clf.predict(X_test)


# In[94]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[95]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename4.csv')


# In[96]:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[97]:


import xgboost as xgb
import pandas as pd


# In[98]:


xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed=123 )


# In[99]:


xg_cl.fit(X_train, y_train)


# In[100]:


y_pred = xg_cl.predict(X_test)


# In[101]:


import numpy as np
accuracy = float(np.sum(y_pred==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


# In[102]:


dataset_dmatrix = xgb.DMatrix(data = X,label = Y)
dataset_dmatrix


# In[103]:


params = {"objective":"reg:logistic", "max_depth":3}
params


# In[104]:


# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "rmse", as_pandas = True, seed = 123)


# In[105]:


print(cv_results)


# In[106]:


print(1-cv_results["test-rmse-mean"].tail(1))


# In[107]:


# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)


# In[108]:


print(cv_results)


# In[109]:


print(cv_results["test-auc-mean"].tail(1))


# In[110]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename5.csv')

