#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from keras.utils import np_utils
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv("../input/train.csv")
train.head()


# In[3]:


# Check for null values
train.isnull().sum().sum()


# In[4]:


train.drop("id", axis=1, inplace=True)
train.columns


# In[5]:


f = plt.figure()
sns.countplot(train['target'])
plt.title("Count plot",fontsize=20)
plt.show()


# In[6]:


y = train['target']
train.drop("target", axis=1, inplace=True)
X = train


# In[7]:


feature_model = ExtraTreesClassifier(n_estimators=100,verbose=1,n_jobs=-1)
feature_model.fit(X,y)


# In[8]:


#feature importance
print(feature_model.feature_importances_)


# In[9]:


feature_select = SelectFromModel(feature_model, prefit=True)
X_new = feature_select.transform(X)


# In[10]:


feature_select = SelectFromModel(feature_model, prefit=True)
X_new = feature_select.transform(X)


# In[11]:


feature_idx = feature_select.get_support()
feature_name = X.columns[feature_idx]


# In[12]:


feature_name


# In[13]:


f,ax = plt.subplots(figsize=(20,15))
sns.heatmap(train[feature_name].corr(), ax=ax,cmap="YlGnBu")
plt.title("Correlation Matrix",fontsize=20)
plt.show()


# In[14]:


plt.figure(figsize=(15,15))
plt.bar(range(len(feature_model.feature_importances_)), feature_model.feature_importances_)
plt.title("Feature Importance")
plt.xticks(np.arange(len(train.columns)),train.columns, rotation=90)
plt.show()


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,auc,roc_curve
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(X_new, y, random_state=42)


# In[17]:


model_KNN = KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=-1,p=2)
model_KNN.fit(x_train, y_train)


# In[18]:


y_pred = model_KNN.predict(x_test)


# In[19]:


print("Accuracy(KNN_Classifier)\t:"+str(accuracy_score(y_test,y_pred)))
print("Precision(KNN_Classifier)\t:"+str(precision_score(y_test,y_pred)))
print("Recall(KNN_Classifier)\t:"+str(recall_score(y_test,y_pred)))


# In[20]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[21]:


from matplotlib.colors import ListedColormap

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X_ = X_new[:, :2][:600]
y_ = y[:600]
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
h = .02  # step size in the mesh

clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=-1,p=2)
clf.fit(X_, y_)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_[:, 0].min() - 1, X_[:, 0].max() + 1
y_min, y_max = X_[:, 1].min() - 1, X_[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15,15))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X_[:, 0], X_[:, 1], c=y_, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification (k = %i)"
              % (5))
plt.show()


# In[22]:


prob=model_KNN.predict_proba(x_test)
prob = prob[:,1]


# In[23]:


fpr,tpr,_ = roc_curve(y_test, prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(14,12))
plt.title('Receiver Operating Characteristic',fontsize=20)
sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[24]:


test = pd.read_csv("../input/test.csv")
test.head()


# In[25]:


test.shape


# In[26]:


id_test = test['id']


# In[27]:


test.drop("id", axis=1, inplace=True)


# In[28]:


test = test[feature_name]


# In[29]:


y_pred_test = model_KNN.predict_proba(test)[:,1]


# In[30]:


my_submission = pd.DataFrame({'id': id_test, 'target': y_pred_test})
my_submission.to_csv('SubmissionVictor2.csv', index=False)

