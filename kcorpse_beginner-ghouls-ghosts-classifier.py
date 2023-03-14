#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


# No missing data, that's a relief. 
data = pd.read_csv("../input/train.csv")
data.info()


# In[3]:


# Let's clean up some data
data = pd.get_dummies(data, columns=["color"])


# In[4]:


class_encoder = LabelEncoder()
data["type"] = class_encoder.fit_transform(data["type"])


# In[5]:


sns.pairplot(data[["bone_length", "type"]])


# In[6]:


class_labels = data["type"]
del data["type"]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(data, class_labels)


# In[8]:


lr = LogisticRegression(C=0.1, penalty='l1')
forest = RandomForestClassifier(criterion="entropy", n_estimators=10, random_state=1, n_jobs=2)


# In[9]:


lr.fit(X_train, y_train)
forest.fit(X_train, y_train)


# In[10]:


print("Logistic Regression: \nTraining accuracy: {}\nTesting Accuracy: {}".format(lr.score(X_train, y_train), lr.score(X_test, y_test)))
print("*" * 10)
print("Random Forests: \nTraining Accuracy: {}\nTesting Accuracy: {}".format(forest.score(X_train, y_train), forest.score(X_test, y_test)))


# In[11]:


# Not the best results, lots of overfitting in Random Forests. 
submission = pd.read_csv("../input/test.csv")
submission = pd.get_dummies(submission, columns=["color"])


# In[12]:


labels = data.columns[1:]
importances = forest.feature_importances_
indices = 


# In[13]:


predictions = [int(forest.predict(row.reshape(1, -1))) for row in submission.values]


# In[14]:



predictions = class_encoder.inverse_transform(predictions)


# In[15]:


final_submission = pd.read_csv("../input/test.csv")
final_submission["type"] = predictions


# In[16]:


final_submission.head()


# In[17]:


x =final_submission[["id", "type"]]


# In[18]:


x.to_csv("predictions.csv")

