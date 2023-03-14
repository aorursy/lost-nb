#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


train_data = pd.read_json('../input/train.json')


# In[ ]:


train_data.head()


# In[ ]:


test_data = pd.read_json('../input/train.json')


# In[ ]:


test_data.head()


# In[ ]:


train_data['cuisine'].value_counts()


# In[ ]:


type(train_data)


# In[ ]:


type(test_data)


# In[ ]:


all_ingredients_list = []
all_ingredients_list = train_data['ingredients']
in_list = []
for each in all_ingredients_list:
    for i in each:
        in_list.append(i)


# In[ ]:


unique_ing = []
for each1 in in_list:
    if each1 not in unique_ing:
        unique_ing.append(each1)
print "Total number of ingredients",len(unique_ing)


# In[ ]:


from collections import Counter
Counter(in_list).most_common(50)


# In[ ]:


total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])
missing_train_data.head(20)


# In[ ]:


import seaborn as sns
plt.figure(figsize=(12,8))
sns.countplot(x="cuisine", data=train_data)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Cuisine', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Column Name", fontsize=15)
plt.show()


# In[ ]:


for each in train_data['ingredients']:
    print each
    break
    


# In[ ]:


# Label Encoding of y - the target kinds of cuisine
from sklearn.preprocessing import LabelEncoder


y_value = train_data['cuisine'].copy()
labelencoder = LabelEncoder()
y_value = labelencoder.fit_transform(y_value)
print(y_value)
print("Total length", len(y_value))


# In[ ]:


labelencoder.inverse_transform(y_value)


# In[ ]:


train_data_without_id = train_data.copy()
train_data_without_id.drop(['id','cuisine'],axis=1,inplace=True)
test_data_without_id = test_data.copy()
test_data_without_id.drop(['id'],axis=1,inplace=True)


# In[ ]:


for each in train_data['ingredients']:
    s = str(''.join(each).lower().strip())
    print(s)
    break


# In[ ]:


train_data_without_id['ingredients'] = [''.join(each).lower().strip() for each in train_data['ingredients']]


# In[ ]:


train_data_without_id['ingredients'].value_counts()


# In[ ]:


test_data_without_id['ingredients'] = [''.join(each).lower().strip() for each in test_data['ingredients']]


# In[ ]:


test_data_without_id['ingredients'].value_counts()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
# TFIDF statiscic applying to the data - resulting in sparse matrix
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_data_without_id['ingredients'])


# In[ ]:


X_train


# In[ ]:


X_test = tfidf.transform(test_data_without_id['ingredients'])


# In[ ]:


X_test


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier


# In[ ]:


dt = DecisionTreeClassifier()
model_dt = OneVsRestClassifier(dt, n_jobs=-1)


# In[ ]:


model_dt.fit(X_train, y_value)
acc_dt = round(model_dt.score(X_train, y_value) * 100, 2)
print(acc_dt)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
model_rf = OneVsRestClassifier(rf, n_jobs=-1)
model_rf.fit(X_train, y_value)
acc_rf = round(model_rf.score(X_train, y_value) * 100, 2)
print(acc_rf)


# In[ ]:


from sklearn.svm import SVC
svc = SVC()
model_svc = OneVsRestClassifier(svc, n_jobs=1)
model_svc.fit(X_train, y_value)
acc_svc = round(model_svc.score(X_train, y_value) * 100, 2)
print(acc_svc)


# In[ ]:


X_test = test_data['ingredients']


# In[ ]:


y_test1 = model_rf.predict(X_test)
y_pred = labelencoder.inverse_transform(y_test1)
print(y_pred)


# In[ ]:



# Submission
sub = pd.DataFrame({'id': test_data['id'], 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('randomForest_output.csv', index=False)


# In[ ]:




