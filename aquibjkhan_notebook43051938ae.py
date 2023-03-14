#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


# In[2]:


dataTrain = pd.read_csv("../input/train.csv")
print (dataTrain.head())
dataTest = pd.read_csv("../input/test.csv")
print ("Test data")
print (dataTest.head())


# In[3]:


print (len(dataTrain))
print (len(dataTest))


# In[4]:


print('DataFrame Train:')
print(dataTrain.isnull().any())
print (" ")
col = np.array(dataTrain.columns)
col = col[2:]
print (col)

print (dataTest["comment_text"].fillna('comment_missing').values)


# In[5]:


print('Dataframe Train:')
for c in col:
    print("The dataframe has '{1}' of comments '{0}' of the total '{2}'.".format(c,dataTrain[c].sum(),len(dataTrain)))


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', lowercase = True, strip_accents='unicode', ngram_range=(1,3), encoding = 'utf-8', decode_error = 'strict', max_features = 50000)


# In[7]:


train_text_comment, test_text_comment = dataTrain["comment_text"], dataTest["comment_text"]
all_comment = pd.concat([train_text_comment, test_text_comment])
print ("total length",len(all_comment))
print (all_comment.head())


# In[8]:


vectorizing_all_comment = vectorizer.fit_transform(all_comment)


# In[9]:


nrow_train = dataTrain.shape[0]
X = vectorizing_all_comment[:nrow_train]
y = dataTrain[col]
print (X.shape)


# In[10]:


#Slpitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)


# In[11]:


from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

classifier = Pipeline([
       ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
print (predicted)
# all_labels = mlb.inverse_transform(predicted)
accuScore = accuracy_score(y_test,predicted)
print (accuScore)


# In[12]:


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
TestVector = vectorizing_all_comment[nrow_train:]
TestVector.shape


# In[13]:


TestPredict = classifier.predict(TestVector)


# In[14]:


TestPredict.shape


# In[15]:


subm = pd.read_csv('../input/sample_submission.csv')
outid = pd.DataFrame({'id': subm["id"]})
out = pd.concat([outid, pd.DataFrame(TestPredict, columns = label_cols)], axis=1)
out.to_csv('mysubmission.csv', index=False)


# In[16]:




