#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk as nk
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")


# In[3]:


train.shape


# In[4]:


test.shape


# In[5]:


freq = train['interest_level'].value_counts()
sns.barplot(freq.index, freq.values, color=color[4])
plt.ylabel('Frequence')
plt.xlabel('Interest level')
plt.show()


# In[6]:


train.head(1)


# In[7]:


from nltk.tokenize import word_tokenize


# In[8]:


from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[10]:


vectorizer = TfidfVectorizer()
desc_text = vectorizer.fit_transform(train['description'])
print(vectorizer.get_feature_names())


# In[11]:


desc_text.shape


# In[12]:


type('features')


# In[13]:


train['features'][:10]


# In[14]:


import itertools as it


# In[15]:


features=list()
train['features'].apply(lambda x: features.append(x))
features=list(it.chain.from_iterable(features))
len(features)


# In[16]:


features[:50]


# In[17]:


uniq_feature_total=set(features)


# In[18]:


for feat in ft: 


# In[19]:





# In[19]:


list(uniq_feature_total)[:10]

