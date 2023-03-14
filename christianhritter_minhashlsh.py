#!/usr/bin/env python
# coding: utf-8

# In[50]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


qa_pairs = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')


# In[4]:


qa_pairs.head()


# In[5]:


qa_pairs.sample(10, random_state=42)


# In[6]:


qa_pairs.info()


# In[14]:


qa_pairs[~qa_pairs['question1'].apply(lambda question: type(question) == str)]


# In[17]:


qa_pairs = qa_pairs.dropna(subset=['question1'])


# In[ ]:





# In[32]:


qa_pairs['is_duplicate'].value_counts()


# In[33]:


sns.countplot(qa_pairs['is_duplicate'])


# In[19]:


token_len_q1 = qa_pairs['question1'].apply(lambda question: len(set(question)))


# In[27]:


fig, ax = plt.subplots(figsize=(16,4))
sns.distplot(token_len_q1, ax=ax)
ax.set_xlabel('token lenght per question')


# In[28]:


sns.boxplot(token_len_q1)
plt.xlabel('token lenght per question')


# In[60]:


pip install datasketch[scipy]


# In[61]:


import datasketch


# In[29]:


qa_pairs.head()


# In[44]:


sents_pairs = pd.concat([qa_pairs[qa_pairs['is_duplicate'] == 0].sample(100, random_state=42), 
                   qa_pairs[qa_pairs['is_duplicate'] == 1].sample(100, random_state=42)]).reset_index(drop=True).sample(frac=1.)
sents_pairs.shape


# In[46]:


sents = pd.concat([sents_pairs['question1'], sents_pairs['question2']])
sents.head()


# In[54]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)


# In[55]:


# Set Representation

'''
set_dict maps question id (eg 'm23') to set representation of questionnorm_dict maps question id (eg 'm23') to actual question.
#May use this dictionary to #evaluate results of LSH output.We loop through each question, convert them into shingles, 
and if the shingle isn’t a stop word,
we add them to a hashset which will be the value for the set_dict dictionary.
'''


set_dict={} 

norm_dict={} 
count=1
for question in tqdm([x for x in sents]):
   temp_list = []
   for shingle in question.split(' '):
       if shingle not in stop_words:
           temp_list.append(shingle.lower())
   set_dict["m{0}".format(count)] = set(temp_list)
   norm_dict["m{0}".format(count)] = question
   count +=1


# In[57]:


set_dict['m1']


# In[62]:


# Create minHash signatures

'''
num_perm is the number of permutations we want for the MinHash algorithm (discussed before). 
The higher the permutations the longer the runtime.Min_dict maps question id (eg 'm23') to min hash signatures.
We loop through all the set representations of questions and calculate the signatures and store them in the min_dict dictionary.
'''


num_perm = 256
min_dict = {}
count2 = 1
for val in tqdm(set_dict.values()):
   m = datasketch.MinHash(num_perm=num_perm)
   for shingle in val:
       m.update(shingle.encode('utf8'))
   min_dict["m{}".format(count2)] = m
   count2+=1


# In[63]:


min_dict['m1']


# In[88]:


elem_test = next(iter(set_dict['m1']))
elem_test


# In[94]:


m1 = datasketch.MinHash(num_perm=num_perm)
m1.update(elem_test.encode('utf8'))
m2 = datasketch.MinHash(num_perm=num_perm)
m2.update(elem_test.encode('utf8'))


# In[95]:


m1 == m2


# In[96]:


m1.jaccard(m2)


# In[97]:


first_digest = m1.digest()
first_digest.shape


# In[99]:


first_digest


# In[100]:


iter_text = iter(set_dict['m1'])
next(iter_text)
elem_test2 = next(iter_text)
elem_test2


# In[101]:


m1.update(elem_test2.encode('utf8'))


# In[103]:


(m1.digest() == first_digest).all()


# In[69]:


list(set_dict['m1']


# In[77]:


# Create LSH index

'''
We set the Jaccard similarity threshold as a parameter in MinHashLSH. 
We loop through the signatures or keys in the min_dict dictionary and store them as bands (as described in the theory section of the article). 
Datasketch stores these in a dictionary format, where the key is a question and the values are all the questions deemed similar based on the threshold. 
But we need them in candidate pairs as they are much easier to evaluate, so we use a function called create_cand_pairs 
which simply changes the format of the dictionary to be a list of lists with each sub-list being a candidate pair.
'''

lsh = datasketch.MinHashLSH(threshold=0.4, num_perm=num_perm)
for key in tqdm(min_dict.keys()):
   lsh.insert(key,min_dict[key]) # insert minhash data structure


# In[104]:


big_list = []
for query in min_dict.keys():
   big_list.append(lsh.query(min_dict[query]))
 


# In[105]:


big_list[0]


# In[109]:


big_list[5]


# In[112]:


norm_dict[big_list[5][0]], norm_dict[big_list[5][1]]


# In[ ]:




