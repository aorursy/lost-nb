#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import pandas as pd 
import six

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


def extract_words(train):
    words = list()
    for index, row in train.iterrows():
        q1 = row['question1']
        q2 = row['question2']
        if not q1 or not q2 or not isinstance(q1, six.string_types)                 or not isinstance(q2, six.string_types):
            continue
        q_words = q1.split()
        for word in q_words:
            words.append(word)
        q_words = q2.split()
        for word in q_words:
            words.append(word)
    return words
vocabulary_size = 50000
words = extract_words(df)
print('Number of words: %d' % len(words))


# In[ ]:




