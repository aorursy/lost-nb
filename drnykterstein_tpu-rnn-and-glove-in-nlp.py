#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[4]:


try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[5]:


train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')


# In[6]:


train.head()


# In[7]:


train.drop(['severe_toxic','obscene','threat','insult','identity_hate','id'],axis=1,inplace=True)


# In[8]:


#after droping 
train.head()


# In[9]:


#checking the maximum number for doing the padding 
train['comment_text'].apply(lambda x:len(str(x).split())).max()


# In[10]:


# Now, let's see the average number of words per sample
plt.figure(figsize=(10, 6))
plt.hist([len(sample) for sample in list(train['comment_text'])], 80)
plt.xlabel('Length of samples')
plt.ylabel('Number of samples')
plt.title('Sample length distribution')
plt.show()


# In[11]:


#splitting the data into training set and validation set 
xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, 
                                                  stratify=train.toxic.values, 
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)


# In[12]:


texts = xtrain
words = [word for text in texts for word in text.split()]
v = sorted(list(set(words)))


# In[13]:


from collections import Counter
word_counts = Counter(words)
print(word_counts.most_common(5))
v_s = sorted(word_counts.items(), key=lambda x: x[1],  reverse=True)
str2idx = {key:val for key,val in v_s}
idx2str = {val:key for key,val in v_s}
str2idx


# In[51]:


token = text.Tokenizer(num_words=None)
max_len = 1000

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

#zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len,padding = 'post')
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len,padding = 'post')

word_index = token.word_index


# In[52]:


len(word_index)


# In[53]:


get_ipython().run_cell_magic('time', '', "with strategy.scope():\n    # A simpleRNN without any pretrained embeddings and one dense layer\n    model = Sequential()\n    model.add(Embedding(len(word_index)+1,\n                     300,\n                     input_length=max_len))\n    model.add(SimpleRNN(100))\n    model.add(Dense(1, activation='sigmoid'))\n    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n    \nmodel.summary()\n    ")


# In[54]:


history = model.fit(xtrain_pad, ytrain,  
                    epochs=5,
                    batch_size=64*strategy.num_replicas_in_sync,
                    validation_data=(xvalid_pad, yvalid))


# In[55]:



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[56]:


#using pretraind Embedding
embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()


# In[57]:


print('Found %s word vectors.' % len(embeddings_index))


# In[58]:


# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index)+1 , 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[60]:


get_ipython().run_cell_magic('time', '', "with strategy.scope():\n    \n    # A simple LSTM with glove embeddings and one dense layer\n    model = Sequential()\n    model.add(Embedding(len(word_index)+1,\n                     300,\n                     weights=[embedding_matrix],\n                     input_length=max_len,\n                     trainable=False))\n\n    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))\n    model.add(Dense(1, activation='sigmoid'))\n    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])\n    \nmodel.summary()")


# In[61]:


history = model.fit(xtrain_pad, ytrain,  
                    epochs=5,
                    batch_size=64*strategy.num_replicas_in_sync,
                    validation_data=(xvalid_pad, yvalid))


# In[63]:


get_ipython().run_cell_magic('time', '', "with strategy.scope():\n    # GRU with glove embeddings and two dense layers\n    model = Sequential()\n    model.add(Embedding(len(word_index) + 1,\n                     300,\n                     weights=[embedding_matrix],\n                     input_length=max_len,\n                     trainable=False))\n     model.add(SpatialDropout1D(0.3))\n     model.add(GRU(300))\n     model.add(Dense(1, activation='sigmoid'))\n\n     model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   \n    \nmodel.summary()")


# In[67]:


history = model.fit(xtrain_pad, ytrain,  
                    epochs=8,
                    batch_size=128*strategy.num_replicas_in_sync,
                    validation_data=(xvalid_pad, yvalid))


# In[ ]:




