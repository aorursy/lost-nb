#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import tensorflow as tf
import math
import timeit
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df=pd.read_csv('../input/train.csv')
data = df.values[:, 2:]
N = data.shape[0]
F = 3
D = 64
C = 99
features = data.reshape(N, D, F)
ids = df.values[:, 0]
names = df.values[:, 1]
name_mapping = {}
name_unique = np.unique(names)
for i in range(C):
    name_mapping[name_unique[i]] = i

labels = np.zeros(N, np.int64)
for i in range(N):
    labels[i] = name_mapping[names[i]]

df=pd.read_csv('../input/test.csv')
testing_ids = df.values[:, 0]
testing_data = df.values[:, 1:]
testing_features = testing_data.reshape(testing_data.shape[0], D, F)
    
print(features.shape, ids.shape, names.shape)
#images = np.zeros(N, )
#for i in range(N):
#    im = Image.open("../input/images/"+str(i + 1)+".jpg")
#    image = np.array(im)
#    print(image.shape)
#    #mage_list.append(image)
#images = np.asarray(image_list)
#print(images.shape)
# Any results you write to the current directory are saved as output.


# In[2]:





# In[2]:


def random_batch(batch_num = 200):
    index = np.random.randint(0, N, 200)
    return (features[index], ids[index], labels[index])


# In[3]:


x = tf.placeholder(tf.float32, shape = (None, D, F))
y = tf.placeholder(tf.int64, shape = (None))

w_conv1 = tf.Variable(tf.random_normal((7, 3, 32)))
b_conv1 = tf.Variable(tf.zeros(32))
conv = tf.nn.conv1d(x, w_conv1, stride = 2, padding = 'VALID') + b_conv1
h1 = tf.nn.relu(conv)

h1_flat = tf.reshape(h1,[-1,928])

w1 = tf.Variable(tf.random_normal((928, 64)))
b1 = tf.Variable(tf.zeros(64))

a2 = tf.matmul(h1_flat, w1) + b1
h2 = tf.nn.relu(a2)

w2 = tf.Variable(tf.random_normal((64, C)))
b2 = tf.Variable(tf.zeros(C))

out = tf.matmul(h2, w2) + b2

y_pred = tf.argmax(out, 1)

mean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y))

optimizer = tf.train.RMSPropOptimizer(learning_rate = 1e-2)

step = optimizer.minimize(mean_loss)

prob = out / tf.reduce_sum(out, axis = 1, keep_dims = True)
prob = tf.maximum(prob, 1e-15)
prob = tf.minimum(prob, 1 - 1e-15)

correct_prediction = tf.equal(y, y_pred)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[4]:


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for k in range(500):
    batch_xs, batch_ids, batch_labels = random_batch()
    loss, _ = sess.run([mean_loss, step], feed_dict={x: batch_xs, y: batch_labels})
    if k % 50 == 0:
        print("loss: ", loss)
    
val_xs, val_ids, val_labels = random_batch()
accuracy = sess.run([accuracy], feed_dict={x: val_xs, y: val_labels})
print("accuracy: ", accuracy)


# In[5]:


prob = sess.run([prob], feed_dict={x: testing_features})
prob_array = np.asarray(prob[0])


# In[6]:


#for i in range(C):
    #print(name_unique[i])


# In[7]:


for i in range(prob_array.shape[0]):
    #print(testing_ids[i], "id")
    for k in range(C):
        #print(prob_array[i][k])
    #print("|")

