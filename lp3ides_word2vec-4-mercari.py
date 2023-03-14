#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ideas and implementation are taken from tensorflow tutorial on word2vec at https://www.tensorflow.org/tutorials/word2vec


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, re
import pickle
import collections
import random
from time import time
import math
import tensorflow as tf


# In[ ]:


df = pd.read_csv("../input/train.tsv", sep = '\t')
df.head()


# In[ ]:


# perform some cleaning of the text fields: remove non-characters, make lower cases, splitting item category into main and sub categories
def clean(text):
    return re.sub(r'[^\w\s]','',text)
def lower(text):
    return text.lower()
# general categories
def split_cat(text): # credit to https://www.kaggle.com/thykhuely
    cats = text.split("/")
    if len(cats) >=3:
        return cats[0:3]
    else: return ("No Label", "No Label", "No Label") 

for column in ['name', 'brand_name', 'item_description']:
    df[column] = df[column].astype(str) 
    df[column] = df[column].apply(clean).apply(lower)
df['category_name'] = df['category_name'].astype(str).apply(lower)
df['general_cat'], df['subcat_1'], df['subcat_2'] = zip(*df['category_name'].apply(lambda x: split_cat(x)))
df.head()


# In[ ]:


# build a corpus from the text fields
# the skip-gram model will be applied to this corpus
corpus = []
for row in range(len(df)):
    for column in ['name', 'general_cat', 'subcat_1', 'subcat_2', 'brand_name', 'item_description']:
        corpus += (df.loc[row, column].split())
print(*corpus[:150], sep = ' ')


# In[ ]:


# get the most frequent 50,000 words used in the corpus
# map these words to integer indices
vocabulary_size = 50000
def build_dataset(corpus, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(corpus).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
    for word in corpus:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(corpus, vocabulary_size)


# In[ ]:


# gauge the coverage of the corpus by the vocabulary
sum = 0
for _, freq in count[1:]:
    sum += freq
print("using the most frequent %5d words captures %2.2f percent of tokens in item descriptions" 
      %(vocabulary_size,sum/len(corpus)*100))


# In[ ]:


print('Most common words (+UNK)', count[:10])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


# In[ ]:


# function to generate a training batch for the skip-gram model.
data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  if data_index == len(data):
      data_index = 0
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
        print("exceeded")
        print(data_index)
        for word in data[:span]:
            buffer.append(word)
        data_index = span
        print("new data_index is set to: ", data_index)
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


# In[ ]:


batch, labels = generate_batch(batch_size=128, num_skips=2, skip_window=2)
for i in range(10):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
print(data_index, len(data))


# In[ ]:


# build and train a skip-gram model.

batch_size = 128
embedding_size = 50  # Dimension of the embedding vector.
skip_window = 5       # How many words to consider left and right.
num_skips = 8         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 10    # Random set of words to evaluate similarity on.
valid_window = 500  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


# In[ ]:


graph = tf.Graph()
with graph.as_default():
    with tf.device('/gpu:0'):
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        # Compute the average NCE loss for the batch.
        loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        init = tf.global_variables_initializer()


# In[ ]:


num_steps = 10**6
check_N = 10**4
with tf.Session(graph=graph) as session:
    init.run()
    average_loss = 0
    time_0 = time()
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        if step % check_N == 0:
            if step > 0:
                average_loss /= check_N
            # The average loss is an estimate of the loss over the last check_N batches.
            print('Average loss at step ', step, ': ', average_loss, 'time: %2.2f' %(time()-time_0))
            average_loss = 0
            time_0 = time()

        if step % 10**4 == 0 and step > 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    print('done training')
    final_embeddings = normalized_embeddings.eval()


# In[ ]:


# visualize the embeddings, looking at the most frequent 500 words
from sklearn.manifold import TSNE
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])


# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

labels = [reverse_dictionary[i] for i in range(plot_only)]
x_plot = [low_dim_embs[i, :][0] for i, label in enumerate(labels)]
y_plot = [low_dim_embs[i, :][1] for i, label in enumerate(labels)]


# In[ ]:


trace1 = go.Scatter(
    x = x_plot,
    y = y_plot,
    mode='markers+text',
    name='Markers and Text',
    text=labels,
    textposition='top'
)
data = [trace1]
layout = go.Layout(
    showlegend=False,
    title = "visualizing word embeddings",
    xaxis = {"visible": False},
    yaxis = {"visible": False}
)
fig = go.Figure(data=data, layout=layout)
plot = py.iplot(fig)

