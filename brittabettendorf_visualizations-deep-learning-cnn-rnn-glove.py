#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import seaborn as sns


# In[2]:


test_labels = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')
print(test_labels.shape)
test_labels.head(2)


# In[3]:


test_labels = test_labels[(test_labels[['toxic','severe_toxic', 'obscene', 'threat', 
                                        'insult', 'identity_hate']] != -1).all(axis=1)]
print(test_labels.shape)
test_labels.head(2)


# In[4]:


df_test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
print(df_test.shape)
df_test.head(2)


# In[5]:


# merge with an inner join
test = pd.merge(test_labels, df_test, on='id', how='inner')
print(test.shape)
test.head(2)


# In[6]:


train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
train.head(2)


# In[7]:


# check the number of records
print('The dataset contains', train.shape[0], 'records and', train.shape[1], 'columns.')


# In[8]:


# check that there are no missing values in either training set
print('The dataset has', train.isna().sum().sum(), 'missing values.')


# In[9]:


# check if there are any duplicates
print('The dataset has', train.duplicated().sum(), 'duplicates.')


# In[10]:


train['comment_text'][4]


# In[11]:


train['comment_text'][13]


# In[12]:


train['comment_text'][1392]


# In[13]:


# creating a list of column names
columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# In[14]:


# to_frame() converts series to DataFrame
frequency = train[columns].sum().to_frame().rename(columns={0: 'count'}).sort_values('count')
frequency.plot.barh(y='count', title='Count of Comments', figsize=(8, 5));


# In[15]:


train.groupby(columns).size().sort_values(ascending=False).reset_index()                      .rename(columns={0: 'count'}).head(15)


# In[16]:


fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Correlation Matrix')
sns.heatmap(train[columns].corr(), annot=True, cmap="YlGnBu", linewidths=.5, ax=ax);


# In[17]:


from matplotlib_venn import venn2
from matplotlib_venn import venn3
from matplotlib_venn import venn3_circles
from matplotlib_venn import venn2_circles


# In[18]:


# build combinations
a = train[(train['toxic']==1) & (train['insult']==0) & (train['obscene']==0)].shape[0]
b = train[(train['toxic']==0) & (train['insult']==1) & (train['obscene']==0)].shape[0]
c = train[(train['toxic']==0) & (train['insult']==0) & (train['obscene']==1)].shape[0]

ab = train[(train['toxic']==1) & (train['insult']==1) & (train['obscene']==0)].shape[0]
ac = train[(train['toxic']==1) & (train['insult']==0) & (train['obscene']==1)].shape[0]
bc = train[(train['toxic']==0) & (train['insult']==1) & (train['obscene']==1)].shape[0]

abc = train[(train['toxic']==1) & (train['insult']==1) & (train['obscene']==1)].shape[0]

# plot venn diagrams
plt.figure(figsize=(8, 8))
plt.title("Venn Diagram for 'toxic', 'insult' and 'obscene' comments")

v=venn3(subsets=(a, b, c, ab, ac, bc, abc), 
        set_labels=('toxic', 'insult', 'obscene'))

vc=venn3_circles(subsets=(a, b, c, ab, ac, bc, abc),
                linestyle='dashed', linewidth=1, color="grey")
vc[1].set_lw(8.0)
vc[1].set_ls('dotted')
vc[1].set_color('skyblue')

plt.show();


# In[19]:


# build combinations
a   = train[(train['toxic']==1) & (train['severe_toxic']==0)].shape[0]
b   = train[(train['toxic']==0) & (train['severe_toxic']==1)].shape[0]
ab = train[(train['toxic']==1) & (train['severe_toxic']==1)].shape[0]

# plot venn diagrams
plt.figure(figsize=(8, 8))
plt.title("Venn Diagram for 'toxic' and 'severe_toxic' comments")
v=venn2(subsets=(a, b, ab), set_labels=('toxic', 'severe_toxic'))

c=venn2_circles(subsets=(a, b, ab),
                linestyle='dashed', linewidth=1, color="grey")
c[1].set_lw(8.0)
c[1].set_ls('dotted')
c[1].set_color('skyblue')

plt.show();


# In[20]:


# import necessary libraries
from wordcloud import WordCloud
from collections import Counter

import re
import string

from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[21]:


# define an empty dictionary
word_counter = {}

# writing a clean_text function
def clean_text(text):
    text = re.sub('[{}]'.format(string.punctuation), ' ', text.lower())
    return ' '.join([word for word in text.split() if word not in (stop)])


# In[22]:


# iterating through all columns in the dataset...
for col in columns:    
    text = Counter()        
    
    # ... applying the clean-function to each column's comments and ...
    train[train[col] == 1]['comment_text'].apply(lambda t: text.update(clean_text(t).split()))
    
    # ... combining all to one dataframe
    word_counter[col] = pd.DataFrame.from_dict(text, orient='index')                                        .rename(columns={0: 'count'})                                        .sort_values('count', ascending=False)


# In[23]:


# iterating through new df word_counter and creating a WordCloud for each column
for col in word_counter:    
    wc_list = word_counter[col]
    
    wordcloud = WordCloud(background_color='white', max_words=150, max_font_size=100, random_state=4)                          .generate_from_frequencies(wc_list.to_dict()['count'])

    fig = plt.figure(figsize=(10, 8))
    plt.title('\n' + col + '\n', fontsize=20, fontweight='bold')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()


# In[24]:


# importing libraries
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence


# In[25]:


X_train = train["comment_text"].values
X_test  = test["comment_text"].values

y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
y_test  = test[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values


# In[26]:


# tokenizing the data
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(X_train))

# turning the tokenized text into sequences
X_train = tokenizer.texts_to_sequences(X_train)
X_test  = tokenizer.texts_to_sequences(X_test)

# padding the sequences
X_train = sequence.pad_sequences(X_train, maxlen=200)
X_test  = sequence.pad_sequences(X_test,  maxlen=200)

print('X_train shape:', X_train.shape)
print('X_test shape: ', X_test.shape)


# In[27]:


from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, SpatialDropout1D, Activation
from keras.layers import Conv1D, Bidirectional, GlobalMaxPool1D, MaxPooling1D, BatchNormalization
from keras.optimizers import Adam


# In[28]:


# number of unique words we want to use (or: number of rows in incoming embedding vector)
max_features = 20000 

# max number of words in a comment to use (or: number of columns in incoming embedding vector)
max_len = 200 

# dimension of the embedding variable (or: number of rows in output of embedding vector)
embedding_dims = 128


# In[29]:


# instantiate NN model
base_model = Sequential()

# add embedding layer 
base_model.add(Embedding(input_dim=max_features, input_length=max_len,
                         output_dim=embedding_dims))

# add pooling layer 
# ... which will extract features from the embeddings of all words in the comment
base_model.add(GlobalMaxPool1D())

# add dense layer to produce an output dimension of 50 and apply relu activation
base_model.add(Dense(50, activation='relu'))

# set the regularizing dropout layer to drop out 30% of the nodes
base_model.add(Dropout(0.3))

# finally add a dense layer
# ... which projects output into six units and squash it with sigmoid activation
base_model.add(Dense(6, activation='sigmoid'))


# In[30]:


base_model.compile(loss='binary_crossentropy',
                   optimizer=Adam(0.01), metrics=['accuracy'])

# check the model with all our layers
base_model.summary()


# In[31]:


base_hist = base_model.fit(X_train, y_train, batch_size=32, 
                           epochs=3, validation_split=0.1)


# In[32]:


# evaluate the algorithm on the test dataset
base_test_loss, base_test_auc = base_model.evaluate(X_test, y_test, batch_size=32)
print('Test Loss:    ', base_test_loss)
print('Test Accuracy:', base_test_auc)


# In[33]:


# instantiate CNN model
cnn_model = Sequential()

# add embedding layer 
cnn_model.add(Embedding(input_dim=max_features, input_length=max_len,
                        output_dim=embedding_dims))
 
# set the dropout layer to drop out 50% of the nodes
cnn_model.add(SpatialDropout1D(0.5))

# add convolutional layer that has ...
# ... 100 filters with a kernel size of 4 so that each convolution will consider a window of 4 word embeddings
cnn_model.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))

# add normalization layer
cnn_model.add(BatchNormalization())

# add pooling layer 
cnn_model.add(GlobalMaxPool1D())

# set the dropout layer to drop out 50% of the nodes
cnn_model.add(Dropout(0.5))

# add dense layer to produce an output dimension of 50 and using relu activation
cnn_model.add(Dense(50, activation='relu'))

# finally add a dense layer
cnn_model.add(Dense(6, activation='sigmoid'))


# In[34]:


cnn_model.compile(loss='binary_crossentropy',
                  optimizer=Adam(0.01),
                  metrics=['accuracy'])

cnn_model.summary()


# In[35]:


cnn_hist = cnn_model.fit(X_train, y_train, batch_size=32, 
                         epochs=3, validation_split=0.1)


# In[36]:


cnn_test_loss, cnn_test_auc = cnn_model.evaluate(X_test, y_test, batch_size=32)
print('Test Loss:    ', cnn_test_loss)
print('Test Accuracy:', cnn_test_auc)


# In[37]:


# instantiate RNN model
rnn_model = Sequential()

# add embedding layer 
rnn_model.add(Embedding(input_dim=max_features, input_length=max_len,
                        output_dim=embedding_dims))

# set the dropout layer to drop out 50% of the nodes
rnn_model.add(SpatialDropout1D(0.5))

# add bidirectional layer and pass in an LSTM()
rnn_model.add(Bidirectional(LSTM(25, return_sequences=True)))

# add normalization layer
rnn_model.add(BatchNormalization())

# add pooling layer 
rnn_model.add(GlobalMaxPool1D())

# set the dropout layer to drop out 50% of the nodes
rnn_model.add(Dropout(0.5))

# add dense layer to produce an output dimension of 50 and using relu activation
rnn_model.add(Dense(50, activation='relu'))

# finally add a dense layer
rnn_model.add(Dense(6, activation='sigmoid'))


# In[38]:


rnn_model.compile(loss='binary_crossentropy',
                  optimizer=Adam(0.01),
                  metrics=['accuracy'])

rnn_model.summary()


# In[39]:


rnn_hist = rnn_model.fit(X_train, y_train, batch_size=32, 
                          epochs=3, validation_split=0.1)


# In[40]:


rnn_test_loss, rnn_test_auc = rnn_model.evaluate(X_test, y_test, batch_size=32)
print('Test Loss:    ', rnn_test_loss)
print('Test Accuracy:', rnn_test_auc)


# In[41]:


# load the glove840B embedding

embeddings_index = dict()
f = open('../input/glove840b300dtxt/glove.840B.300d.txt')

for line in f:
    # Note: use split(' ') instead of split() if you get an error
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))


# In[42]:


# create a weight matrix
embedding_matrix = np.zeros((len(tokenizer.word_index)+1, 300))

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[43]:


# instantiate pretrained glove model
glove_model = Sequential()

# add embedding layer 
glove_model.add(Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len,
                          output_dim=embedding_matrix.shape[1], 
                          weights=[embedding_matrix], trainable=False))
 
# set the dropout layer to drop out 50% of the nodes
glove_model.add(SpatialDropout1D(0.5))

# add convolutional layer that has ...
# ... 100 filters with a kernel size of 4 so that each convolution will consider a window of 4 word embeddings
glove_model.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))

# add normalization layer
glove_model.add(BatchNormalization())

# add pooling layer 
glove_model.add(GlobalMaxPool1D())

# set the dropout layer to drop out 50% of the nodes
glove_model.add(Dropout(0.5))

# add dense layer to produce an output dimension of 50 and using relu activation
glove_model.add(Dense(50, activation='relu'))

# finally add a dense layer
glove_model.add(Dense(6, activation='sigmoid'))


# In[44]:


glove_model.compile(loss='binary_crossentropy',
                    optimizer=Adam(0.01),
                    metrics=['accuracy'])

glove_model.summary()


# In[45]:


glove_hist = glove_model.fit(X_train, y_train, batch_size=32, 
                             epochs=3, validation_split=0.1)


# In[46]:


glove_test_loss, glove_test_auc = glove_model.evaluate(X_test, y_test, batch_size=32)
print('Test Loss:    ', glove_test_loss)
print('Test Accuracy:', glove_test_auc)


# In[47]:


# instantiate pretrained glove model
glove_2_model = Sequential()

# add embedding layer 
glove_2_model.add(Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len,
                          output_dim=embedding_matrix.shape[1], 
                          weights=[embedding_matrix], trainable=False))

# set the dropout layer to drop out 50% of the nodes
glove_2_model.add(SpatialDropout1D(0.5))

# add bidirectional layer and pass in an LSTM()
glove_2_model.add(Bidirectional(LSTM(25, return_sequences=True)))

# add normalization layer
glove_2_model.add(BatchNormalization())

# add pooling layer 
glove_2_model.add(GlobalMaxPool1D())

# set the dropout layer to drop out 50% of the nodes
glove_2_model.add(Dropout(0.5))

# add dense layer to produce an output dimension of 50 and using relu activation
glove_2_model.add(Dense(50, activation='relu'))

# finally add a dense layer
glove_2_model.add(Dense(6, activation='sigmoid'))


# In[48]:


glove_2_model.compile(loss='binary_crossentropy',
                    optimizer=Adam(0.01),
                    metrics=['accuracy'])

glove_2_model.summary()


# In[49]:


glove_2_hist = glove_2_model.fit(X_train, y_train, batch_size=32, 
                                 epochs=3, validation_split=0.1)


# In[50]:


glove_2_test_loss, glove_2_test_auc = glove_2_model.evaluate(X_test, y_test, batch_size=32)
print('Test Loss:    ', glove_2_test_loss)
print('Test Accuracy:', glove_2_test_auc)


# In[51]:


# concat all training, validation and testing accuracy scores
accuracy_nn = ['Plain NN', 
               np.mean(base_hist.history['acc']), 
               np.mean(base_hist.history['val_acc']), 
               base_test_auc]

accuracy_cnn = ['CNN', 
                np.mean(cnn_hist.history['acc']), 
                np.mean(cnn_hist.history['val_acc']), 
                cnn_test_auc]

accuracy_rnn = ['RNN', 
                np.mean(rnn_hist.history['acc']), 
                np.mean(rnn_hist.history['val_acc']), 
                rnn_test_auc]

accuracy_glove = ['Glove CNN', 
                  np.mean(glove_hist.history['acc']), 
                  np.mean(glove_hist.history['val_acc']), 
                  glove_test_auc]

accuracy_glove_2 = ['Glove RNN', 
                    np.mean(glove_2_hist.history['acc']), 
                    np.mean(glove_2_hist.history['val_acc']), 
                    glove_2_test_auc]

# create dataframe
comparison = pd.DataFrame([accuracy_nn])
# append all other scores
comparison = comparison.append([accuracy_cnn, accuracy_rnn, accuracy_glove, accuracy_glove_2])


# In[52]:


# beautify the new dataframe
comparison.columns = ['Algorithm', 'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy']
comparison.set_index(['Algorithm'], inplace=True)
comparison

