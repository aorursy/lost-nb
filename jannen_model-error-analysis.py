#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import pandas as pd
import numpy as np
import gc
import os
from sklearn.metrics import f1_score
import re
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm
tqdm.pandas(desc='Progress')

#pd.set_option('display.max_colwidth', -1)  


# In[ ]:


EMBEDDINGS_PATH = '../input/embeddings/'
EMBEDDING_FILE_GLOVE = f'{EMBEDDINGS_PATH}/glove.840B.300d/glove.840B.300d.txt'


# In[ ]:


embed_size = 300 # how big is each word vector
max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use
batch_size = 1024 # how many samples to process at once
n_epochs = 2 # how many times to iterate over all samples
SEED = 1006


# In[ ]:


# REPEATABILITY
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()
# kernel https://www.kaggle.com/hengzheng/pytorch-starter


# In[ ]:


os.environ['OMP_NUM_THREADS'] = '4'


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[ ]:


# Randomize
np.random.seed(SEED)
trn_idx = np.random.permutation(len(df_train))
df_train = df_train.iloc[trn_idx]

df = pd.concat([df_train ,df_test],sort=True)


# In[ ]:


df_train.head(3)


# In[ ]:


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

df_train["question_text"] = df_train["question_text"].progress_apply(lambda x: clean_text(x))
df_test["question_text"] = df_test["question_text"].apply(lambda x: clean_text(x))


# In[ ]:


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

df_train["question_text"] = df_train["question_text"].progress_apply(lambda x: clean_numbers(x))
df_test["question_text"] = df_test["question_text"].apply(lambda x: clean_numbers(x))


# In[ ]:


specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}

def clean_special_chars(text):
    
    for s in specials:
        text = text.replace(s, specials[s])    
    return text

df_train["question_text"] = df_train["question_text"].progress_apply(lambda x: clean_special_chars(x))
df_test["question_text"] = df_test["question_text"].apply(lambda x: clean_special_chars(x))


# In[ ]:


df_train["question_text"] = df_train["question_text"].apply(lambda x: x.lower())
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.lower())


# In[ ]:


list_sentences_train = df_train['question_text']
list_sentences_test = df_test['question_text']
list_sentences_combined = list_sentences_train.append(list_sentences_test, ignore_index=True)
tokenizer = Tokenizer(num_words=max_features, filters='\t\n')
tokenizer.fit_on_texts(list(list_sentences_combined))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)


# In[ ]:


train_x = pad_sequences(list_tokenized_train, maxlen=maxlen)
test_x = pad_sequences(list_tokenized_test, maxlen=maxlen)
train_y = df_train['target'].values


# In[ ]:


start = time.time()
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE_GLOVE))
end = time.time()
print(end-start)


# In[ ]:


# Get embedding mean and st deviation for giving random value near mean for words that were not in glove
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


# In[ ]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:


seed_everything()
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


def train_validate_test_split(df, df_y, train_percent=.6, validate_percent=.2, random_state=10):
    np.random.seed(random_state)
    perm = np.random.permutation(len(df))
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df[perm[:train_end]]
    validate = df[perm[train_end:validate_end]]
    test = df[perm[validate_end:]]
    
    train_y = df_y[perm[:train_end]]
    validate_y = df_y[perm[train_end:validate_end]]
    test_y = df_y[perm[validate_end:]]

    return train, validate, test, train_y, validate_y, test_y, perm


# In[ ]:


# Train / Val / Test -split
seed_everything()
X_tra, X_val, X_test, y_tra, y_val, y_test, permutation = train_validate_test_split(train_x, train_y, 
                                train_percent=0.95, validate_percent=0.04, random_state=SEED+2)


# In[ ]:


print(len(X_tra))
print(len(X_val))
print(len(X_test))


# In[ ]:


seed_everything()
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=n_epochs, validation_data=(X_val, y_val)); #verbose=2


# In[ ]:


train_preds = model.predict(X_tra, batch_size=1024)
print(len(train_preds))


# In[ ]:


# https://www.kaggle.com/ziliwang/baseline-pytorch-bilstm
def bestThresshold(train_y,train_preds):
    tmp = [0,0,0] # idx, cur, max
    delta = 0
    for tmp[0] in tqdm(np.arange(0.1, 0.501, 0.01)):
        tmp[1] = f1_score(train_y, np.array(train_preds)>tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
    return delta
delta = bestThresshold(y_tra,train_preds)


# In[ ]:


val_preds = model.predict(X_val, batch_size=1024)
print(len(val_preds))
print(f1_score(y_val, np.array(val_preds)>delta))


# In[ ]:


test_preds = model.predict(X_test, batch_size=1024)
print(len(test_preds))
print(f1_score(y_test, np.array(test_preds)>delta))


# In[ ]:


final_preds = model.predict(test_x, batch_size=1024)


# In[ ]:


submission = df_test[['qid']].copy()
submission['prediction'] = (final_preds > delta).astype(int)
submission.to_csv('submission.csv', index=False)


# In[ ]:


get_ipython().system('head submission.csv')


# In[ ]:


train_preds


# In[ ]:


# predictions
predicted = pd.DataFrame(train_preds)
predicted.columns = ['predicted']
predicted.to_csv('train_preds.csv', index=False)


# In[ ]:


# save the processed form of train-data
df_train_preproc = df_train
df_train_preproc.to_csv('df_train_preprocessed.csv')


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_train_preproc = pd.read_csv("df_train_preprocessed.csv")
train_preds = pd.read_csv('train_preds.csv')


# In[ ]:


y_train = y_tra; y_train[0:10]


# In[ ]:


train_preds[0:10].T


# In[ ]:


combined = pd.concat([df_train, train_preds], axis=1, sort=False)
combined.head()


# In[ ]:


# SIZE of error - |true class - predicted|
combined['error'] = abs(combined['target'] - combined['predicted']) 
combined.head()


# In[ ]:


# Display whole text of dataframe field and don't cut it
pd.set_option('display.max_colwidth', -1)    


# In[ ]:


combined.head()


# In[ ]:


# List of biggest errors in decreasing order
sorted = combined.sort_values(by=['error'], ascending=False)


# In[ ]:


sorted[ sorted['target']==0] [0:14]


# In[ ]:


pd.options.display.float_format = "{:.8f}".format


# In[ ]:


# pick texts where true target was 1
insincere = sorted[sorted['target']==1]
insincere[0:14]


# In[ ]:


# List of errors in increasing order
sorted_increasing = combined.sort_values(by=['error'], ascending=True)
sorted_increasing[0:10]


# In[ ]:


# add new filed 'question_length' in characters
combined['question_length']=combined['question_text'].apply(lambda x: len(x))
# sort by that field
sorted_len = combined.sort_values(by=['question_length'], ascending=True)
sorted_len[0:20]


# In[ ]:


# reverse order - start from longest
# This print is very wide, so omit printing qid-field
sorted_len.drop('qid',axis=1).iloc[::-1][0:8]


# In[ ]:


sorted_len[sorted_len['qid']=='4d2e2796dd1ced2c8e64']

