#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation,  Conv1D
from tensorflow.keras.layers import GRU,LSTM
from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D, MaxPool1D
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.callbacks import EarlyStopping


# In[3]:


from tqdm import tqdm
tqdm.pandas()


# In[4]:


df_train=pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv' )


# In[5]:


df_train.head()


# In[6]:


df_test=pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv' )


# In[7]:


from zipfile import ZipFile


# In[8]:


'''embeddings1_index = {}

with ZipFile('/kaggle/input/quora-insincere-questions-classification/embeddings.zip') as myzip:
  with myzip.open('glove.840B.300d/glove.840B.300d.txt') as myfile:
    lines = myfile.readlines()
    for line in lines:
      values = line.decode().split(" ")
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings1_index[word] = coefs

print('Found %s word vectors.' % len(embeddings1_index))
'''


# In[9]:


import io
import zipfile

dim=300
embeddings1_index={}

with zipfile.ZipFile("../input/quora-insincere-questions-classification/embeddings.zip") as zf:
    with io.TextIOWrapper(zf.open("glove.840B.300d/glove.840B.300d.txt"), encoding="utf-8") as f:
        for line in tqdm(f):
            values=line.split(' ') # ".split(' ')" only for glove-840b-300d; for all other files, ".split()" works
            word=values[0]
            vectors=np.asarray(values[1:],'float32')
            embeddings1_index[word]=vectors


# In[10]:


#del vectors,word,values,line,f,zf,io


# In[11]:


#del ZipFile


# In[12]:


print('Found %s word vectors.' % len(embeddings1_index))


# In[13]:


del zipfile


# In[14]:


import gc
gc.collect()


# In[ ]:





# In[15]:


## Creating the vocabulary of words
def build_vocab(sentences,verbose=True):

    vocab={}
    for sentence in tqdm(sentences,disable= (not verbose)):
        for word in sentence:
            try:
                vocab[word] +=1
            except:
                vocab[word] =1
    return vocab    


# In[16]:


sentences=df_train['question_text'].progress_apply(lambda x : x.split()).values
vocab=build_vocab(sentences)
print({k : vocab[k] for k in list(vocab)[:5]})


# In[17]:


import operator
def check_coverage(vocab, embeddings_index):
    oov={}
    a={}
    i,k=0,0
    for word in tqdm(vocab):
        try:
            a[word]=embeddings_index[word]
            k+= vocab[word]
        except:
            oov[word]=vocab[word]
            i+=vocab[word]
            pass
    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x   


# In[18]:


oov = check_coverage(vocab,embeddings1_index)


# In[19]:


oov[:20]


# In[20]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }


# In[21]:


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


# In[22]:


df_train['question_text']=df_train['question_text'].progress_apply(lambda x: clean_contractions(x,contraction_mapping))
df_test['question_text']=df_test['question_text'].progress_apply(lambda x: clean_contractions(x,contraction_mapping))

sentences= df_train['question_text'].apply(lambda x : x.split())
vocab = build_vocab(sentences)


# In[23]:


oov = check_coverage(vocab,embeddings1_index)


# In[24]:


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


# In[25]:


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown


# In[26]:


print(unknown_punct(embeddings1_index, punct))


# In[27]:


punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }


# In[28]:


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text


# In[29]:


df_train['question_text'] = df_train['question_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
df_test['question_text'] = df_test['question_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

sentences= df_train['question_text'].apply(lambda x : x.split())
vocab = build_vocab(sentences)


# In[30]:


oov = check_coverage(vocab,embeddings1_index)


# In[31]:


oov[:10]


# In[32]:


mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}


# In[33]:


def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


# In[34]:


df_train['question_text'] = df_train['question_text'].progress_apply(lambda x: correct_spelling(x, mispell_dict))
df_test['question_text'] = df_test['question_text'].apply(lambda x: correct_spelling(x, mispell_dict))

sentences= df_train['question_text'].progress_apply(lambda x : x.split())
sentences = [[word for word in sentence] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)


# In[35]:


oov = check_coverage(vocab,embeddings1_index)


# In[36]:


#del sentences,build_vocab,vocab,oov,mispell_dict,punct,contraction_mapping
#del check_coverage


# In[37]:


gc.collect()


# In[38]:


len_voc = 95000
max_len =60


# In[ ]:





# In[39]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[40]:


t = Tokenizer(num_words=len_voc, filters='')
t.fit_on_texts(df_train['question_text'])
X = t.texts_to_sequences(df_train['question_text'])
X_test=t.texts_to_sequences(df_test['question_text'])
X = pad_sequences(X, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
word_index=t.word_index


# In[41]:


y = df_train['target'].values


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=420)


# In[44]:


del df_train


# In[45]:


gc.collect()


# In[46]:


def make_embed_matrix(embeddings_index, word_index, len_voc):
 all_embs = np.stack(embeddings_index.values())
 emb_mean,emb_std = all_embs.mean(), all_embs.std()
 embed_size = all_embs.shape[1]
 word_index = word_index
 embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))
 
 for word, i in word_index.items():
     if i >= len_voc:
         continue
     embedding_vector = embeddings_index.get(word)
     if embedding_vector is not None: 
         embedding_matrix[i] = embedding_vector
 
 return embedding_matrix


# In[47]:


embedding = make_embed_matrix(embeddings1_index, word_index, len_voc)

del word_index
gc.collect()


# In[48]:


early = EarlyStopping(monitor='val_loss', mode="min", patience=2)


# In[49]:


lstm = Sequential()
lstm.add(Embedding(len_voc, 300, weights=[embedding], trainable=False))
lstm.add(Bidirectional(LSTM(units = 256, return_sequences= True)))
lstm.add(Dropout(rate = 0.2))
#lstm.add(Bidirectional(LSTM(units = 60)))
#lstm.add(Dropout(rate = 0.2))
lstm.add(Conv1D(128, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform"))

lstm.add(GlobalMaxPooling1D())
#lstm.add(Dense(units = 64, activation = 'relu'))
lstm.add(Dropout(rate = 0.2))
lstm.add(Dense(units = 1, activation = 'sigmoid'))

lstm.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[50]:


lstm.summary()


# In[51]:


epochs = 3
batch_size = 128


# In[52]:


hist = lstm.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val,y_val),callbacks=[early])


# In[53]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[54]:


from sklearn.metrics import confusion_matrix,classification_report


# In[55]:


pred_val_y = lstm.predict([X_val], batch_size=1024, verbose=0)


# In[56]:


pred_val_y = (pred_val_y > 0.50).astype(int)


# In[57]:


print(classification_report(y_val,pred_val_y))


# In[58]:


print(confusion_matrix(y_val,pred_val_y))


# In[59]:


pred_test=lstm.predict([X_test], batch_size=1024, verbose=0)


# In[60]:


pred_test = (pred_test > 0.50).astype(int)


# In[61]:


out_df = pd.DataFrame({"qid":df_test["qid"].values})


# In[62]:


out_df['prediction'] = pred_test


# In[63]:


out_df.to_csv("submission.csv", index=False)

