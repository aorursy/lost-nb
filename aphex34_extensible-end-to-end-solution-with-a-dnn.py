#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import nltk
import random
import gensim
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Embedding
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.layers import LSTM, Bidirectional
from keras.layers import Input, Dense, Dropout
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[ ]:


stopwords = set(nltk.corpus.stopwords.words("english"))
 
def preprocess(text, min_length=2, swords=set()):
    """
    Does preprocessing on an input string by lowering it, tokenizing, filtering out stopwords,
    tokens shorter than min_length and tokens consisting of not English letters.
    """
    text = str(text).lower()
    words = map(lambda word: word.lower(), nltk.word_tokenize(text))
    words = [word for word in words if word not in swords]
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, words))
    return filtered_tokens

def build_vocab(tokenlists, max_size=20000, emb_model=None):
    """
    Builds a vocabulary of at most max_size words from the supplied list of lists of tokens.
    If a word embedding model is provided, adds only the words present in the model vocabulary.
    """

    all_words = list(itertools.chain.from_iterable(tokenlists))
    counter = Counter(all_words)
    if emb_model:
        counter = Counter(x for x in counter if x in emb_model)
            
    vocab = counter.most_common(max_size-2)

    voc_words = [k[0] for k in vocab]

    voc = {}
    voc['NULL'] = 0
    voc['UNKN'] = 1
    for i, k in enumerate(voc_words):
        voc[k] = i+2

    rvoc = {v: k for k, v in voc.items()}

    return voc, rvoc

def vectorize_tokens(tokens, token_to_id, max_len):
    """
    Converts a list of tokens to a list of token ids using the supplied dictionary.
    Pads resulting list with NULL identifiers up to max_len length.
    """
    ids = []
    for token in tokens:
        ids.append(token_to_id.get(token, voc["UNKN"]))

    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += (max_len-len(ids))*[token_to_id["NULL"]]

    return ids

def vectorize(tok_lists, token_to_id, max_len=150):
    """
    Converts a list of lists of tokens to a numpy array of token identifiers
    """
    
    token_matrix = []
        
    for tok_list in tok_lists:
        token_ids = vectorize_tokens(tok_list, token_to_id, max_len)
        token_matrix.append(token_ids)
    
    token_matrix = np.array(token_matrix)
        
    return token_matrix

def get_embeddings(model, rev_voc, dim=300):

    myembeddings = []
    for key in sorted(rev_voc.keys()):
        val = rev_voc[key]
        if val == 'NULL':
            myembeddings.append(np.zeros((dim,)))
        elif val == 'UNKN':
            myembeddings.append(np.random.normal(size=(dim,)))
        else:
            try:
                myembeddings.append(model[val])
            except KeyError:
                print("OOV: {}".format(val))
                myembeddings.append(np.random.normal(size=(dim,)))

    myembeddings = np.array(myembeddings)
    return myembeddings


# In[ ]:


training_data = pd.read_csv("/kaggle/input/train.csv")
testing_data = pd.read_csv("/kaggle/input/test.csv")
labels = np.array(list(training_data['is_duplicate']))


# In[ ]:


tr_q1_preprocessed = [preprocess(t, swords=stopwords) for t in training_data['question1']]
tr_q2_preprocessed = [preprocess(t, swords=stopwords) for t in training_data['question2']]


# In[ ]:


ts_q1_preprocessed = [preprocess(t, swords=stopwords) for t in testing_data['question1']]
ts_q2_preprocessed = [preprocess(t, swords=stopwords) for t in testing_data['question2']]


# In[ ]:


emb_mod = gensim.models.Word2Vec.load_word2vec_format("./assets/GoogleNews-vectors-negative300.bin", 
                                                      binary=True)


# In[ ]:


all_texts = tr_q1_preprocessed+tr_q2_preprocessed+ts_q1_preprocessed+ts_q2_preprocessed


# In[ ]:


emb_mod = gensim.models.Word2Vec(all_texts, min_count=7, size=128)


# In[ ]:


# You might want to train the model more to get better results
n_epochs = 5
for i in range(n_epochs)
    emb_mod.train(all_texts)


# In[ ]:


voc, rev_voc = build_vocab(all_texts, 
                           75000, emb_mod)
embs_m = get_embeddings(emb_mod, rev_voc, emb_mod.vector_size)


# In[ ]:


v_tr_q1 = vectorize(tr_q1_preprocessed, voc, max_len=24)
v_tr_q2 = vectorize(tr_q2_preprocessed, voc, max_len=24)


# In[ ]:


v_ts_q1 = vectorize(ts_q1_preprocessed, voc, max_len=24)
v_ts_q2 = vectorize(ts_q2_preprocessed, voc, max_len=24)


# In[ ]:





# In[ ]:


pickle.dump([v_tr_q1, v_tr_q2], open("vectorized_train", "wb"))
pickle.dump([v_ts_q1, v_ts_q2], open("vectorized_test", "wb"))
pickle.dump([voc, rev_voc], open("voc_rvoc", "wb"))
pickle.dump(embs_m, open("embedding_matrix", "wb"))


# In[ ]:


v_tr_q1, v_tr_q2 = pickle.load(open("./assets/vectorized_train", "rb"))
v_ts_q1, v_ts_q2 = pickle.load(open("./assets/vectorized_test", "rb"))
voc, rev_voc = pickle.load(open("./assets/voc_rvoc", "rb"))
embs_m = pickle.load(open("./assets/embedding_matrix", "rb"))


# In[ ]:





# In[ ]:


MAXLEN = 24
DROPOUT = 0.5
LSTM_UNITS = 600
DENSE_UNITS = 600


# In[ ]:


def pairwise_dis(vests):
    x, y = vests
    return x-y

def pairwise_mul(vests):
    x, y = vests
    return x*y

def cosine_similarity(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.sum((x * y), axis=-1, keepdims=True)

def cosine_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))


# In[ ]:


def build_rnn_mk1_encoder(embs_matrix):
    """
    Basic Bidirectional LSTM encoder. 
    Word embedding layer is frozen to prevent overfitting.
    """
    inp = Input(shape=(MAXLEN,))
    emb = Embedding(embs_matrix.shape[0], embs_matrix.shape[1], input_length=MAXLEN, 
                    weights=[embs_matrix], trainable = False)(inp)
    ls1 = Bidirectional(LSTM(LSTM_UNITS))(emb)
    mod = Model(inputs=inp, outputs=ls1)
    return mod

def build_sim_net(input_shape):
    """
    MLP combining the representations of two question into one vector.
    Takes into account distanse and angle between the input vectors.
    For more information check out the blog post
    https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
    """
    input_a = Input(shape=(input_shape[1],))
    input_b = Input(shape=(input_shape[1],))
    
    mul_layer = Lambda(pairwise_mul, name='MultiplicationLayer')([input_a, input_b])
    dis_layer = Lambda(pairwise_dis, name='SubstractionLayer')([input_a, input_b])

    mer = concatenate([mul_layer, dis_layer])
    bnr = BatchNormalization()(mer)
    
    dr1 = Dropout(DROPOUT)(bnr)
    fc1 = Dense(DENSE_UNITS, activation='relu')(dr1)
    
    mod = Model(inputs=[input_a, input_b], outputs=fc1)
    return mod

def build_model(embs_matrix):
    """
    Combines the modules above into an end-to-end model
    predicting similarity scores for pairs of questions.
    
    Keep in mind that you can plug in just about anything in place of the encoder.
    As long as it predicts a fixed-length vector for each sentence, it should just work.
    """
    
    encoder = build_rnn_mk1_encoder(embs_matrix)
    simnet = build_sim_net(encoder.layers[-1].output_shape)
    
    input_a = Input(shape=(MAXLEN,))
    input_b = Input(shape=(MAXLEN,))
    
    enc_a = encoder(input_a)
    enc_b = encoder(input_b)
    
    fc1 = simnet([enc_a, enc_b])
    
    fc2 = Dense(1, activation='sigmoid')(fc1)
    
    model = Model(inputs=[input_a, input_b], outputs=fc2)
    feature_model = Model(inputs=[input_a, input_b], outputs=fc1)

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    return model, feature_model, encoder


# In[ ]:


mmod, fmod, encmod = build_model(embs_m)


# In[ ]:


encmod.summary()


# In[ ]:


idx = list(range(len(v_tr_q1)))
random.shuffle(idx)
train_idx, val_idx = train_test_split(idx, train_size=0.9)


# In[ ]:


train_X = [v_tr_q1[train_idx], v_tr_q2[train_idx]]
train_Y = labels[train_idx]

val_X = [v_tr_q1[val_idx], v_tr_q2[val_idx]]
val_Y = labels[val_idx]


# In[ ]:


checkpointer = ModelCheckpoint(filepath="quora_bilstm.hdf5",
                                       verbose=0, save_best_only=True)


# In[ ]:


hist = mmod.fit(train_X, train_Y, validation_data=(val_X, val_Y), 
                batch_size=256, epochs=20, 
                callbacks=[checkpointer])


# In[ ]:


mmod.load_weights("quora_bilstm.hdf5")


# In[ ]:


predictions = mmod.predict([v_ts_q1, v_ts_q2]).reshape(-1,)


# In[ ]:


sub = pd.DataFrame({'test_id': testing_data['test_id'], 'is_duplicate': predictions})
sub.to_csv('sample_submission.csv', index=False)
sub.head()


# In[ ]:





# In[ ]:




