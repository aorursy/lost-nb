#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files areavailable in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from gensim import models

import re
from collections import Counter
import gensim
import heapq
from operator import itemgetter
from multiprocessing import Pool




data = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_data = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
display(data.head())
data.target.value_counts()




stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

def removeStopWords(tokens): 
    return [word for word in tokens if word not in stop_words]

def lower_token(tokens): 
    return [w.lower() for w in tokens]   

#https://towardsdatascience.com/nlp-learning-series-part-1-text-preprocessing-methods-for-deep-learning-20085601684b 

def clean_numbers(sen):
    res = []
    for word in sen:
        if bool(re.search(r'\d', word)):
            word = re.sub('[0-9]{5,}', '#####', word)
            word = re.sub('[0-9]{4}', '####', word)
            word = re.sub('[0-9]{3}', '###', word)
            word = re.sub('[0-9]{2}', '##', word)
            word = re.sub('[0-9]{1}', '#', word)
        res.append(word)
    return res


word2vec_path = '../input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
words = word2vec.index2word

w_rank = {}
for i,word in enumerate(words):
    w_rank[word] = i

WORDS = w_rank

def words(text): return re.findall(r'\w+', text.lower())

def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def correct(sen):
    return [misspell_dict[word] if word in misspell_dict else word for word in sen]

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab




def process(data):
    data, filtered_nums = process_no_misspells(data)
    return process_misspells(data, filtered_nums)

def process_no_misspells(data):
    tokens = [tokenizer.tokenize(sen) for sen in data['question_text']]
    lower_tokens = [lower_token(token) for token in tokens]
    filtered_nums = [clean_numbers(sen) for sen in lower_tokens]
    data['basic'] = [' '.join(sen) for sen in filtered_nums]
    return data, filtered_nums

def process_misspells(data, filtered_nums):
    no_stopwords = [removeStopWords(sen) for sen in filtered_nums]
    no_misspells = [correct(sen) for sen in no_stopwords]
    data['no_misspels'] = [' '.join(sen) for sen in no_misspells]
    return data, no_misspells 




import time
from tqdm import tqdm_notebook
data, filtered_nums = process_no_misspells(data)

start_time = time.time()
vocab = build_vocab(filtered_nums)
top_90k_words = dict(heapq.nlargest(90000, vocab.items(), key=itemgetter(1)))
corrected_words = map(correction,list(top_90k_words.keys()))
print("vocab and correction %s seconds" % (time.time() - start_time),  flush=True)

misspell_dict = {}
start_time = time.time()
for _, (word,corrected_word) in tqdm_notebook(enumerate(zip(top_90k_words,corrected_words))):
    if word!=corrected_word:
        corrected_num, real_num = 0, 0
        try:
            corrected_num = vocab[corrected_word]
        except:
            pass
        real_num = vocab[word]
        if corrected_num > 2 * real_num:
            misspell_dict[word] = corrected_word
print("misspell dict %s seconds" % (time.time() - start_time))




data, no_misspells = process_misspells(data, filtered_nums)
display(data.head())
test_data, test_no_misspells = process(test_data)
display(test_data.head())




display(test_data.head(50))




data.to_csv('processed_train.csv')
test_data.to_csv('processed_test.csv')




'''
def _get_mispell(misspell_dict):
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))
    return misspell_dict, misspell_re

mispellings, mispellings_re = _get_mispell(misspell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

# Usage
print(replace_typical_misspell("Whta is demonitisation"))
print(replace_typical_misspell("become"))
try:
    print(misspell_dict['whta'])
except:
    pass
'''
print(correction('whta'))
print(correction('becoe'))
print(correction('become'))




'''
print(correction('whta'))
print(correction('becoe'))
print(correction('become'))
filtered_misspellings = [correct(sen) for sen in filtered_nums]
no_stopwords = [removeStopWords(sen) for sen in filtered_misspellings]
data['Text_Final_corrected'] = [' '.join(sen) for sen in no_stopwords]
data['tokens_corrected'] = no_stopwords
data.to_csv('tokens.csv')
''''




data.head(50)




''''
data_train, data_test = train_test_split(data, 
                                         test_size=0.10, 
                                         random_state=42)
                                    '''




'''
all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))
'''




'''
all_test_words = [word for tokens in data_test["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
print("Max sentence length is %s" % max(test_sentence_lengths))
'''




def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, data, generate_missing=False):
    embeddings = data['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    data['embeddings'] = embedings
    return list(embeddings)




#training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)




MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300

