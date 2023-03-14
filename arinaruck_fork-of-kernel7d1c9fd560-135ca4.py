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
#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer 
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
data.target.value_counts




stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer() 

def lower_token(tokens): 
    return [w.lower() for w in tokens]   

def lemmatize_words(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def remove_stop_words(tokens): 
    return [word for word in tokens if word not in stop_words]

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




import time
from tqdm import tqdm_notebook

def process_basic(data):
    tokens = [tokenizer.tokenize(sen) for sen in data['question_text']]
    lower_tokens = [lower_token(token) for token in tokens]
    lemmatized_tokens = [lemmatize_words(token) for token in lower_tokens]
    filtered_nums = [clean_numbers(sen) for sen in lemmatized_tokens]
    no_stopwords = [remove_stop_words(sen) for sen in filtered_nums]
    data['basic'] = [' '.join(sen) for sen in no_stopwords]
    return data




data = process_basic(data)
display(data.head())
test_data = process_basic(test_data)
display(test_data.head())




lemmatizer.lemmatize('do')




display(test_data.head(50))




data.to_csv('processed_train_lemmatized.csv')
test_data.to_csv('processed_test_lemmatized.csv')

