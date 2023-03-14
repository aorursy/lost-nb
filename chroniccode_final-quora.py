#!/usr/bin/env python
# coding: utf-8



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




import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from tqdm import tqdm




get_ipython().system('unzip ../input/quora-question-pairs/train.csv.zip')
get_ipython().system('unzip ../input/quora-question-pairs/test.csv.zip')




train = pd.read_csv("/kaggle/working/train.csv")
test = pd.read_csv("/kaggle/working/test.csv")




print(train.isnull().sum())
print(test.isnull().sum())




train = train.fillna('empty')
test = test.fillna('empty')




stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']




def text_to_wordlist(text):
    text = ''.join([c for c in text if c not in punctuation])
    text = text.split()
    text = [w for w in text if not w in stop_words]
    text = " ".join(text)
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return(text)




def modify_column(dataframe,column):
    temp = []
    for ques in tqdm(dataframe[column]):
        temp.append(text_to_wordlist(ques))
    return temp




train['question1'] = modify_column(train,'question1')
train['question2'] = modify_column(train,'question2')
test['question1'] = modify_column(test,'question1')
test['question2'] = modify_column(test,'question2')




train.head() 




from sklearn.feature_extraction.text import CountVectorizer
import itertools
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split




df_all = pd.concat((train, test))
counts_vectorizer = CountVectorizer(max_features=10000-1).fit(
    itertools.chain(df_all['question1'], df_all['question2']))
other_index = len(counts_vectorizer.vocabulary_)




words_tokenizer = re.compile(counts_vectorizer.token_pattern)




def create_padded_seqs(texts, max_len=10):
    seqs = texts.apply(lambda s: 
        [counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
         for w in words_tokenizer.findall(s.lower())])
    return pad_sequences(seqs, maxlen=max_len)




X1_train, X1_val, X2_train, X2_val, y_train, y_val =     train_test_split(create_padded_seqs(df_all[df_all['id'].notnull()]['question1']), 
                     create_padded_seqs(df_all[df_all['id'].notnull()]['question2']),
                     df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     stratify=df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     test_size=0.3, random_state=1989)




import keras.layers as lyr
from keras.models import Model




input1 = lyr.Input(X1_train.shape[1:])
input2 = lyr.Input(X2_train.shape[1:])

words_embedding_layer = lyr.Embedding(X1_train.max() + 1, 100)
seq_embedding_layer = lyr.LSTM(256, activation='tanh')

seq_embedding = lambda x: seq_embedding_layer(words_embedding_layer(x))

merge_layer = lyr.multiply([seq_embedding(input1), seq_embedding(input2)])

dense1_layer = lyr.Dense(16, activation='sigmoid')(merge_layer)
ouput_layer = lyr.Dense(1, activation='sigmoid')(dense1_layer)

model = Model([input1, input2], ouput_layer)

model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()




model.fit([X1_train, X2_train], y_train, 
          validation_data=([X1_val, X2_val], y_val), 
          batch_size=128, epochs=6, verbose=2)




features_model = Model([input1, input2], merge_layer)
features_model.compile(loss='mse', optimizer='adam')




F_train = features_model.predict([X1_train, X2_train], batch_size=128)
F_val = features_model.predict([X1_val, X2_val], batch_size=128)




import xgboost as xgb




dTrain = xgb.DMatrix(F_train, label=y_train)
dVal = xgb.DMatrix(F_val, label=y_val)




xgb_params = {
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'eval_metric': 'logloss',
    'eta': 0.1, 
    'max_depth': 9,
    'subsample': 0.9,
    'colsample_bytree': 1 / F_train.shape[1]**0.5,
    'min_child_weight': 5,
    'silent': 1
}
bst = xgb.train(xgb_params, dTrain, 1000,  [(dTrain,'train'), (dVal,'val')], 
                verbose_eval=10, early_stopping_rounds=10)




X1_test = create_padded_seqs(df_all[df_all['test_id'].notnull()]['question1'])
X2_test = create_padded_seqs(df_all[df_all['test_id'].notnull()]['question2'])




F_test = features_model.predict([X1_test, X2_test], batch_size=128)




dTest = xgb.DMatrix(F_test)




df_sub = pd.DataFrame({
        'test_id': df_all[df_all['test_id'].notnull()]['test_id'].values,
        'is_duplicate': bst.predict(dTest, ntree_limit=bst.best_ntree_limit)
    }).set_index('test_id')




df_sub.head()




df_sub.to_csv("submission.csv")

