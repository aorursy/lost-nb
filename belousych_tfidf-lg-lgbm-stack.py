#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import json
from sklearn.cross_validation import train_test_split

# Any results you write to the current directory are saved as output.




import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier




import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')

# Text Data Features
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

def getSentences(data):
    text_data= []
    for doc in data:
        text_data.append(doc['ingredients'])
    return text_data 


train_text = generate_text(train)
test_text = generate_text(test)

Y = [doc['cuisine'] for doc in train]
_Id = [doc['id'] for doc in test]


train_sentences = getSentences(train)
test_sentences = getSentences(test)
sentences = train_sentences + test_sentences




tfidf = TfidfVectorizer(max_df=0.9, min_df=2)




X = tfidf.fit_transform(train_text)




X_train_source = tfidf.transform((' '.join(i) for i in train_sentences))
X_test_source = tfidf.transform((' '.join(i) for i in test_sentences))




from sklearn.model_selection import StratifiedKFold




Y = np.array(Y)
Y.shape




probaFeature = np.zeros((X.shape[0], 40))
probaTest = []




lg = LogisticRegression(
    penalty='l2',
    C=10, 
    n_jobs=-1, verbose=1, 
    solver='sag', multi_class='multinomial',
    max_iter=300
)




X = csr_matrix(X).toarray()




from lightgbm import LGBMClassifier

params = {
    'multi_class': 'ovr',
    'solver': 'lbfgs'
}

lgbm_params = {
    'n_estimators': 250,
    'max_depth': 25,
    'learning_rate': 0.2,
    'objective': 'multiclass',
    'n_jobs': 7
}

model = LGBMClassifier(**lgbm_params)
# model = LogisticRegression(**params)




skf = StratifiedKFold(n_splits=5, random_state=2, shuffle=False)
for train_index, test_index in skf.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
#     LG
    lg.fit(X_train, Y_train)
    probaLg = lg.predict_proba(X_test)
    
    
    score = lg.score(X_test, Y_test)
    print(score)
    
#     lgbm
    model.fit(X_train, Y_train)
    probaLGBM = model.predict_proba(X_test)
    score = model.score(X_test, Y_test)
    print(score)
    
    
#     test proba
    probaLg_test = lg.predict_proba(X_test_source)
    probaLGBM_test = model.predict_proba(X_test_source)
#     MLP
#     clfMLP.fit(X_train, Y_train)
#     probaMlp = clfMLP.predict_proba(X_test)
#     score = clfMLP.score(X_test, Y_test)
#     print(score)
    
#     clfMLP = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(500,), random_state=1)
#     clfMLP.fit(X_train.dot(model.wv.vectors), Y_train)
#     probaMlp2 = clfMLP.predict_proba(X_test.dot(model.wv.vectors))
    
    probaStack = np.hstack((probaLg, probaLGBM))
    
    probaTestStack = np.hstack((probaLg_test, probaLGBM_test))
    
    probaTest.append(probaTestStack)
    probaFeature[test_index] = probaStack




probaFeature




probaTest = np.array(probaTest)
probaFeatureTest = np.mean(probaTest, axis=0)
probaFeatureTest.shape




from sklearn.neighbors import KNeighborsClassifier
clfKNN = KNeighborsClassifier(
    n_neighbors=10, 
    p=2, # < Степень в "формуле вычисления расстояния"
    metric='minkowski',
    
)




outTest = []




for train_index, test_index in skf.split(probaFeature, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = probaFeature[train_index], probaFeature[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    clfKNN.fit(X_train, Y_train)
    
    score = clfKNN.score(X_test, Y_test)
    print(score)
    
    predict = clfKNN.predict_proba(probaFeatureTest)
    outTest.append(predict)




outTest = np.array(outTest)
outTestMean = np.mean(outTest, axis=0)
indexClass = outTestMean.argmax(axis=1)




indexClass




results = []




for i in indexClass:
    results.append(clfKNN.classes_[i])




with open('tfidf_lg_lgbm_stack4.csv', 'w') as f:
    f.write('id,cuisine\n')
    for _id, y  in zip(_Id, results):
        f.write('%s,%s\n' % (_id, y))





















