#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import random
import nltk




sample_sub = pd.read_csv("../input/sample_submission.csv")
sample_sub.head(5)




train = pd.read_csv("../input/train.csv")
train.head(5)




test = pd.read_csv("../input/test.csv")
test.head(5)




eap = train.loc[train.author == "EAP"]
hpl = train.loc[train.author == "HPL"]
mws = train.loc[train.author == "MWS"]




results = pd.DataFrame()
results["id"] = test.id
results["EAP"] = 0
results["HPL"] = 0
results["MWS"] = 0




for i in results.T:
    r = random.randint(1, 3)
    if r == 1:
        results.at[i, "EAP"] = 1
    if r == 2:
        results.at[i, "HPL"] = 1
    if r == 3:
        results.at[i, "MWS"] = 1

results.head(5)




# Kaggle Score: 15.77882
# results.to_csv("../results/random.csv")




eap_words = []
hpl_words = []
mws_words = []

for i in train.T:
    if train.author[i] == "EAP":
        eap_words.append(nltk.word_tokenize(train.text[i]))
    if train.author[i] == "HPL":
        hpl_words.append(nltk.word_tokenize(train.text[i]))
    if train.author[i] == "MWS":
        mws_words.append(nltk.word_tokenize(train.text[i]))




stopwords = nltk.corpus.stopwords.words('english')




eap_words = [for word in wl in eap_words]
    
for phrase in hpl_words:
    phrase = [word for word in phrase if word.lower() not in stopwords]

for phrase in mws_words:
    phrase = [word for word in phrase if word.lower() not in stopwords]




print(eap_words[0:4])
#print(hpl_words[0:4])
#print(mws_words[0:4])






