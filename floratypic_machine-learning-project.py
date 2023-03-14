#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk as nk
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")




train.shape




test.shape




freq = train['interest_level'].value_counts()
sns.barplot(freq.index, freq.values, color=color[4])
plt.ylabel('Frequence')
plt.xlabel('Interest level')
plt.show()




train.head(1)




from nltk.tokenize import word_tokenize




from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)




from sklearn.feature_extraction.text import TfidfVectorizer




vectorizer = TfidfVectorizer()
desc_text = vectorizer.fit_transform(train['description'])
print(vectorizer.get_feature_names())




desc_text.shape




type('features')




train['features'][:10]




import itertools as it




features=list()
train['features'].apply(lambda x: features.append(x))
features=list(it.chain.from_iterable(features))
len(features)




features[:50]




uniq_feature_total=set(features)




for feat in ft: 









list(uniq_feature_total)[:10]

