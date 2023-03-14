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




from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score




train=pd.read_csv(r"/kaggle/input/fake-news/train.csv")
test=pd.read_csv(r"/kaggle/input/fake-news/test.csv")




train=train.dropna()




X=train.drop('label',axis=1)
y=train['label']




y.value_counts()




voc_size=5000




messages=X.copy()
messages.reset_index(inplace=True)




import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')




from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def data_preprocessing(messages):
    corpus = []
    for i in range(0, len(messages)):
    
        review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
        review = review.lower()
        review = review.split()
    
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
        #corpus
    return [one_hot(words,voc_size)for words in corpus] 

onehot_repr=data_preprocessing(messages)




sent_length=20
def embedding_representation(onehot_repr):
    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    return embedded_docs

embedded_docs=embedding_representation(onehot_repr)




embedding_vector_features=50

model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())




X_final=np.array(embedded_docs)
y_final=np.array(y)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=77)




model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=15,batch_size=128)




y_pred=model.predict_classes(X_test)




confusion_matrix(y_test,y_pred)




accuracy_score(y_test,y_pred)




test_messages=test.copy()
test_messages = test_messages.fillna(' ')




test_messages.reset_index(inplace=True)
test_onehot_repr=data_preprocessing(test_messages)
test_embedded_docs=embedding_representation(test_onehot_repr)




test_final=np.array(test_embedded_docs)




model_prediction = model.predict_classes(test_final)
model_prediction = model_prediction.ravel()




submission = pd.DataFrame({'id':test_messages['id'], 'label':model_prediction})
submission.head()




submission.to_csv('submit.csv', index = False)




kaggle competitions submit -c fake-news -f submission.csv -m "Message"

