# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import string
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, GlobalAvgPool1D, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model


#from zipfile import ZipFile
#file_name = "train.csv.zip"

#with ZipFile(file_name,'r') as zip:
#  zip.extractall()
#  print("DOne")

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
## Parameters 
embed_size = 300 # how big is each word vector
max_features = 60000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 26 # max number of words in a question to use
batch_size = 3636

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

train_y = train_df['target'].values

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_X, train_y)

# Predicting the Test set results
y_pred = classifier.predict(train_X)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(train_y, y_pred)

from sklearn.metrics import accuracy_score

acc = accuracy_score(train_y, y_pred)

print("Accuracy on the Quora dataset: {:.2f}".format(acc*100))

from sklearn.metrics import classification_report
target_names = ['0','1']
print(classification_report(y_pred,train_y, target_names=target_names))

from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics # for the check the error and accuracy of the model
# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then insteda of model_selection we can use cross_validation

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_X, train_y)

# Predicting the Test set results
y_pred = classifier.predict(train_X)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print(metrics.accuracy_score(y_pred,train_y))

model_DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
model_DT.fit(train_X, train_y)
prediction = model_DT.predict(train_X)
metrics.accuracy_score(prediction,train_y)

from sklearn.metrics import classification_report
target_names = ['0','1']
print(classification_report(prediction,train_y, target_names=target_names))

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(train_X)
prediction = kmeans.predict(train_X)
metrics.accuracy_score(prediction,train_y)

from sklearn.metrics import classification_report
target_names = ['0','1']
print(classification_report(prediction,train_y, target_names=target_names))

from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
eclf1 = VotingClassifier(estimators=[('model_GaussianNB', classifier),('model_DT', model_DT),('kmeans', kmeans)], voting='hard')
eclf1 = eclf1.fit(train_X, train_y)
prediction=eclf1.predict(train_X)
print(metrics.accuracy_score(prediction,train_y),"voting classifier hard method")

from sklearn.metrics import classification_report
target_names = ['0','1']
print(classification_report(prediction,train_y, target_names=target_names))

ids = test_df["qid"]

target = eclf1.predict(test_X)

sample_submission["prediction"]=target

 sample_submission = sample_submission.loc[:, ~sample_submission.columns.str.contains('^Unnamed')]

sample_submission.head()


