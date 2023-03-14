#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import sklearn as sk
import sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing,	sklearn.linear_model, sklearn.decomposition, sklearn.neural_network
import re
import numpy as np




from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
loss_df = train_df.loss.copy(deep=True)
log_loss = np.log(loss_df.copy(deep=True))
train_df = train_df.drop('id', axis=1)#.drop('loss', axis=1)
test_df = test_df.drop('id', axis=1)

ntrain = train_df.shape[0]
ntest = test_df.shape[0]

train_test = pd.concat((train_df,test_df)).reset_index(drop=True)




reg = re.compile('cat.*')
for col in train_test.columns:
    if reg.match(col):
        cats = []
        for col in train_df.columns:
            #Label encode
            #label_encoder.fit(labels[i])
            label_encoder = LabelEncoder()
            feature = label_encoder.fit_transform(train_df[col])#dataset.iloc[:,i])
            #feature = feature.reshape(dataset.shape[0], 1)
            #One hot encode
            ohe = OneHotEncoder(sparse=False,n_values=len(train_df))
            feature = ohe.fit_transform(feature)
            print(feature)






