#!/usr/bin/env python
# coding: utf-8



#Libraries
import os
from time import time
import math
import random
import gc
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import Ridge

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import Callback
from numpy.random import seed
from urllib.parse import urlparse
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import keras.backend.tensorflow_backend as KTF
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import TensorBoard

from time import gmtime, strftime
import warnings
warnings.filterwarnings("ignore")
seed(42)
random.seed(42)




train = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv', dtype={'Id':str})            .dropna().reset_index(drop=True) # to make things easy
reveal_ID = pd.read_csv('../input/trends-assessment-prediction/reveal_ID_site2.csv', dtype={'Id':str})
ICN_numbers = pd.read_csv('../input/trends-assessment-prediction/ICN_numbers.csv')
loading = pd.read_csv('../input/trends-assessment-prediction/loading.csv', dtype={'Id':str})
fnc = pd.read_csv('../input/trends-assessment-prediction/fnc.csv', dtype={'Id':str})
sample_submission = pd.read_csv('../input/trends-assessment-prediction/sample_submission.csv', dtype={'Id':str})




# Config
OUTPUT_DICT = ''
ID = 'Id'
TARGET_COLS = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
SEED = 42




sample_submission['ID_num'] = sample_submission[ID].apply(lambda x: int(x.split('_')[0]))
test = pd.DataFrame({ID: sample_submission['ID_num'].unique().astype(str)})
del sample_submission['ID_num']; gc.collect()




# merge
train = train.merge(loading, on=ID, how='left')
train = train.merge(fnc, on=ID, how='left')

test = test.merge(loading, on=ID, how='left')
test = test.merge(fnc, on=ID, how='left')




len(loading.columns)




len(fnc.columns)




len(train.columns)




def outlier_2s(df):
    for i in range(1, len(df.columns)-1):
        col = df.iloc[:,i]
        average = np.mean(col)
        sd = np.std(col)
        outlier_min = average - (sd) * 2.2
        outlier_max = average + (sd) * 2.2
        col[col < outlier_min] = outlier_min
        col[col > outlier_max] = outlier_max
    return df

from sklearn import preprocessing
def scaler(df):
    for i in range(5, len(df.columns)-5):
        col = df.iloc[:,i]
        col = preprocessing.minmax_scale(col)
    return df

def mean_diff1(df):
    for i in range(7, 7+len(loading.columns)):
        dfa = df.iloc[:,7:7+len(loading.columns)]
        average = np.mean(np.mean(dfa))
        col = df.iloc[:,i]
        for j in range(1,len(train)):
            val = df.iloc[j]
            val = col - average
    return df

def mean_diff2(df):
    for i in range(7+len(loading.columns), 7+len(loading.columns)+len(fnc.columns)-7):
        dfa = df.iloc[:,7+len(loading.columns):7+len(loading.columns)+len(fnc.columns)]
        average = np.mean(np.mean(dfa))
        col = df.iloc[:,i]
        for j in range(1,len(train)):
            val = df.iloc[j]
            val = col - average
    return df




#diff1 = mean_diff1(train)
#diff2 = mean_diff2(train)
#train = train.merge(diff1, on=ID, how='left')
#train = train.merge(diff2, on=ID, how='left')


#diff1 = mean_diff1(test)
#diff2 = mean_diff2(test)
#test = test.merge(diff1, on=ID, how='left')
#test = test.merge(diff2, on=ID, how='left')




train = outlier_2s(train)
train = scaler(train)
train = train.dropna(how='all').dropna(how='all', axis=1)




X_train = train.drop('Id', axis=1).drop(TARGET_COLS, axis=1)
y_train = train.drop('Id', axis=1)[TARGET_COLS]
X_test = test.drop('Id', axis=1)




np.random.seed(1964)
epochs= 16
batch_size = 128
verbose = 1
validation_split = 0.25
input_dim = X_train.shape[1]
n_out = y_train.shape[1]

model_1 = Sequential([
               #input
               Dense(2048, input_shape=(input_dim,)),
               Activation('relu'),
               Dropout(0.1),
               Dense(2048),
               Activation('relu'),
               Dropout(0.1),
#               Dense(256),
#               Activation('relu'),
#               Dropout(0.1),
#               Dense(128),
#               Activation('relu'),
#               Dropout(0.1),
               #output
               Dense(n_out),
               Activation('relu'),
        ])

model_1.compile(loss='mse',
                 optimizer='adam',
                 metrics=['mse'])
hist_1 = model_1.fit(X_train, y_train,
                        batch_size = batch_size, epochs = epochs,
                        callbacks = [],
                        verbose=verbose, validation_split=validation_split)




prediction_dict = model_1.predict(X_test)
prediction_dict = pd.DataFrame(prediction_dict)
prediction_dict.columns = y_train.columns
prediction_dict.head(10)




pred_df = pd.DataFrame()

for TARGET in TARGET_COLS:
    tmp = pd.DataFrame()
    tmp[ID] = [f'{c}_{TARGET}' for c in test[ID].values]
    tmp['Predicted'] = prediction_dict[TARGET]
    pred_df = pd.concat([pred_df, tmp])

print(pred_df.shape)
print(sample_submission.shape)

pred_df.head()




submission = pd.merge(sample_submission, pred_df, on = 'Id')
submission




submission = pd.merge(sample_submission, pred_df, on = 'Id')[['Id', 'Predicted_y']]
submission.columns = ['Id', 'Predicted']




submission.to_csv('submission.csv', index=False)
submission.head()

