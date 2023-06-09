#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")




train_identity.head(5)




train_transaction.head(5)




train_identity.shape




train_transaction.shape




train_transaction.columns




train_identity.columns




test_transaction.shape




test_identity.shape




fc = train_transaction['isFraud'].value_counts(normalize=True).to_frame()
fc.plot.bar()




fc




train=train_transaction.merge(train_identity,how='left',left_index=True,right_index=True)
y_train=train['isFraud'].astype('uint8')
print('Train shape',train.shape)
del train_transaction,train_identity
print("Data set merged ")




train.head(3)




test=test_transaction.merge(test_identity,how='left',left_index=True,right_index=True)
print('Test shape',test.shape)
del test_transaction,test_identity
print("Test Data set merged ")




test.head(3)




get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage2(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df')




get_ipython().run_cell_magic('time', '', 'train = reduce_mem_usage2(train)')




get_ipython().run_cell_magic('time', '', 'test = reduce_mem_usage2(test)')




not_fraud=train[train.isFraud==0]
fraud=train[train.isFraud==1]




from sklearn.utils import resample

# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
X = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
X.isFraud.value_counts()




len(X)




y = X['isFraud']




#X = X.drop(['isFraud'], axis=1)




X["TransactionDay"] = X["TransactionDT"] // (24*60*60)
X["TransactionWeek"] = X["TransactionDay"] // 7




prod = list(set(X["ProductCD"]))
print(prod)




X.groupby("ProductCD")['isFraud'].value_counts().to_frame().plot.bar()




X.groupby(['card4', 'card3'])['isFraud'].value_counts().to_frame().plot.bar()




X.groupby(['card4'])['isFraud'].value_counts().to_frame().plot.bar()




X.groupby(['card4', 'ProductCD'])['isFraud'].value_counts().to_frame().plot.bar()




fraud = X[X['isFraud'] == 1]
print("max trans amount happend during fraud:",max(fraud["TransactionAmt"]))




print("Min trans amount happend during fraud:",min(fraud["TransactionAmt"]))




fraud[fraud["TransactionAmt"] == 0.292]




X['addr1']




X['addr2']




fraud.groupby(['addr1', 'addr2'])['isFraud'].value_counts().to_frame().plot.bar()




null_percent = train.isnull().sum()/train.shape[0]*100

cols_to_drop = np.array(null_percent[null_percent > 50].index)

cols_to_drop




X = X.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop,axis=1)




X.columns




null_percent = test.isnull().sum()/X.shape[0]*100
null_percent[null_percent > 0]




cols_to_drop_again = np.array(null_percent[null_percent > 0.001].index)

cols_to_drop_again




X = X.drop(cols_to_drop_again, axis=1)
test = test.drop(cols_to_drop_again,axis=1)




list(X.columns)




sns.distplot(a=X["TransactionAmt"])




log_trans = X['TransactionAmt'].apply(np.log)
sns.distplot(a=log_trans)




X['TransactionAmt'] = X['TransactionAmt'].apply(np.log)
test['TransactionAmt'] = test['TransactionAmt'].apply(np.log)




cols_to_drop = ['TransactionDT','TransactionID_x']
X = X.drop(cols_to_drop, axis=1)




cat_data = X.select_dtypes(include='object')
num_data = X.select_dtypes(exclude='object')

cat_cols = cat_data.columns.values
num_cols = num_data.columns.values




cat_cols




from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm




for i in tqdm(cat_cols): 
    label = LabelEncoder()
    label.fit(list(X[i].values)+list(test[i].values))
    X[i] = label.transform(list(X[i].values))
    test[i] = label.transform(list(test[i].values))




X.shape




X.head(10)




corr = X.corr()




plt.figure(figsize=(20,20))
sns.heatmap(corr)




col_corr = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if (corr.iloc[i, j] >= 0.95) and (corr.columns[j] not in col_corr):
            colname = corr.columns[i] # getting the name of column
            col_corr.add(colname)




col_corr




final_columns = []

cols = X.columns

for i in cols:
    if i in col_corr:
        continue
    else:
        final_columns.append(i)
        
        




final_columns




X_final = X[final_columns]




test_final = test[final_columns]




print(X_final.shape)
print(test_final.shape)




from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()




scaler.fit(X_final)




dfX = scaler.transform(X_final)




feature_name = X_final.columns




dfX = pd.DataFrame(dfX, columns=feature_name)




from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Input
from keras import backend as K
import keras

n_features = dfX.shape[1]
dim = 15

def build_model(dropout_rate=0.15, activation='tanh'):
    main_input = Input(shape=(n_features, ), name='main_input')
    
    x = Dense(dim*2, activation=activation)(main_input)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(dim*2, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate/2)(x)
    
    x = Dense(dim, activation=activation)(x)
    x = Dropout(dropout_rate/4)(x)

    encoded = Dense(2, activation='tanh')(x)

    input_encoded = Input(shape=(2, ))
    
    x = Dense(dim, activation=activation)(input_encoded)
    x = Dense(dim, activation=activation)(x)
    x = Dense(dim*2, activation=activation)(x)
    
    decoded = x = Dense(n_features, activation='linear')(x)

    encoder = Model(main_input, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(main_input, decoder(encoder(main_input)), name="autoencoder")
    return encoder, decoder, autoencoder

K.clear_session()
c_encoder, c_decoder, c_autoencoder = build_model()
c_autoencoder.compile(optimizer='nadam', loss='mse')

c_autoencoder.summary()




dfX.shape




get_ipython().run_cell_magic('time', '', "epochs = 50\nbatch_size = 9548\nhistory = c_autoencoder.fit(dfX, y,\n                    epochs=epochs,\n                    batch_size=batch_size,\n                    shuffle=True,\n                    verbose=1)\n\nloss_history = history.history['loss']\nplt.figure(figsize=(10, 5))\nplt.plot(loss_history);")




params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
         }




from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
seed = 123
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
pred_test_full =0
cv_score =[]
i=1
predictions = np.zeros(test_final.shape[0])
print('5 Fold Stratified Cross Validation')
print('-----------------------------------')
for train_index,test_index in kf.split(X_final,y):
    print('{} of KFold {}'.format(i,kf.n_splits))
    xtr,xv = X_final.loc[train_index],X_final.loc[test_index]
    ytr,yv = y.loc[train_index],y.loc[test_index]
    
    #model
    #clf = LogisticRegression(C=2)
    #clf.fit(xtr,ytr)
    #clf = CatBoostClassifier(task_type='GPU', eval_metric='AUC', loss_function='Logloss', use_best_model=True,
    #                      silent=True, class_weights=[0.05, 0.95],
    #                     random_state=42, iterations=5000, od_type='Iter', od_wait=200, grow_policy='Lossguide', max_depth=8)
    dtrain = lgb.Dataset(xtr, label=ytr)
    dvalid = lgb.Dataset(xv, label=yv)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
    
    y_pred_valid = clf.predict(xv)
    score = roc_auc_score(yv,clf.predict(xv))
    print('ROC AUC score:',score)
    cv_score.append(score)    
    pred_test = clf.predict(test_final)/5
    pred_test_full +=pred_test
    i+=1
    print('-------------------------------------')




print('Mean AUC Score for CatBoost : {}'.format(np.array(score).mean()))




sub = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')




sub['isFraud'] = pred_test_full




sub.head()




sub.to_csv('submission.csv', index=False)

