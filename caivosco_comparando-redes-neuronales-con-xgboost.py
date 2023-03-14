#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(" python -c 'import keras; keras.backend.backend()'")


# In[ ]:


get_ipython().system(" ls | grep -i '.*base.*csv'")


# In[ ]:


get_ipython().system(" pip freeze | grep -E 'keras|pandas|numpy|sklearn|tensorflow'")


# In[ ]:


get_ipython().system(" python -c 'import keras; print(keras.__version__)'")


# In[ ]:


import keras
keras.__version__


# In[ ]:


import pandas as pd


# In[ ]:


data1 = pd.read_csv('../input/bbva-dataset/01dataBaseTrainTrxRec.csv')


# In[ ]:


data1.head(5)


# In[ ]:


data1 = data1[['codCliente', 'codEstab', 'ratingMonto']]


# In[ ]:


data1.head()


# In[ ]:


data1.info()


# In[ ]:


type(data1)


# In[ ]:


data1.head()


# In[ ]:


data_normalized = (data1[['codCliente', 'codEstab']]-data1[['codCliente', 'codEstab']].mean())/data1[['codCliente', 'codEstab']].std()


# In[ ]:


data_normalized.head()


# In[ ]:


data_normalized_train = pd.concat([data_normalized, data1[['ratingMonto']]], axis=1)


# In[ ]:


data_normalized_train.head(5)


# In[ ]:


data_train = data_normalized_train.values


# In[ ]:


type(data_train)


# In[ ]:


data_train[0:4, :]


# In[ ]:


X = data_train[:, 0:2]


# In[ ]:


Y = data_train[:, 2]


# In[ ]:


X[0:4, :]


# In[ ]:


Y[0:4,]


# In[ ]:


X.shape, Y.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[ ]:


def model1():
    model = Sequential()
    model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[ ]:


import numpy as np


# In[ ]:


seed = 7
np.random.seed(seed)


# In[ ]:


estimator = KerasRegressor(build_fn=model1, epochs=1, batch_size=100, verbose=0)


# In[ ]:


kfold = KFold(n_splits=10, random_state=seed)


# In[ ]:


import time


# In[ ]:


ti = time.time()
results = cross_val_score(estimator, X, Y, cv=kfold)
tf = time.time()
print((tf - ti)/60)


# In[ ]:


print('Results: %.2f (%.2f) MSE'%(results.mean(), results.std()))


# In[ ]:


data5 = pd.read_csv('../input/bbva-dataset/05dataBaseTestKeyRec.csv')


# In[ ]:


data5.head(5)


# In[ ]:


data5.info()


# In[ ]:


data_normalized_test = (data5 - data5.mean())/ data5.std()


# In[ ]:


data_normalized_test.head(5)


# In[ ]:


data_test = data_normalized_test.values


# In[ ]:


type(data_test)


# In[ ]:


data_test.shape


# In[ ]:


data_test[0:4,]


# In[ ]:


X_test = data_test


# In[ ]:


ti = time.time()
estimator.fit(X,Y)
tf = time.time()
print((tf - ti)/60)


# In[ ]:


pred = estimator.predict(X_test)


# In[ ]:


pred[0:10,]


# In[ ]:


pred_df = pd.DataFrame(data=pred, columns=['ratingMonto'])


# In[ ]:


pred_df.head(5)


# In[ ]:


data5.head(5)


# In[ ]:


data_pre_submit = pd.concat([data5, pred_df], axis=1)


# In[ ]:


data_pre_submit.head(5)


# In[ ]:


ti = time.time()
data_pre_submit['codClienteCodEstab'] = data_pre_submit[data_pre_submit.columns[0:2]].apply(lambda x: ''.join(x.astype(str)), axis=1)
tf = time.time()
print((tf-ti)/60)


# In[ ]:


data_pre_submit.head()


# In[ ]:


data_pre_submit = data_pre_submit[['codClienteCodEstab','ratingMonto']]


# In[ ]:


data_pre_submit.head(5)


# In[ ]:


data_pre_submit.info()


# In[ ]:


data_pre_submit['codClienteCodEstab'] = data_pre_submit['codClienteCodEstab'].astype(str).astype(int)


# In[ ]:


data_pre_submit.info()


# In[ ]:


data3 = pd.read_csv('../input/bbva-dataset/03dataBaseTestRec.csv')


# In[ ]:


data3.head(5)


# In[ ]:


data3.info()


# In[ ]:


data_submit = data3.merge(data_pre_submit, on='codClienteCodEstab')


# In[ ]:


data_submit.head(5)


# In[ ]:


data_submit.to_csv('submitNN_1.csv', sep=',', index=False)


# In[ ]:


cat submitNN_1.csv | head -5


# In[ ]:


# LB = 0.04433


# In[ ]:


get_ipython().system(" pip freeze | grep 'boost'")


# In[ ]:


from xgboost import XGBRegressor
import pandas as pd


# In[ ]:


data1 = pd.read_csv('../input/bbva-dataset/01dataBaseTrainTrxRec.csv')


# In[ ]:


data1.head(5)


# In[ ]:


data1 = data1[['codCliente', 'codEstab', 'ratingMonto']]


# In[ ]:


data1.head()


# In[ ]:


X = data1.drop('ratingMonto', axis=1)
Y = data1['ratingMonto']


# In[ ]:


type(X), type(Y)


# In[ ]:


search_params = {'max_depth': list(range(10,12,2)), 'n_estimators': list(range(300,400,100))}


# In[ ]:


search_params


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


model = XGBRegressor(**search_params)


# In[ ]:


model


# In[ ]:


search = RandomizedSearchCV(model, search_params, n_iter=1)


# In[ ]:


import time


# In[ ]:


ti = time.time()
search.fit(X,Y)
tf = time.time()
print((tf-ti)/60)


# In[ ]:


print(search.best_params_)


# In[ ]:


data5 = pd.read_csv('../input/bbva-dataset/05dataBaseTestKeyRec.csv')


# In[ ]:


data5.head(5)


# In[ ]:


data5.info()


# In[ ]:


X_test = data5


# In[ ]:


pred = search.predict(X_test)


# In[ ]:


type(pred)


# In[ ]:


df_pred = pd.DataFrame(pred, columns=['ratingMonto'])


# In[ ]:


df_pred.head(5)


# In[ ]:


df_pred.info()


# In[ ]:


X_test.head(5)


# In[ ]:


len(df_pred), len(X_test)


# In[ ]:


type(df_pred), type(X_test)


# In[ ]:


data_pre_submit = pd.concat([X_test, df_pred], axis=1)


# In[ ]:


data_pre_submit.head(5)


# In[ ]:


data_pre_submit.info()


# In[ ]:


ti = time.time()
data_pre_submit['codClienteCodEstab'] = data_pre_submit[data_pre_submit.columns[0:2]].apply(lambda x: ''.join(x.astype(str)), axis=1)
tf = time.time()
print((tf-ti)/60)


# In[ ]:


data_pre_submit.head(5)


# In[ ]:


data_pre_submit = data_pre_submit[['codClienteCodEstab','ratingMonto']]


# In[ ]:


data_pre_submit.head(5)


# In[ ]:


data_pre_submit.info()


# In[ ]:


data_pre_submit['codClienteCodEstab'] = data_pre_submit['codClienteCodEstab'].astype(str).astype(int)


# In[ ]:


data_pre_submit.info()


# In[ ]:


data3 = pd.read_csv('../input/bbva-dataset/03dataBaseTestRec.csv')


# In[ ]:


data3.head(5)


# In[ ]:


data3.info()


# In[ ]:


data3.shape, data_pre_submit.shape


# In[ ]:


data_submit = data3.merge(data_pre_submit, on='codClienteCodEstab')


# In[ ]:


data_submit.head()


# In[ ]:


data_submit.info()


# In[ ]:


data_submit.to_csv('submitXGB_1.csv', sep=',', index=False)


# In[ ]:


get_ipython().system(' cat submitXGB_1.csv | head -5')


# In[ ]:


# LB = 0.04191


# In[ ]:




