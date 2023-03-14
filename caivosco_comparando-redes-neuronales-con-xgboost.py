#!/usr/bin/env python
# coding: utf-8



get_ipython().system(" python -c 'import keras; keras.backend.backend()'")




get_ipython().system(" ls | grep -i '.*base.*csv'")




get_ipython().system(" pip freeze | grep -E 'keras|pandas|numpy|sklearn|tensorflow'")




get_ipython().system(" python -c 'import keras; print(keras.__version__)'")




import keras
keras.__version__




import pandas as pd




data1 = pd.read_csv('../input/bbva-dataset/01dataBaseTrainTrxRec.csv')




data1.head(5)




data1 = data1[['codCliente', 'codEstab', 'ratingMonto']]




data1.head()




data1.info()




type(data1)




data1.head()




data_normalized = (data1[['codCliente', 'codEstab']]-data1[['codCliente', 'codEstab']].mean())/data1[['codCliente', 'codEstab']].std()




data_normalized.head()




data_normalized_train = pd.concat([data_normalized, data1[['ratingMonto']]], axis=1)




data_normalized_train.head(5)




data_train = data_normalized_train.values




type(data_train)




data_train[0:4, :]




X = data_train[:, 0:2]




Y = data_train[:, 2]




X[0:4, :]




Y[0:4,]




X.shape, Y.shape




from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor




from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold




def model1():
    model = Sequential()
    model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model




import numpy as np




seed = 7
np.random.seed(seed)




estimator = KerasRegressor(build_fn=model1, epochs=1, batch_size=100, verbose=0)




kfold = KFold(n_splits=10, random_state=seed)




import time




ti = time.time()
results = cross_val_score(estimator, X, Y, cv=kfold)
tf = time.time()
print((tf - ti)/60)




print('Results: %.2f (%.2f) MSE'%(results.mean(), results.std()))




data5 = pd.read_csv('../input/bbva-dataset/05dataBaseTestKeyRec.csv')




data5.head(5)




data5.info()




data_normalized_test = (data5 - data5.mean())/ data5.std()




data_normalized_test.head(5)




data_test = data_normalized_test.values




type(data_test)




data_test.shape




data_test[0:4,]




X_test = data_test




ti = time.time()
estimator.fit(X,Y)
tf = time.time()
print((tf - ti)/60)




pred = estimator.predict(X_test)




pred[0:10,]




pred_df = pd.DataFrame(data=pred, columns=['ratingMonto'])




pred_df.head(5)




data5.head(5)




data_pre_submit = pd.concat([data5, pred_df], axis=1)




data_pre_submit.head(5)




ti = time.time()
data_pre_submit['codClienteCodEstab'] = data_pre_submit[data_pre_submit.columns[0:2]].apply(lambda x: ''.join(x.astype(str)), axis=1)
tf = time.time()
print((tf-ti)/60)




data_pre_submit.head()




data_pre_submit = data_pre_submit[['codClienteCodEstab','ratingMonto']]




data_pre_submit.head(5)




data_pre_submit.info()




data_pre_submit['codClienteCodEstab'] = data_pre_submit['codClienteCodEstab'].astype(str).astype(int)




data_pre_submit.info()




data3 = pd.read_csv('../input/bbva-dataset/03dataBaseTestRec.csv')




data3.head(5)




data3.info()




data_submit = data3.merge(data_pre_submit, on='codClienteCodEstab')




data_submit.head(5)




data_submit.to_csv('submitNN_1.csv', sep=',', index=False)




cat submitNN_1.csv | head -5




# LB = 0.04433




get_ipython().system(" pip freeze | grep 'boost'")




from xgboost import XGBRegressor
import pandas as pd




data1 = pd.read_csv('../input/bbva-dataset/01dataBaseTrainTrxRec.csv')




data1.head(5)




data1 = data1[['codCliente', 'codEstab', 'ratingMonto']]




data1.head()




X = data1.drop('ratingMonto', axis=1)
Y = data1['ratingMonto']




type(X), type(Y)




search_params = {'max_depth': list(range(10,12,2)), 'n_estimators': list(range(300,400,100))}




search_params




from sklearn.model_selection import RandomizedSearchCV




model = XGBRegressor(**search_params)




model




search = RandomizedSearchCV(model, search_params, n_iter=1)




import time




ti = time.time()
search.fit(X,Y)
tf = time.time()
print((tf-ti)/60)




print(search.best_params_)




data5 = pd.read_csv('../input/bbva-dataset/05dataBaseTestKeyRec.csv')




data5.head(5)




data5.info()




X_test = data5




pred = search.predict(X_test)




type(pred)




df_pred = pd.DataFrame(pred, columns=['ratingMonto'])




df_pred.head(5)




df_pred.info()




X_test.head(5)




len(df_pred), len(X_test)




type(df_pred), type(X_test)




data_pre_submit = pd.concat([X_test, df_pred], axis=1)




data_pre_submit.head(5)




data_pre_submit.info()




ti = time.time()
data_pre_submit['codClienteCodEstab'] = data_pre_submit[data_pre_submit.columns[0:2]].apply(lambda x: ''.join(x.astype(str)), axis=1)
tf = time.time()
print((tf-ti)/60)




data_pre_submit.head(5)




data_pre_submit = data_pre_submit[['codClienteCodEstab','ratingMonto']]




data_pre_submit.head(5)




data_pre_submit.info()




data_pre_submit['codClienteCodEstab'] = data_pre_submit['codClienteCodEstab'].astype(str).astype(int)




data_pre_submit.info()




data3 = pd.read_csv('../input/bbva-dataset/03dataBaseTestRec.csv')




data3.head(5)




data3.info()




data3.shape, data_pre_submit.shape




data_submit = data3.merge(data_pre_submit, on='codClienteCodEstab')




data_submit.head()




data_submit.info()




data_submit.to_csv('submitXGB_1.csv', sep=',', index=False)




get_ipython().system(' cat submitXGB_1.csv | head -5')




# LB = 0.04191






