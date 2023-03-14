#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




df = pd.read_csv('../input/train.csv')




df.head()




str_features = df.select_dtypes(include=['object'])
str_features.head()




dummy_features = pd.get_dummies(str_features[['X0','X5','X6']])
dummy_features.shape




number_features  = df.select_dtypes(include=['number'])
number_features.head()




from sklearn.decomposition import PCA
number_features = number_features.drop(['ID','y'],axis=1)
number_features.head(1)
pca = PCA(n_components=7)
pca.fit(number_features)
pca_samples = pca.transform(number_features)
ps = pd.DataFrame(pca_samples)




pca2 = PCA(n_components=6)
pca2.fit(dummy_features)
pca2_samples = pca2.transform(dummy_features)
pdf = pd.DataFrame(pca2_samples,columns=['cat0','cat1','cat2','cat3','cat4','cat5'])
pdf = dummy_features




ps_y = ps.copy()
ps_y['y'] = df['y']




ps_y.corr()




from sklearn.model_selection import train_test_split
   
X_train, X_test, y_train, y_test = train_test_split(ps, df['y'], test_size=0.33, random_state=42)




from sklearn import linear_model
clf  = linear_model.SGDRegressor()
clf.fit(X_train,y_train)




X2_train, X2_test, y2_train, y2_test = train_test_split(pdf, df['y'], test_size=0.33, random_state=42)




clf2 =  linear_model.SGDRegressor()
clf2.fit(X2_train,y2_train)




val = clf.predict(X_test)




from sklearn.metrics import r2_score
r2_score(y_test,val)




val = clf2.predict(X2_test)




from sklearn.metrics import r2_score
r2_score(y2_test,val)




pcall = pd.concat([ps,pdf],axis=1)




X3_train, X3_test, y3_train, y3_test = train_test_split(pcall, df['y'], test_size=0.33, random_state=42)




clf3 =  linear_model.SGDRegressor()
clf3.fit(X3_train,y3_train)




val = clf3.predict(X3_test)




from sklearn.metrics import r2_score
r2_score(y3_test,val)




test = pd.read_csv('../input/test.csv')
tid = test['ID']
test = test.drop(['ID'],axis=1)
test.head()




from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(ps)
df_normalized = pd.DataFrame(np_scaled)
df_normalized.shape
pcall = pd.concat([ps,pdf,axis=1)




X3_train, X3_test, y3_train, y3_test = train_test_split(pcall, df['y'], test_size=0.33, random_state=42)




X3_train.head()




from keras.layers import Dense, Activation,Dropout, BatchNormalization
from keras.models import Sequential
nodes = 512
model = Sequential()
model.add(Dense(nodes, input_dim=95))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(int(nodes/2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(int(nodes/4)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(int(nodes/4)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))
#model.add(Activation('softmax'))




model.compile(loss='mean_absolute_error',
              optimizer='adam')




model.summary()




# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(np.array(X3_train), np.array(y3_train), epochs=100, batch_size=64)




val = model.predict(np.array(X3_test))




from sklearn.metrics import r2_score
r2_score(y3_test,val)




import xgboost




model = xgboost.XGBRegressor(n_estimators=100,max_depth=3,
                             learning_rate=0.29,booster='dart')
model.fit(X3_train,y3_train)
val = model.predict(X3_test)
r2_score(y3_test,val)






