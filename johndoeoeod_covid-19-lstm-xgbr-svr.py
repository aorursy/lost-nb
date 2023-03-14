#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




#dependencies
import pandas as pd
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, svm  
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
import math
from keras import metrics
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from xgboost import XGBRegressor




df_train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv', index_col=0)




#df_train['Fatalities'].plt.show()
df_train.drop(columns=['Province_State'], inplace=True)




df_train.fillna(0, inplace=True)
#df_train.set_index('Date', inplace=True)




len(df_train)*0.80




df_train




le = preprocessing.LabelEncoder()
df_train['Country_Region'] = le.fit_transform(df_train['Country_Region'])
df_train['Date'] = le.fit_transform(df_train['Date'])
df_train




X = df_train.drop(columns=['Fatalities','ConfirmedCases']) 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(X) #t_scaled_data = preprocessing.scale(X)
X= np.array(X)
X = preprocessing.scale(X)




scaler.scale_
scale=1/ 1.51515152e-02




y = df_train.drop(columns=['Date','Country_Region','Fatalities'])
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(y)
y = np.array(y)
y = preprocessing.scale(y)




19698-15523




len(y)




X.shape




X_train,X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20)




X_train.shape[1]




#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#X_train.shape




model = XGBRegressor(n_estimators=1000)




model.fit(X_train, y_train) #,batch_size = 50, epochs= 20)




# Use the forest's predict method on the test data
prediction_s = model.predict(X_test)
# Calculate the absolute errors
errors_s = abs(prediction_s - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors_s), 2), 'degrees.')




# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors_s - y_test)
#
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')




accuracy = model.score(X_test, y_test) #test Accuracy squared error for linreg




print(accuracy)




df_t = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv',index_col=0)




df_t.drop(columns=['Province_State'], inplace=True)




df_t




df_t['Country_Region'] = le.fit_transform(df_t['Country_Region'])
df_t['Date'] = le.fit_transform(df_t['Date'])




#df_t.drop(columns=['ForecastId'], inplace=True)
#subt_data = np.array(df_t)
subt_data = scaler.fit_transform(df_t)
subt_data= np.array(subt_data)
subt_data = preprocessing.scale(subt_data)
#subt_data  = np.reshape(subt_data, (subt_data.shape[0], subt_data.shape[1], 1))
subt_data.shape




#df_t = np.array(scaled_data)
#t_scaled_data = scaler.fit_transform(df_t)
#t_scaled_data = preprocessing.scale(df_t)




test_predictions = model.predict(subt_data)
test_predictions.shape




#test_predictions = test_predictions.reshape(12642,)
#test_predictions = test_predictions.reshape(-1, 3)




#test_predictions = scaler.inverse_transform(test_predictions)




#INVERSE TRANSFORM
#test_predictions = test_predictions.reshape(12642,)
#test_predictions_c = test_predictions* scale




#test_predictions = test_predictions.reshape(12642,)
test_predictions = test_predictions_c




test_predictions_c.max()




df_sub = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')
df_sub.drop(columns=['Fatalities','ConfirmedCases'], inplace=True)

save_file_c = pd.DataFrame(test_predictions_c, columns=[['ConfirmedCases']])

result_c = pd.merge(df_sub, save_file_c,left_index=True, right_index=True)
result_c.columns = ['ForecastId','ConfirmedCases']




df_t




result_c = pd.merge(df_t,result_c ,on='ForecastId')
df_t.drop(columns=['Country_Region'], inplace=True)
#result_c.drop(columns=['Country_Region_y','Country_Region_x'], inplace=True




result_c.set_index('ForecastId', inplace=True)




result_c['ConfirmedCases'] = [0 if result_c.loc[i, 'ConfirmedCases'] <= -0 
                                else result_c.loc[i, 'ConfirmedCases'] for i in result_c.index]




#Fatalities X
X = df_train.drop(columns=['Fatalities', ]) 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(X) #t_scaled_data = preprocessing.scale(X)
X= np.array(X)
X = preprocessing.scale(X)




#Fatalities y
y = df_train.drop(columns=['Date','Country_Region','ConfirmedCases'])
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(y)
y = np.array(y)
y = preprocessing.scale(y)




X_train,X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20)




#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape




model1 = XGBRegressor(n_estimators=1000)




#Compile the model, because this is a binary classification problem, accuracy can be used
#model1.compile(optimizer='Adam', loss= 'mean_squared_error')




model1.fit(X_train, y_train)#, batch_size = 50, epochs= 20)




train_c = result_c
train_c = scaler.fit_transform(train_c)
train_c= np.array(train_c)
train_c = preprocessing.scale(train_c)
#train_c  = np.reshape(train_c, (train_c.shape[0], train_c.shape[1], 1))




pred_f = model1.predict(train_c)
#pred_f = pred_f.reshape(-1,3)




#INVERSE TRANSFORM
#pred_f = pred_f*scale




pred_f.max()




save_file_f = pd.DataFrame(pred_f, columns=['Fatalities'])
save_file_f.index += 1 




result = pd.merge(result_c, save_file_f, left_index=True, right_index=True)




#result.set_index('ForecastId', inplace=True)




result




#result= result[['ConfirmedCases','Fatalities']].round(0)




submission = result




submission




submission['Fatalities'] = [0 if submission.loc[i, 'Fatalities'] < 0 
                                else submission.loc[i, 'Fatalities'] for i in submission.index]




#submission['ConfirmedCases'] = [0 if submission.loc[i, 'ConfirmedCases'] <= -0 
            #                   else submission.loc[i, 'ConfirmedCases'] for i in submission.index]




submission




submission.drop(columns=['Country_Region','Date'], inplace=True)




submission['Fatalities'].sum()




submission




len(submission)




submission.to_csv('submission.csv')

