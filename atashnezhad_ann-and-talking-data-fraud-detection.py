#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import time
import csv
import os
import seaborn as sns
import random
import gc
from sklearn import preprocessing
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.stats as st
import missingno as msno
import math
import copy
from matplotlib import pyplot
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split 
import gc
import time
get_ipython().run_line_magic('matplotlib', 'inline')

before = time.time()




def null_table(df):
    print("Training Data\n\n")
    print(pd.isnull(df).sum()) 

def some_useful_data_insight(df):
    print(df.head(5))
    print(df.dtypes)
    print(null_table(df))
    print('data length=', len(df))
    
def Plot_Hist_column(df, x):
    pyplot.hist(df[x], log = True)
    pyplot.title(x)
    pyplot.show()
    
def Plot_Hist_columns(df, xlist):
    [Plot_Hist_column(df, x) for x in xlist]  
    pyplot.show()
    
def Make_X_Y(df):
    Y = pd.DataFrame()
    Y['is_attributed'] = df['is_attributed']
    X = df.copy()
    X.drop(labels = ["is_attributed"], axis = 1, inplace = True)
    return X, Y

def Train_Test_training_valid(X, Y, ratio):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio)
    X_training, X_valid, y_training, y_valid =     train_test_split(X_train, y_train, test_size=ratio, random_state=0)
    return X_training, y_training, X_valid, y_valid

def Drop_cols(df, x):
    df.drop(labels = x, axis = 1, inplace = True)
    return df

def Normalized(df):
    df_col_names = df.columns
    x = df.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = df_col_names
    return df


def Parse_time(df):
    df['day'] = df['click_time'].dt.day.astype('uint8')
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['minute'] = df['click_time'].dt.minute.astype('uint8')
    df['second'] = df['click_time'].dt.second.astype('uint8')
    
def Merge_data(df1, df2):
    frames = [df1, df2]
    df = pd.concat(frames)
    return df




def read_train_test_data(address_train, train_nrows, address_test, test_nrows, Skip_range_low, Skip_range_Up, nrows):
    df_train = pd.read_csv(address_train, parse_dates=['click_time'], skiprows=range(Skip_range_low,Skip_range_Up), nrows = nrows)
    df_test = pd.read_csv(address_test, parse_dates=['click_time'])#, nrows = 100)#, nrows = test_nrows)
    return df_train, df_test

def read_train_test_data2(address_train, address_test):
    df_train = pd.read_csv(address_train, parse_dates=['click_time'])
    df_test = pd.read_csv(address_test, parse_dates=['click_time'])
    return df_train, df_test

def read_train_test_data_balanced(address_train, address_test):
    #Read Training data, all class 1 and add same amount 0
    iter_csv = pd.read_csv(address_train, iterator=True, chunksize=10000000, parse_dates=['click_time'])
    df_train_1 = pd.concat([chunk[chunk['is_attributed'] > 0] for chunk in iter_csv])
    iter_csv = pd.read_csv(address_train, iterator=True, chunksize=10000000, parse_dates=['click_time'], nrows=2000000)
    df_train_0 = pd.concat([chunk[chunk['is_attributed'] == 0] for chunk in iter_csv])
    #seperate same number values as train data with class 1
    df_train_0 = df_train_0.head(len(df_train_1))
    #Merge 0 and 1 data
    df_train = Merge_data(df_train_1, df_train_0)
    #Read Test data
    df_test = pd.read_csv(address_test, parse_dates=['click_time'])
    return df_train, df_test

def read_train_test_data_balanced_oversample1(address_train, address_test):
    #Read Training data, all class 1 and add same amount 0
    iter_csv = pd.read_csv(address_train, iterator=True, chunksize=10000000, parse_dates=['click_time'])
    df_train_1 = pd.concat([chunk[chunk['is_attributed'] > 0] for chunk in iter_csv])
    iter_csv = pd.read_csv(address_train, iterator=True, chunksize=10000000, parse_dates=['click_time'], skiprows=range(1,120000000), nrows=14000000)
    df_train_0 = pd.concat([chunk[chunk['is_attributed'] == 0] for chunk in iter_csv])
    count_class_0 = len(df_train_0)
    df_train_1_over = df_train_1.sample(count_class_0, replace=True)
    df_train_over = pd.concat([df_train_1_over, df_train_0], axis=0)
    print(df_train_over.is_attributed.value_counts())
    #Read Test data
    df_test = pd.read_csv(address_test, parse_dates=['click_time'])
    return df_train_over, df_test

def check_memory():
    mem=str(os.popen('free -t -m').readlines())
    T_ind=mem.index('T')
    mem_G=mem[T_ind+14:-4]
    S1_ind=mem_G.index(' ')
    mem_T=mem_G[0:S1_ind]
    mem_G1=mem_G[S1_ind+8:]
    S2_ind=mem_G1.index(' ')
    mem_U=mem_G1[0:S2_ind]
    mem_F=mem_G1[S2_ind+8:]
    print('Free Memory = ' + mem_F +' MB')


def Feature_engineering(df, ip_count):
    df = pd.merge(df, ip_count, on='ip', how='left', sort=False)
    df['clicks_by_ip'] = df['clicks_by_ip'].astype('uint16')
    return df


def predict_And_Submit_using_xgb(df, Trained_Model):
    data_to_submit = pd.DataFrame()
    data_to_submit['click_id'] = range(0, len(df))
    dtest = xgb.DMatrix(df)
    del df
    predict = Trained_Model.predict(dtest, ntree_limit=Trained_Model.best_ntree_limit)
    data_to_submit['is_attributed'] = predict
    pyplot.hist(data_to_submit['is_attributed'], log = True)
    return data_to_submit


def predict_And_Submit(df, Trained_Model):
    pred = Trained_Model.predict(df)
    print('pred Done.')
    predict = pd.DataFrame(pred)
    data_to_submit = pd.DataFrame()
    data_to_submit['click_id'] = range(0, len(df))
    data_to_submit['is_attributed'] = predict
    print(Num_of_line*'=')
    print('data_to_submit = \n', data_to_submit.head(5))
    pyplot.hist(data_to_submit['is_attributed'], log = True)
    return data_to_submit


def generate_ip_count(df_train, df_test):
    
    
    df_train2 = df_train.copy()
    df_test2 = df_test.copy()
    # Drop the IP and the columns from target
    y = df_train2['is_attributed']
    df_train2.drop(['is_attributed'], axis=1, inplace=True)
    # Drop IP and ID from test rows
    sub = pd.DataFrame()
    #sub['click_id'] = test['click_id'].astype('int')
    df_test2.drop(['click_id'], axis=1, inplace=True)
    gc.collect()
    nrow_df_train2 = df_train2.shape[0]
    merge = pd.concat([df_train2, df_test2])

    del df_train2, df_test2
    gc.collect()
    
    # Count the number of clicks by ip
    ip_count = merge.groupby(['ip'])['channel'].count().reset_index()
    ip_count.columns = ['ip', 'clicks_by_ip']
    merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
    merge['clicks_by_ip'] = merge['clicks_by_ip'].astype('uint16')
    merge.drop('ip', axis=1, inplace=True)

    df_train2 = merge[:nrow_df_train2]
    df_test2 = merge[nrow_df_train2:]
    del df_test2, merge
    gc.collect()
    
    return ip_count




Start_time = time.time()
#Address to data
address_train = '../input/talkingdata-adtracking-fraud-detection/train.csv'
address_test = '../input/talkingdata-adtracking-fraud-detection/test.csv'
address_train_sample = '../input/talkingdata-adtracking-fraud-detection/train_sample.csv'
address_test_supplement = '../input/talkingdata-adtracking-fraud-detection/test_supplement.csv'
print('Reading data...!'); check_memory()
#df_train, df_test = read_train_test_data_balanced(address_train, address_test)    
#df_train, df_test = read_train_test_data_balanced_oversample1(address_train, address_test) 
df_train, df_test = read_train_test_data2(address_train_sample, address_test)
print(len(df_train))
print('Reading Done!')
check_memory()



#Parse time
print('Parse, training data...'); check_memory(); Parse_time(df_train); print('Parse, training data, Done!'); 
check_memory()
    
#Feature_engineering data
ip_count = generate_ip_count(df_train, df_test)
df_train = Feature_engineering(df_train, ip_count); df_train.head(); null_table(df_train);  df_train.head(); #df_train = df_train.dropna()
    
#Drop and normalize 
print('Drop colum and normalize, training data...!'); check_memory()
colmn_names = ['attributed_time','click_time', 'ip']; df_train = Drop_cols(df_train, colmn_names)
#df_train = Normalized(df_train)
print('Drop colum and normalize, training data, Done!'); check_memory()




df_train




'''xlist = ['app', 'device', 'os', 'channel',          'is_attributed', 'day', 'hour',          'minute', 'second', 'clicks_by_ip']
Plot_Hist_columns(df_train, xlist)'''




df_train.head(2)




df_columns_name = []
for col in df_train.columns: 
    df_columns_name.append(col)
    #print(col)
    
df_columns_name.remove('is_attributed')
print(df_columns_name)




'''for col1 in df_columns_name:
    for col in df_columns_name:
        name = col1 + '*' + col
        df_train[name] = df_train[col1]*df_train[col]'''




df_train.head(2)




'''for col2 in df_columns_name:
    for col1 in df_columns_name:
        for col in df_columns_name:
            name = col2 + '*' + col1 + '*' + col
            df_train[name] = df_train[col2]*df_train[col1]*df_train[col]'''




df_train.head(2)




#Devide training data, X-Y
print('Begin devide training data, X_Y...'); check_memory()
X, Y = Make_X_Y(df_train); X_training, y_training, X_valid, y_valid = Train_Test_training_valid(X, Y, 0.25)
print('Begin devide training data, X_Y, Done!'); check_memory()
print('Cleaning before training'); del df_train; gc.collect(); check_memory()
print('Begin training...'); check_memory()




from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing




X_training.to_numpy(); y_training.to_numpy(); 
X_valid.to_numpy(); y_valid.to_numpy()

X_training = preprocessing.scale(X_training)
y_training = to_categorical(y_training)
X_valid = preprocessing.scale(X_valid)
y_valid = to_categorical(y_valid)




X_training




#model
model = Sequential()
model.add(Dense(5, input_dim=9, activation='relu'))
#model.add(Dense(4, input_dim=5, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])




'''#model
model = Sequential()
model.add(Dense(125, activation='relu', input_shape=(9,)))
model.add(Dense(75, activation='relu', input_shape=(5,)))
#model.add(Dropout(0.5, input_shape=(2,)))
#model.add(Dense(50, activation='relu', input_shape=(5,)))
#model.add(Dense(25, activation='relu', input_shape=(5,)))
model.add(Dense(10, activation='relu', input_shape=(5,)))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])'''




model.summary()




#fitting the model
history = model.fit(X_training, y_training, validation_data=(X_valid, y_valid), epochs=70, batch_size=1000000)




# list all data in history
#print(history.history.keys())




# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




#Read Test data
df_test = pd.read_csv(address_test, parse_dates=['click_time'])

#Parse time
print('Parse, test data...'); 
check_memory(); 
Parse_time(df_test); 
df_test.head(); 
null_table(df_test); 
print('Parse, test data, Done!'); 
check_memory()




#Feature_engineering data
df_test = pd.merge(df_test, ip_count, on='ip', how='left', sort=False)


#Drop and normalize
print('Drop colum and normalize, test data...!'); check_memory()
colmn_names = ["click_time", "click_id", "ip"]; 
df_test = Drop_cols(df_test, colmn_names); 
df_test.head(); null_table(df_test);
df_test.head()


df_test_columns_name = []
for col in df_test.columns: 
    df_test_columns_name.append(col)

    
df_test_columns_name



'''for col1 in df_test_columns_name:
    for col in df_test_columns_name:
        name = col1 + '*' + col
        df_test[name] = df_test[col1]*df_test[col]'''


'''for col2 in df_test_columns_name:
    for col1 in df_test_columns_name:
        for col in df_test_columns_name:
            name = col2 + '*' + col1 + '*' + col
            df_test[name] = df_test[col2]*df_test[col1]*df_test[col]'''

check_memory()
df_test.head()





#df_test = Normalized(df_test)
df_test = preprocessing.scale(df_test)

print('Drop colum and normalize, test data, Done!'); check_memory()
print('Cleaning before prediction'); del X, Y, X_training, y_training, X_valid, y_valid, ip_count; gc.collect(); check_memory()




#df_test.to_numpy()
y_pred = model.predict_classes(df_test)




y_pred = pd.DataFrame(y_pred)
y_pred.shape




data_to_submit = pd.DataFrame()
data_to_submit['click_id'] = range(0, len(y_pred))
data_to_submit['is_attributed'] = y_pred
pyplot.hist(data_to_submit['is_attributed'], log = True)




data_to_submit




data_to_submit.to_csv('Amin_csv_to_submit.csv', index = False)




After = time.time()
After - before

