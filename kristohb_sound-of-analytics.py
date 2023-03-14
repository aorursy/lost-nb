#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import copy 
import re
from sklearn import preprocessing 
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output




# LOad data ###
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
songs = pd.read_csv('../input/songs.csv')
song_info = pd.read_csv('../input/song_extra_info.csv')
members = pd.read_csv('../input/members.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
print("| fin")




print("Shape of Training set: ",train.shape)
print("Column names: ",train.columns.values)
users = train.loc[:,'msno'].values
unique_users = list(set(users))
print("Number of unique Users: ",len(unique_users))




## Merge in data into the training set 

train = pd.merge(train,members,how="left",on="msno")
print("Shape new train: ",train.shape)
train = pd.merge(train,songs,how="left",on="song_id")
print("Shape new train: ",train.shape)
train = pd.merge(train,song_info,how="left",on="song_id")
print("Shape new train: ",train.shape)




# Merge test data as well 
test = pd.merge(test,members,how="left",on="msno")
print("Shape new test: ",test.shape)
test = pd.merge(test,songs,how="left",on="song_id")
print("Shape new test: ",test.shape)
test = pd.merge(test,song_info,how="left",on="song_id")

print("Shape new test: ",test.shape)




## Missing data analysis curtesy https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data_train = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print("| Top Missing data in Training data|")
print(missing_data_train.head(20))

##
total = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print("| Top Missing data in Test data |")
print(missing_data_test.head(20))




train_objects = train.columns.to_series().groupby(train.dtypes).groups

kys = list(train_objects.keys())
print(train_objects)
train_objects[kys[0]]
train_cpy = train
#for col in train_cpy.columns.values:
#    print(train_cpy[col].dtype)
#    print("Col: ",col," Number of values: ",train_cpy[col].value_counts().shape[0])
#    print(train_cpy[col].value_counts())




## Experiment with the regular expressions on the artist name

'''
print(re.split("(\w+)?[feat||]*(\w+?[\s\w+])","sia feat kanye|rkelly|lil wayne|beyonce"))
print(re.split("(\w+)?[featuring|feat|feat.||]*(\w+?[\s\w+])","sia featuring kanye|rkelly|lil wayne|beyonce"))
print(re.split("[?\s+]feat?uring[?\s+]|\|","sia feat. kanye | lil wayne"))

fit_label = preprocessing.LabelEncoder().fit(['one','two','three'])
print(fit_label.transform(['one','one']))
'''




'''
## Study the string objects
train["artist_name"] = train["artist_name"].str.lower()
counts = train["artist_name"].value_counts()
uniq = set(list((counts.index)))
#print(uniq)
#print(re.findall("feat|(s+) ?\|","sia feat kanye, rkelly"))

#Have | and feat seperating artists
where = train["artist_name"].str.find("feat")>0 
print("..")
split_train = train["artist_name"].str.split("[?\s+]feat?uring.[?\s+]|\|",expand=True)
print("..")
split_train = split_train.apply(lambda x: x.str.strip())
uniq_names = list(pd.unique(split_train.values.ravel('K')))
print("..")
la = split_train.apply(lambda x: [uniq_names.index(v) for v in x] ,axis = 1)
print("..")
la.colnames = ['MainArtist'] + ["FeatArtist"+str(c) for c in la.colnames.values[1:]]
print(la)

train[la.colnames] = la
''''''




string_objs = ["source_system_tab","source_screen_name",'source_type','gender','artist_name',
              'composer','lyricist','name']
for col in string_objs:
    train[col] = train[col].astype('category').cat.codes
train_objects = train.columns.to_series().groupby(train.dtypes).groups
print(train_objects)




date_objs = ['registration_init_time','expiration_date']

train['registration_init_time']=pd.to_datetime(train['registration_init_time'], format='%Y%m%d')
train['expiration_date']=pd.to_datetime(train['expiration_date'], format='%Y%m%d')




isrc = ['isrc']
print("| Example extraction from ISRC |")
print('ISRC:      ',train[isrc].values[0][0])
print('Country:   ',train[isrc].values[0][0][0:2])
print('Reg code:  ',train[isrc].values[0][0][2:5])
print('Year:      ',train[isrc].values[0][0][5:7])
print('Unique id: ',train[isrc].values[0][0][7:])
print("|------------------------------|")
#print(train['isrc'].str.extract('(.{2,2})' * 4))
isrc_cols = ['isrc_c','isrc_rc','isrc_yr','isrc_ud']
train['isrc_c'],train['isrc_rc'],train['isrc_yr'],train['isrc_ud'] = map(train['isrc'].str.slice, [0, 2, 5,7], [2, 5, 7,12])
train[isrc_cols] = train[isrc_cols].fillna(value=-1)

train['isrc_c'] = train['isrc_c'].astype('category').cat.codes
train['isrc_rc'] = train['isrc_rc'].astype('category').cat.codes
train['isrc_yr'] = train['isrc_yr'].astype("int")
train['isrc_ud'] =train['isrc_ud'].astype('category').cat.codes
#print([i[0:2] for i in train[isrc].values])




print(train['genre_ids'].value_counts())
genres_new = train['genre_ids'].str.split('|', expand=True)
genres_new = genres_new.fillna(value=-1)
genres_new.columns = ["genre_"+str(c) for c in genres_new.columns]

train[genres_new.columns] = genres_new.astype("int64")




train_objects = train.columns.to_series().groupby(train.dtypes).groups
print(train_objects)




train = train.drop(['genre_ids','isrc'],axis=1)
print(train.columns.values)




## Feature Engineering ###
# Days since registration
diff_days = (train['expiration_date']-train['registration_init_time']).dt.days
train['diff_days'] = diff_days




# Number of Genres 
train['num_genres'] = (train[genres_new.columns]!=-1).sum(axis=1)




# Split song length
train['song_length_bins'] = pd.qcut(train['song_length'],10,labels=range(10))




# Age bins

## Adjust ages ##
#print(train.loc[train['bd'] >90,'bd'])
print("| Ages less than 16: ", train.loc[train['bd'] <-1,'bd'].shape[0])
print("| Ages greater than 90: ", train.loc[train['bd'] >90,'bd'].shape[0])
train.loc[train['bd'] >90,'bd'] = -1
train.loc[train['bd'] <10,'bd']= -1
#train['bd_age'] = pd.qcut(train['bd'],bins,labels=False)




var = 'artist_name'
random_user = train['msno'].iloc[0]
random_user_data = train[train['msno']==random_user]
random_user_data = random_user_data.drop(["msno","song_id",'registration_init_time', 'expiration_date'],axis=1)
print(random_user_data.columns.values)
print("| Correlation study |")
print(random_user_data.corrwith(random_user_data['target'],axis=0).sort_values(ascending=False))
tar_0 = random_user_data[random_user_data['target']==0]
tar_1 = random_user_data[random_user_data['target']==1]
val_0_var =tar_0[var].value_counts()
val_0_var = val_0_var/val_0_var.sum(axis=0)
val_1_var = tar_1[var].value_counts()
val_1_var = val_1_var/val_1_var.sum(axis=0)
comp = pd.concat([val_0_var, val_1_var],axis=1,names=['0','1'])
comp.columns = ['target 0','target 1']
comp['absdiff'] = abs(comp['target 0'] - comp['target 1'])
print()
print("| ",var," vs. target |")
print(comp.sort_values(by="absdiff",ascending=False))
#plt.scatter(random_user_data[var],random_user_data['target'])
print(train[var].value_counts())




'''
## Run V fold CV for XGBOOST ##
kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X_data, y_data)):
    print("fold ",i," of ",kfold)
    X_train, X_valid = X_data[train_index, :], X_data[test_index, :]
    y_train, y_valid = y_data[train_index],    y_data[test_index]
    
    
    d_train = xgb.DMatrix(X_train, y_train,missing=-999.0)
    d_valid = xgb.DMatrix(X_valid, y_valid,missing=-999.0)
    #d_test = xgb.DMatrix(test_data)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Model XGBOOST
    print(" - model")
    xgb_mdl = xgb.train(xgb_params, d_train,100, watchlist, early_stopping_rounds=100, feval='auc', maximize=True,
                        verbose_eval=1)
    print(" - predict")
    xgb_predict = xgb_mdl.predict(xgb.DMatrix(X_valid),ntree_limit=xgb_mdl.best_ntree_limit)
'''

