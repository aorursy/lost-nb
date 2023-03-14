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

# Any results you write to the current directory are saved as output.




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
plt.style.use('ggplot')




print(os.listdir('../input'))




folder_path = '../input/ieee-fraud-detection/'
train_id = pd.read_csv(f'{folder_path}train_identity.csv')
train_tr = pd.read_csv(f'{folder_path}train_transaction.csv')
test_id = pd.read_csv(f'{folder_path}test_identity.csv')
test_tr = pd.read_csv(f'{folder_path}test_transaction.csv')
#sub = pd.read_csv(f'{folder_path}sample_submission.csv')
# let's combine the data and work with the whole dataset
train = pd.merge(train_tr, train_id,on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')




train_id.columns




train_tr.columns




train_id.head()




train_tr.head()




train.head()




train.shape[0]




train['isFraud'].mean()




train['DeviceType'].fillna('nan').value_counts()




train['DeviceType'] = train['DeviceType'].fillna('nan')
train.groupby(['DeviceType'])['isFraud'].mean()




(train[train['isFraud']==1]['DeviceType']=='mobile').astype(int).sum() ,(train[train['isFraud']==1]['DeviceType']=='desktop').astype(int).sum()




train['isFraud'].mean()




(train['isFraud']==1).astype(int).sum()-5657-5554




train_mobile = train[train['DeviceType']=='mobile']
train_desktop = train[train['DeviceType']=='desktop']
train_nan = train[train['DeviceType']=='nan']




train_mobile.groupby('ProductCD')['isFraud']     .mean()     .sort_index()     .plot(kind='barh',
          figsize=(15, 3),
         title='Percentage of Fraud by ProductCD')
plt.show()




train_desktop.groupby('ProductCD')['isFraud']     .mean()     .sort_index()     .plot(kind='barh',
          figsize=(15, 3),
         title='Percentage of Fraud by ProductCD')
plt.show()




train_nan.groupby('ProductCD')['isFraud']     .mean()     .sort_index()     .plot(kind='barh',
          figsize=(15, 3),
         title='Percentage of Fraud by ProductCD')
plt.show()




train['ProductCD'].unique()




train['addr1'].unique()




train['addr2'].unique()




train.shape[0]




train['M9'].unique()




for i in range(12,39):
    print(train['id_'+str(i)].isnull().any())




for i in range(12,39):
    print(train['id_'+str(i)].nunique())




for col, values in train.iteritems():
    num_uniques = values.nunique()
    print ('{name}: {num_unique}'.format(name=col, num_unique=num_uniques))
    print (values.unique())
    print ('\n')




for i in range(1,10):
    print(train['M'+str(i)].unique())




# M1-M9: match, such as names on card and address, etc.




for i in range(1,16):
    print(train['D'+str(i)].nunique())




train['D1'].head()




train['D2'].fillna(-1.).value_counts()




train['P_emaildomain'].unique()




test['P_emaildomain'].fillna('0').value_counts()




train['P_emaildomain'].fillna('0').value_counts()




train['R_emaildomain'].unique()




train['R_emaildomain'].fillna('0').value_counts()




test['R_emaildomain'].fillna('0').value_counts()




train['P_emaildomain'].isnull().any()




train_id.columns




train_id['id_03'].unique()




train_id['id_04'].unique()




train_id.head()




train['ProductCD'].nunique()




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('V')]].isnull().sum()[:20]




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('V')]].isnull().sum()[20:40]




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('V')]].isnull().sum().index




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('V')]].isnull().sum().value_counts()




train['V103'].hist(bins=100)




(train['V2'].isnull()==train['V10'].isnull()).any()




train.head()




print(train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('V')]].head(10))




V_variables = [i for i in list(train) if 'V' in i]
na_value = train[V_variables].isnull().sum()
na_list = na_value.unique()
na_value = na_value.to_dict()
cols_same_null = []
for i in range(len(na_list)):
    cols_same_null.append([k for k,v in na_value.items() if v == na_list[i]])
print(cols_same_null)




train['D15'].unique()




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==0].head(10)




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==0].head(30)




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==1].head(30)




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==0]['D1'].describe()




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==1]['D1'].describe()




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==1]['D2'].describe()




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==0]['D2'].describe()




train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]][train_tr.isFraud==0]['D2'].value_counts()




train_tr['card4'].isnull().sum()




for col1 in train_tr.columns[train_tr.columns.str.startswith('D')]:
    for col2 in train_tr.columns[train_tr.columns.str.startswith('D')]:
        if col1==col2:
            continue
        else:
            tmp = train_tr[train_tr[col1].isnull()]
            k = tmp[col2].isnull().sum() - tmp.shape[0]
            if abs(k)<50 :
                print('%s -> %s %d'%(col1,col2,k))
            




tmp = train_tr[train_tr[col1].isnull()]
k = tmp[col2].isnull().sum() - tmp.shape[0]




for col1 in test_tr.columns[test_tr.columns.str.startswith('D')]:
    for col2 in test_tr.columns[test_tr.columns.str.startswith('D')]:
        if col1==col2:
            continue
        else:
            tmp = test_tr[test_tr[col1].isnull()]
            k = tmp[col2].isnull().sum() - tmp.shape[0]
            if abs(k)<50 :
                print('%s -> %s %d'%(col1,col2,k))
            




k= train_tr.loc[:,train_tr.columns[train_tr.columns.str.startswith('D')]]




k_F =  k[-train_tr['D8'].isnull()][train.isFraud==1]
k_NF =  k[-train_tr['D8'].isnull()][train.isFraud==0]




k_NF['D9'].hist()




k_F['D9'].hist()




k_F['D8'].hist()




k_NF['D8'].hist()




plt.scatter(x=k_F['D1'],y=k_F['D2'])




plt.scatter(x=k_NF['D1'],y=k_NF['D2'])




sns.heatmap(k.corr())




plt.scatter(x=k_NF['D2'],y=k_NF['D4'])




plt.scatter(x=k_F['D2'],y=k_F['D4'])




plt.scatter(x=k_F['D1'],y=k_F['D4'])




plt.scatter(x=k_NF['D1'],y=k_NF['D4'])




tmp = train.corr()




sns.heatmap(train.corr())




plt.figure(figsize=(100,100))
sns.heatmap(tmp)




['V305','V107']




corr_no_V = train.drop(train_tr.columns[train_tr.columns.str.startswith('V')],axis=1).corr()




plt.figure(figsize=(32,32))
sns.heatmap(corr_no_V)






