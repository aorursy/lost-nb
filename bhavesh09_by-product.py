#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from plotnine import *
import warnings
warnings.filterwarnings('ignore') 




train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')




train_transaction[['ProductCD','isFraud']] = train_transaction[['ProductCD','isFraud']].astype('category')




#train_transaction['cents'] = train_transaction.TransactionAmt.astype('str').str.split('.').str[1].astype('int8')
train_transaction['cents'] = np.round( train_transaction['TransactionAmt'] - np.floor(train_transaction['TransactionAmt']),2 )




train_transaction['cents'].describe()




ggplot(train_transaction, aes(x='TransactionDT',y='cents', color='isFraud'))+geom_point(size=0.1, alpha=0.3) +facet_wrap( facets='ProductCD', nrow=5) +theme(figure_size=(12, 20))




train_transaction.groupby('ProductCD')[['dist1','dist2']].count()




df = train_transaction.groupby('ProductCD').count().reset_index()
df = df.set_index('ProductCD').stack().reset_index(name='Value')
df['Value'] = df['Value'] / df['ProductCD'].map(train_transaction['ProductCD'].value_counts()).astype('int')




ggplot(df, aes(x='level_1',y='Value'))+geom_bar(stat='identity')+facet_wrap( facets='ProductCD', nrow=5) +theme(figure_size=(12, 20))






