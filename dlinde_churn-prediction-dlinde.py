#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




train = pd.read_csv('../input/train_v2.csv')
transactions = pd.read_csv('../input/transactions_v2.csv')
members = pd.read_csv('../input/members_v3.csv')




#user_logs = pd.read_csv('../input/user_logs.csv')
transactions.head(1)




print(transactions.msno.nunique())
print(train.msno.nunique())




train_transactions = pd.merge(train, transactions, on='msno', how='left')
train_transactions.head(1)




composite = pd.merge(train_transactions, members, on='msno', how='left')
composite.head(1)




print(composite.msno.nunique())
composite.shape




composite['membership_expire_date'] = pd.to_datetime(composite['membership_expire_date'], format='%Y%m%d')
composite['transaction_date'] = pd.to_datetime(composite['transaction_date'], format='%Y%m%d')




mask = composite['registration_init_time'].notnull()
composite.loc[composite[mask].index, 'registration_init_time'
             ] = composite.loc[composite[mask].index, 'registration_init_time'].astype('int')




composite.head(1)




composite['registered_via'].value_counts()




def churn_rate(df, col):
    col_rate = df.groupby(col)[['is_churn']].mean().reset_index().sort_values(
        'is_churn', ascending=False)
    sns.barplot(x=col, y='is_churn', data=col_rate)
    print(plt.show())
    return col_rate




rv = churn_rate(composite,'registered_via')




rv




composite.is_churn.mean()




cancel = churn_rate(composite,'is_cancel')




cancel




composite['registration_init_time'] = pd.to_datetime(
    composite['registration_init_time'], format='%Y%m%d')




composite.head(1)




most_recent = composite['registration_init_time'].max()




composite['registration_init_weeks'] = (
    most_recent - composite.registration_init_time)/ np.timedelta64(1, 'W')


composite.head(1)




composite['registration_init_weeks'] = composite['registration_init_weeks'].round(decimals=0)




weeks = composite.groupby('registration_init_weeks')[
    ['is_churn']].mean().reset_index().sort_values('is_churn',ascending=False)
weeks 
weeks.head(10)




sns.barplot(x='registration_init_weeks', y='is_churn', data=weeks.head(10))
plt.show()




mask = (composite['registration_init_weeks']>=0)&(composite['registration_init_weeks']<=12)
composite[mask]['registration_init_weeks'].value_counts()




composite['registration_init_weeks'].value_counts(ascending=False).head(10)




fig, ax = plt.subplots(figsize=(10,6))
ax.plot_date(composite['registration_init_time'], composite['is_churn', color="blue", linestyle="-")

ax.set(xlabel='date', ylabel='churn',title='Click Rate Over Time')
ax.legend()

plt.show()




#composite['registration_init_year'] = composite.registration_init_time.dt.year
#composite['registration_init_month'] = composite.registration_init_time.dt.month
#composite['registration_init_day'] = composite.registration_init_time.dt.day
#composite = composite.drop('registration_init_time', axis=1)




#year = churn_rate(composite, 'registration_init_year')




#year




#composite.registration_init_year.value_counts()




#month = churn_rate(composite, 'registration_init_month')




#month




#day = churn_rate(composite, 'registration_init_day')




#composite.isnull().sum()




composite.shape




for col in transactions.columns:
    print(col + ' has ' + str(transactions[col].nunique()) + ' unique_values.')




#transactions = transactions.drop_duplicates()
#transactions['payment_method_id', 'payment_plan_days']




#def churn_rate(df, col):
#    col_rate = df.groupby(col)[['is_churn']].mean().reset_index().sort_values(
#        'is_churn', ascending=False)
#    sns.barplot(x=col, y='is_churn', data=col_rate)
#    print(plt.show())
#    return col_rate









#pm_id = churn_rate(transactions, 'payment_method_id')




#pm_id.dtypes




#members = pd.merge(members, train, on='msno', how='left')

transactions.msno.nunique()


#members.shape






