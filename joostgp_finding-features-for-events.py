# This Python 3 environment comes with many he
lpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv('../input/gender_age_train.csv')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
events = pd.read_csv('../input/events.csv', parse_dates=['timestamp'], date_parser=dateparse)

# Here I only care about devices with events -> inner join
train = pd.merge(train, events, how='inner', on='device_id')
def unpack_date(df, date_col):
    # Create columns for elements of date_col
    di=pd.DatetimeIndex(df[date_col])
    
    df[date_col+'_h'] = di.hour
    df[date_col+'_d'] = di.day
    
    return df
train = unpack_date(train, 'timestamp')

train.shape


