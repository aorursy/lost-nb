#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd
import gc

# Lots of stuff in here copied from other kernels. Not sure who to credit because I saw them in 
# multiple places but I'll try to mark them


# Specify data types for the columns to save space
# Taken from a thread on advice for saving memory
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

# Read input, use dask to sample a fraction rather than doing first n rows
data = dd.read_csv('../input/train.csv', dtype=dtypes, 
                  usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']).sample(0.1).compute()

train_len = len(data)
print(train_len)




# Load test data, append to train
# Use this larger collection for counting features by IP and other columns
# Idea taken from some other kernel
test_data = pd.read_csv("../input/test.csv", dtype=dtypes)
data = data.append(test_data)
del test_data
gc.collect()

print("Total rows: {:d}".format(len(data)))




data['click_time'] = dd.to_datetime(data['click_time'])
data['hour'] = data['click_time'].dt.hour
data['minute'] = data['click_time'].dt.minute
data['second'] = data['click_time'].dt.second
data['day'] = data['click_time'].dt.day

data = data.drop(columns=['click_time'])




import matplotlib.pyplot as plt

data.hist('hour', by='is_attributed', bins=24, normed=True)




data.hist('minute', by='is_attributed', bins=60, normed=True)




data.hist('second', by='is_attributed', bins=60, normed=True)




data['minute_0'] = (data['minute']==0)

gp = data[['ip', 'minute_0']].groupby(by=['ip']).count().reset_index().rename(columns={'minute_0':'minute_0_count'})
data = data.merge(gp, on='ip', how='left')
del gp
gc.collect()




data[data.minute_0_count > 10000].hist('minute_0_count', by='is_attributed', bins=10, normed=True)




# Smallest possible frequency this could have been scheduled for if on a repeating schedule: 60/gcd(60,x)
minInterval = {0: 1.0, 1: 60.0, 2: 30.0, 3: 20.0, 4: 15.0, 5: 12.0, 6: 10.0, 7: 60.0, 8: 15.0, 9: 20.0, 10: 6.0, 11: 60.0,
12: 5.0, 13: 60.0, 14: 30.0, 15: 4.0, 16: 15.0, 17: 60.0, 18: 10.0, 19: 60.0, 20: 3.0, 21: 20.0, 22: 30.0, 23: 60.0, 24: 5.0, 
25: 12.0, 26: 30.0, 27: 20.0, 28: 15.0, 29: 60.0, 30: 2.0, 31: 60.0, 32: 15.0, 33: 20.0, 34: 30.0, 35: 12.0, 36: 5.0, 37: 60.0, 
38: 30.0, 39: 20.0, 40: 3.0, 41: 60.0, 42: 10.0, 43: 60.0, 44: 15.0, 45: 4.0, 46: 30.0, 47: 60.0, 48: 5.0, 49: 60.0, 50: 6.0, 
51: 20.0, 52: 15.0, 53: 60.0, 54: 10.0, 55: 12.0, 56: 15.0, 57: 20.0, 58: 30.0, 59: 60.0}
data['minute_interval'] = data['minute'].map(minInterval)




gp = data[['ip','os','app','minute_interval']].groupby(by=['ip','os','app']).mean().reset_index().rename(columns={'minute_interval':'minute_interval_avg'})
data = data.merge(gp, on=['ip','os','app'], how='left')
del gp
gc.collect()




data.hist('minute_interval', by='is_attributed')
data.hist('minute_interval_avg', by='is_attributed')

