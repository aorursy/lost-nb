#!/usr/bin/env python
# coding: utf-8

# In[1]:


import hashlib
import pandas as pd


file_names = open('../input/category_names.csv', 'rb').read()
md5_names  = hashlib.md5(file_names).hexdigest()
sha1_names = hashlib.sha1(file_names).hexdigest()
del file_names

file_subex = open('../input/sample_submission.csv', 'rb').read()
md5_subex  = hashlib.md5(file_subex).hexdigest()
sha1_subex = hashlib.sha1(file_subex).hexdigest()
del file_subex

res_cmd = get_ipython().getoutput('md5sum ../input/test.bson')
md5_test   = str(res_cmd).split('\'')[1].split(' ')[0]
res_cmd = get_ipython().getoutput('sha1sum ../input/test.bson')
sha1_test  = str(res_cmd).split('\'')[1].split(' ')[0]

res_cmd = get_ipython().getoutput('md5sum ../input/train.bson')
md5_train  = str(res_cmd).split('\'')[1].split(' ')[0]
res_cmd = get_ipython().getoutput('sha1sum ../input/train.bson')
sha1_train = str(res_cmd).split('\'')[1].split(' ')[0]

file_ex    = open('../input/train_example.bson', 'rb').read()
md5_ex     = hashlib.md5(file_ex).hexdigest()
sha1_ex    = hashlib.sha1(file_ex).hexdigest()
del file_ex

df = pd.DataFrame([
    pd.Series([md5_names, sha1_names], index=['MD5', 'SHA-1']),
    pd.Series([md5_subex, sha1_subex], index=['MD5', 'SHA-1']),
    pd.Series([md5_test,  sha1_test],  index=['MD5', 'SHA-1']),
    pd.Series([md5_train, sha1_train], index=['MD5', 'SHA-1']),
    pd.Series([md5_ex,    sha1_ex],    index=['MD5', 'SHA-1']),
])

df.index = ['category_names.csv', 'sample_submission.csv', 'test.bson', 'train.bson', 'train_example.bson']
df.to_csv('hashsum_cdiscount.csv')
df

