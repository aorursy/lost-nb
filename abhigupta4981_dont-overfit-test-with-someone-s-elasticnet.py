#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet




get_ipython().run_line_magic('pinfo2', 'ElasticNet')




PATH = Path('../input')




train = pd.read_csv(PATH/'train.csv')
test = pd.read_csv(PATH/'test.csv').drop(columns=['id'])




train_Y = train['target']
train_X = train.drop(columns=['target', 'id'])




best_parameters = {
    'alpha': 0.2,
    'l1_ratio': 0.31,
    'precompute': True,
    'selection': 'random',
    'tol': 0.001, 
    'random_state': 2
}




net = ElasticNet(**best_parameters)
net.fit(train_X, train_Y)




sub = pd.read_csv(PATH/'sample_submission.csv')
sub['target'] = net.predict(test)




sub.head()




sub.to_csv('submission.csv', index=False)




FileLink('submission.csv')






