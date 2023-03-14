#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd

def pdf(x, mean, cov):
    """Probability density for a multivariate normal distribution
    with given mean and covariance.
    """
    return np.linalg.det(2*np.pi*cov)**(-1/2) *        np.exp(-1/2 * (x-mean).T.dot(np.linalg.inv(cov)).dot(x-mean))




df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# As other answers have pointed out, `color` appears to be mostly noise
df_train = df_train.drop(['id', 'color'], axis=1)




# Calculate means and covariances by type
mean = df_train.groupby('type').mean()
cov = df_train.groupby('type').cov()

def classify(x):
    """Given a 4d input, find the monster type whose density is greatest."""
    pdfs = {}
    for monster_type in ('Ghost', 'Ghoul', 'Goblin'):
        # Note: For whatever reason, applying cov ends up shuffling the columns
        monster_cov = cov.T[monster_type].as_matrix()[[0, 3, 1, 2]].T[[0, 3, 1, 2]].T
        pdfs[monster_type] = pdf(x, mean.T[monster_type], monster_cov)
    return max(pdfs, key=pdfs.get)




# Let's check if this procedure makes any sense at all on the training set
df_train['prediction'] = df_train[df_train.columns[:4]].apply(classify, axis=1)
len(df_train[df_train.type == df_train.prediction])/len(df_train)




# That could have been a lot worse! Let's move on to the test data.
df_test['type'] = df_test[df_test.columns[1:5]].apply(classify, axis=1)
df_test[['id', 'type']].to_csv('submission.csv', index=False)






