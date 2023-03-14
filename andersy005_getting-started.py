#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




# Get the list of training files 
train = os.listdir('../input/train')
# Get the list of test files
test = os.listdir('../input/test')

print("Total number of training images: ",len(train))
print("Toal number of test images: ",len(test))




sample = pd.read_csv('../input/sample_submission.csv')
print(sample.shape)
sample.head()




# load training labels into a pandas dataframe
train_labels = pd.read_csv('../input/train.csv')
train_labels.head()




train_labels.info()




all_labels = train_labels['Id']
unique_labels = all_labels.unique()




print("There are {} unique IDs".format(unique_labels.shape[0]))




print("There are {} non unique IDs".format(all_labels.shape[0]))




print("Average number of labels per image {}".format(1.0*all_labels.shape[0]/train_labels.shape[0]))




all_ids = [item for sublist in list(train_labels['Id'].apply(lambda row: row.split(" ")).values) for item in sublist]
print('total of {} non-unique tags in all training images'.format(len(all_ids)))
print('average number of labels per image {}'.format(1.0*len(all_ids)/train_labels.shape[0]))




ids_counted_and_sorted = pd.DataFrame({'Id': all_labels}).groupby('Id')                            .size().reset_index().sort_values(0, ascending=False)
ids_counted_and_sorted.head(20)




from scipy.stats import bernoulli




id_probas = ids_counted_and_sorted[0].values / (ids_counted_and_sorted[0].values.sum())
indicators = np.hstack([bernoulli.rvs(p, 0, sample.shape[0]).reshape(sample.shape[0], 1) for p in id_probas])




indicators = np.array(indicators)
indicators.shape




indicators[:10,:]




sorted_ids = ids_counted_and_sorted['Id'].values
all_test_ids = []




for index in range(indicators.shape[0]):
    all_test_ids.append(' '.join(list(sorted_ids[np.where(indicators[index, :] == 1)[0]])))




len(all_test_ids)




sample['Id'] = all_test_ids
sample.head()
sample.to_csv('bernoulli_submission.csv', index=False)




get_ipython().system('ls')






