#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




#point this to your data path where you download your data
path = "../input/"

#taken from https://www.kaggle.com/orangutan/keras-vgg19-starter
df_train = pd.read_csv(path+"labels.csv")
df_test = pd.read_csv(path+"sample_submission.csv")
targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse=True)




get_ipython().system('mkdir {path+"valid"}')




#creates all of the directories you will need
for i in range(targets_series.unique().shape[0]):
    get_ipython().system('mkdir {path+"train/"+targets_series.unique()[i]}/')
    get_ipython().system('mkdir {path+"valid/"+targets_series.unique()[i]}/')




#Change valid percent to how many of your files you want in a validation set
valid_percent = 20
for i in range(df_train.shape[0]):
    if i%100>=valid_percent:
        get_ipython().system('mv {path+"train/"+df_train["id"][i]+".jpg"} {path+"train/"+df_train["breed"][i]+"/"}')
    else:
        get_ipython().system('mv {path+"train/"+df_train["id"][i]+".jpg"} {path+"valid/"+df_train["breed"][i]+"/"}')
    if i%1000==0:
        print(i)

