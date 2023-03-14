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




def resize_data(dataset): 
    dataset.replace(' NA', -99, inplace=True)
    dataset.fillna(-99, inplace=True)
    
    for col in list(dataset.columns):
        if dataset[col].dtype == 'int64' or dataset[col].dtype == 'float64':
            dataset[col] = dataset[col].astype(np.int8)    
                
    return dataset




reader = pd.read_csv('../input/train_ver2.csv', chunksize=10000)
df = pd.concat([resize_data(chunk) for chunk in reader])




df.





























































