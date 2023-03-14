#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scikit-learn as sk # scikit learn, regression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




import json
data = dict(json.load(open('../input/train.json')))
#print(data.keys())

from matplotlib import pyplot as plt

beds = list(data['price'].items())
interest = list(data['interest_level'].items())

ls = []
il = []
for i, k in enumerate(interest):
    if interest[i][1]=='low':
        il.append(1)
        ls.append(int(beds[i][1]))
    elif interest[i][1]=='medium':
        il.append(2)
        ls.append(int(beds[i][1]))
    else:
        il.append(3)
        ls.append(int(beds[i][1]))

#print(len(ls))
plt.hist(ls, alpha=0.5)
plt.show()
plt.scatter(ls,il,s=np.array(ls)/1000,c=il, alpha=0.5)
plt.show()




data = pd.read_json('../input/train.json')
data.drop(['building_id','created','latitude','longitude','listing_id','manager_id','photos'],
          axis=1, inplace=True)
#dat = pd.DataFrame(data[data['price'] >= 10000])
#print(dat)

data.head()




import re
#print ([d['features'] for d in data.iterrows()])
features = []
for index, row in data.iterrows():
    features += (row['features'])

features = list(set(features))

feature_set = []
for feature in features:
    feature_set += ([f for f in re.split(re.compile('[\+\-//\*\<\>\%\(\)]'), feature) if f!=''])
    
#print (feature_set)
feature_set = list(set(feature_set))

feature_set += list(set(list(data)))

print(feature_set)




new_data = 

