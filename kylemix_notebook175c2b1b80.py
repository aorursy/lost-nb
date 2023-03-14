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




import s2sphere as s2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss




df = pd.read_json(open("../input/train.json", "r"))




def convertHour(hour, am=False, pm=False):
    if am == True:
        if hour < 12:
            return 1
        return 0
    if pm == True:
        if hour > 12:
            return 1
        return 0
def convertDay(day, fst =False, snd=False,thrd=False):
    if fst==True:
        if day<=10:
            return 1
        return 0
    if snd==True:
        if 10<day<=20:
            return 1
        return 0
    if thrd == True:
        if day>20:
            return 1
        return 0
def countCaps(desc):
    counter = 0:
        for i in list(desc):
            if i == upper(i):
                counter +=1




df['created'] = pd.to_datetime(df.created)
df['day'] = df.created.map(lambda x: x.day)
df['hour'] = df.created.map(lambda x: x.hour)
df['am'] = df.hour.map(lambda x:convertHour(x,am=True))
df['pm'] = df.hour.map(lambda x:convertHour(x,pm=True))
df['first_tri'] = df.day.map(lambda x: convertDay(x,fst=True))
df['second_tri'] = df.day.map(lambda x: convertDay(x,snd=True))
df['third_tri'] = df.day.map(lambda x: convertDay(x,thrd=True))
df["photo_count"] = df["photos"].apply(len)




df.created[10].hour




num_featuresA = ['bathrooms', 'bedrooms','price','day','hour',"photo_count",
                'description_wrdlen','latitude','longitude']
num_featuresB = ['bathrooms', 'bedrooms','price','first_tri', 'second_tri', 'third_tri','am',
       'pm',"photo_count",
                'description_wrdlen','description_len','latitude','longitude']




df = df[['bathrooms', 'bedrooms', 'building_id', 'created', 'description',
       'display_address', 'features', 'interest_level', 'latitude',
       'listing_id', 'longitude', 'manager_id', 'photos', 'price',
       'street_address', 'description_len', 'feature_len',
       'description_wrdlen', 'day', 'hour', 'am',
       'pm', 'first_tri', 'second_tri', 'third_tri', 'high', 'low', 'medium']]




X = df[num_featuresB]
y = df["interest_level"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)




clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)
log_loss(y_val, y_val_pred)




df['description_wrdlen'] = df.description.map(lambda x: len(x.split()))
df['feature_len'] = df.features.map(len)




int_dummies = pd.get_dummies(data=df.interest_level)




df = pd.concat([df,int_dummies],axis=1)




type(df.created[10])




df.apply(lambda r: s2.CellId.from_lat_lng(s2.LatLng(r.latitude,r.longitude)).parent(15).to_token(), axis=1)




df.building_id.value_counts()[df.building_id.value_counts()>50].shape






