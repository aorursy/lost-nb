# load the files
from graphlab import SFrame
import graphlab as gl

Train = SFrame(data='~/Documents/Kaggle/CrimeClassification/train.csv')
Test = SFrame(data='~/Documents/Kaggle/CrimeClassification/test.csv')


import numpy as np
import datetime as dt
import pandas as pd

DayOfWeekList = list(set(Train['DayOfWeek']))
DayOfWeekDict = dict(zip(DayOfWeekList,xrange(7)))
PdList = list(set(Train['PdDistrict']))
PdDict = dict(zip(PdList,xrange(len(PdList))))

CrimeList = list(set(Train['Category']))
CrimeDict = dict(zip(CrimeList,xrange(len(CrimeList))))

import datetime as dt

# create and train the model
TrainData = Train
TrainData['Hour'] = TrainData['Dates'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour)
train_data, validation_data = TrainData.random_split(0.8)

model = gl.random_forest_classifier.create(train_data,
                                           features = ['DayOfWeek','Hour','PdDistrict','Address','X','Y'],
                                           target='Category',
                                           validation_set = validation_data,
                                           num_trees=10,
                                           max_depth = 5)


# make the predictions

ResultRandomForest = model.classify(Test)

print ResultRandomForest[1]
def Result2CSV(result):
    prob = result['probability']
    category = result['class']
    remainProb = (1-prob)/40
    Dict = dict(zip(CrimeList,remainProb*np.ones(len(CrimeList))))
    Dict[category] = prob
    return Dict

Sub2CSV = ResultRandomForest.apply(lambda x: Result2CSV(x))

test = gl.SFrame(Sub2CSV)
test = test.unpack('X1',column_name_prefix="")
test['Id'] = Test['Id']
test.save('~/Documents/Kaggle/CrimeClassification/submission.csv',format='csv')
