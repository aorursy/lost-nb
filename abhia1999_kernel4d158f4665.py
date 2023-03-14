#!/usr/bin/env python
# coding: utf-8



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler




train=pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')




train.head()




test=pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')




test.head()




train.isnull().sum()




test.isnull().sum()




submission=pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')




submission.head()




train.shape+test.shape




data = pd.concat([train, test])




data.head()




data.isnull().sum()




data['Province_State']=data['Province_State'].fillna('PS',inplace=True)




data.isnull().sum()




train_len=len(train)




data.describe()




trg=['ConfirmedCases','Fatalities']
features = ["past__{}".format(col) for col in trg]
    









for cols in data.columns:
    if (data[cols].dtype==np.number):
        continue
    data[cols]=LabelEncoder().fit_transform(data[cols])




train=data[:train_len]




train = train.drop('ForecastId',axis=1)




test=data[train_len:]




drop=['Id','ConfirmedCases','Fatalities']
test = test.drop(drop,axis=1)




train.head()




test.head()




from sklearn.linear_model import LogisticRegression




model = LogisticRegression(random_state=71,n_jobs=-1,verbose=0)




x_train=train.drop(labels=['Fatalities','ConfirmedCases','Id'],axis=1)
y_train1=train['ConfirmedCases']
y_train2=train['Fatalities']




m1=model.fit(x_train,y_train1)




m2=model.fit(x_train,y_train2)




x_test=test.drop(labels=['ForecastId'],axis=1)




pred1=m1.predict(x_test)




pred2=m2.predict(x_test)




data_to_submit = pd.DataFrame({
    'ForecastId':submission['ForecastId'],
    'ConfirmedCases':pred1,
    'Fatalities':pred2
})
data_to_submit.to_csv('submission.csv', index = False)






