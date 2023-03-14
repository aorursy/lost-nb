#!/usr/bin/env python
# coding: utf-8



#Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt




def corr_plot(dataframe, top_n, target, fig_x, fig_y):
    corrmat = dataframe.corr()
    #top_n - top n correlations +1 since price is included
    top_n = top_n + 1 
    cols = corrmat.nlargest(top_n, target)[target].index
    cm = np.corrcoef(dataa[cols].values.T)
    return cols,cm


dataa = pd.read_csv("./input/train.csv")
dataa2 = pd.read_csv("./input/macro.csv")

dataa = pd.merge(dataa, dataa2, on='timestamp')

del dataa["timestamp"]

dataa3 = dataa


#Treating missing data
total = dataa.isnull().sum().sort_values(ascending=False) #Calculate the total of the missing values for dataa
percent = (dataa.isnull().sum()/dataa.isnull().count()).sort_values(ascending=False) #Calculate the percentage of missing values for each variable of dataa
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



dataa = dataa.drop((missing_data[missing_data['Total'] > 10000]).index,1)

dataa = dataa.dropna(thresh=dataa.shape[1])

print dataa.shape

dtype_df = dataa.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

tab = []

for x in range(0,dtype_df.shape[0]):
    if(dtype_df["Column Type"][x] == "object"):
        tab.append(dtype_df["Count"][x])
for x in range(0,15):
    dataa[tab[x]] = pd.factorize(dataa[tab[x]])[0]





corr_20,cm = corr_plot(dataa, 150, 'price_doc', 10,10)

corr_20 = corr_20[0:6]


dataa3 = dataa3[corr_20].copy()

test = pd.read_csv("./input/test.csv")

dataa2 = pd.read_csv("./input/macro.csv")

test = pd.merge(test, dataa2, on='timestamp')

del test["timestamp"]
print test.shape
test = test[corr_20[1:corr_20.shape[0]]].copy()


total = dataa3.isnull().sum().sort_values(ascending=False) #Calculate the total of the missing values for dataa3
percent = (dataa3.isnull().sum()/dataa3.isnull().count()).sort_values(ascending=False) #Calculate the percentage of missing values for each variable of dataa3 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


dataa3 = dataa3.drop((missing_data[missing_data['Total'] > 10000]).index,1)



dataa3 = dataa3.dropna(thresh=dataa3.shape[1])
dtype_df = dataa3.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

tab = []

for x in range(0,dtype_df.shape[0]):
    if(dtype_df["Column Type"][x] == "object"):
        tab.append(dtype_df["Count"][x])
        #print dtype_df["Count"][x]

for x in range(0,len(tab)):
    dataa3[tab[x]] = pd.factorize(dataa3[tab[x]])[0]
    test[tab[x]] = pd.factorize(test[tab[x]])[0]

total = test.isnull().sum().sort_values(ascending=False) #Calculate the total of the missing values for test
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False) #Calculate the percentage of missing values for each variable of test 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


test  = test.fillna(test.mean())


total = test.isnull().sum().sort_values(ascending=False) #Calculate the total of the missing values for test
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False) #Calculate the percentage of missing values for each variable of test 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print missing_data



dataa3 = dataa3[dataa3.price_doc > 100]
dataa3 = dataa3[dataa3.price_doc < 0.2e8]





price = dataa3.price_doc
del dataa3["price_doc"]
dataa3 = dataa3[corr_20[1:corr_20.shape[0]]].copy()


modeleReg=LinearRegression()


modeleReg.fit(dataa3,price) #Make a linear regression
y_predicted = modeleReg.predict(test)
#print(y_test)

id_test = range(30474,38136)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predicted})
output.head()
output.to_csv('submission.csv', index=False) #Submission file

