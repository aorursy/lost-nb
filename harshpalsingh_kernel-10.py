#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
train.head()




train.shape




train.describe()




train.dtypes




train["Date"] = pd.to_datetime(train["Date"])




train.dtypes




num_countries = len(train.Country_Region.unique())
min_date = min(train.Date)
max_date = max(train.Date)

print("COVID19 has affected {a} countries between {b} and {c}".format(a = num_countries, b = min_date, c = max_date))




#Global confirmed cases and fatalities
worldwide_cases = train.groupby('Date')['ConfirmedCases', 'Fatalities'].sum().reset_index()
print(worldwide_cases.head())

plt.figure(figsize=(12, 8))
ax = plt.gca()

worldwide_cases.plot(x = "Date", y = "ConfirmedCases", kind = "line", color = 'blue', ax = ax)
worldwide_cases.plot(x = "Date", y = "Fatalities", kind = "line", color = 'red', ax = ax)
plt.show()




#Confirmed cases and Fatalities by Country
cases_by_country = train[train['Date'] == max_date].groupby('Country_Region')['ConfirmedCases', 'Fatalities'].sum().reset_index()
cases_by_country.head()




most_cases = cases_by_country.sort_values('ConfirmedCases', ascending = False)
most_cases.head(10)




most_fatalities = cases_by_country.sort_values("Fatalities", ascending = False)
most_fatalities.head(10)




train[train["Country_Region"] == "China"].head()




len(train[train["Country_Region"] == "China"])




len(train["Province_State"][train["Country_Region"] == "China"].unique())




china_cases = train[train['Country_Region'] == 'China'].groupby('Date')["ConfirmedCases", "Fatalities"].sum().reset_index()
china_cases.head()




plt.figure(figsize=(12,8))
ax = plt.gca()
 
china_cases.plot(kind='line', x='Date', y='ConfirmedCases', color='red', ax=ax)
china_cases.plot(kind='line', x='Date', y='Fatalities', color='blue', ax=ax)
plt.show()




india_cases = train[train["Country_Region"] == "India"].groupby('Date')['ConfirmedCases', 'Fatalities'].sum().reset_index()
india_cases.tail()




plt.figure(figsize = (12,8))
ax = plt.gca()

india_cases.plot(kind = 'line', x = "Date", y = "ConfirmedCases", color = 'blue', ax = ax)
india_cases.plot(kind = 'line', x = "Date", y = "Fatalities", color = 'red', ax = ax)
plt.show()




def apply_log(x):
    try:
        return (np.log(x+1))
    except:
        return (0)

india_cases['log_ConfirmedCases'] = india_cases['ConfirmedCases'].apply(apply_log)
india_cases['log_Fatalities'] = india_cases['Fatalities'].apply(apply_log)




print(india_cases.tail())

plt.figure(figsize = (12,8))
ax = plt.gca()

india_cases.plot(kind = 'line', x = "Date", y = "log_ConfirmedCases", color = 'blue', ax = ax)
india_cases.plot(kind = 'line', x = "Date", y = "log_Fatalities", color = 'red', ax = ax)
plt.show()




train["Country_State"] = train['Country_Region'] + ('-' + train['Province_State']).fillna('')
train["log_ConfirmedCases"] = np.log(train['ConfirmedCases'] + 1)
train["log_Fatalities"] = np.log(train['Fatalities'] + 1)

train.sample(10)




train['T'] = (train['Date'] - min_date).dt.days + 1
train.head()




cases_df = train[train['log_ConfirmedCases'] != 0]
cases_df.head()




fatalities_df = train[train['log_Fatalities'] != 0]
fatalities_df.head()




from scipy import stats




#Regression of T against log_Confirmed Cases

Confirmedcases_regress_df = pd.DataFrame.from_dict({y:np.polyfit(x['T'],x['ConfirmedCases'],3) for y, x in train.groupby('Country_State')},'index').     rename(columns={0:'power3_c',1:'power2_c',2:'power1_c', 3:'intercept_c'})

Confirmedcases_regress_df = Confirmedcases_regress_df.rename_axis('Country_State').reset_index()

#Confirmedcases_regress_df.rename(columns = {'Slope':'Slope_ConfirmedCases', 'Intercept':'Intercept_ConfirmedCases'}, inplace = True) 
Confirmedcases_regress_df [:10]




#Regression of T against log_Fatalities

Fatalities_regress_df = pd.DataFrame.from_dict({y:np.polyfit(x['T'],x['Fatalities'],3) for y, x in train.groupby('Country_State')},'index').     rename(columns={0:'power3_f',1:'power2_f',2: 'power1_f', 3:'intercept_f'})

Fatalities_regress_df = Fatalities_regress_df.rename_axis('Country_State').reset_index()
#Fatalities_regress_df.rename(columns = {'Slope':'Slope_Fatalities', 'Intercept':'Intercept_Fatalities'}, inplace = True) 
Fatalities_regress_df [-10:]




#Test data
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
test.head()




test["Date"] = pd.to_datetime(test["Date"])
test["Country_State"] = test['Country_Region'] + ('-' + test['Province_State']).fillna('')
test['T'] = (test['Date'] - min_date).dt.days + 1
test.head()




test_1 = pd.merge(test, Confirmedcases_regress_df, on = 'Country_State', how = 'left')
test_1.head()




test_final = pd.merge(test_1, Fatalities_regress_df, on = 'Country_State', how = 'left')
test_final.head(5)




test_final['ConfirmedCases'] = test_final['intercept_c'] + (test_final['power1_c'] * test_final['T']) + (test_final['power2_c'] * test_final['T'] * test_final['T']) + (test_final['power3_c'] * test_final['T'] * test_final['T'] * test_final['T'])
test_final['Fatalities'] = test_final['intercept_f'] + (test_final['power1_f'] * test_final['T']) + (test_final['power2_f'] * test_final['T'] * test_final['T']) + (test_final['power3_f'] * test_final['T'] * test_final['T'] * test_final['T'])
test_final.sample(10)




test_final['ConfirmedCases'][test_final['ConfirmedCases'] < 0] = 0
test_final['Fatalities'][test_final['Fatalities'] < 0] = 0




test_final['ConfirmedCases'] = round(test_final['ConfirmedCases'], 0)
test_final['Fatalities'] = round(test_final['Fatalities'], 0)




for country in test_final['Country_State'].unique():
    i = test_final[test_final['Country_State'] == country].index.values[0]
    #print(country, i)
    for i in range(i, i+42):
        if (test_final.loc[i, "ConfirmedCases"] > test_final.loc[i+1, "ConfirmedCases"]):
            test_final.loc[i+1, "ConfirmedCases"] = test_final.loc[i, "ConfirmedCases"]
        else:
            pass




header = ['ForecastId', 'ConfirmedCases', 'Fatalities']
test_final.to_csv("submission.csv", columns = header, index=False)






