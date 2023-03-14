#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


from sklearn.model_selection import train_test_split
import random
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


# In[3]:


Input = '../input/covid19-global-forecasting-week-3'
covid_train = pd.read_csv(f'{Input}/train.csv')
covid_train = covid_train.rename(columns={'Country_Region':'Country'})


# In[4]:


covid_train['Country'].nunique()


# In[ ]:





# In[5]:


covid_lat = pd.read_csv('/kaggle/input/time-series-covid-19-confirmedcsv/time_series_covid_19_confirmed.csv')


# In[6]:


col = ['Country/Region','Lat', 'Long']
covid_lat = covid_lat[col]


# In[7]:


covid_lat = covid_lat.rename(columns={'Country/Region':'Country'})


# In[8]:


covid_lat.info()


# In[9]:


# result_lat = [x for x in ]


# In[10]:


# df = pd.merge(covid_train, covid_lat[['Lat', 'Long']], on='Country_Region', how='left')


# In[11]:


# covid_train.merge(covid_lat,left_on='Country_Region',right_on='Country/Region')


# In[12]:


# result = pd.merge(covid_train, covid_lat.unique(), how='right', on='Country')


# In[13]:


# result.info()


# In[14]:


covid_train['Lat'] = np.nan
covid_train['Long'] = np.nan
covid_train['Popu_dens'] = np.nan
covid_train['life_expectacy'] = np.nan


# In[15]:


covid_train.nunique()


# In[16]:


# count = 0
# for country in covid_train['Country'][:10]:
#     for country1 in covid_lat['Country'][:10]:
#         if country == country1:
#             count += 1
# print(count)


# In[17]:


country_list = covid_lat['Country'].unique()


# In[18]:


country_list


# In[ ]:





# In[19]:


covid_train.head()


# In[20]:


# what about so many countries in the covid train, how to get them


# In[21]:


count = 0
for country in covid_train['Country']:
    for country1 in country_list:
        if country == country1:
#             print("Comutry", country)
#             print("Comutry1", country1)
            covid_train['Lat'][count] = covid_lat['Lat'][covid_lat['Country']== country].iloc[0]
            covid_train['Long'][count] = covid_lat['Long'][covid_lat['Country']== country].iloc[0]
            count += 1
    
print(count)


# In[22]:


covid_lat['Lat'].nunique


# In[23]:


covid_train['Lat'].nunique()


# In[24]:


covid_train.info()


# In[25]:


# for country1 in country_list:
#     covid_lat['Lat']&covid_lat['Country'== country1]


# In[26]:


# [lat for lat in covid_lat['Lat'] if covid_train['Country']==covid_lat['Country']]


# In[27]:


# covid_lat['Lat'].where("n")


# In[28]:


#  result = pd.merge(covid_train, covid_lat, how='left', on=['Country'])


# In[29]:


# result['Province_State'].nunique()


# In[30]:


# covid_train['Province_State'].nunique()


# In[31]:


# covid_lat['Country'].nunique()


# In[32]:


# covid_lat['Lat'][covid_lat['Country']== 'China'].iloc[0]


# In[33]:


# covid_train['Lat'].nunique()


# In[34]:


# in covid lat there are 180 countries and 246 lat, so more latitudes than country
# in covid train there are 180 countires and 173 lat, so why are lat less than country


# In[35]:


# find rows where lat is same but not countries
# df = covid_train.groupby(['Lat', 'Country']).count()


# In[36]:


# df.index


# In[37]:


# covid_train[covid_train['Country'] == 'Syria']


# In[38]:


# covid_lat[covid_lat['Country'] == 'Syria']


# In[39]:


# covid_lat['Lat'][covid_lat['Country']== 'Syria'].iloc[0]


# In[40]:


covid_popu_dens = pd.read_csv("/kaggle/input/population-density/Popu_dens.csv",skiprows=3)
# covid_population.head()


# In[41]:


covid_popu_dens.head()


# In[42]:


col = ['Country Name','2018']
covid_popu_dens = covid_popu_dens[col]


# In[43]:


covid_popu_dens = covid_popu_dens.rename(columns={'Country Name':'Country'})


# In[44]:


covid_popu_dens.head()


# In[45]:


covid_Life_expectancy = pd.read_csv('/kaggle/input/life-expectancy/Life_Expectancy.csv',skiprows=3)
covid_Life_expectancy.head()


# In[46]:


col = ['Country Name','2017']
covid_Life_expectancy = covid_Life_expectancy[col]
covid_Life_expectancy = covid_Life_expectancy.rename(columns={'Country Name':'Country'})


# In[47]:


country_dense = covid_popu_dens['Country'].unique()
country_le = covid_Life_expectancy['Country'].unique()


# In[48]:


covid_train.head()


# In[49]:


count = 0
for country in covid_train['Country']:
    for country1 in country_dense:
        if country == country1:
#             print("Comutry", country)
#             print("Comutry1", country1)
            covid_train['Popu_dens'][count] = covid_popu_dens['2018'][covid_popu_dens['Country']== country].iloc[0]
            count += 1


# In[50]:


covid_popu_dens['2018'][covid_popu_dens['Country']== 'Afghanistan'].iloc[0]


# In[51]:


count = 0
for country in covid_train['Country']:
    for country1 in country_le:
        if country == country1:
#             print("Comutry", country)
#             print("Comutry1", country1)
            covid_train['life_expectacy'][count] = covid_Life_expectancy['2017'][covid_Life_expectancy['Country']== country].iloc[0]
            count += 1


# In[52]:


covid_Life_expectancy['2017'][covid_Life_expectancy['Country']== 'Afghanistan'].iloc[0]


# In[53]:


df_1 = covid_train.groupby(['life_expectacy', 'Country']).count()


# In[54]:


df_1.head()


# In[55]:


covid_train.head()


# In[56]:


# covi


# In[57]:


covid_train['Date'] = pd.to_datetime(covid_train['Date'])


# In[58]:


covid_train['Date'] = (pd.to_datetime(covid_train['Date'], unit='s').astype(int)/10**9).astype(int)


# In[59]:


covid_train = covid_train.drop(['Province_State'], axis=1)


# In[60]:


covid_test = pd.read_csv(f'{Input}/test.csv', parse_dates=['Date'])
covid_test = covid_test.rename(columns={'Country_Region':'Country'})


# In[61]:


covid_test['Lat'] = np.nan
covid_test['Long'] = np.nan
covid_test['Popu_dens'] = np.nan
covid_test['life_expectacy'] = np.nan


# In[62]:


count = 0
for country in covid_test['Country']:
    for country1 in country_list:
        if country == country1:
#             print("Comutry", country)
#             print("Comutry1", country1)
            covid_test['Lat'][count] = covid_lat['Lat'][covid_lat['Country']== country].iloc[0]
            covid_test['Long'][count] = covid_lat['Long'][covid_lat['Country']== country].iloc[0]
            count += 1
    
print(count)


# In[63]:


count = 0
for country in covid_test['Country']:
    for country1 in country_dense:
        if country == country1:
#             print("Comutry", country)
#             print("Comutry1", country1)
            covid_test['Popu_dens'][count] = covid_popu_dens['2018'][covid_popu_dens['Country']== country].iloc[0]
            count += 1


# In[64]:


count = 0
for country in covid_test['Country']:
    for country1 in country_le:
        if country == country1:
#             print("Comutry", country)
#             print("Comutry1", country1)
            covid_test['life_expectacy'][count] = covid_Life_expectancy['2017'][covid_Life_expectancy['Country']== country].iloc[0]
            count += 1


# In[65]:


covid_test['Date'] = (pd.to_datetime(covid_test['Date'], unit='s').astype(int)/10**9).astype(int)


# In[66]:


covid_test = covid_test.drop(['Province_State'], axis=1)


# In[67]:


covid_test.head()


# In[68]:


################################################################################################
##Start the categorical encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[69]:


categorical_cols = [cname for cname in covid_train.columns if 
                    covid_train[cname].dtype == "object"]
categorical_cols


# In[70]:


covid_train['Country'] = le.fit_transform(covid_train['Country'])


# In[71]:


covid_test['Country'] = le.fit_transform(covid_test['Country'])


# In[72]:


covid_train = covid_train.fillna(0)
covid_test = covid_test.fillna(0)


# In[73]:


################################################################################################
##Start the training phase and deal with the model


# In[74]:


cols = [col for col in covid_train.columns if col not in ['Id','ConfirmedCases','Fatalities']]
X = covid_train[cols]
y1 = covid_train['ConfirmedCases']
y2 = covid_train['Fatalities']


# In[75]:


X_test = covid_test.iloc[:,1:]


# In[ ]:





# In[76]:


from sklearn.ensemble import RandomForestRegressor


# In[77]:


rf = RandomForestRegressor(random_state = 42,bootstrap = False, max_depth= 80, max_features = 2,
                 min_samples_leaf = 5, min_samples_split = 8, n_estimators = 100)


# In[78]:


def train_and_predict(X, y, X_test):
    
    # Bundle preprocessing and modeling code in a pipeline
    rf_classifier_model = Pipeline(steps=[
                          ('model', rf)
                         ])
    
    #Fit the model
#     grid_rf_classifier_model.fit(X_train, y)
    rf_classifier_model.fit(X, y)
#     print(grid_rf_classifier.best_params_) #for finding best parameters
    
    
    #get predictions
    y_pred = rf_classifier_model.predict(X_test)
    
    y_pred = np.around(y_pred)
    y_pred = y_pred.astype(int)

    return y_pred


# In[79]:


confirmer_y_pred = train_and_predict(X, y1, X_test)
    
fatality_y_pred = train_and_predict(X, y2, X_test)


# In[80]:


pred=pd.DataFrame()
pred['ForecastId']=covid_test['ForecastId']
pred['ConfirmedCases']=confirmer_y_pred
pred['Fatalities']=fatality_y_pred
pred.to_csv('submission.csv',index=False)


# In[ ]:




