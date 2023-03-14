#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install pycountry_convert')




# Imports .......
import pandas as pd
import numpy as np
import datetime as dt
import pycountry
import pycountry_convert as pc
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV
from scipy.optimize import curve_fit
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error, mean_squared_error
import tensorflow.keras.layers as KL
from datetime import timedelta
import datetime
import gc
from tqdm import tqdm




df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')




display(df_train.head())
display(df_train.describe())
display(df_train.info())




df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')
df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')




print('Minimum date from training set: {}'.format(df_train['Date'].min()))
print('Maximum date from training set: {}'.format(df_train['Date'].max()))




print('Minimum date from test set: {}'.format(df_test['Date'].min()))
print('Maximum date from test set: {}'.format(df_test['Date'].max()))




class country_utils():
    def __init__(self):
        self.d = {}
    
    def get_dic(self):
        return self.d
    
    def get_country_details(self,country):
        """Returns country code(alpha_3) and continent"""
        try:
            country_obj = pycountry.countries.get(name=country)
            continent_code = pc.country_alpha2_to_continent_code(country_obj.alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj.alpha_3, continent
        except:
            if 'Congo' in country:
                country = 'Congo'
            elif country == 'Diamond Princess' or country == 'Laos' or country == 'MS Zaandam'            or country == 'Holy See' or country == 'Timor-Leste':
                return country, country
            elif country == 'Korea, South':
                country = 'Korea, Republic of'
            elif country == 'Taiwan*':
                country = 'Taiwan'
            elif country == 'Burma':
                country = 'Myanmar'
            elif country == 'West Bank and Gaza':
                country = 'Gaza'
            country_obj = pycountry.countries.search_fuzzy(country)
            continent_code = pc.country_alpha2_to_continent_code(country_obj[0].alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj[0].alpha_3, continent
    
    def get_iso3(self, country):
        return self.d[country]['code']
    
    def get_continent(self,country):
        return self.d[country]['continent']
    
    def add_values(self,country):
        self.d[country] = {}
        self.d[country]['code'],self.d[country]['continent'] = self.get_country_details(country)
    
    def fetch_iso3(self,country):
        if country in self.d.keys():
            return self.get_iso3(country)
        else:
            self.add_values(country)
            return self.get_iso3(country)
        
    def fetch_continent(self,country):
        if country in self.d.keys():
            return self.get_continent(country)
        else:
            self.add_values(country)
            return self.get_continent(country)




df_tm = df_train.copy()
df_tm = df_tm[:25500]
date = df_tm.Date.max()#get current date
df_tm = df_tm[df_tm['Date']==date]
obj = country_utils()
df_tm.Province_State.fillna('',inplace=True)
df_tm['continent'] = df_tm.apply(lambda x: obj.fetch_continent(x['Country_Region']), axis=1)
df_tm["world"] = "World" # in order to have a single root node
fig = px.treemap(df_tm, path=['world', 'continent', 'Country_Region','Province_State'], values='ConfirmedCases',
                  color='ConfirmedCases', hover_data=['Country_Region'],
                  color_continuous_scale='dense', title='Current share of Worldwide COVID19 Cases')
fig.show()




fig = px.treemap(df_tm, path=['world', 'continent', 'Country_Region','Province_State'], values='Fatalities',
                  color='Fatalities', hover_data=['Country_Region'],
                  color_continuous_scale='matter', title='Current share of Worldwide COVID19 Deaths')
fig.show()




def add_daily_measures(df):
    df.loc[0,'Daily Cases'] = df.loc[0,'ConfirmedCases']
    df.loc[0,'Daily Deaths'] = df.loc[0,'Fatalities']
    for i in range(1,len(df)):
        df.loc[i,'Daily Cases'] = df.loc[i,'ConfirmedCases'] - df.loc[i-1,'ConfirmedCases']
        df.loc[i,'Daily Deaths'] = df.loc[i,'Fatalities'] - df.loc[i-1,'Fatalities']
    #Make the first row as 0 because we don't know the previous value
    df.loc[0,'Daily Cases'] = 0
    df.loc[0,'Daily Deaths'] = 0
    return df




df_world = df_train.copy()
df_world = df_world.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_world = add_daily_measures(df_world)




fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_world['Date'], y=df_world['Daily Cases']),
    go.Bar(name='Deaths', x=df_world['Date'], y=df_world['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Worldwide daily Case and Death count')
fig.show()




df_map = df_train.copy()
df_map = df_map[:24500]
df_map['Date'] = df_map['Date'].astype(str)
df_map = df_map.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()




df_map['iso_alpha'] = df_map.apply(lambda x: obj.fetch_iso3(x['Country_Region']), axis=1)




df_map['ln(ConfirmedCases)'] = np.log(df_map.ConfirmedCases + 1)
df_map['ln(Fatalities)'] = np.log(df_map.Fatalities + 1)




px.choropleth(df_map, 
              locations="iso_alpha", 
              color="ln(ConfirmedCases)", 
              hover_name="Country_Region", 
              hover_data=["ConfirmedCases"] ,
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.dense, 
              title='Total Confirmed Cases growth(Logarithmic Scale)')




px.choropleth(df_map, 
              locations="iso_alpha", 
              color="ln(Fatalities)", 
              hover_name="Country_Region",
              hover_data=["Fatalities"],
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.OrRd,
              title = 'Total Deaths growth(Logarithmic Scale)')




#Get the top 10 countries
last_date = df_train.Date.max()
df_countries = df_train[df_train['Date']==last_date]
df_countries = df_countries.groupby('Country_Region', as_index=False)['ConfirmedCases','Fatalities'].sum()
df_countries = df_countries.nlargest(10,'ConfirmedCases')
#Get the trend for top 10 countries
df_trend = df_train.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()
df_trend = df_trend.merge(df_countries, on='Country_Region')
df_trend.drop(['ConfirmedCases_y','Fatalities_y'],axis=1, inplace=True)
df_trend.rename(columns={'Country_Region':'Country', 'ConfirmedCases_x':'Cases', 'Fatalities_x':'Deaths'}, inplace=True)
#Add columns for studying logarithmic trends
df_trend['ln(Cases)'] = np.log(df_trend['Cases']+1)# Added 1 to remove error due to log(0).
df_trend['ln(Deaths)'] = np.log(df_trend['Deaths']+1)




px.line(df_trend, x='Date', y='Cases', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries')




px.line(df_trend, x='Date', y='Deaths', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries')




px.line(df_trend, x='Date', y='ln(Cases)', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries (Logarithmic Scale)')




px.line(df_trend, x='Date', y='ln(Deaths)', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries (Logarithmic Scale)')




df_map['Mortality Rate%'] = round((df_map.Fatalities/df_map.ConfirmedCases)*100,2)




px.choropleth(df_map, 
                    locations="iso_alpha", 
                    color="Mortality Rate%", 
                    hover_name="Country_Region",
                    hover_data=["ConfirmedCases","Fatalities"],
                    animation_frame="Date",
                    color_continuous_scale=px.colors.sequential.Magma_r,
                    title = 'Worldwide Daily Variation of Mortality Rate%')




df_trend['Mortality Rate%'] = round((df_trend.Deaths/df_trend.Cases)*100,2)
px.line(df_trend, x='Date', y='Mortality Rate%', color='Country', title='Variation of Mortality Rate% \n(Top 10 worst affected countries)')




# Dictionary to get the state codes from state names for US
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}




df_us = df_train[df_train['Country_Region']=='US']
df_us['Date'] = df_us['Date'].astype(str)
df_us['state_code'] = df_us.apply(lambda x: us_state_abbrev.get(x.Province_State,float('nan')), axis=1)
df_us['ln(ConfirmedCases)'] = np.log(df_us.ConfirmedCases + 1)
df_us['ln(Fatalities)'] = np.log(df_us.Fatalities + 1)




px.choropleth(df_us,
              locationmode="USA-states",
              scope="usa",
              locations="state_code",
              color="ln(ConfirmedCases)",
              hover_name="Province_State",
              hover_data=["ConfirmedCases"],
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.Darkmint,
              title = 'Total Cases growth for USA(Logarithmic Scale)')




px.choropleth(df_us,
              locationmode="USA-states",
              scope="usa",
              locations="state_code",
              color="ln(Fatalities)",
              hover_name="Province_State",
              hover_data=["Fatalities"],
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.OrRd,
              title = 'Total deaths growth for USA(Logarithmic Scale)')




df_usa = df_train.query("Country_Region=='US'")
df_usa = df_usa.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_usa = add_daily_measures(df_usa)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_usa['Date'], y=df_usa['Daily Cases']),
    go.Bar(name='Deaths', x=df_usa['Date'], y=df_usa['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(USA)')
fig.show()




df_train.Province_State.fillna('NaN', inplace=True)
df_plot = df_train.groupby(['Date','Country_Region','Province_State'], as_index=False)['ConfirmedCases','Fatalities'].sum()




df = df_plot.query("Country_Region=='Italy'")
df.reset_index(inplace = True)
df = add_daily_measures(df)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df['Date'], y=df['Daily Cases']),
    go.Bar(name='Deaths', x=df['Date'], y=df['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count (Italy)')
fig.show()




df_train.Province_State.fillna('NaN', inplace=True)
df_plot = df_train.groupby(['Date','Country_Region','Province_State'], as_index=False)['ConfirmedCases','Fatalities'].sum()




df = df_plot.query("Country_Region=='Spain'")
df.reset_index(inplace = True)
df = add_daily_measures(df)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df['Date'], y=df['Daily Cases']),
    go.Bar(name='Deaths', x=df['Date'], y=df['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count (Spain)')
fig.show()




df = df_plot.query("Country_Region=='China'")
px.line(df, x='Date', y='ConfirmedCases', color='Province_State', title='Total Cases growth for China')




px.line(df, x='Date', y='Fatalities', color='Province_State', title='Total Deaths growth for China')




df_ch = df_train.query("Country_Region=='China'")
df_ch = df_ch.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_ch = add_daily_measures(df_ch)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_ch['Date'], y=df_ch['Daily Cases']),
    go.Bar(name='Deaths', x=df_ch['Date'], y=df_ch['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count (China)')
fig.show()




df_train.Province_State.fillna('NaN', inplace=True)
df_plot = df_train.groupby(['Date','Country_Region','Province_State'], as_index=False)['ConfirmedCases','Fatalities'].sum()




df = df_plot.query("Country_Region=='Australia'")
px.line(df, x='Date', y='ConfirmedCases', color='Province_State', title='Total Cases growth for Australia')




px.line(df, x='Date', y='Fatalities', color='Province_State', title='Total Deaths growth for Australia')




df_AU = df_train.query("Country_Region=='Australia'")
df_AU = df_AU.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_AU = add_daily_measures(df_AU)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_AU['Date'], y=df_AU['Daily Cases']),
    go.Bar(name='Deaths', x=df_AU['Date'], y=df_AU['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count (Australia)')
fig.show()




# set the session to pool available GPU
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 10} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)

# here we will train a dense network using tensorflow backend on keras as part of our regressors

# defining helper functions ...
def main_for_train(save_model_train=False, save_public_test=False):
    train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
    train['Province_State'].fillna('', inplace=True)
    train['Date'] = pd.to_datetime(train['Date'])
    train['day'] = train.Date.dt.dayofyear
    train['my_geoloc'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]
    train

    test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
    test['Province_State'].fillna('', inplace=True)
    test['Date'] = pd.to_datetime(test['Date'])
    test['day'] = test.Date.dt.dayofyear
    test['my_geoloc'] = ['_'.join(x) for x in zip(test['Country_Region'], test['Province_State'])]
    test

    day_min = train['day'].min()
    train['day'] -= day_min
    test['day'] -= day_min

    min_test_val_day = test.day.min()
    max_test_val_day = train.day.max()
    max_test_day = test.day.max()
    num_days = max_test_day + 1

    min_test_val_day, max_test_val_day, num_days

    train['ForecastId'] = -1
    test['Id'] = -1
    test['ConfirmedCases'] = 0
    test['Fatalities'] = 0

    debug = False

    data = pd.concat([train,
                      test[test.day > max_test_val_day][train.columns]
                     ]).reset_index(drop=True)
    if debug:
        data = data[data['my_geoloc'] >= 'France_'].reset_index(drop=True)
    #del train, test
    gc.collect()

    dates = data[data['my_geoloc'] == 'France_'].Date.values

    if 0:
        gr = data.groupby('my_geoloc')
        data['ConfirmedCases'] = gr.ConfirmedCases.transform('cummax')
        data['Fatalities'] = gr.Fatalities.transform('cummax')

    my_geoloc_data = data.pivot(index='my_geoloc', columns='day', values='ForecastId')
    num_my_geoloc = my_geoloc_data.shape[0]
    my_geoloc_data

    my_geoloc_id = {}
    for i,g in enumerate(my_geoloc_data.index):
        my_geoloc_id[g] = i


    ConfirmedCases = data.pivot(index='my_geoloc', columns='day', values='ConfirmedCases')
    Fatalities = data.pivot(index='my_geoloc', columns='day', values='Fatalities')

    if debug:
        cases = ConfirmedCases.values
        deaths = Fatalities.values
    else:
        cases = np.log1p(ConfirmedCases.values)
        deaths = np.log1p(Fatalities.values)


    def load_my_dataset(start_pred, num_train, lag_period):
        days = np.arange( start_pred - num_train + 1, start_pred + 1)
        lag_cases = np.vstack([cases[:, d - lag_period : d] for d in days])
        lag_deaths = np.vstack([deaths[:, d - lag_period : d] for d in days])
        target_cases = np.vstack([cases[:, d : d + 1] for d in days])
        target_deaths = np.vstack([deaths[:, d : d + 1] for d in days])
        my_geoloc_ids = np.vstack([my_geoloc_ids_base for d in days])
        country_ids = np.vstack([country_ids_base for d in days])
        return lag_cases, lag_deaths, target_cases, target_deaths, my_geoloc_ids, country_ids, days

    def update_valid_dataset(data, pred_death, pred_case):
        lag_cases, lag_deaths, target_cases, target_deaths, my_geoloc_ids, country_ids, days = data
        day = days[-1] + 1
        new_lag_cases = np.hstack([lag_cases[:, 1:], pred_case])
        new_lag_deaths = np.hstack([lag_deaths[:, 1:], pred_death]) 
        new_target_cases = cases[:, day:day+1]
        new_target_deaths = deaths[:, day:day+1] 
        new_my_geoloc_ids = my_geoloc_ids  
        new_country_ids = country_ids  
        new_days = 1 + days
        return new_lag_cases, new_lag_deaths, new_target_cases, new_target_deaths, new_my_geoloc_ids, new_country_ids, new_days

    def infer_model(lr_death, lr_case, data, start_lag_death, end_lag_death, num_lag_case, fit, score):
        lag_cases, lag_deaths, target_cases, target_deaths, my_geoloc_ids, country_ids, days = data

        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], country_ids])
        X_death = np.hstack([lag_deaths[:, -num_lag_case:], country_ids])
        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], lag_deaths[:, -num_lag_case:], country_ids])
        y_death = target_deaths
        y_death_prev = lag_deaths[:, -1:]
        if fit:
            if 0:
                keep = (y_death > 0).ravel()
                X_death = X_death[keep]
                y_death = y_death[keep]
                y_death_prev = y_death_prev[keep]
            lr_death.fit(X_death, y_death)
        y_pred_death = lr_death.predict(X_death)
        y_pred_death = np.maximum(y_pred_death, y_death_prev)

        X_case = np.hstack([lag_cases[:, -num_lag_case:], my_geoloc_ids])
        X_case = lag_cases[:, -num_lag_case:]
        y_case = target_cases
        y_case_prev = lag_cases[:, -1:]
        if fit:
            lr_case.fit(X_case, y_case)
        y_pred_case = lr_case.predict(X_case)
        y_pred_case = np.maximum(y_pred_case, y_case_prev)

        if score:
            death_score = val_score(y_death, y_pred_death)
            case_score = val_score(y_case, y_pred_case)
        else:
            death_score = 0
            case_score = 0

        return death_score, case_score, y_pred_death, y_pred_case

    def train_model(train, valid, start_lag_death, end_lag_death, num_lag_case, num_val, score=True):
        alpha = 2
        lr_death = Ridge(alpha=alpha, fit_intercept=False)
        lr_case = Ridge(alpha=alpha, fit_intercept=True)

        (train_death_score, train_case_score, train_pred_death, train_pred_case,
        ) = infer_model(lr_death, lr_case, train, start_lag_death, end_lag_death, num_lag_case, fit=True, score=score)

        death_scores = []
        case_scores = []

        death_pred = []
        case_pred = []

        for i in range(num_val):

            (valid_death_score, valid_case_score, valid_pred_death, valid_pred_case,
            ) = infer_model(lr_death, lr_case, valid, start_lag_death, end_lag_death, num_lag_case, fit=False, score=score)

            death_scores.append(valid_death_score)
            case_scores.append(valid_case_score)
            death_pred.append(valid_pred_death)
            case_pred.append(valid_pred_case)

            if 0:
                print('val death: %0.3f' %  valid_death_score,
                      'val case: %0.3f' %  valid_case_score,
                      'val : %0.3f' %  np.mean([valid_death_score, valid_case_score]),
                      flush=True)
            valid = update_valid_dataset(valid, valid_pred_death, valid_pred_case)

        if score:
            death_scores = np.sqrt(np.mean([s**2 for s in death_scores]))
            case_scores = np.sqrt(np.mean([s**2 for s in case_scores]))
            if 0:
                print('train death: %0.3f' %  train_death_score,
                      'train case: %0.3f' %  train_case_score,
                      'val death: %0.3f' %  death_scores,
                      'val case: %0.3f' %  case_scores,
                      'val : %0.3f' % ( (death_scores + case_scores) / 2),
                      flush=True)
            else:
                print('%0.4f' %  case_scores,
                      ', %0.4f' %  death_scores,
                      '= %0.4f' % ( (death_scores + case_scores) / 2),
                      flush=True)
        death_pred = np.hstack(death_pred)
        case_pred = np.hstack(case_pred)
        return death_scores, case_scores, death_pred, case_pred

    countries = [g.split('_')[0] for g in my_geoloc_data.index]
    countries = pd.factorize(countries)[0]

    country_ids_base = countries.reshape((-1, 1))
    ohe = OneHotEncoder(sparse=False)
    country_ids_base = 0.2 * ohe.fit_transform(country_ids_base)
    country_ids_base.shape

    my_geoloc_ids_base = np.arange(num_my_geoloc).reshape((-1, 1))
    ohe = OneHotEncoder(sparse=False)
    my_geoloc_ids_base = 0.1 * ohe.fit_transform(my_geoloc_ids_base)
    my_geoloc_ids_base.shape

    def val_score(true, pred):
        pred = np.log1p(np.round(np.expm1(pred) - 0.2))
        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))

    def val_score(true, pred):
        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))



    start_lag_death, end_lag_death = 14, 6,
    num_train = 6
    num_lag_case = 14
    lag_period = max(start_lag_death, num_lag_case)

    def load_outputs_fit(start_val_delta=0):   
        start_val = min_test_val_day + start_val_delta
        last_train = start_val - 1
        num_val = max_test_val_day - start_val + 1
        print(dates[start_val], start_val, num_val)
        train_data = load_my_dataset(last_train, num_train, lag_period)
        valid_data = load_my_dataset(start_val, 1, lag_period)
        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 
                                                            start_lag_death, end_lag_death, num_lag_case, num_val)

        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()
        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)
        pred_deaths = pred_deaths.stack().reset_index()
        pred_deaths.columns = ['my_geoloc', 'day', 'Fatalities']
        pred_deaths

        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()
        pred_cases.iloc[:, :] = np.expm1(val_case_preds)
        pred_cases = pred_cases.stack().reset_index()
        pred_cases.columns = ['my_geoloc', 'day', 'ConfirmedCases']
        pred_cases

        sub = train[['Date', 'Id', 'my_geoloc', 'day']]
        sub = sub.merge(pred_cases, how='left', on=['my_geoloc', 'day'])
        sub = sub.merge(pred_deaths, how='left', on=['my_geoloc', 'day'])
        #sub = sub.fillna(0)
        sub = sub[sub.day >= start_val]
        sub = sub[['Id', 'ConfirmedCases', 'Fatalities']].copy()
        return sub


    if save_model_train:
        for start_val_delta, date in zip(range(3, -8, -3),
                                  ['2020-04-27', '2020-04-24', '2020-04-21', '2020-04-18']):
            print(date, end=' ')
            outputs_fit = load_outputs_fit(start_val_delta)
            outputs_fit.to_csv('../submissions/cpmp-%s.csv' % date, index=None)

    def get_sub(start_val_delta=0):   
        start_val = min_test_val_day + start_val_delta
        last_train = start_val - 1
        num_val = max_test_val_day - start_val + 1
        print(dates[last_train], start_val, num_val)
        num_lag_case = 14
        train_data = load_my_dataset(last_train, num_train, lag_period)
        valid_data = load_my_dataset(start_val, 1, lag_period)
        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 
                                                            start_lag_death, end_lag_death, num_lag_case, num_val)

        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()
        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)
        pred_deaths = pred_deaths.stack().reset_index()
        pred_deaths.columns = ['my_geoloc', 'day', 'Fatalities']
        pred_deaths

        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()
        pred_cases.iloc[:, :] = np.expm1(val_case_preds)
        pred_cases = pred_cases.stack().reset_index()
        pred_cases.columns = ['my_geoloc', 'day', 'ConfirmedCases']
        pred_cases

        sub = test[['Date', 'ForecastId', 'my_geoloc', 'day']]
        sub = sub.merge(pred_cases, how='left', on=['my_geoloc', 'day'])
        sub = sub.merge(pred_deaths, how='left', on=['my_geoloc', 'day'])
        sub = sub.fillna(0)
        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]
        return sub
        return sub


    known_test = train[['my_geoloc', 'day', 'ConfirmedCases', 'Fatalities']
              ].merge(test[['my_geoloc', 'day', 'ForecastId']], how='left', on=['my_geoloc', 'day'])
    known_test = known_test[['ForecastId', 'ConfirmedCases', 'Fatalities']][known_test.ForecastId.notnull()].copy()
    known_test

    unknow_test = test[test.day > max_test_val_day]
    unknow_test

    def get_final_sub():   
        start_val = max_test_val_day + 1
        last_train = start_val - 1
        num_val = max_test_day - start_val + 1
        print(dates[last_train], start_val, num_val)
        num_lag_case = num_val + 3
        train_data = load_my_dataset(last_train, num_train, lag_period)
        valid_data = load_my_dataset(start_val, 1, lag_period)
        (_, _, val_death_preds, val_case_preds
        ) = train_model(train_data, valid_data, start_lag_death, end_lag_death, num_lag_case, num_val, score=False)

        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()
        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)
        pred_deaths = pred_deaths.stack().reset_index()
        pred_deaths.columns = ['my_geoloc', 'day', 'Fatalities']
        pred_deaths

        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()
        pred_cases.iloc[:, :] = np.expm1(val_case_preds)
        pred_cases = pred_cases.stack().reset_index()
        pred_cases.columns = ['my_geoloc', 'day', 'ConfirmedCases']
        pred_cases
        print(unknow_test.shape, pred_deaths.shape, pred_cases.shape)

        sub = unknow_test[['Date', 'ForecastId', 'my_geoloc', 'day']]
        sub = sub.merge(pred_cases, how='left', on=['my_geoloc', 'day'])
        sub = sub.merge(pred_deaths, how='left', on=['my_geoloc', 'day'])
        #sub = sub.fillna(0)
        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]
        sub = pd.concat([known_test, sub])
        return sub

    if save_public_test:
        sub = get_sub()
    else:
        sub = get_final_sub()
    return sub



# here we will load data and make it ready for training
def load_deep_nn():
    df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
    dataframe_for_submission = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

    co_train_data = pd.read_csv("../input/mytrainweek7/train (3).csv").rename(columns={"Country/Region": "Country_Region"})
    co_train_data = co_train_data.groupby("Country_Region")[["Lat", "Long"]].mean().reset_index()
    co_train_data = co_train_data[co_train_data["Country_Region"].notnull()]

    loc_group = ["Province_State", "Country_Region"]


    def preprocess(df):
        df["Date"] = df["Date"].astype("datetime64[ms]")
        df["days"] = (df["Date"] - pd.to_datetime("2020-01-01")).dt.days
        df["weekend"] = df["Date"].dt.dayofweek//5

        df = df.merge(co_train_data, how="left", on="Country_Region")
        df["Lat"] = (df["Lat"] // 30).astype(np.float32).fillna(0)
        df["Long"] = (df["Long"] // 60).astype(np.float32).fillna(0)

        for col in loc_group:
            df[col].fillna("none", inplace=True)
        return df

    df = preprocess(df)
    dataframe_for_submission = preprocess(dataframe_for_submission)

    print(df.shape)

    TARGETS = ["ConfirmedCases", "Fatalities"]

    for col in TARGETS:
        df[col] = np.log1p(df[col])

    NUM_SHIFT = 5

    features = ["Lat", "Long"]

    for s in range(1, NUM_SHIFT+1):
        for col in TARGETS:
            df["prev_{}_{}".format(col, s)] = df.groupby(loc_group)[col].shift(s)
            features.append("prev_{}_{}".format(col, s))

    df = df[df["Date"] >= df["Date"].min() + timedelta(days=NUM_SHIFT)].copy()

    TEST_FIRST = dataframe_for_submission["Date"].min() # pd.to_datetime("2020-03-13") #
    TEST_DAYS = (df["Date"].max() - TEST_FIRST).days + 1

    dev_df, test_df = df[df["Date"] < TEST_FIRST].copy(), df[df["Date"] >= TEST_FIRST].copy()

    def nn_block(input_layer, size, dropout_rate, activation):
        out_layer = KL.Dense(size, activation=None)(input_layer)
        #out_layer = KL.BatchNormalization()(out_layer)
        out_layer = KL.Activation(activation)(out_layer)
        out_layer = KL.Dropout(dropout_rate)(out_layer)
        return out_layer


    def get_model():
        inp = KL.Input(shape=(len(features),))

        hidden_layer = nn_block(inp, 208, 0.0, "relu")
        gate_layer = nn_block(hidden_layer, 104, 0.0, "hard_sigmoid")
        hidden_layer = nn_block(hidden_layer, 104, 0.0, "relu")
        hidden_layer = KL.multiply([hidden_layer, gate_layer])

        out = KL.Dense(len(TARGETS), activation="linear")(hidden_layer)

        model = tf.keras.models.Model(inputs=[inp], outputs=out)
        return model

    get_model().summary()

    def get_input(df):
        return [df[features]]

    NUM_MODELS = 100


    def train_models(df, save=False):
        models = []
        for i in range(NUM_MODELS):
            model = get_model()
            model.compile(loss="mean_squared_error", optimizer=Nadam(lr=1e-4))
            hist = model.fit(get_input(df), df[TARGETS],
                             batch_size=2250, epochs=1000, verbose=0, shuffle=True)
            if save:
                model.save_weights("model{}.h5".format(i))
            models.append(model)
        return models

    models = train_models(dev_df)


    prev_targets = ['prev_ConfirmedCases_1', 'prev_Fatalities_1']

    def predict_one(df, models):
        pred = np.zeros((df.shape[0], 2))
        for model in models:
            pred += model.predict(get_input(df))/len(models)
        pred = np.maximum(pred, df[prev_targets].values)
        pred[:, 0] = np.log1p(np.expm1(pred[:, 0]) + 0.1)
        pred[:, 1] = np.log1p(np.expm1(pred[:, 1]) + 0.01)
        return np.clip(pred, None, 15)

    print([mean_squared_error(dev_df[TARGETS[i]], predict_one(dev_df, models)[:, i]) for i in range(len(TARGETS))])


    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def evaluate(df):
        error = 0
        for col in TARGETS:
            error += rmse(df[col].values, df["pred_{}".format(col)].values)
        return np.round(error/len(TARGETS), 5)


    def predict(test_df, first_day, num_days, models, val=False):
        temp_df = test_df.loc[test_df["Date"] == first_day].copy()
        y_pred = predict_one(temp_df, models)

        for i, col in enumerate(TARGETS):
            test_df["pred_{}".format(col)] = 0
            test_df.loc[test_df["Date"] == first_day, "pred_{}".format(col)] = y_pred[:, i]

        print(first_day, np.isnan(y_pred).sum(), y_pred.min(), y_pred.max())
        if val:
            print(evaluate(test_df[test_df["Date"] == first_day]))


        y_prevs = [None]*NUM_SHIFT

        for i in range(1, NUM_SHIFT):
            y_prevs[i] = temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]].values

        for d in range(1, num_days):
            date = first_day + timedelta(days=d)
            print(date, np.isnan(y_pred).sum(), y_pred.min(), y_pred.max())

            temp_df = test_df.loc[test_df["Date"] == date].copy()
            temp_df[prev_targets] = y_pred
            for i in range(2, NUM_SHIFT+1):
                temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]] = y_prevs[i-1]

            y_pred, y_prevs = predict_one(temp_df, models), [None, y_pred] + y_prevs[1:-1]


            for i, col in enumerate(TARGETS):
                test_df.loc[test_df["Date"] == date, "pred_{}".format(col)] = y_pred[:, i]

            if val:
                print(evaluate(test_df[test_df["Date"] == date]))

        return test_df

    test_df = predict(test_df, TEST_FIRST, TEST_DAYS, models, val=True)
    print(evaluate(test_df))

    for col in TARGETS:
        test_df[col] = np.expm1(test_df[col])
        test_df["pred_{}".format(col)] = np.expm1(test_df["pred_{}".format(col)])

    models = train_models(df, save=True)

    dataframe_for_submission_public = dataframe_for_submission[dataframe_for_submission["Date"] <= df["Date"].max()].copy()
    dataframe_for_submission_private = dataframe_for_submission[dataframe_for_submission["Date"] > df["Date"].max()].copy()

    pred_cols = ["pred_{}".format(col) for col in TARGETS]
    #dataframe_for_submission_public = dataframe_for_submission_public.merge(test_df[["Date"] + loc_group + pred_cols].rename(columns={col: col[5:] for col in pred_cols}), 
    #                                    how="left", on=["Date"] + loc_group)
    dataframe_for_submission_public = dataframe_for_submission_public.merge(test_df[["Date"] + loc_group + TARGETS], how="left", on=["Date"] + loc_group)

    SUB_FIRST = dataframe_for_submission_private["Date"].min()
    SUB_DAYS = (dataframe_for_submission_private["Date"].max() - dataframe_for_submission_private["Date"].min()).days + 1

    dataframe_for_submission_private = df.append(dataframe_for_submission_private, sort=False)

    for s in range(1, NUM_SHIFT+1):
        for col in TARGETS:
            dataframe_for_submission_private["prev_{}_{}".format(col, s)] = dataframe_for_submission_private.groupby(loc_group)[col].shift(s)

    dataframe_for_submission_private = dataframe_for_submission_private[dataframe_for_submission_private["Date"] >= SUB_FIRST].copy()

    dataframe_for_submission_private = predict(dataframe_for_submission_private, SUB_FIRST, SUB_DAYS, models)

    for col in TARGETS:
        dataframe_for_submission_private[col] = np.expm1(dataframe_for_submission_private["pred_{}".format(col)])

    dataframe_for_submission = dataframe_for_submission_public.append(dataframe_for_submission_private, sort=False)
    dataframe_for_submission["ForecastId"] = dataframe_for_submission["ForecastId"].astype(np.int16)

    return dataframe_for_submission[["ForecastId"] + TARGETS]

# get the output of the DNN as part of our submission
sub1 = main_for_train()
sub1['ForecastId'] = sub1['ForecastId'].astype('int')
sub2 = load_deep_nn()

sub1.sort_values("ForecastId", inplace=True)
sub2.sort_values("ForecastId", inplace=True)


from sklearn.metrics import mean_squared_error

TARGETS = ["ConfirmedCases", "Fatalities"]

[np.sqrt(mean_squared_error(np.log1p(sub1[t].values), np.log1p(sub2[t].values))) for t in TARGETS]

dataframe_for_submission = sub1.copy()
for t in TARGETS:
    dataframe_for_submission[t] = np.expm1(np.log1p(sub1[t].values)*0.5 + np.log1p(sub2[t].values)*0.5)




# We will train XG-BOOST here
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
train['Date'] = pd.to_datetime(train['Date'])
def dealing_with_null_values(dataset):
    dataset = dataset
    for i in dataset.columns:
        replace = []
        data  = dataset[i].isnull()
        count = 0
        for j,k in zip(data,dataset[i]):
            if (j==True):
                count = count+1
                replace.append('No Information Available')
            else:
                replace.append(k)
        print("Num of null values (",i,"):",count)
        dataset[i] = replace
    return dataset
train = dealing_with_null_values(train)
def fillState(state, country):
    if state == 'No Information Available': return country
    return state
train['Province_State'] = train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
train.loc[:, 'Date'] = train.Date.dt.strftime("%m%d")
train["Date"]  = train["Date"].astype(int)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

train.Country_Region = le.fit_transform(train.Country_Region)
train.Province_State = le.fit_transform(train.Province_State)
data = pd.DataFrame()
data['Province_State']=train['Province_State']
data['Country_Region']=train['Country_Region']
data['Date']=train['Date']

xgb_model1 = XGBRegressor(n_estimators=1000) 
xgb_model1.fit(data,train['ConfirmedCases'])
xgb_model2 = XGBRegressor(n_estimators=1000) 
xgb_model2.fit(data,train['Fatalities'])
test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')
test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)
test = dealing_with_null_values(test)
test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
test.loc[:, 'Date'] = test.Date.dt.strftime("%m%d")
test["Date"]  = test["Date"].astype(int)
le = preprocessing.LabelEncoder()

test.Country_Region = le.fit_transform(test.Country_Region)
test.Province_State = le.fit_transform(test.Province_State)

test_data = pd.DataFrame()
test_data['Province_State']  = test['Province_State'] 
test_data['Country_Region']  = test['Country_Region']
test_data['Date'] = test['Date']
from warnings import filterwarnings
filterwarnings('ignore')

xgb_sub =  pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
for i in test['Country_Region'].unique():
    s = test[test['Country_Region'] == i].Province_State.unique()
    for j in s:
        xgb_sub_train = data[(data['Country_Region'] == i) & (data['Province_State'] == j)]
        xgb_sub_test = test_data[(test_data['Country_Region']==i) & (test_data['Province_State'] == j)]
    
        xgb_sub_train_with_labels = train[(train['Country_Region'] == i) & (train['Province_State'] == j)]
    
        index_test = test[(test['Country_Region']==i) & (test['Province_State'] == j)]
        index_test = index_test['ForecastId']
    
        xgb_model1 = XGBRegressor(n_estimators=2000)
        xgb_model1.fit(xgb_sub_train,xgb_sub_train_with_labels['ConfirmedCases'])
    
        xgb_model2 = XGBRegressor(n_estimators=2000)
        xgb_model2.fit(xgb_sub_train,xgb_sub_train_with_labels['Fatalities'])
                                        
        y1_xpred_xgb_sub = xgb_model1.predict(xgb_sub_test)
        y2_xpred_xgb_sub = xgb_model2.predict(xgb_sub_test)
    
    
        xgb_xgb_subs = pd.DataFrame()
        xgb_xgb_subs['ForecastId'] = index_test
        xgb_xgb_subs['ConfirmedCases'] = y1_xpred_xgb_sub
        xgb_xgb_subs['Fatalities']=y2_xpred_xgb_sub
    
        xgb_sub = pd.concat([xgb_sub, xgb_xgb_subs], axis=0)
        
xgb_sub['ForecastId']= xgb_sub['ForecastId'].astype('int')




import pandas as pd
from pathlib import Path
from pandas_profiling import ProfileReport
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score




dataset_path = Path('/kaggle/input/covid19-global-forecasting-week-4')
train = pd.read_csv(dataset_path/'train.csv')
test = pd.read_csv(dataset_path/'test.csv')
dtree_sub = pd.read_csv(dataset_path/'submission.csv')




train_profile = ProfileReport(train, title='COVID19 WEEK 4 Profiling Report', html={'style':{'full_width':True}},progress_bar=False);
train_profile




def fill_state(state,country):
    if pd.isna(state) : return country
    return state




train['Province_State'] = train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fill_state(x['Province_State'], x['Country_Region']), axis=1)
test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fill_state(x['Province_State'], x['Country_Region']), axis=1)
train['Date'] = pd.to_datetime(train['Date'],infer_datetime_format=True)
test['Date'] = pd.to_datetime(test['Date'],infer_datetime_format=True)
train['Day_of_Week'] = train['Date'].dt.dayofweek
test['Day_of_Week'] = test['Date'].dt.dayofweek
train['Month'] = train['Date'].dt.month
test['Month'] = test['Date'].dt.month
train['Day'] = train['Date'].dt.day
test['Day'] = test['Date'].dt.day
train['Day_of_Year'] = train['Date'].dt.dayofyear
test['Day_of_Year'] = test['Date'].dt.dayofyear
train['Week_of_Year'] = train['Date'].dt.weekofyear
test['Week_of_Year'] = test['Date'].dt.weekofyear
train['Quarter'] = train['Date'].dt.quarter  
test['Quarter'] = test['Date'].dt.quarter  
train.drop('Date',1,inplace=True)
test.drop('Date',1,inplace=True)




dtree_sub=pd.DataFrame(columns=dtree_sub.columns)
l1=LabelEncoder()
l2=LabelEncoder()
l1.fit(train['Country_Region'])
l2.fit(train['Province_State'])




countries=train['Country_Region'].unique()
for country in countries:
    country_df=train[train['Country_Region']==country]
    provinces=country_df['Province_State'].unique()
    for province in provinces:
            train_df=country_df[country_df['Province_State']==province]
            train_df.pop('Id')
            x=train_df[['Province_State','Country_Region','Day_of_Week','Month','Day','Day_of_Year','Week_of_Year','Quarter']]
            x['Country_Region']=l1.transform(x['Country_Region'])
            x['Province_State']=l2.transform(x['Province_State'])
            y1=train_df[['ConfirmedCases']]
            y2=train_df[['Fatalities']]
            model_1=DecisionTreeClassifier()
            model_2=DecisionTreeClassifier()
            model_1.fit(x,y1)
            model_2.fit(x,y2)
            test_df=test.query('Province_State==@province & Country_Region==@country')
            test_id=test_df['ForecastId'].values.tolist()
            test_df.pop('ForecastId')
            test_x=test_df[['Province_State','Country_Region','Day_of_Week','Month','Day','Day_of_Year','Week_of_Year','Quarter']]
            test_x['Country_Region']=l1.transform(test_x['Country_Region'])
            test_x['Province_State']=l2.transform(test_x['Province_State'])
            test_y1=model_1.predict(test_x)
            test_y2=model_2.predict(test_x)
            test_res=pd.DataFrame(columns=dtree_sub.columns)
            test_res['ForecastId']=test_id
            test_res['ConfirmedCases']=test_y1
            test_res['Fatalities']=test_y2
            dtree_sub=dtree_sub.append(test_res)




# ENSEMBLE .................
dtree_confirmed=dtree_sub["ConfirmedCases"]
dtree_fatal=dtree_sub["Fatalities"]
boost_confirmed = xgb_sub["ConfirmedCases"]
boost_fatal = xgb_sub["Fatalities"]
deep_confirmed = dataframe_for_submission["ConfirmedCases"]
deep_fatal = dataframe_for_submission["Fatalities"]
dataframe_for_submission["ConfirmedCases"] = 0.1 * boost_confirmed.values +  0.70 * deep_confirmed.values + 0.20 *dtree_confirmed.values
dataframe_for_submission["Fatalities"] = 0.1 * boost_fatal.values  +  0.70 * deep_fatal.values + 0.20  * dtree_fatal.values
dataframe_for_submission.to_csv('submission.csv',index=False)

