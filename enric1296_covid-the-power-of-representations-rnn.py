#!/usr/bin/env python
# coding: utf-8

# In[1]:


############################################
# 1. Libraries

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm, tqdm_notebook
import os
from datetime import datetime, date, timedelta

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from scipy import spatial

import matplotlib.pyplot as plt

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.constraints import MaxNorm

pd.set_option('max_rows', 1000)

############################################


# In[2]:


############################################
# 2. Global Variables

WINDOW_X_SIZE = 60
WINDOW_Y_SIZE = 1

PUBLIC_LEADERBOARD = False

DRIVE = False
KAGGLE = True

FILE_NAME = 'submission.csv'

# paths

if DRIVE:
    path = '/content/drive/My Drive/PROYECTOS/COVID-19/'
elif KAGGLE:
    path = '../input/covid19-global-forecasting-week-2/'
else:
    path = './'

if KAGGLE:
  path_input = path
  path_output = '../working/'
  path_population = '../input/locations-population/'
else:
    path_input = path + 'input/'
    path_output = path + 'output/'
    path_population = path_input

############################################


# In[3]:


############################################
# 3. Load Data

df_data = pd.read_csv(path_input + 'train.csv', sep=',')
df_test = pd.read_csv(path_input + 'test.csv', sep=',')
df_sample_sub = pd.read_csv(path_input + 'submission.csv', sep=',')

df_population = pd.read_csv(path_population + 'locations_population.csv', sep=',')

############################################


# In[4]:


############################################
# 4. Data Summary

date_format = "%Y-%m-%d"
print(f'Train Dates: {df_data.Date.min()} - {df_data.Date.max()} // QT_DAYS: {(datetime.strptime(df_data.Date.max(), date_format) - datetime.strptime(df_data.Date.min(), date_format)).days} ')
print(f'Test Dates: {df_test.Date.min()} - {df_test.Date.max()} // QT_DAYS: {(datetime.strptime(df_test.Date.max(), date_format) - datetime.strptime(df_test.Date.min(), date_format)).days} ')
print (f'Window days to be predicted: {(datetime.strptime(df_test.Date.max(), date_format) - datetime.strptime(df_data.Date.max(), date_format)).days}')

window_test_days = (datetime.strptime(df_test.Date.max(), date_format) - datetime.strptime(df_data.Date.max(), date_format)).days

############################################


# In[5]:


############################################
# 5. Functions

def parseProvinces(df_data, df_test, df_population):
  df_data['Province_State'] = df_data['Province_State'].fillna('#NF')
  df_test['Province_State'] = df_test['Province_State'].fillna('#NF')
  df_population['Province.State'] = df_population['Province.State'].fillna('#NF')

  return df_data, df_test, df_population


def isNaN(num):
    return num != num


def getDaysElapsedSinceDeltaCases(df, feature_names=['qt_day_100cases'], deltas=[100]):

    for feature_name in feature_names:
      df[feature_name] = 0

    for i, row in tqdm(df.iterrows()):

        date = row['Date']
        country = row['Country_Region']
        province = row['Province_State']

        for pos, feature_name in enumerate(feature_names):

          date_delta = df_data['Date'][(df_data['Country_Region']==country) & (df_data['Province_State']==province) & (df_data['Date']<=date) & (df_data['ConfirmedCases']>=deltas[pos])].min()

          if isNaN(date_delta):
            value = 0
          else:
            value = (datetime.strptime(date, date_format) - datetime.strptime(date_delta, date_format)).days + 1
          if value < 0:
            value = 0

          df[feature_name][i] = value

    return df


def normalizeDataModel(vector, feature, axis, dict_normalization, mode='train'):
  uni_data_mean = vector.mean(axis=axis)
  uni_data_std = vector.std(axis=axis)

  if mode=='train':
    dict_normalization[feature] = {'Mean': uni_data_mean, 'Std': uni_data_std}

  return dict_normalization, (vector - uni_data_mean)/uni_data_std


def createCountryDicts(df_data):
  unique_countries = df_data['Country_Region'].unique()
  dict_countries = {}
  dict_countries_inv = {}

  for i, country in enumerate(unique_countries):
    dict_countries[country] = i
    dict_countries_inv[i] = country

  return dict_countries, dict_countries_inv


def createProvinceDicts(df_data):
  unique_provinces = df_data['Province_State'].unique()
  dict_provinces = {}
  dict_provinces_inv = {}

  for i, province in enumerate(unique_provinces):
    dict_provinces[province] = i
    dict_provinces_inv[i] = province

  return dict_provinces, dict_provinces_inv


def getCountryRepresentation(df, scale=True):
  dict_latitudes = {'France': [46.887, 2.552]}

  vector = np.zeros((len(dict_countries.keys()), 5))

  for i, country in enumerate(dict_countries):

    population = df['Population'][df['Country_Region']==country].values[0]
      
    # One province
    if df['co_province'][df['Country_Region']==country].unique().shape[0] == 1:
        last_confirmed, last_fatalities = df[['ConfirmedCases', 'Fatalities']][df['Country_Region']==country].values[-1]
        std_last_confirmed, std_last_fatalities = df[['ConfirmedCases', 'Fatalities']][df['Country_Region']==country].values[-5:].std(axis=0)
    else:
        last_confirmed, last_fatalities = df[['Date', 'ConfirmedCases', 'Fatalities']][df['Country_Region']==country].groupby(['Date']).sum().values[-1]
        std_last_confirmed, std_last_fatalities = df[['Date', 'ConfirmedCases', 'Fatalities']][df['Country_Region']==country].groupby(['Date']).sum().values[-5:].std(axis=0)

    vector[i] = np.array([population, std_last_confirmed, std_last_fatalities, last_confirmed, last_fatalities])

  if scale:
    scaler = StandardScaler()
    vector = scaler.fit_transform(vector)

  return vector


def getProvinceRepresentation(df, scale=True):
  vector = np.zeros((len(dict_provinces.keys()), 13))
  countries_raw = getCountryRepresentation(df, scale=False)

  for i, province in enumerate(dict_provinces):

      if province == '#NF':
          lat_, long_ = 0, 0
          last_confirmed, last_fatalities = countries_raw[:, -2:].mean(axis=0)
          std_last_confirmed, std_last_fatalities = countries_raw[:, -2:].mean(axis=0)
          qt_1, qt_100, qt_1_000 = df_data[['qt_days_since_1_case', 'qt_days_since_100_cases', 'qt_days_since_1000_cases']]                                          [(df_data['Date']==df_data.Date.max()) & (df_data['Province_State']=='#NF')].values.mean(axis=0).round()
          population = df_data['Population'][(df_data['Date']==df_data.Date.max()) & (df_data['Province_State']=='#NF')].values.mean(axis=0).round()
          vec_country = countries_raw.mean(axis=0)
      else:
          last_confirmed, last_fatalities = df[['ConfirmedCases', 'Fatalities']][df['Province_State']==province].values[-1]
          std_last_confirmed, std_last_fatalities = df[['ConfirmedCases', 'Fatalities']][df['Province_State']==province].values[-5:].std(axis=0)
          country = df['Country_Region'][df['Province_State']==province].values[0]
          qt_1, qt_100, qt_1_000 = df_data[['qt_days_since_1_case', 'qt_days_since_100_cases', 'qt_days_since_1000_cases']][df_data['Province_State']==province].values[-1:].squeeze()
          population = df_data['Population'][df_data['Province_State']==province].values[0]
          vec_country = countries_raw[dict_countries[country]]    

      vector[i] = np.hstack([np.array([population, std_last_confirmed, std_last_fatalities, last_confirmed, last_fatalities, qt_1, qt_100, qt_1_000]), vec_country])

  if scale:
    scaler = StandardScaler()
    vector = scaler.fit_transform(vector)

  return vector

#################################
# Name: uniVariatedata
# input: 
# output:  
#################################

def uniVariateData(df, feature, window_x_size, window_y_size, train=True, return_all_series=False):
    num_series = df['Country_Province'].unique().shape[0]
    X = np.empty((num_series, window_x_size, 1))
    y = np.empty((num_series, window_x_size, window_y_size))
    if return_all_series:
      all_data = X = np.empty((num_series, window_x_size + window_y_size))

    unique_series = df_data['Country_Province'].unique()
    if return_all_series:
        for i, serie in tqdm(enumerate(unique_series)):
            all_data[i] = df_data[feature][df_data['Country_Province']==serie].values[-(WINDOW_X_SIZE+WINDOW_Y_SIZE):]
        return all_data
    else:
      for i, serie in tqdm(enumerate(unique_series)):
          data = df_data[feature][df_data['Country_Province']==serie].values
          if train:
            X[i] = data[:window_x_size].reshape(-1, 1)
          else:
            X[i] = data[-window_x_size:].reshape(-1, 1)
          if train:
            for step_ahead in range(1, window_y_size + 1):
                y[i, :, step_ahead - 1] = data[step_ahead:step_ahead + window_x_size]
      
    if train:
      return X, y
    else:
      return X

##################
# Plots

def create_time_steps(length):
  return list(range(-length, 0))

def baseline(history):
  return np.mean(history[-1:])

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']

    time_steps = create_time_steps(plot_data[0].shape[0])

    if delta:
      future = delta
    else:
      future = 0

    plt.title(title)

    for i, x in enumerate(plot_data):
      if i:
        plt.plot(future, plot_data[i], marker[i], markersize=10,
                label=labels[i])
      else:
        plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])

    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    plt.show()
  
def multi_step_plot(history, true_future, prediction):
      plt.figure(figsize=(12, 6))
      num_in = create_time_steps(len(history))
      num_out = true_future.shape[0]

      plt.plot(num_in, np.array(history), label='History')
      plt.plot(np.arange(num_out)/1, np.array(true_future), 'bo',
              label='True Future')
      if prediction.any():
        plt.plot(np.arange(num_out)/1, np.array(prediction), 'ro',
                label='Predicted Future')
      plt.legend(loc='upper left')
      plt.show()

def testStepPlot(history, prediction):
      plt.figure(figsize=(12, 6))
      num_in = create_time_steps(len(history))
      num_out = prediction.shape[0]

      plt.plot(num_in, np.array(history), label='History')
      plt.plot(np.arange(num_out)/1, np.array(prediction), 'ro',
              label='Predicted Future')
      plt.legend(loc='upper left')
      plt.show()

def lastTimeStepMse(y_true, y_pred):
    return mean_squared_error(y_true[: , -1], y_pred[:, -1])

def moovingAverage(array, size_window, weights=None):
    if weights:
      assert size_window==len(weights)
    new_array = np.empty(array.shape[0])
    for i in range(size_window, array.shape[0]):
      if weights:
        new_array[i] = np.sum(array[i-size_window:i] * np.array(weights))
      else:
        new_array[i] = np.mean(array[i-size_window:i])
    return new_array


#################################
# Name: buildModel
# input: lr(float), summary(bool)
# output:  
#################################

def buildModel(lr=0.001, summary=False):

    input_country = Input(shape=[1], name='country')
    input_province = Input(shape=[1], name='province')
    input_confirmed_cases = Input(shape=[WINDOW_X_SIZE, 1], name='in_confirmedcases')
    input_fatalities = Input(shape=[WINDOW_X_SIZE, 1], name='in_fatalities')
    input_trend_confirmed_cases = Input(shape=[WINDOW_X_SIZE-1, 1], name='in_trend_confirmedcases')
    input_trend_fatalities = Input(shape=[WINDOW_X_SIZE-1, 1], name='in_trend_fatalities')
    input_delta_confirmed_cases = Input(shape=[WINDOW_X_SIZE-1, 1], name='in_delta_confirmedcases')
    input_delta_fatalities = Input(shape=[WINDOW_X_SIZE-1, 1], name='in_delta_fatalities')

    country_index = country_representation.shape[0]
    province_index = province_representation.shape[0]

    embedding_country = Embedding(country_index,
                                  country_representation.shape[1],
                                  weights=[country_representation],
                                  trainable=False)(input_country)

    embedding_province =  Embedding(province_index,
                                    province_representation.shape[1],
                                    weights=[province_representation],
                                    trainable=False)(input_province)

    #####################

    embeddings = concatenate([Flatten()(embedding_country), Flatten()(embedding_province)])

    lstm_confirmed_cases = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input_confirmed_cases)
    lstm_trend_confirmed_cases = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input_trend_confirmed_cases)

    lstm_fatalities = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input_fatalities)
    lstm_trend_fatalities = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input_trend_fatalities)

    lstm_confirmed_cases = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(lstm_confirmed_cases)
    lstm_trend_confirmed_cases = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(lstm_trend_confirmed_cases)

    lstm_fatalities = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(lstm_fatalities)
    lstm_trend_fatalities = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(lstm_trend_fatalities)

    conf_cases = concatenate([lstm_confirmed_cases, lstm_trend_confirmed_cases, lstm_trend_fatalities, embeddings])  
    conf_cases = Dropout(0.4)(conf_cases)
    conf_cases = Dense(128, activation='selu')(conf_cases)
    conf_cases = Dropout(0.3)(conf_cases)
    conf_cases = Dense(64, activation='selu')(conf_cases)
    conf_cases = Dropout(0.2)(conf_cases)

    fatalities = concatenate([lstm_fatalities, lstm_trend_fatalities, lstm_trend_confirmed_cases, embeddings])  
    fatalities = Dropout(0.4)(fatalities)
    fatalities = Dense(128, activation='selu')(fatalities)
    fatalities = Dropout(0.3)(fatalities)
    fatalities = Dense(64, activation='selu')(fatalities)
    fatalities = Dropout(0.2)(fatalities)

    output_confirmed_cases = Dense(WINDOW_Y_SIZE, activation='relu', name='confirmed_cases')(conf_cases)
    output_fatalities = Dense(WINDOW_Y_SIZE, activation='relu', name='fatalities')(fatalities)

    model = Model(inputs=[input_country, input_province, input_confirmed_cases, input_fatalities, input_trend_confirmed_cases, input_trend_fatalities,
                          input_delta_confirmed_cases, input_delta_fatalities],
                  outputs=[output_confirmed_cases, output_fatalities])

    model.compile(loss_weights=[1, 1], loss='mse', optimizer=Adam(learning_rate=lr, clipvalue=1.0), metrics=[lastTimeStepMse, 'mape'])

    if summary:
      print(model.summary())

    return model

############################################


# In[6]:


############################################
# 6. Workflow

# Parse data 

if PUBLIC_LEADERBOARD:
    df_data = df_data[df_data['Date']<='2020-03-18']

df_data['dt_datetime'] = pd.to_datetime(df_data['Date'])
df_data['co_weekday'] = df_data.dt_datetime.dt.weekday
df_test['dt_datetime'] = pd.to_datetime(df_test['Date'])
df_test['co_weekday'] = df_test.dt_datetime.dt.weekday
df_data, df_test, df_population = parseProvinces(df_data, df_test, df_population)

df_data = df_data.merge(df_population[['Province.State', 'Country.Region', 'Population']], how='left', left_on=['Province_State', 'Country_Region'], right_on=['Province.State', 'Country.Region'])
df_test = df_test.merge(df_population[['Province.State', 'Country.Region', 'Population']], how='left', left_on=['Province_State', 'Country_Region'], right_on=['Province.State', 'Country.Region'])
df_data['Population'][df_data['Population'].isna()] = df_data['Population'].median()
df_test['Population'][df_test['Population'].isna()] = df_test['Population'].median()

# Feature Engineering

df_data = getDaysElapsedSinceDeltaCases(df_data, ['qt_days_since_1_case', 'qt_days_since_100_cases', 'qt_days_since_1000_cases'], deltas=[1, 100, 1_000])
df_data['Country_Province'] = df_data.apply(lambda x: x['Province_State'] + '_' + x['Country_Region'], axis=1)

df_data_original = df_data.copy()
df_test_original = df_test.copy()

# Create dicts

dict_countries, dict_countries_inv = createCountryDicts(df_data)
dict_provinces, dict_provinces_inv = createProvinceDicts(df_data)

# Apply dictionaries

df_data['co_country'] = df_data['Country_Region'].apply(lambda x: dict_countries[x])
df_test['co_country'] = df_test['Country_Region'].apply(lambda x: dict_countries[x])

df_data['co_province'] = df_data['Province_State'].apply(lambda x: dict_provinces[x])
df_test['co_province'] = df_test['Province_State'].apply(lambda x: dict_provinces[x])

# Create representations

country_representation = getCountryRepresentation(df_data)
province_representation = getProvinceRepresentation(df_data)

# Normalize data

df_data['ConfirmedCases'] = np.log1p(df_data['ConfirmedCases'])
df_data['Fatalities'] = np.log1p(df_data['Fatalities'])

# Create time Series Data

data_confirmed_cases = uniVariateData(df_data, 'ConfirmedCases', WINDOW_X_SIZE, WINDOW_Y_SIZE, return_all_series=True)
data_fatalities = uniVariateData(df_data, 'Fatalities', WINDOW_X_SIZE, WINDOW_Y_SIZE, return_all_series=True)

X_confirmedcases = data_confirmed_cases[:, :WINDOW_X_SIZE]
y_confirmedcases = data_confirmed_cases[:, WINDOW_X_SIZE:]
X_fatalities = data_fatalities[:, :WINDOW_X_SIZE]
y_fatalities = data_fatalities[:, WINDOW_X_SIZE:]

X_countries = uniVariateData(df_data, 'co_country', WINDOW_X_SIZE, WINDOW_Y_SIZE, train=False)
X_province = uniVariateData(df_data, 'co_province', WINDOW_X_SIZE, WINDOW_Y_SIZE, train=False)

X_countries = np.expand_dims(X_countries[:, 0, 0], axis=1)
X_province = np.expand_dims(X_province[:, 0, 0], axis=1)

# Trends

X_trend_confirmed_cases = np.expm1(X_confirmedcases[:, 1:]) - np.expm1(X_confirmedcases[:, :-1])
X_trend_fatalities = np.expm1(X_fatalities[:, 1:]) - np.expm1(X_fatalities[:, :-1])
for i in range(X_trend_confirmed_cases.shape[0]): 
  X_trend_confirmed_cases[i] = moovingAverage(X_trend_confirmed_cases[i], size_window=7)
  X_trend_fatalities[i] = moovingAverage(X_trend_fatalities[i], size_window=7)

# Deltas

X_delta_confirmed_cases = np.nan_to_num((np.expm1(X_confirmedcases[:, 1:]) - np.expm1(X_confirmedcases[:, :-1]))/np.expm1(X_confirmedcases[:, :-1]), nan=0, posinf=0)
X_delta_fatalities = np.nan_to_num((np.expm1(X_fatalities[:, 1:]) - np.expm1(X_fatalities[:, :-1]))/np.expm1(X_fatalities[:, :-1]), nan=0, posinf=0)

## Series with no confirmed cases
# We will assume that in the future they will not have any confirmed cases, this way our
# model will focus on the countries that are more critically affected.

mean_series = X_confirmedcases.mean(axis=1)
series_with_no_confimred_cases = {i for i, serie in enumerate(mean_series) if serie==0}
series_with_confimred_cases = [i for i in range(X_confirmedcases.shape[0]) if i not in series_with_no_confimred_cases]

X_confirmedcases = np.expand_dims(X_confirmedcases, axis=2)
X_fatalities = np.expand_dims(X_fatalities, axis=2)
X_trend_confirmed_cases = np.expand_dims(X_trend_confirmed_cases, axis=2)
X_trend_fatalities = np.expand_dims(X_trend_fatalities, axis=2)
X_delta_confirmed_cases = np.expand_dims(X_delta_confirmed_cases, axis=2)
X_delta_fatalities = np.expand_dims(X_delta_fatalities, axis=2)

# Model inputs

print(X_confirmedcases.mean(), y_confirmedcases.mean(), X_confirmedcases.std(), y_confirmedcases.std())
print(X_fatalities.mean(), y_fatalities.mean(), X_fatalities.std(), y_fatalities.std())

x_train = [X_countries[series_with_confimred_cases], X_province[series_with_confimred_cases], X_confirmedcases[series_with_confimred_cases], 
           X_fatalities[series_with_confimred_cases], X_trend_confirmed_cases[series_with_confimred_cases], X_trend_fatalities[series_with_confimred_cases],
           X_delta_confirmed_cases[series_with_confimred_cases], X_delta_fatalities[series_with_confimred_cases]]
           
y_train = [y_confirmedcases[series_with_confimred_cases], y_fatalities[series_with_confimred_cases]]

############################################


# In[7]:


candidates_samples = ['#NF_Spain', '#NF_Italy', '#NF_Germany', '#NF_Mexico', '#NF_Algeria', '#NF_Brazil', 'New York_US', 'California_US', 'Alaska_US']
samples = {x:country_prov for x, country_prov in enumerate(df_data['Country_Province'].unique()) if country_prov in candidates_samples}

for i, sample in enumerate(samples):
    print('==='*20)
    print(samples[sample])
    print('==='*20)
    show_plot([X_confirmedcases[sample], y_confirmedcases[sample, 0], baseline(X_confirmedcases[sample])], 0, 'Sample Example')
    plt.plot(X_trend_confirmed_cases[sample])
    plt.show()


# In[8]:


############################################
# 7. Train Model

model = buildModel(lr=0.0006, summary=True)

history = model.fit(x_train, y_train,
                    # validation_split=0.1,
                    shuffle=False,
                    batch_size=16,
                    epochs=200,
                    verbose=1)

# Evaluation 

predictions = model.predict(x_train)
pred_confirmed_cases = predictions[0]
pred_fatalities = predictions[1]

for sample in samples:
   multi_step_plot(X_confirmedcases[sample], y_confirmedcases[sample, :], pred_confirmed_cases[sample, :])
   multi_step_plot(X_fatalities[sample], y_fatalities[sample, :], pred_fatalities[sample, :])
   print(y_confirmedcases[sample, :], pred_confirmed_cases[sample, :])


############################################


# In[9]:


###############################################
# 8. Test Prediction 

dt_ini = datetime.strptime(df_data_original.Date.max(), date_format) - timedelta(WINDOW_X_SIZE)
dt_end = datetime.strptime(df_data_original.Date.max(), date_format)
dt_max_test = df_test['Date'].max()

X_test_confirmedcases = X_confirmedcases
X_test_fatalities = X_fatalities
X_test_countries = X_countries
X_test_province = X_province

X_test_trend_confirmed_cases = X_trend_confirmed_cases
X_test_trend_fatalities = X_trend_fatalities
X_test_delta_confirmed_cases = X_delta_confirmed_cases
X_test_delta_fatalities = X_delta_fatalities

i = 0

X_final_confirmedcases = X_test_confirmedcases
X_final_fatalities = X_test_fatalities


while dt_end.strftime(date_format) < dt_max_test:
    print('==='*20)
    print(f'dt_ini: {dt_ini.strftime(date_format)}')
    print(f'dt_end: {dt_end.strftime(date_format)}')
    print('==='*20)

    x_test = [X_test_countries, X_test_province, X_test_confirmedcases, X_test_fatalities, X_test_trend_confirmed_cases, X_test_trend_fatalities, 
              X_delta_confirmed_cases, X_delta_fatalities]
    y_test_predictions = model.predict(x_test)

    y_pred_confirmedcases_unscaled = y_test_predictions[0]
    y_pred_fatalities_unscaled = y_test_predictions[1]

    y_pred_confirmedcases_unscaled[list(series_with_no_confimred_cases)] = 0
    y_pred_fatalities_unscaled[list(series_with_no_confimred_cases)] = 0

    if i==0:
      X_final_confirmedcases = np.concatenate([X_final_confirmedcases.squeeze(), y_pred_confirmedcases_unscaled], axis=1)
      X_final_fatalities = np.concatenate([X_final_fatalities.squeeze(), y_pred_fatalities_unscaled], axis=1)
    else:
      X_final_confirmedcases = np.concatenate([X_final_confirmedcases, y_pred_confirmedcases_unscaled], axis=1)
      X_final_fatalities = np.concatenate([X_final_fatalities, y_pred_fatalities_unscaled], axis=1)

    X_test_confirmedcases = np.expand_dims(X_final_confirmedcases[: ,-WINDOW_X_SIZE:], axis=2)
    X_test_fatalities = np.expand_dims(X_final_fatalities[:, -WINDOW_X_SIZE:], axis=2)

    X_test_trend_confirmed_cases = np.expm1(X_test_confirmedcases[:, 1:]) - np.expm1(X_test_confirmedcases[:, :-1])
    X_test_trend_fatalities = np.expm1(X_test_fatalities[:, 1:]) - np.expm1(X_test_fatalities[:, :-1])
    for i in range(X_trend_confirmed_cases.shape[0]): 
      X_test_trend_confirmed_cases[i] = np.expand_dims(moovingAverage(X_test_trend_confirmed_cases[i], size_window=5), axis=1)
      X_test_trend_fatalities[i] = np.expand_dims(moovingAverage(X_test_trend_fatalities[i], size_window=5), axis=1)
        
    X_test_delta_confirmed_cases = np.nan_to_num((np.expm1(X_test_confirmedcases[:, 1:]) - np.expm1(X_test_confirmedcases[:, :-1]))/np.expm1(X_test_confirmedcases[:, :-1]), nan=0, posinf=0)
    X_test_delta_fatalities = np.nan_to_num((np.expm1(X_test_fatalities[:, 1:]) - np.expm1(X_test_fatalities[:, :-1]))/np.expm1(X_test_fatalities[:, :-1]), nan=0, posinf=0)

    dt_ini += timedelta(WINDOW_Y_SIZE)
    dt_end += timedelta(WINDOW_Y_SIZE)
    i += 1

###############################################


# In[10]:


for sample in samples:
    print('=='*30)
    print(df_data['Country_Province'].unique()[sample])
    print('=='*30)
    testStepPlot(np.expm1(X_final_confirmedcases[sample, :WINDOW_X_SIZE]), np.expm1(X_final_confirmedcases[sample, WINDOW_X_SIZE:]))


# In[11]:


##############################################
# 9. Submission

df_test = pd.merge(df_test_original, df_data[['Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'Fatalities']], how='left', on=['Province_State', 'Country_Region', 'Date'])
df_test['ConfirmedCases'][df_test['Date']>df_data['Date'].max()] = X_final_confirmedcases[:, WINDOW_X_SIZE:].flatten()
df_test['Fatalities'][df_test['Date']>df_data['Date'].max()] = X_final_fatalities[:, WINDOW_X_SIZE:].flatten()
submission = df_test[['ForecastId', 'ConfirmedCases', 'Fatalities']][df_test.Date>=df_test.Date.min()].reset_index(drop = True)

if not PUBLIC_LEADERBOARD:
    assert submission.shape[0] == df_sample_sub.shape[0]
    assert submission.shape[1] == df_sample_sub.shape[1]

submission['ConfirmedCases'] = np.expm1(submission['ConfirmedCases'])
submission['Fatalities'] = np.expm1(submission['Fatalities'])


print(submission.describe())

submission.to_csv(path_output + FILE_NAME, sep=',', index=False, header=True)


###############################################


# In[12]:


print(submission.shape)
print(df_test.shape)

