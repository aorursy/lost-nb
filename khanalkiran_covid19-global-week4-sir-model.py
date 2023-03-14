#!/usr/bin/env python
# coding: utf-8



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




import pandas as pd
import numpy as np
import re
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, DayLocator, WeekdayLocator
import datetime as dt
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib import gridspec
from matplotlib import dates
from IPython.display import Image
from scipy.optimize import curve_fit
   
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)




train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
submission_file = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
population_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv') # External resources 
parameter_data = pd.read_csv('/kaggle/input/covid19-global-forecast-sir-jhu-timeseries-fit/per_location_fitted_params.csv') # External resources 




display(train_data.head())
display(train_data.describe())
print("Number of countries:", train_data['Country_Region'].nunique())
print('Training data are from', min(train_data['Date']), 'to', max(train_data['Date']))
print("Total number of days: ", train_data['Date'].nunique())




train_data.shape, test_data.shape,submission_file.shape




display(test_data.head())
print('Test data are from', test_data['Date'].min(), 'to', test_data['Date'].max())
print("Number of days", pd.date_range(test_data['Date'].min(),test_data['Date'].max()).shape[0])




print(train_data.isna().any().any(), test_data.isna().any().any())
display(train_data.isna().any())
display(test_data.isna().any())




train_data_covid = train_data.copy()
test_data_covid = test_data.copy()
test_data_covid = test_data_covid.fillna('NA')
train_data_covid = train_data_covid.fillna('NA')




train_series_cc = train_data_covid.fillna('NA').groupby(['Country_Region','Province_State','Date'])['ConfirmedCases'].sum()             .groupby(['Country_Region','Province_State']).max().sort_values()             .groupby('Country_Region').sum().sort_values(ascending = False)
train_series_fatal = train_data_covid.fillna('NA').groupby(['Country_Region','Province_State','Date'])['Fatalities'].sum()             .groupby(['Country_Region','Province_State']).max().sort_values()             .groupby('Country_Region').sum().sort_values(ascending = False)




train_large10_cc = pd.DataFrame(train_series_cc).head(10)
display(train_large10_cc.head())
train_large10_fatal= pd.DataFrame(train_series_fatal).head(10)
display(train_large10_fatal.head())
print("Toal number of people infected by Coronavirus in the world from", min(train_data['Date']),                                    "to", max(train_data['Date']), 'are:',                                                 int(sum(train_series_cc))) 
print("Toal number of people deceased by cronavirus in the world from", min(train_data['Date']),                                    "to", max(train_data['Date']), 'are:',                                                 int(sum(train_series_fatal))) 




fig, (ax1, ax2) = plt.subplots(1,2, figsize = (24,8))
fig.suptitle('Number of Confirmed Cases and Fatalities in the World', fontsize = 30)

#Left plot
ax1.bar(train_large10_cc.index, train_large10_cc['ConfirmedCases'], color = 'purple')
ax1.set(xlabel = 'Countries',
        ylabel = 'Number of ConfirmedCases')
ax1.legend(['ConfirmedCases'])
ax1.grid()
#Right plot
ax2.bar(train_large10_fatal.index, train_large10_fatal['Fatalities'], color = 'orange')
ax2.set(xlabel = 'Countries',
        ylabel = 'Number of Fatalities')
ax2.legend(['Fatalities'])
ax2.grid()

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "20"
plt.show()




train_series_date = train_data_covid.groupby(['Date'])[['ConfirmedCases']].sum().                      sort_values('ConfirmedCases')
display(train_series_date.head(),)
train_series_date_fata = train_data_covid.groupby(['Date'])[['Fatalities']].sum().                      sort_values('Fatalities')
display(train_series_date_fata.head())




fig, (ax1, ax2) = plt.subplots(1,2, figsize = (24,8))
fig.suptitle('Trends of Confirmed Cases and Fatalities in the World', fontsize = 30)

#Left plot
ax1.plot(train_series_date.index, train_series_date['ConfirmedCases'], color = 'purple', marker = 'o',linewidth = 2)
ax1.set(xlabel = 'Date',
        ylabel = 'ConfirmedCases')
ax1.set_xticks(np.arange(0, 80,  step = 12))
ax1.legend(['ConfirmedCases'])
ax1.grid()
#Right plot
ax2.plot(train_series_date_fata.index, train_series_date_fata['Fatalities'], color = 'orange', marker = 'o', linewidth = 2)
ax2.set(xlabel = 'Date',
        ylabel = 'Fatalities')
ax2.set_xticks(np.arange(0, 80,  step = 12))
ax2.legend(['Fatalities'])
ax2.grid()

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"
plt.show()




def country_fun(country_name):
    df_country = train_data_covid.loc[(train_data_covid['Country_Region']== country_name)]
    df_confirmed =  df_country.groupby(['Date'])[['ConfirmedCases']].sum().sort_values('ConfirmedCases') 
    df_fatal = df_country.groupby(['Date'])[['Fatalities']].sum().sort_values('Fatalities') 
    df_confirmed_fatal =  df_confirmed.join((df_fatal), how = 'outer')
    return df_confirmed_fatal




def country_plot_fun(country_name):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (24,8))
    #fig.suptitle('Trends of Confirmed Cases and Fatalities in', fontsize = 30) 
    fig.suptitle(country_name, fontsize = 30)
    #Left plot
    ax1.plot(country_fun(country_name).index, country_fun(country_name)['ConfirmedCases'], color = 'purple', marker = 'o',linewidth = 2)
    ax1.set(xlabel = 'Date',
            ylabel = 'ConfirmedCases')
    ax1.set_xticks(np.arange(0, 80,  step = 16))
    ax1.legend(['ConfirmedCases'])
    ax1.grid()
    #Right plot
    ax2.plot(country_fun(country_name).index, country_fun(country_name)['Fatalities'], color = 'orange', marker = 'o', linewidth = 2)
    ax2.set(xlabel = 'Date',
            ylabel = 'Fatalities')
    ax2.set_xticks(np.arange(0, 80,  step = 16))
    ax2.legend(['Fatalities'])
    ax2.grid()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"
    plt.show()    




country_plot_fun('China')
country_plot_fun('US')
country_plot_fun('Italy')
country_plot_fun('Spain')




fig, (ax1, ax2) = plt.subplots(1,2, figsize = (22,8))
fig.suptitle('Trends of Confirmed Cases and Fatalities', fontsize = 30)

#Left plot
ax1.plot(country_fun('Italy').index, country_fun('Italy')['ConfirmedCases'], color = 'purple', marker = 'o',linewidth = 2)
ax1.plot(country_fun('US').index, country_fun('US')['ConfirmedCases'], color = 'blue', marker = 'o',linewidth = 2)
ax1.plot(country_fun('China').index, country_fun('China')['ConfirmedCases'], color = 'red', marker = 'o',linewidth = 2)
ax1.plot(country_fun('Spain').index, country_fun('Spain')['ConfirmedCases'], color = 'green', marker = 'o',linewidth = 2)

ax1.set(xlabel = 'Date',
        ylabel = 'ConfirmedCases')
ax1.set_xticks(np.arange(0, 80,  step = 20))
ax1.legend(['Italy', 'US', 'China','Spain'])
ax1.grid()
#Right plot
ax2.plot(country_fun('Italy').index, country_fun('Italy')['Fatalities'], color = 'purple', marker = 'o',linewidth = 2)
ax2.plot(country_fun('US').index, country_fun('US')['Fatalities'], color = 'blue', marker = 'o',linewidth = 2)
ax2.plot(country_fun('China').index, country_fun('China')['Fatalities'], color = 'red', marker = 'o',linewidth = 2)
ax2.plot(country_fun('Spain').index, country_fun('Spain')['Fatalities'], color = 'green', marker = 'o',linewidth = 2)
ax2.set(xlabel = 'Date',
        ylabel = 'Fatalities')
ax2.set_xticks(np.arange(0, 80,  step = 20))
ax2.legend(['Italy', 'US', 'China','Spain'])
ax2.grid()

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "20"

plt.show()




def country_state_fun(country_name, state_name):
    df_country = train_data_covid.loc[(train_data_covid['Country_Region'] == country_name)]
    df_country_state = df_country.loc[(df_country['Province_State']) == state_name]
    state_conf =  df_country_state.groupby(['Date'])[['ConfirmedCases']].sum().sort_values('ConfirmedCases')
    state_fatal = df_country_state.groupby(['Date'])[['Fatalities']].sum().sort_values('Fatalities') 
    state_confirmed_fatal = state_conf.join((state_fatal), how = 'outer')
    return state_confirmed_fatal




fig, (ax1, ax2) = plt.subplots(1,2, figsize = (22,8))
fig.suptitle('Trends of Confirmed Cases and Fatalities in US states', fontsize = 30)

#Left plot
#ax1.plot(country_state_fun("US", "New York").index, country_state_fun("US", "New York")['ConfirmedCases'], color = 'purple', marker = 'o',linewidth = 2)
ax1.plot(country_state_fun("US", "Washington").index, country_state_fun("US", "Washington")['ConfirmedCases'], color = 'blue', marker = 'o',linewidth = 2)
ax1.plot(country_state_fun("US", "Illinois").index, country_state_fun("US", "Illinois")['ConfirmedCases'], color = 'red', marker = 'o',linewidth = 2)
ax1.plot(country_state_fun("US", "California").index, country_state_fun("US", "California")['ConfirmedCases'], color = 'green', marker = 'o',linewidth = 2)
ax1.plot(country_state_fun("US", "Florida").index, country_state_fun("US", "Florida")['ConfirmedCases'], color = 'black', marker = 'o',linewidth = 2)

ax1.set(xlabel = 'Date',
        ylabel = 'ConfirmedCases')
ax1.set_xticks(np.arange(0, 90,  step = 20))
ax1.legend(["Washington", "Illinois","California", "Florida"])
ax1.grid()
#Right plot
#ax2.plot(country_state_fun("US", "New York").index, country_state_fun("US", "New York")['Fatalities'], color = 'purple', marker = 'o',linewidth = 2)
ax2.plot(country_state_fun("US", "Washington").index, country_state_fun("US", "Washington")['Fatalities'], color = 'blue', marker = 'o',linewidth = 2)
ax2.plot(country_state_fun("US", "Illinois").index, country_state_fun("US", "Illinois")['Fatalities'], color = 'red', marker = 'o',linewidth = 2)
ax2.plot(country_state_fun("US", "California").index, country_state_fun("US", "California")['Fatalities'], color = 'green', marker = 'o',linewidth = 2)
ax2.plot(country_state_fun("US", "Florida").index, country_state_fun("US", "Florida")['Fatalities'], color = 'black', marker = 'o',linewidth = 2)


ax2.set(xlabel = 'Date',
        ylabel = 'Fatalities')
ax2.set_xticks(np.arange(0, 90,  step = 20))
ax2.legend(["Washington", "Illinois","California", "Florida"])
ax2.grid()

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "20"

plt.show()




#Using population dataset from kaggle
population_df = population_data.copy()
population_df = population_df.fillna('NA')
display(population_df.head())
all_countries = list(train_data["Country_Region"].unique())
display(len(all_countries))




def population_country(country_name, state_name):
    pop_country = population_df.loc[(population_df['Country.Region']==country_name)]
    pop_state = pop_country.loc[(pop_country['Province.State'] == state_name)]
    pop = pop_state.Population
    return pop.values[0]
print(population_country('Botswana', 'NA'))    




def country_state_fun(country_name, state_name):
    df_country = train_data_covid.loc[(train_data_covid['Country_Region'] == country_name)]
    df_country_state = df_country.loc[(df_country['Province_State']) == state_name]
    state_conf =  df_country_state.groupby(['Date'])[['ConfirmedCases']].sum().sort_values('ConfirmedCases')
    state_fatal = df_country_state.groupby(['Date'])[['Fatalities']].sum().sort_values('Fatalities') 
    state_confirmed_fatal = state_conf.join((state_fatal), how = 'outer')
    return state_confirmed_fatal




display(country_state_fun("US", "Illinois"))




display(country_state_fun("Afghanistan", 'NA'))




get_ipython().system('ls ../input/sir-model/')




Image("../input/sir-model/sir_model_image.png")




param_data_new = parameter_data.copy()
param_data_new = param_data_new.fillna('NA')
for i in range(len(param_data_new)):
    pro_txt = param_data_new['Province'][i]
    pro_txt = pro_txt.replace('_', ' ')
    param_data_new['Province'][i] = pro_txt
    
para_state_list = param_data_new['Province'].unique()
para_country_list = param_data_new['Country'].unique()




def parameter_extract(country_name, state_name):
    para_country = param_data_new.loc[(param_data_new['Country']==country_name)]
    para_state = para_country[(para_country['Province'] == state_name)]
    beta_in = para_state.Beta.values[0]
    gamma_in = para_state.Gamma.values[0] 
    return beta_in, gamma_in




# SIR Model 
def SIR_DEQ(y, time, beta, k, N):
    DS = -beta * y[0] * y[1]/N
    DI = (beta * y[0] * y[1] - k * y[1])/N
    DR = k * y[1]/N
    return [DS, DI, DR]
# Parameters
t0 = 0 
tmax = pd.date_range(test_data['Date'].min(),test_data['Date'].max()).shape[0]
dt = 1
# Rate of infection
#beta = 0.165
# Rate of recovery
#k = 1/12
time = np.arange(t0, tmax, dt)

df_final = pd.DataFrame(columns = ['ConfirmedCases', 'Fatalities'])

for cout in all_countries:
    all_state_test = test_data_covid.copy()
    all_states = all_state_test.loc[(all_state_test["Country_Region"] == cout)]
    all_states_list = list(all_states["Province_State"].unique())
    for char in all_states_list:
        df_new = pd.DataFrame(columns = ['ConfirmedCases', 'Fatalities'])
        N = population_country(cout, char)
        I0 = int(country_state_fun(cout, char)[country_state_fun(cout, char).index == test_data['Date'].min()]['ConfirmedCases'])
        R0 = int(country_state_fun(cout, char)[country_state_fun(cout, char).index == test_data['Date'].min()]['Fatalities'])
        S0 = N-I0-R0 # initial population of susceptible individual
        init_state = [S0, I0, R0]
        if char in para_state_list and cout in para_country_list:
            beta = parameter_extract(cout, char)[0]
            k = parameter_extract(cout, char)[1]
        else:
            beta = 0.165
            k = 1/12
        args = (beta, k, N)
        solution = odeint(SIR_DEQ, init_state, time, args)
        df_new = pd.DataFrame({'ConfirmedCases':solution[:,1], 'Fatalities': solution[:,2]})
        df_final_all = pd.concat([df_final, df_new], axis = 0)         
        df_final = df_final_all




print(df_final.head())
display(len(df_final))




len(submission_file), len(df_final)




submission_file['ConfirmedCases'] = df_final['ConfirmedCases'].values
submission_file['Fatalities'] = df_final['Fatalities'].values
display(submission_file.head())
display(submission_file.tail())




submission_file.to_csv('submission.csv', index=False)

