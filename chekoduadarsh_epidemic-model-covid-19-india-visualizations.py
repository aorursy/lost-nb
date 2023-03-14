#!/usr/bin/env python
# coding: utf-8



'''
COVID-19 Data analysis
'''

from IPython.core.display import HTML
import folium
import datetime
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import lxml.html as lh
import pandas as pd
import re
import time
import psutil
import json

import numpy as np
from PIL import Image
import os
from os import path
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import plotly.graph_objects as go
from pandas.plotting import register_matplotlib_converters
import plotly.express as px
from IPython.display import display, Markdown, Latex
import matplotlib as plot
from matplotlib.pyplot import figure
import seaborn as sns

register_matplotlib_converters()
from IPython.display import Markdown


dataset = pd.DataFrame()




import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")




def statelat(sate):
    lat = {
        "Maharashtra":19.7515,
        "Delhi":28.7041,
        "Tamil Nadu":11.1271,
        "Rajasthan":27.0238,
        "Madhya Pradesh":22.9734,
        "Telangana":18.1124,
        "Gujarat":22.2587,
        "Uttar Pradesh":26.8467,
        "Andhra Pradesh":15.9129,
        "Kerala":10.8505,
        "Jammu and Kashmir":33.7782,
        "Karnataka":15.3173,
        "Haryana":29.0588,
        "Punjab":31.1471,
        "West Bengal":22.9868,
        "Bihar":25.0961,
        "Odisha":20.9517,
        "Uttarakhand":30.0668,
        "Himachal Pradesh":31.1048,
        "Assam":26.2006,
        "Chhattisgarh":22.0797,
        "Chandigarh":30.7333,
        "Jharkhand":23.6102,
        "Ladakh":34.152588,
        "Andaman and Nicobar Islands":11.7401,
        "Goa":15.2993,
        "Puducherry":11.9416,
        "Manipur":24.6637,
        "Tripura":23.9408,
        "Mizoram":23.1645,
        "Arunachal Pradesh":28.2180,
        "Dadra and Nagar Haveli":20.1809,
        "Nagaland":26.1584,
        "Daman and Diu":20.4283,
        "Lakshadweep":8.295441,
        "Meghalaya":25.4670,
        "Sikkim":27.5330
    }
    return lat[sate]




def statelong(sate):
    long = {
        "Maharashtra":75.7139,
        "Delhi":77.1025,
        "Tamil Nadu":78.6569,
        "Rajasthan":74.2179,
        "Madhya Pradesh":78.6569,
        "Telangana":79.0193,
        "Gujarat":71.1924,
        "Uttar Pradesh":80.9462,
        "Andhra Pradesh":79.7400,
        "Kerala":76.2711,
        "Jammu and Kashmir":76.5762,
        "Karnataka":75.7139,
        "Haryana":76.0856,
        "Punjab":75.3412,
        "West Bengal":87.8550,
        "Bihar":85.3131,
        "Odisha":85.0985,
        "Uttarakhand":79.0193,
        "Himachal Pradesh":77.1734,
        "Assam":92.9376,
        "Chhattisgarh":82.1409,
        "Chandigarh":76.7794,
        "Jharkhand":85.2799,
        "Ladakh":77.577049,
        "Andaman and Nicobar Islands":92.6586,
        "Goa":74.1240,
        "Puducherry":79.8083,
        "Manipur":93.9063,
        "Tripura":91.9882,
        "Mizoram":92.9376,
        "Arunachal Pradesh":94.7278,
        "Dadra and Nagar Haveli":73.0169,
        "Nagaland":94.5624,
        "Daman and Diu":72.8397,
        "Lakshadweep":73.048973,
        "Meghalaya":91.3662,
        "Sikkim":88.5122
    }
    return long[sate]




get_ipython().run_cell_magic('HTML', '', '<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2061549" data-url="https://flo.uri.sh/visualisation/2061549/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')




df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
data = df.copy()
data['Date'] = data['Date'].apply(pd.to_datetime)
data.drop(['Sno', 'Time'],axis=1,inplace=True)

# collect present data
from datetime import date
data_apr = data[data['Date'] > pd.Timestamp(date(2020,4,12))]

# prepaing data state wise
state_cases = data_apr.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()
state_cases['Active'] = state_cases['Confirmed'] - (state_cases['Deaths']- state_cases['Cured'])
state_cases["Death Rate (per 100)"] = np.round(100*state_cases["Deaths"]/state_cases["Confirmed"],2)
state_cases["Cure Rate (per 100)"] = np.round(100*state_cases["Cured"]/state_cases["Confirmed"],2)




indiaLiveJson = 'https://api.covid19india.org/data.json'
r = requests.get(indiaLiveJson)
indiaData = r.json()


display(Markdown("# Todays Condition in India: Testing for of 2019-nCoV"))
if r.json()['tested'][len(r.json()['tested'])-1]['samplereportedtoday'] != '':
    display(Markdown("**Time Stamp**                   :<font color='red'>{}</font>".format(r.json()['tested'][len(r.json()['tested'])-1]['updatetimestamp'])))
    display(Markdown("**Individals Tested**            :<font color='red'>{}</font>".format(r.json()['tested'][len(r.json()['tested'])-1]['samplereportedtoday'])))
    display(Markdown("**Individals Found Positive**    :<font color='red'>{}</font>".format(r.json()['tested'][len(r.json()['tested'])-1]['positivecasesfromsamplesreported'])))
else:
    display(Markdown("**Time Stamp**                   :<font color='red'>{}</font>".format(r.json()['tested'][len(r.json()['tested'])-2]['updatetimestamp'])))
    display(Markdown("**Individals Tested**            :<font color='red'>{}</font>".format(r.json()['tested'][len(r.json()['tested'])-2]['samplereportedtoday'])))
    display(Markdown("**Individals Found Positive**    :<font color='red'>{}</font>".format(r.json()['tested'][len(r.json()['tested'])-2]['positivecasesfromsamplesreported'])))
    
    
display(Markdown("# Todays Condition in India: 2019-nCoV"))
display(Markdown("**Time Stamp**               :<font color='red'>{}</font>".format(r.json()['cases_time_series'][len(r.json()['cases_time_series'])-1]['date'])))
display(Markdown("**Positive Reported Today**  :<font color='red'>{}</font>".format(r.json()['cases_time_series'][len(r.json()['cases_time_series'])-1]['dailyconfirmed'])))
display(Markdown("**Deceased Today**           :<font color='red'>{}</font>".format(r.json()['cases_time_series'][len(r.json()['cases_time_series'])-1]['dailydeceased'])))
display(Markdown("**Recoverd Today**           :<font color='red'>{}</font>".format(r.json()['cases_time_series'][len(r.json()['cases_time_series'])-1]['dailyrecovered'])))         




testingHistory = pd.DataFrame()
testingNO = []
testedPos = []
timeStamp = []
for index in range(len(indiaData['tested'])):
    try:
        testingNO.append(int(re.sub(',','',indiaData['tested'][index]['totalindividualstested'])))
        testedPos.append(int(re.sub(',','',indiaData['tested'][index]['totalpositivecases'])))
    except:
        testingNO.append(testingNO[len(testingNO)-1])
        testedPos.append(testedPos[len(testedPos)-1])
        
    timeStamp.append(indiaData['tested'][index]['updatetimestamp'][:-9])
    
testingHistory['testing_no'] = testingNO[:-1]
testingHistory['testing_pos'] = testedPos
testingHistory['time_stamp'] = timeStamp

testingHistory.drop_duplicates(subset ="time_stamp", 
                     keep = False, inplace = True) 


fig = go.Figure()

fig = fig.add_trace(go.Scatter(y=testingHistory['testing_no'], x=testingHistory['time_stamp'],
                    mode='lines+markers',
                    name='Testing Pattern'))

fig = fig.add_trace(go.Scatter(y=testingHistory['testing_pos'], x=testingHistory['time_stamp'],
                    mode='lines+markers',
                    name='Tested Positive'))

fig = fig.update_layout(
    title="India COVID-19 Testing History",
    xaxis_title="Testing",
    yaxis_title="Date",
    
)


fig.show()




total_test = pd.read_csv('../input/globaltestcovid19/full-list-total-tests-for-covid-19.csv')
total_test['Date'] = total_test['Date'].apply(pd.to_datetime)
total_test.set_index(["Entity"], inplace = True)
total_test = total_test.loc[['India','South Korea']]
total_test.reset_index(inplace = True)
total_test.sort_values('Date', ascending= True,inplace=True)

# plot
'''
fig = go.Figure()

fig = fig.add_trace(go.Scatter(y=total_test['Cumulative total tests'], x=total_test['Date'],
                    mode='lines+markers',
                    name='sk'))

fig = fig.add_trace(go.Scatter(y=testingHistory['testing_pos'], x=testingHistory['time_stamp'],
                    mode='lines+markers',
                    name='in'))
'''


fig = px.scatter(total_test, 
                 x='Date', 
                 y='Cumulative total tests', 
                 color='Entity')

fig.update_traces(marker=dict(size=3.5),
                  mode='lines+markers')


fig = fig.add_trace(go.Scatter(y=testingHistory['testing_no'], x=testingHistory['time_stamp'],
                    mode='lines+markers',
                    name='Testing Pattern'))

fig.add_annotation( # add a text callout with arrow
    text="Initial Rapid Testing pattern", x='2020-03-01', y=100000, arrowhead=1, showarrow=True
)


fig.add_annotation( # add a text callout with arrow
    text="Lack of inital rapid testing", x='2020-04-03', y=69245, arrowhead=4, ax=0,
            ay=-40,showarrow=True
)


fig.update_layout(template = 'plotly_white', title_text = '<b>Total Tests for COVID-19</b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))
fig.show()




indiaConfirmed = []
indiaRecovered = []
indiaDeseased = []
timeStamp = []
for index in range(len(indiaData['cases_time_series'])):
    indiaConfirmed.append(int(re.sub(',','',indiaData['cases_time_series'][index]['totalconfirmed'])))
    indiaRecovered.append(int(re.sub(',','',indiaData['cases_time_series'][index]['totalrecovered'])))
    indiaDeseased.append(int(re.sub(',','',indiaData['cases_time_series'][index]['totaldeceased'])))
    
    timeStamp.append(indiaData['cases_time_series'][index]['date'])
    

fig = go.Figure()
#fig = px.scatter(testingHistory,x="time_stamp", y="testing_no")

fig = fig.add_trace(go.Scatter(x=timeStamp, y=indiaConfirmed,
                    mode='lines+markers',
                    name='Confirmed Cases'))
fig = fig.add_trace(go.Scatter(x=timeStamp, y=indiaRecovered,
                    mode='lines+markers',
                    name='Recoverd Patients'))
fig = fig.add_trace(go.Scatter(x=timeStamp, y=indiaDeseased,
                    mode='lines+markers',
                    name='Deseased Patients'))

fig = fig.update_layout(
    title="India COVID-19 ",
    xaxis_title="Date",
    yaxis_title="Testing",
    
)


fig.show()




indiaPrediction = pd.DataFrame()
indiaConfirmed = []
indiaRecovered = []
indiaDeseased = []
timeStamp = []
for index in range(len(indiaData['cases_time_series'])):
    indiaConfirmed.append(int(re.sub(',','',indiaData['cases_time_series'][index]['totalconfirmed'])))
    indiaRecovered.append(int(re.sub(',','',indiaData['cases_time_series'][index]['totalrecovered'])))
    indiaDeseased.append(int(re.sub(',','',indiaData['cases_time_series'][index]['totaldeceased'])))
    
    timeStamp.append(datetime.strptime(datetime.strptime(indiaData['cases_time_series'][index]['date']+'2020',"%d %B %Y").strftime("%m/%d/%Y"),'%m/%d/%Y'))

    #print datetime.datetime.strptime("%d %B %Y","April 28,2015").strftime("%m/%d/%y")
indiaPrediction['total confirmed'] = indiaConfirmed
indiaPrediction['total recovered'] = indiaRecovered
indiaPrediction['total deceased'] = indiaDeseased
indiaPrediction['Date'] = timeStamp

indiaPrediction['Date'] = indiaPrediction['Date'].map(datetime.toordinal).tolist()



x = np.array(indiaPrediction['Date'])
y = np.array(indiaPrediction['total confirmed'])

z = np.polyfit(x, y,15)
f = np.poly1d(z)



x_new = np.linspace(x[0], x[-1]+(150),50)
y_new = f(x_new)


x = np.array(indiaPrediction['Date'])
y = np.array(indiaPrediction['total recovered'])

z = np.polyfit(x, y,15)
f = np.poly1d(z)



x_new2 = np.linspace(x[0], x[-1]+(150),50)
y_new2 = f(x_new)


x = np.array(indiaPrediction['Date'])
y = np.array(indiaPrediction['total deceased'])

z = np.polyfit(x, y,15)
f = np.poly1d(z)



x_new3 = np.linspace(x[0], x[-1]+(150),50)
y_new3 = f(x_new)


indiaPrediction['Date'] = indiaPrediction['Date'].apply(lambda x: datetime.fromordinal(x))


pred1 = pd.DataFrame(columns = ['Date'])
pred1['Date'] = x_new

y1 = pd.DataFrame(columns = ['Count'])
y1['Count'] = y_new


pred2 = pd.DataFrame(columns = ['Date'])
pred2['Date'] = x_new2

y2 = pd.DataFrame(columns = ['Count'])
y2['Count'] = y_new2


pred3 = pd.DataFrame(columns = ['Date'])
pred3['Date'] = x_new3

y3 = pd.DataFrame(columns = ['Count'])
y3['Count'] = y_new3

pred1['Date'] = pred1['Date'].apply(lambda x: datetime.fromordinal(int(x)))
pred2['Date'] = pred2['Date'].apply(lambda x: datetime.fromordinal(int(x)))
pred3['Date'] = pred3['Date'].apply(lambda x: datetime.fromordinal(int(x)))

#fig = px.scatter(indiaPrediction, y="total confirmed", x="Date")



fig = go.Figure()
#fig = px.scatter(testingHistory,x="time_stamp", y="testing_no")

fig = fig.add_trace(go.Scatter(x=timeStamp, y=indiaConfirmed,
                    mode='markers',
                    name='Confirmed Cases'))
fig = fig.add_trace(go.Scatter(x=timeStamp, y=indiaRecovered,
                    mode='markers',
                    name='Recoverd Patients'))
fig = fig.add_trace(go.Scatter(x=timeStamp, y=indiaDeseased,
                    mode='markers',
                    name='Deseased Patients'))

Hospitalbeds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
Hospitalbeds = Hospitalbeds[-1:]
totalBeds = sum((Hospitalbeds['NumRuralBeds_NHP18']+Hospitalbeds['NumUrbanBeds_NHP18']).tolist())

fig = fig.add_shape(
        # Line Horizontal
            type="line",
            x0=min(pred1['Date'].tolist()),
            y0=totalBeds,
            x1=max(pred1['Date'].tolist()),
            y1=totalBeds,
            line=dict(
                color="LightSeaGreen",
                width=4,
                dash="dashdot",
            ),
    )

fig = fig.add_trace(go.Scatter(
    x=pred1['Date'].tolist(),
    y=y1['Count'].tolist(),
    mode='lines',
    name='Coronavirus Infected Trend Line'
))

fig = fig.add_trace(go.Scatter(
    x=pred2['Date'].tolist(),
    y=y2['Count'].tolist(),
    mode='lines',
    name='Coronavirus Cured Trend Line'
))

fig = fig.add_trace(go.Scatter(
    x=pred3['Date'].tolist(),
    y=y3['Count'].tolist(),
    mode='lines',
    name='Coronavirus deceased Trend Line'
))

#fig = add_trace(trendline)

fig = fig.add_annotation( # add a text callout with arrow
    text="Total Hospital beds availble in India", x='2020-07-30', y=709250, arrowhead=4, ax=0,
            ay=-40,showarrow=True
)

fig = fig.update_layout(
    title="India COVID-19 ",
    xaxis_title="Date",
    yaxis_title="Cases"  
)


fig.show()




def base_seir_model(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (beta*S[-1]*I[-1])*dt
        next_E = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return [S,E, I,R]




# Define parameters
t_max = 100
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
init_vals = 1 - 1/N, 1/N, 0, 0
alpha = 0.2
beta = 1.75
gamma = 0.5
params = alpha, beta, gamma
# Run simulation
results = base_seir_model(init_vals, params, t)
#results

fig = go.Figure()
#fig = px.scatter(testingHistory,x="time_stamp", y="testing_no")

fig = fig.add_trace(go.Scatter(x=t, y=results[0],
                    mode='lines',
                    name='Susceptibles'))
fig = fig.add_trace(go.Scatter(x=t, y=results[1],
                    mode='lines',
                    name='Exposed'))
fig = fig.add_trace(go.Scatter(x=t, y=results[2],
                    mode='lines',
                    name='Infectious'))
fig = fig.add_trace(go.Scatter(x=t, y=results[3],
                    mode='lines',
                    name='Recovered'))
fig = fig.update_layout(
    yaxis_title ="Populaton Fraction",
    xaxis_title ="Days"  
)

fig.show()




def seir_model_with_soc_dist(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (rho*beta*S[-1]*I[-1])*dt
        next_E = E[-1] + (rho*beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return [S, E, I, R]




#sum((Hospitalbeds['NumRuralBeds_NHP18']+Hospitalbeds['NumUrbanBeds_NHP18']).tolist())/1352600000




# Define parameters
t_max = 100
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
init_vals = 1 - 1/N, 1/N, 0, 0
alpha = 0.2
beta = 1.75
gamma = 0.5
rho = 0.6
params = alpha, beta, gamma, rho
# Run simulation
results = seir_model_with_soc_dist(init_vals, params, t)
#results

fig = go.Figure()

'''Hospitalbeds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
Hospitalbeds = Hospitalbeds[-1:]
totalBedsperpatient = 5/1000
fig = fig.add_shape(
        # Line Horizontal
            type="line",
            x0=0,
            y0=totalBedsperpatient,
            x1=t_max,
            y1=totalBedsperpatient,
            line=dict(
                color="LightSeaGreen",
                width=4,
                dash="dashdot",
            ),
    )'''


'''fig = fig.add_annotation( # add a text callout with arrow
    text="Total Hospital beds availble in India", x='2020-07-30', y=709250, arrowhead=4, ax=0,
            ay=-40,showarrow=True
)'''

rho = 1
params = alpha, beta, gamma, rho
# Run simulation
results = seir_model_with_soc_dist(init_vals, params, t)

'''fig = fig.add_trace(go.Scatter(x=t, y=results[0],
                    mode='lines',
                    name='Susceptibles (ρ  = 1)'))'''
fig = fig.add_trace(go.Scatter(x=t, y=results[1],
                    mode='lines',
                    name='Exposed (ρ  = 1)',
                   line = dict(color='red', width=1, dash='dash')))
fig = fig.add_trace(go.Scatter(x=t, y=results[2],
                    mode='lines',
                    name='Infectious (ρ  = 1)',
                   line = dict(color='red', width=1, dash='dash')))
'''fig = fig.add_trace(go.Scatter(x=t, y=results[3],
                    mode='lines',
                    name='Recovered (ρ  = 1)'))'''


rho = 0.8
params = alpha, beta, gamma, rho
# Run simulation
results = seir_model_with_soc_dist(init_vals, params, t)

'''fig = fig.add_trace(go.Scatter(x=t, y=results[0],
                    mode='lines',
                    name='Susceptibles (ρ  = 0.8)'))'''
fig = fig.add_trace(go.Scatter(x=t, y=results[1],
                    mode='lines',
                    name='Exposed (ρ  = 0.8)',
                   line = dict(color='blue', width=1, dash='dash')))
fig = fig.add_trace(go.Scatter(x=t, y=results[2],
                    mode='lines',
                    name='Infectious (ρ  = 0.8)',
                   line = dict(color='blue', width=1, dash='dash')))
'''fig = fig.add_trace(go.Scatter(x=t, y=results[3],
                    mode='lines',
                    name='Recovered (ρ  = 0.8)'))
'''
rho = 0.5
params = alpha, beta, gamma, rho
# Run simulation
results = seir_model_with_soc_dist(init_vals, params, t)

'''fig = fig.add_trace(go.Scatter(x=t, y=results[0],
                    mode='lines',
                    name='Susceptibles (ρ  = 0.5)'))'''
fig = fig.add_trace(go.Scatter(x=t, y=results[1],
                    mode='lines',
                    name='Exposed (ρ  = 0.5)',
                   line = dict(color='green', width=1)))
fig = fig.add_trace(go.Scatter(x=t, y=results[2],
                    mode='lines',
                    name='Infectious (ρ  = 0.5)',
                   line = dict(color='green', width=1)))
'''fig = fig.add_trace(go.Scatter(x=t, y=results[3],
                    mode='lines',
                    name='Recovered (ρ  = 0.5)'))'''

fig = fig.update_layout(
    xaxis_title="Populaton Fraction",
    yaxis_title="Days"  
)
fig.show()




def base_sird_model(init_vals, params, t):
    S_0, E_0, I_0, R_0,D_0 = init_vals
    S, E, I, R,D = [S_0], [E_0], [I_0], [R_0], [D_0]
    alpha, beta, gamma, myu = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (beta*S[-1]*I[-1])*dt
        next_E = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt - (myu*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        next_D = D[-1] + (myu*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
        D.append(next_D)
    return [S,I,R,D]




# Define parameters
t_max = 100
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
init_vals = 1 - 1/N, 1/N, 0, 0,0
alpha = 0.2
beta = 1.75
gamma = 0.5
myu = 0.05
params = alpha, beta, gamma,myu
# Run simulation
results = base_sird_model(init_vals, params, t)
#results

fig = go.Figure()
#fig = px.scatter(testingHistory,x="time_stamp", y="testing_no")

fig = fig.add_trace(go.Scatter(x=t, y=results[0],
                    mode='lines',
                    name='Susceptibles'))
fig = fig.add_trace(go.Scatter(x=t, y=results[1],
                    mode='lines',
                    name='Infectious'))
fig = fig.add_trace(go.Scatter(x=t, y=results[2],
                    mode='lines',
                    name='Recovered'))
fig = fig.add_trace(go.Scatter(x=t, y=results[3],
                    mode='lines',
                    name='Dead'))
fig = fig.update_layout(
    yaxis_title ="Populaton Fraction",
    xaxis_title ="Days"  
)

fig.show()




def sird_model_with_soc_dist(init_vals, params, t):
    S_0, E_0, I_0, R_0,D_0 = init_vals
    S, E, I, R, D = [S_0], [E_0], [I_0], [R_0], [D_0]
    alpha, beta, gamma, rho, myu = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (rho*beta*S[-1]*I[-1])*dt
        next_E = E[-1] + (rho*beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        next_D = D[-1] + (myu*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return [S, I, R, D]




# Define parameters
t_max = 100
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
init_vals = 1 - 1/N, 1/N, 0, 0, 0
alpha = 0.2
beta = 1.75
gamma = 0.5
rho = 0.6
myu = 0.05
params = alpha, beta, gamma, rho, myu
# Run simulation
results = sird_model_with_soc_dist(init_vals, params, t)
#results

fig = go.Figure()

'''Hospitalbeds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
Hospitalbeds = Hospitalbeds[-1:]
totalBedsperpatient = 5/1000
fig = fig.add_shape(
        # Line Horizontal
            type="line",
            x0=0,
            y0=totalBedsperpatient,
            x1=t_max,
            y1=totalBedsperpatient,
            line=dict(
                color="LightSeaGreen",
                width=4,
                dash="dashdot",
            ),
    )'''


'''fig = fig.add_annotation( # add a text callout with arrow
    text="Total Hospital beds availble in India", x='2020-07-30', y=709250, arrowhead=4, ax=0,
            ay=-40,showarrow=True
)'''

rho = 1
params = alpha, beta, gamma, rho, myu
# Run simulation
results = sird_model_with_soc_dist(init_vals, params, t)

'''fig = fig.add_trace(go.Scatter(x=t, y=results[0],
                    mode='lines',
                    name='Susceptibles (ρ  = 1)'))'''
fig = fig.add_trace(go.Scatter(x=t, y=results[1],
                    mode='lines',
                    name='Infectious (ρ  = 1)',
                   line = dict(color='red', width=1, dash='dash')))
fig = fig.add_trace(go.Scatter(x=t, y=results[2],
                    mode='lines',
                    name='Recovered (ρ  = 1)',
                   line = dict(color='red', width=1, dash='dash')))
fig = fig.add_trace(go.Scatter(x=t, y=results[3],
                    mode='lines',
                    name='Dead (ρ  = 1)'))


rho = 0.8
params = alpha, beta, gamma, rho, myu
# Run simulation
results = sird_model_with_soc_dist(init_vals, params, t)

'''fig = fig.add_trace(go.Scatter(x=t, y=results[0],
                    mode='lines',
                    name='Susceptibles (ρ  = 0.8)'))'''
fig = fig.add_trace(go.Scatter(x=t, y=results[1],
                    mode='lines',
                    name='Infectious (ρ  = 0.8)',
                   line = dict(color='blue', width=1, dash='dash')))
fig = fig.add_trace(go.Scatter(x=t, y=results[2],
                    mode='lines',
                    name='Recovered (ρ  = 0.8)',
                   line = dict(color='blue', width=1, dash='dash')))
fig = fig.add_trace(go.Scatter(x=t, y=results[3],
                    mode='lines',
                    name='Dead (ρ  = 0.8)'))

rho = 0.5
params = alpha, beta, gamma, rho, myu
# Run simulation
results = sird_model_with_soc_dist(init_vals, params, t)

'''fig = fig.add_trace(go.Scatter(x=t, y=results[0],
                    mode='lines',
                    name='Susceptibles (ρ  = 0.5)'))'''
fig = fig.add_trace(go.Scatter(x=t, y=results[1],
                    mode='lines',
                    name='Infectious (ρ  = 0.5)',
                   line = dict(color='green', width=1)))
fig = fig.add_trace(go.Scatter(x=t, y=results[2],
                    mode='lines',
                    name='Recovered (ρ  = 0.5)',
                   line = dict(color='green', width=1)))
fig = fig.add_trace(go.Scatter(x=t, y=results[3],
                    mode='lines',
                    name='Dead (ρ  = 0.5)'))

fig = fig.update_layout(
    xaxis_title="Populaton Fraction",
    yaxis_title="Days"  
)
fig.show()




display(Markdown("** STATE WISE CONFIRMED, DEATH AND CURED CASES of 2019-nCoV**"))
state_cases.sort_values('Confirmed', ascending= False).fillna(0).style.background_gradient(cmap='YlOrBr',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Cured"])                        .background_gradient(cmap='Blues',subset=["Active"])                        .background_gradient(cmap='Purples',subset=["Death Rate (per 100)"])                        .background_gradient(cmap='Greens',subset=["Cure Rate (per 100)"])




states = []
active = []
confirmed = []
deaths = []
for index in range(len(indiaData['statewise'])):
    if index == 0:
        continue
    states.append(str(re.sub(',','',indiaData['statewise'][index]['state'])))
    active.append(int(re.sub(',','',indiaData['statewise'][index]['active'])))
    confirmed.append(int(re.sub(',','',indiaData['statewise'][index]['confirmed'])))
    deaths.append(int(re.sub(',','',indiaData['statewise'][index]['deaths'])))
    
sates = state_cases
india_map = pd.DataFrame()




india_map['States'] = states
india_map['lat'] = india_map['States'].apply(lambda x : statelat(x))
india_map['long'] = india_map['States'].apply(lambda x : statelong(x))
india_map['Confirmed'] = confirmed
india_map['Recovered'] = list(np.array(confirmed) - np.array(active))
india_map['Deaths'] = deaths




indiaMap = folium.Map(location=[23,80], tiles="Stamen Toner", zoom_start=4)

for lat, lon, value1,value2,value3, name in zip(india_map['lat'], india_map['long'], india_map['Confirmed'],india_map['Recovered'],india_map['Deaths'], india_map['States']):
    folium.CircleMarker([lat, lon],
                        radius= (int((np.log(value1+1.00001))))*4,
                        popup = ('<strong>States</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases</strong>: ' + str(value1) + '<br>'),
                        color='#ff6600',
                        
                        fill_color='#ff8533',
                        fill_opacity=0.5 ).add_to(indiaMap)
    folium.CircleMarker([lat, lon],
                        radius= (int((np.log(value2+1.00001))))*4,
                        popup = ('<strong>States</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Recovered</strong>: ' + str(value2) + '<br>'),
                        color='#008000',
                        
                        fill_color='#008000',
                        fill_opacity=0.4 ).add_to(indiaMap)
    folium.CircleMarker([lat, lon],
                        radius= (int((np.log(value3+1.00001))))*4,
                        popup = ('<strong>States</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Deaths</strong>: ' + str(value3) + '<br>'),
                        color='#0000A0',
                        
                        fill_color='#0000A0',
                        fill_opacity=0.4 ).add_to(indiaMap)
indiaMap




df = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%y')
df['Date'] = df['Date'].dt.date
df.rename(columns={'Date':'date','State/UnionTerritory':'state','ConfirmedIndianNational':'confirmed_in',                   'ConfirmedForeignNational':'confirmed_fr'}, inplace=True)
df.drop(['Sno','Time'],axis=1,inplace=True)
df['state'] = df.apply(lambda x: 'Nagaland' if x['state']=='Nagaland#' else 'Jharkhand' if x['state']=='Jharkhand#' else x['state'], axis=1)
df = df[df['state']!='Unassigned']
df.reset_index(inplace=True)
df_states = df.copy()
def add_days(df,new_col,basis):
    states = {}
    df[new_col] = 0
    for i in range(len(df_states)):
        if df_states.loc[i,'state'] in states:
            df_states.loc[i,new_col] = (df_states.loc[i,'date'] - states[df_states.loc[i,'state']]).days
        else:
            if df_states.loc[i,basis] > 0:
                states[df_states.loc[i,'state']] = df_states.loc[i,'date']
    return df
df_states = add_days(df_states,'day_since_inf','Confirmed')
df_states = add_days(df_states,'day_since_death','Deaths')
df_states = add_days(df_states,'day_since_cure','Cured')



fig = px.line(df_states,x='day_since_inf',y='Confirmed',color='state',title='Cumulative cases over time')
fig.update_layout(template="simple_white")
fig.update_xaxes(title_text='Days since first infection was reported')
fig.update_yaxes(title_text='No. of Confirmed cases')
fig.show()


fig = px.line(df_states,x='day_since_death',y='Deaths',color='state',title='Cumulative deaths over time')
fig.update_layout(template="simple_white")
fig.update_xaxes(title_text='Days since first death was reported')
fig.update_yaxes(title_text='No. of Confirmed deaths')
fig.show()

fig = px.line(df_states,x='day_since_cure',y='Cured',color='state',title='Cumulative recoveries over time')
fig.update_layout(template="simple_white")
fig.update_xaxes(title_text='Days since first recovery was reported')
fig.update_yaxes(title_text='No. of Confirmed recoveries')
fig.show()




def add_daily_measures(df):
    has_state=False
    if 'state' in df.columns:
        states = []
        has_state = True
    df.loc[0,'Daily Cases'] = df.loc[0,'Confirmed']
    df.loc[0,'Daily Deaths'] = df.loc[0,'Deaths']
    df.loc[0,'Daily Cured'] = df.loc[0,'Cured']
    for i in range(1,len(df)):
        if has_state:
            if df.loc[i,'state'] in states:
                df.loc[i,'Daily Cases'] = df.loc[i,'Confirmed'] - df.loc[i-1,'Confirmed']
                df.loc[i,'Daily Deaths'] = df.loc[i,'Deaths'] - df.loc[i-1,'Deaths'] 
                df.loc[i,'Daily Cured'] = df.loc[i,'Cured'] - df.loc[i-1,'Cured']
            else:
                states.append(df.loc[i,'state'])
                df.loc[i,'Daily Cases'] = df.loc[i,'Confirmed']
                df.loc[i,'Daily Deaths'] = df.loc[i,'Deaths']
                df.loc[i,'Daily Cured'] = df.loc[i,'Cured']
        else:
            df.loc[i,'Daily Cases'] = df.loc[i,'Confirmed'] - df.loc[i-1,'Confirmed']
            df.loc[i,'Daily Deaths'] = df.loc[i,'Deaths'] - df.loc[i-1,'Deaths'] 
            df.loc[i,'Daily Cured'] = df.loc[i,'Cured'] - df.loc[i-1,'Cured']
    #Make the first row as 0 because we don't know the previous value
    df.loc[0,'Daily Cases'] = 0
    df.loc[0,'Daily Deaths'] = 0
    df.loc[0,'Daily Cured'] = 0
    return df




df_states.sort_values(by=['state','date'],inplace=True)
df_states.reset_index(inplace=True,drop=True)
df_states_daily = add_daily_measures(df_states)
df_states_daily.fillna(0,inplace=True)

states = df_states_daily['state'].unique().tolist()
df_roll = pd.DataFrame()
for state in states:
    df_state = df_states_daily[df_states_daily['state']==state]
    df_state['roll_avg_c'] = np.round(df_state['Daily Cases'].rolling(7).mean())
    df_state['roll_avg_d'] = np.round(df_state['Daily Deaths'].rolling(7).mean())
    df_state['roll_avg_r'] = np.round(df_state['Daily Cured'].rolling(7).mean())
    df_roll = df_roll.append(df_state,ignore_index=True)
    
fig = px.line(df_roll,x='day_since_inf',y='roll_avg_c',color='state',title='Daily cases over time')
fig.update_layout(template="simple_white")
fig.update_xaxes(title_text='Days since first infection was reported')
fig.update_yaxes(title_text='7-Day Rolling average')
fig.show()

fig = px.line(df_roll,x='day_since_inf',y='roll_avg_d',color='state',title='Daily deaths over time')
fig.update_layout(template="simple_white")
fig.update_xaxes(title_text='Days since first infection was reported')
fig.update_yaxes(title_text='7-Day Rolling average')
fig.show()

fig = px.line(df_roll,x='day_since_inf',y='roll_avg_r',color='state',title='Daily recoveries over time')
fig.update_layout(template="simple_white")
fig.update_xaxes(title_text='Days since first infection was reported')
fig.update_yaxes(title_text='7-Day Rolling average')
fig.show()




state_cases.sort_values('Confirmed', ascending= False).head(15).style.background_gradient(cmap='YlOrBr',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Cured"])                        .background_gradient(cmap='Blues',subset=["Active"])                        .background_gradient(cmap='Purples',subset=["Death Rate (per 100)"])                        .background_gradient(cmap='Greens',subset=["Cure Rate (per 100)"])




Hospitalbeds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
Hospitalbeds = Hospitalbeds[:-1]


states = []
active = []

for index in range(len(indiaData['statewise'])):
    if index == 0:
        continue
    states.append(str(re.sub(',','',indiaData['statewise'][index]['state'])))
    active.append(int(re.sub(',','',indiaData['statewise'][index]['active'])))

indiaActive = pd.DataFrame()  
indiaActive['States'] = states
indiaActive['active'] = active



state_cases['State/UnionTerritory'] = state_cases['State/UnionTerritory'].apply(lambda x: re.sub(' and ',' & ',x))
activelist = []
for state in Hospitalbeds['State/UT'].tolist():
    try:
        activelist.append(indiaActive[indiaActive['States'] == state]['active'].values[0])
    except:
        try:
            activelist.append(state_cases[state_cases['State/UnionTerritory'] == state]['Active'].values[0])
        except:
            activelist.append(0)

    
Hospitalbeds['active'] = activelist

fig = go.Figure(data=[go.Bar(
            y= (Hospitalbeds['NumRuralBeds_NHP18']+Hospitalbeds['NumUrbanBeds_NHP18']).tolist(), 
            x=Hospitalbeds['State/UT'].tolist(),
            name='Beds availible in states',
            marker_color='#000000'),
            
            go.Bar(
            y=Hospitalbeds['active'].tolist(), 
            x=Hospitalbeds['State/UT'].tolist(),
            name='Positve Cases',
            marker_color='#FF0000')
                     ])

# Change the bar mode
fig.update_layout(barmode='stack', template="ggplot2", title_text = '<b>Sample Tested for COVID-19 in India (Day Wise)</b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))
fig.show()




medicalFacility = pd.read_csv('../input/indian-medical-facility-dataset/phcdoclabasstpharma2012mar.csv')
medicalFacility = medicalFacility.head(len(medicalFacility)-1)
fig = go.Figure(data=[go.Bar(
            y= (medicalFacility['Number of PHCs functioning with 4+ doctors']).tolist(), 
            x=medicalFacility['State/UT'].tolist(),
            name='Number of PHCs functioning with 4+ doctors',
            marker_color='#00ff00'),
            
            go.Bar(
            y= (medicalFacility['Number of PHCs functioning with 3 doctors']).tolist(), 
            x=medicalFacility['State/UT'].tolist(),
            name='Number of PHCs functioning with 3 doctors',
            marker_color='#7bff00'),
                
            go.Bar(
            y= (medicalFacility['Number of PHCs functioning with 2 doctors']).tolist(), 
            x=medicalFacility['State/UT'].tolist(),
            name='Number of PHCs functioning with 2 doctors',
            marker_color='#e5ff00'),
                
            go.Bar(
            y= (medicalFacility['Number of PHCs functioning with 1 doctor']).tolist(), 
            x=medicalFacility['State/UT'].tolist(),
            name='Number of PHCs functioning with 1 doctors',
            marker_color='#ffb300'),
                
            go.Bar(
            y= (medicalFacility['Number of PHCs functioning without doctor']).tolist(), 
            x=medicalFacility['State/UT'].tolist(),
            name='Number of PHCs functioning without doctor',
            marker_color='#ff2f00')
                     ])

# Change the bar mode
fig.update_layout(barmode='stack', template="ggplot2", title_text = '<b>Sample Tested for COVID-19 in India (Day Wise)</b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))
fig.show()




def makeitstring(line):
    newLine = []
    for x in line:
        count = True
        y = ''
        for l in str(x):
            if l == '.' and count == True:
                count = False
                y = y+l
            elif l == '.' and count == False:
                continue
            else:
                y = y+l
                
        newLine.append(str(y))
        
    return newLine

def makeitfloat(line):
    newLine = []
    for x in line:
        newLine.append(float(x))
        
    return newLine




medicalFacility = pd.read_csv('../input/indian-medical-facility-dataset/geocode_health_centre.csv')


medicalFacility['ActiveFlag_C'] = medicalFacility['ActiveFlag_C'].apply(lambda x: 1 if x=='Y' else 0)

#medicalFacility['Latitude'] = makeitfloat(makeitstring(medicalFacility['Latitude']))
#medicalFacility['Longitude'] = makeitfloat(makeitstring(medicalFacility['Latitude']))

fig = px.density_mapbox(medicalFacility, lat='Latitude', lon='Longitude', z='ActiveFlag_C', radius=1,
                        center=dict(lat=20.05, lon=75.9), zoom=5,
                        mapbox_style="stamen-terrain")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},    width=1100,height=1100)

#fig.update_geos()
fig.show()






