#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')




age_details = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')
india_covid_19 = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
hospital_beds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
state_wise_details = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')
individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
icmr_testing = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')
icmr_deatils = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')
india_population = pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')
world_population = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv')




print('Age Details \n',age_details.head(2))
print('India Covid case \n',india_covid_19.head(2))
print('Hospital beds details \n',hospital_beds.head(2))
print('state wise deatils \n',state_wise_details.head(2))
print('indiviual details \n',individual_details.head(2))
print('icmr testing \n',icmr_testing.head(2))
print('icmr details \n',icmr_deatils.head(2))
print('india population details \n',india_population.head(2))
print('Word population details \n',world_population.head(2))
print('confrom cases \n',confirmed_df.head(2))
print('death cases \n',deaths_df.head(2))
print('recoved cases \n',recovered_df.head(2))
print('latest data \n',latest_data.head(2))




print(age_details.isna().any().sum())
print(state_wise_details.isna().any().sum())
print(india_covid_19.isna().any().sum())
print(hospital_beds.isna().any().sum())
print(state_wise_details.isna().any().sum())
print(india_population.isna().any().sum())
print(confirmed_df.isna().any().sum())
print(deaths_df.isna().any().sum())
print(recovered_df.isna().any().sum())




india_covid_19['Date'] = pd.to_datetime(india_covid_19['Date'])
state_wise_details.Date = pd.to_datetime(state_wise_details.Date)
icmr_deatils.dtypes




#world update
confirmed_df.head()




world_confirmed = confirmed_df[confirmed_df.columns[-1:]].sum()--53218964
world_recoverd =  recovered_df[recovered_df.columns[-1:]].sum()
world_death = deaths_df[deaths_df.columns[-1:]].sum()
world_active = world_confirmed - (world_recoverd - world_death)
labels = 'Active','Recovered','Deceased'
sizes = [world_active,world_recoverd,world_death]
color = ['red','green','black']
explode = (0.05, 0.05, 0.05)  # only "explode" the 2nd slice (i.e. 'Hogs')
plt.figure(figsize=(14,14))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('World COVID-19 Cases',fontsize = 20)
plt.axis('equal')  
plt.tight_layout()
plt.show()




import matplotlib.dates as mdates





hotspots = ['China','Germany','Iran','Italy','Spain','US','Korea, South','France','Turkey','United Kingdom','India']
dates = list(confirmed_df.columns[4:])
dates = list(pd.to_datetime(dates))
dates_india = dates[8:]

df1 = confirmed_df.groupby('Country/Region').sum().reset_index()
df2 = deaths_df.groupby('Country/Region').sum().reset_index()
df3 = recovered_df.groupby('Country/Region').sum().reset_index()

global_confirmed = {}
global_deaths = {}
global_recovered = {}
global_active= {}

for country in hotspots:
    k =df1[df1['Country/Region'] == country].loc[:,'1/30/20':]
    global_confirmed[country] = k.values.tolist()[0]
    
    k =df2[df2['Country/Region'] == country].loc[:,'1/30/20':]
    global_deaths[country] = k.values.tolist()[0]
    
    k =df3[df3['Country/Region'] == country].loc[:,'1/30/20':]
    global_recovered[country] = k.values.tolist()[0]
for country in hotspots:
    k = list(map(int.__sub__,global_confirmed[country],global_deaths[country]))
    global_active[country] = list(map(int.__sub__,k,global_recovered[country]))
    #print(global_active[country])
    fig = plt.figure(figsize= (15,15))
    plt.suptitle('Active, Recovered, Deaths in Hotspot Countries and India as of May 15',fontsize = 20,y=1.0)
k=0
for i in range(1,12):
    ax = fig.add_subplot(6,2,i)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.bar(dates_india,global_active[hotspots[k]],color = 'green',alpha = 0.6,label = 'Active');
    ax.bar(dates_india,global_recovered[hotspots[k]],color='grey',label = 'Recovered');
    ax.bar(dates_india,global_deaths[hotspots[k]],color='red',label = 'Death');
    plt.title(hotspots[k])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    k=k+1
plt.tight_layout(pad=3.0)
plt.show()









sizes = list(age_details['TotalCases'])
labels = list(age_details['AgeGroup'])
explode = []

for i in labels:
    explode.append(0.05)
    
plt.figure(figsize= (15,10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode)
centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('India - Age Group wise Distribution',fontsize = 20)
plt.axis('equal')  
plt.tight_layout()




print(sizes)
print(labels)









size1 = []
size1.append(individual_details['gender'].isnull().sum())
size1.append(list(individual_details['gender'].value_counts())[0])
size1.append(list(individual_details['gender'].value_counts())[1])




lables = ['missing','male','female']
explode = (0,0.1,0)
plt.figure(figsize= (15,10))
plt.title('Percentage of Gender',fontsize = 20)
plt.pie(size1,explode=explode,labels=lables,autopct='%1.1f%%',shadow=True, startangle=90)
plt.axis('equal')
plt.tight_layout()




#now ignorning the missing data 
size = []
size.append(list(individual_details['gender'].value_counts())[0])
size.append(list(individual_details['gender'].value_counts())[1])
lables = ['male','female']
explode = (0,0.1)
plt.figure(figsize= (12,8))
plt.title('Percentage of Gender(ignoring missing data)',fontsize = 20)
plt.pie(size,explode=explode,labels=lables,autopct='%1.1f%%',shadow=True, startangle=90)
plt.axis('equal')
plt.tight_layout()
#mens are more affected but remember we have 80% data missing.




dates = list(confirmed_df.columns[4:])
dates = pd.to_datetime(dates)
india_dates = dates[8:]
india_dates




df1 = confirmed_df.groupby('Country/Region').sum().reset_index()
df2 = recovered_df.groupby('Country/Region').sum().reset_index()
df3 = deaths_df.groupby('Country/Region').sum().reset_index()
k = df1[df1['Country/Region']=='India'].loc[:,'1/30/20':]
india_confirmed = k.values.tolist()[0]
k = df2[df2['Country/Region']=='India'].loc[:,'1/30/20':]
india_recoverd = k.values.tolist()[0]
k = df3[df3['Country/Region']=='India'].loc[:,'1/30/20':]
india_death = k.values.tolist()[0]
plt.figure(figsize=(12,8))
plt.title('India Confirmed,Recoverd and Death cases',fontsize=20)
plt.xticks(rotation = 90 ,fontsize = 11)
plt.yticks(fontsize = 10)
plt.xlabel('Dates',fontsize = 20)
plt.ylabel('Total Cases',fontsize=20)
plt.grid()
ax1 = plt.plot_date(y = india_confirmed,x=india_dates,label = 'Confirmed',linestyle ='-',color = 'b')
ax2 = plt.plot_date(x= india_dates,y=india_recoverd,label='Recoverd',linestyle='-',color='g')
ax3 = plt.plot_date(x=india_dates,y=india_death,label='Death',linestyle='-',color='r')
plt.legend()
plt.tight_layout()




countries = ['China','India','US', 'Italy', 'Spain', 'France']
global_confirmed = []
global_recoverd = []
global_death = []
global_active = []
for country in countries:
    k = df1[df1['Country/Region']==country].loc[:,'1/30/20':]
    global_confirmed.append(k.values.tolist()[0])
    k = df2[df2['Country/Region']==country].loc[:,'1/30/20':]
    global_recoverd.append(k.values.tolist()[0])
    k = df3[df3['Country/Region']==country].loc[:,'1/30/20':]
    global_death.append(k.values.tolist()[0])
plt.figure(figsize= (10,8))
plt.xticks(rotation = 90 ,fontsize = 11)
plt.yticks(fontsize = 10)
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total cases',fontsize = 20)
plt.title("Comparison with other Countries confirmed cases" , fontsize = 20)
for i in range(len(countries)):
    plt.plot_date(x= india_dates,y = global_confirmed[i],label = countries[i],linestyle ='-')
    #plt.plot_date(x= india_dates,y = global_recoverd[i],label = countries[i],linestyle ='-')
    #plt.plot_date(x= india_dates,y = global_death[i],label = countries[i],linestyle ='-')
plt.legend();




countries = ['China','India','US', 'Italy', 'Spain', 'France']
global_confirmed = []
global_recoverd = []
global_death = []
global_active = []
for country in countries:
    k = df1[df1['Country/Region']==country].loc[:,'1/30/20':]
    global_confirmed.append(k.values.tolist()[0])
    k = df2[df2['Country/Region']==country].loc[:,'1/30/20':]
    global_recoverd.append(k.values.tolist()[0])
    k = df3[df3['Country/Region']==country].loc[:,'1/30/20':]
    global_death.append(k.values.tolist()[0])
plt.figure(figsize= (10,8))
plt.xticks(rotation = 90 ,fontsize = 11)
plt.yticks(fontsize = 10)
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total cases',fontsize = 20)
plt.title("Comparison with other Countries Recoverd cases" , fontsize = 20)
for i in range(len(countries)):
    plt.plot_date(x= india_dates,y = global_recoverd[i],label = countries[i],linestyle ='-')
    #plt.plot_date(x= india_dates,y = global_recoverd[i],label = countries[i],linestyle ='-')
    #plt.plot_date(x= india_dates,y = global_death[i],label = countries[i],linestyle ='-')
plt.legend();




countries = ['China','India','US', 'Italy', 'Spain', 'France']
global_confirmed = []
global_recoverd = []
global_death = []
global_active = []
for country in countries:
    k = df1[df1['Country/Region']==country].loc[:,'1/30/20':]
    global_confirmed.append(k.values.tolist()[0])
    k = df2[df2['Country/Region']==country].loc[:,'1/30/20':]
    global_recoverd.append(k.values.tolist()[0])
    k = df3[df3['Country/Region']==country].loc[:,'1/30/20':]
    global_death.append(k.values.tolist()[0])
plt.figure(figsize= (15,10))
plt.xticks(rotation = 90 ,fontsize = 11)
plt.yticks(fontsize = 10)
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total cases',fontsize = 20)
plt.title("Comparison with other Countries Deaths cases" , fontsize = 20)
for i in range(len(countries)):
    plt.plot_date(x= india_dates,y = global_death[i],label = countries[i],linestyle ='-')
    #plt.plot_date(x= india_dates,y = global_recoverd[i],label = countries[i],linestyle ='-')
    #plt.plot_date(x= india_dates,y = global_death[i],label = countries[i],linestyle ='-')
plt.legend();




india_covid_19.head()
states = india_covid_19.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()
states['Active'] = states['Confirmed'] - (states['Cured'] + states['Deaths'])
states['Death rate per 100'] = np.round(100*states['Deaths']/states['Confirmed'],2)
states['Recover rate per 100'] = np.round(100*states['Cured']/states['Confirmed'],2)
states.sort_values('Confirmed',ascending=False).fillna(0).style.background_gradient(cmap='Blues',subset=["Confirmed"]).background_gradient(cmap='Blues',subset=['Deaths']).background_gradient(cmap='Blues',subset=['Cured']).background_gradient(cmap='Blues',subset=['Active']).background_gradient(cmap='Blues',subset=['Death rate per 100']).background_gradient(cmap='Blues',subset=['Recover rate per 100']).background_gradient(cmap='Blues',subset=['Confirmed'])




hospital_beds.drop([36])




hospital_beds




hospital_beds.drop([36],inplace=True)
col_objects = list(hospital_beds.columns[2:8])
col_objects
for each in col_objects:
    hospital_beds[each] = hospital_beds[each].astype(int,errors='ignore')
hospital_beds['NumPrimaryHealthCenters_HMIS'] = hospital_beds['NumPrimaryHealthCenters_HMIS'].astype(int)
top_10_primary = hospital_beds.nlargest(10,'NumPrimaryHealthCenters_HMIS')
top_10_community = hospital_beds.nlargest(10,'NumCommunityHealthCenters_HMIS')
top_10_district_hospitals = hospital_beds.nlargest(10,'NumDistrictHospitals_HMIS')
top_10_public_facility = hospital_beds.nlargest(10,'TotalPublicHealthFacilities_HMIS')
top_10_public_beds = hospital_beds.nlargest(10,'NumPublicBeds_HMIS')




plt.figure(figsize=(15,10))
plt.suptitle('Top 10 States in each Health Facility',fontsize=20)
plt.subplot(221)
plt.title('Primary health care centre')
plt.barh(top_10_primary['State/UT'],top_10_primary['NumPrimaryHealthCenters_HMIS'],color ='#87479d');

plt.subplot(222)
plt.title('Community health care centre')
plt.barh(top_10_community['State/UT'],top_10_community['NumCommunityHealthCenters_HMIS']);

plt.subplot(223)
plt.title('District health care centre')
plt.barh(top_10_district_hospitals['State/UT'],top_10_district_hospitals['NumDistrictHospitals_HMIS']);

plt.subplot(224)
plt.title('Public health care centre')
plt.barh(top_10_public_facility['State/UT'],top_10_public_facility['NumPublicBeds_HMIS']);




state_wise_details.head()




state_test = pd.pivot_table(state_wise_details,values=['TotalSamples','Negative','Positive'],index='State', aggfunc='max')
state_name = list(state_test.index)
state_test['State'] = state_name
plt.figure(figsize=(15,10))
sns.set_color_codes("pastel")
sns.barplot(x="TotalSamples", y= state_name, data=state_test,label="Total Samples", color = '#9370db')
sns.barplot(x="Positive", y= state_name, data=state_test,label="Positive Samples")
plt.legend()




value = list(icmr_testing['state'].value_counts())
name = list(icmr_testing['state'].value_counts().index)
plt.figure(figsize=(15,10))
sns.set_color_codes("pastel")
plt.title('ICMR Testing Centers in each State', fontsize = 20)
sns.barplot(x= value, y= name,color = '#9370db');






