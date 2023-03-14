#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pycountry_convert


# In[2]:


#Libraries to import
import pandas as pd
import numpy as np
import datetime as dt
import requests
import sys
from itertools import chain
import pycountry
import pycountry_convert as pc
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV


# In[3]:


df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')


# In[4]:


display(df_train.head())
display(df_train.describe())
display(df_train.info())


# In[5]:


df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')
df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')


# In[6]:


train_date_min = df_train['Date'].min()
train_date_max = df_train['Date'].max()
print('Minimum date from training set: {}'.format(train_date_min))
print('Maximum date from training set: {}'.format(train_date_max))


# In[7]:


test_date_min = df_test['Date'].min()
test_date_max = df_test['Date'].max()
print('Minimum date from test set: {}'.format(test_date_min))
print('Maximum date from test set: {}'.format(test_date_max))


# In[8]:


class country_utils():
    def __init__(self):
        self.d = {}
    
    def get_dic(self):
        return self.d
    
    def get_country_details(self,country):
        """Returns country code(alpha_3) and continent"""
        try:
            country_obj = pycountry.countries.get(name=country)
            if country_obj is None:
                c = pycountry.countries.search_fuzzy(country)
                country_obj = c[0]
            continent_code = pc.country_alpha2_to_continent_code(country_obj.alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj.alpha_3, continent
        except:
            if 'Congo' in country:
                country = 'Congo'
            elif country == 'Diamond Princess' or country == 'Laos' or country == 'MS Zaandam'            or country == 'Holy See' or country == 'Timor-Leste':
                return country, country
            elif country == 'Korea, South' or country == 'South Korea':
                country = 'Korea, Republic of'
            elif country == 'Taiwan*':
                country = 'Taiwan'
            elif country == 'Burma':
                country = 'Myanmar'
            elif country == 'West Bank and Gaza':
                country = 'Gaza'
            else:
                return country, country
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


# In[9]:


df_tm = df_train.copy()
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


# In[10]:


fig = px.treemap(df_tm, path=['world', 'continent', 'Country_Region','Province_State'], values='Fatalities',
                  color='Fatalities', hover_data=['Country_Region'],
                  color_continuous_scale='matter', title='Current share of Worldwide COVID19 Deaths')
fig.show()


# In[11]:


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


# In[12]:


df_world = df_train.copy()
df_world = df_world.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_world = add_daily_measures(df_world)


# In[13]:


fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_world['Date'], y=df_world['Daily Cases']),
    go.Bar(name='Deaths', x=df_world['Date'], y=df_world['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Worldwide daily Case and Death count')
fig.show()


# In[14]:


df_map = df_train.copy()
df_map['Date'] = df_map['Date'].astype(str)
df_map = df_map.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()


# In[15]:


df_map['iso_alpha'] = df_map.apply(lambda x: obj.fetch_iso3(x['Country_Region']), axis=1)


# In[16]:


df_map['ln(ConfirmedCases)'] = np.log(df_map.ConfirmedCases + 1)
df_map['ln(Fatalities)'] = np.log(df_map.Fatalities + 1)


# In[17]:


px.choropleth(df_map, 
              locations="iso_alpha", 
              color="ln(ConfirmedCases)", 
              hover_name="Country_Region", 
              hover_data=["ConfirmedCases"] ,
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.dense, 
              title='Total Confirmed Cases growth(Logarithmic Scale)')


# In[18]:


px.choropleth(df_map, 
              locations="iso_alpha", 
              color="ln(Fatalities)", 
              hover_name="Country_Region",
              hover_data=["Fatalities"],
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.OrRd,
              title = 'Total Deaths growth(Logarithmic Scale)')


# In[19]:


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


# In[20]:


px.line(df_trend, x='Date', y='Cases', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries')


# In[21]:


px.line(df_trend, x='Date', y='Deaths', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries')


# In[22]:


px.line(df_trend, x='Date', y='ln(Cases)', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries(Logarithmic Scale)')


# In[23]:


px.line(df_trend, x='Date', y='ln(Deaths)', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries(Logarithmic Scale)')


# In[24]:


df_map['Mortality Rate%'] = round((df_map.Fatalities/df_map.ConfirmedCases)*100,2)


# In[25]:


px.choropleth(df_map, 
                    locations="iso_alpha", 
                    color="Mortality Rate%", 
                    hover_name="Country_Region",
                    hover_data=["ConfirmedCases","Fatalities"],
                    animation_frame="Date",
                    color_continuous_scale=px.colors.sequential.Magma_r,
                    title = 'Worldwide Daily Variation of Mortality Rate%')


# In[26]:


df_trend['Mortality Rate%'] = round((df_trend.Deaths/df_trend.Cases)*100,2)
px.line(df_trend, x='Date', y='Mortality Rate%', color='Country', title='Variation of Mortality Rate% \n(Top 10 worst affected countries)')


# In[27]:


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


# In[28]:


df_us = df_train[df_train['Country_Region']=='US']
df_us['Date'] = df_us['Date'].astype(str)
df_us['state_code'] = df_us.apply(lambda x: us_state_abbrev.get(x.Province_State,float('nan')), axis=1)
df_us['ln(ConfirmedCases)'] = np.log(df_us.ConfirmedCases + 1)
df_us['ln(Fatalities)'] = np.log(df_us.Fatalities + 1)


# In[29]:


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


# In[30]:


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


# In[31]:


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


# In[32]:


df_train.Province_State.fillna('NaN', inplace=True)
df_plot = df_train.groupby(['Date','Country_Region','Province_State'], as_index=False)['ConfirmedCases','Fatalities'].sum()


# In[33]:


df = df_plot.query("Country_Region=='India'")
df.reset_index(inplace = True)
df = add_daily_measures(df)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df['Date'], y=df['Daily Cases']),
    go.Bar(name='Deaths', x=df['Date'], y=df['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(India)',
                 annotations=[dict(x='2020-03-23', y=106, xref="x", yref="y", text="Lockdown Imposed(23rd March)", showarrow=True, arrowhead=1, ax=-100, ay=-100)])
fig.show()


# In[34]:


df_ita = df_plot.query("Country_Region=='Italy'")
df_ita.reset_index(inplace = True)
df_ita = add_daily_measures(df_ita)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_ita['Date'], y=df_ita['Daily Cases']),
    go.Bar(name='Deaths', x=df_ita['Date'], y=df_ita['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(Italy)',
                 annotations=[dict(x='2020-03-09', y=1797, xref="x", yref="y", text="Lockdown Imposed(9th March)", showarrow=True, arrowhead=1, ax=-100, ay=-200)])
fig.show()


# In[35]:


df_esp = df_plot.query("Country_Region=='Spain'")
df_esp.reset_index(inplace = True)
df_esp = add_daily_measures(df_esp)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_esp['Date'], y=df_esp['Daily Cases']),
    go.Bar(name='Deaths', x=df_esp['Date'], y=df_esp['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(Spain)',
                 annotations=[dict(x='2020-03-15', y=1407, xref="x", yref="y", text="Lockdown Imposed(15th March)", showarrow=True, arrowhead=1, ax=-100, ay=-200)])
fig.show()


# In[36]:


df_ch = df_plot.query("Country_Region=='China'")
px.line(df_ch, x='Date', y='ConfirmedCases', color='Province_State', title='Total Cases growth for China')


# In[37]:


px.line(df_ch, x='Date', y='Fatalities', color='Province_State', title='Total Deaths growth for China')


# In[38]:


r = requests.get(url='https://raw.githubusercontent.com/deldersveld/topojson/master/countries/china/china-provinces.json')
topology = r.json()


# In[39]:


#Convert topology json into geojson
#The code is from https://gist.github.com/perrygeo/1e767e42e8bc54ad7262
def rel2abs(arc, scale=None, translate=None):
    """Yields absolute coordinate tuples from a delta-encoded arc.
    If either the scale or translate parameter evaluate to False, yield the
    arc coordinates with no transformation."""
    if scale and translate:
        a, b = 0, 0
        for ax, bx in arc:
            a += ax
            b += bx
            yield scale[0]*a + translate[0], scale[1]*b + translate[1]
    else:
        for x, y in arc:
            yield x, y

def coordinates(arcs, topology_arcs, scale=None, translate=None):
    """Return GeoJSON coordinates for the sequence(s) of arcs.
    
    The arcs parameter may be a sequence of ints, each the index of a
    coordinate sequence within topology_arcs
    within the entire topology -- describing a line string, a sequence of 
    such sequences -- describing a polygon, or a sequence of polygon arcs.
    
    The topology_arcs parameter is a list of the shared, absolute or
    delta-encoded arcs in the dataset.
    The scale and translate parameters are used to convert from delta-encoded
    to absolute coordinates. They are 2-tuples and are usually provided by
    a TopoJSON dataset. 
    """
    if isinstance(arcs[0], int):
        coords = [
            list(
                rel2abs(
                    topology_arcs[arc if arc >= 0 else ~arc],
                    scale, 
                    translate )
                 )[::arc >= 0 or -1][i > 0:] \
            for i, arc in enumerate(arcs) ]
        return list(chain.from_iterable(coords))
    elif isinstance(arcs[0], (list, tuple)):
        return list(
            coordinates(arc, topology_arcs, scale, translate) for arc in arcs)
    else:
        raise ValueError("Invalid input %s", arcs)

def geometry(obj, topology_arcs, scale=None, translate=None):
    """Converts a topology object to a geometry object.
    
    The topology object is a dict with 'type' and 'arcs' items, such as
    {'type': "LineString", 'arcs': [0, 1, 2]}.
    See the coordinates() function for a description of the other three
    parameters.
    """
    return {
        "type": obj['type'], 
        "coordinates": coordinates(
            obj['arcs'], topology_arcs, scale, translate )}

from shapely.geometry import asShape

topojson_path = sys.argv[1]
geojson_path = sys.argv[2]


# file can be renamed, the first 'object' is more reliable
layername = list(topology['objects'].keys())[0]  

features = topology['objects'][layername]['geometries']
scale = topology['transform']['scale']
trans = topology['transform']['translate']

fc = {'type': "FeatureCollection", 'features': []}

for id, tf in enumerate(features):
    f = {'id': id, 'type': "Feature"}
    f['properties'] = tf['properties'].copy()

    geommap = geometry(tf, topology['arcs'], scale, trans)
    geom = asShape(geommap).buffer(0)
    assert geom.is_valid
    f['geometry'] = geom.__geo_interface__

    fc['features'].append(f) 


# In[40]:


df_ch = df_ch[df_ch['Date']==train_date_max]


# In[41]:


fig = px.choropleth(df_ch,
                    geojson=fc,
                    locations='Province_State',
                    featureidkey="properties.NAME_1",
                    color_continuous_scale=px.colors.sequential.Darkmint,
                    hover_name='Province_State',
                    range_color=(0, df_ch['ConfirmedCases'].max()),
                    color='ConfirmedCases', 
                    title='China: Total Current Cases per Province'
                   )

fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
fig.show()


# In[42]:


fig = px.choropleth(df_ch,
                    geojson=fc,
                    locations='Province_State',
                    featureidkey="properties.NAME_1",
                    color_continuous_scale=px.colors.sequential.OrRd,
                    hover_name='Province_State',
                    range_color=(0, df_ch['Fatalities'].max()),
                    color='Fatalities', 
                    title='China: Total Current Deaths per Province'
                   )

fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
fig.show()


# In[43]:


df_ch = df_train.query("Country_Region=='China'")
df_ch = df_ch.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_ch = add_daily_measures(df_ch)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_ch['Date'], y=df_ch['Daily Cases']),
    go.Bar(name='Deaths', x=df_ch['Date'], y=df_ch['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(China)')
fig.show()


# In[44]:


df_pd = pd.read_csv('/kaggle/input/countries-dataset-2020/Pupulation density by countries.csv') 
df_pd['iso_code3'] = df_pd.apply(lambda x: obj.fetch_iso3(x['Country (or dependent territory)'].strip()), axis=1)
df = df_train[df_train['Date']==train_date_max]
df = df.groupby('Country_Region', as_index=False)['ConfirmedCases','Fatalities'].sum()
df['iso_code3'] = df.apply(lambda x:obj.fetch_iso3(x['Country_Region']), axis=1)
df = df.merge(df_pd, how='left', on='iso_code3')


# In[45]:


def convert(pop):
    if pop == float('nan'):
        return 0.0
    return float(pop.replace(',',''))

df['Population'].fillna('0', inplace=True)
df['Population'] = df.apply(lambda x: convert(x['Population']),axis=1)
df['Density pop./km2'].fillna('0', inplace=True)
df['Density pop./km2'] = df.apply(lambda x: convert(x['Density pop./km2']),axis=1)


# In[46]:


q3 = np.percentile(df.ConfirmedCases,75)
q1 = np.percentile(df.ConfirmedCases,25)
IQR = q3-q1
low = q1 - 1.5*IQR
high = q3 + 1.3*IQR
df = df[(df['ConfirmedCases']>low) & (df['ConfirmedCases']>high)]
df['continent'] = df.apply(lambda x: obj.fetch_continent(x['Country_Region']), axis=1)


# In[47]:


px.scatter(df,x='ConfirmedCases',y='Density pop./km2', size = 'Population', color='continent',hover_data=['Country_Region'], title='Variation of Population density wrt Confirmed Cases')


# In[48]:


px.scatter(df,x='Fatalities',y='Density pop./km2', size = 'Population', color='continent',hover_data=['Country_Region'],title='Variation of Population density wrt Fatalities')


# In[49]:


df.corr()


# In[50]:


#Add continent column to training set
df_train['Continent'] = df_train.apply(lambda X: obj.fetch_continent(X['Country_Region']), axis=1)


# In[51]:


def categoricalToInteger(df):
    #convert NaN Province State values to a string
    df.Province_State.fillna('NaN', inplace=True)
    #Define Ordinal Encoder Model
    oe = OrdinalEncoder()
    df[['Province_State','Country_Region','Continent']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region','Continent']])
    return df


# In[52]:


def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df


# In[53]:


def train_dev_split(df, days):
    #Last days data as dev set
    date = df['Date'].max() - dt.timedelta(days=days)
    return df[df['Date'] <= date], df[df['Date'] > date]


# In[54]:


def avoid_data_leakage(df, date=test_date_min):
    return df[df['Date']<date]


# In[55]:


df_train = avoid_data_leakage(df_train)
df_train = categoricalToInteger(df_train)
df_train = create_features(df_train)


# In[56]:


df_train, df_dev = train_dev_split(df_train,0)


# In[57]:


columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','Continent','ConfirmedCases','Fatalities']
df_train = df_train[columns]
df_dev = df_dev[columns]


# In[58]:


#Apply the same transformation to test set that were applied to the training set
df_test['Continent'] = df_test.apply(lambda X: obj.fetch_continent(X['Country_Region']), axis=1)
df_test = categoricalToInteger(df_test)
df_test = create_features(df_test)
#Columns to select
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','Continent']


# In[59]:


submission = []
#Loop through all the unique countries
for country in df_train.Country_Region.unique():
    #Filter on the basis of country
    df_train1 = df_train[df_train["Country_Region"]==country]
    #Loop through all the States of the selected country
    for state in df_train1.Province_State.unique():
        #Filter on the basis of state
        df_train2 = df_train1[df_train1["Province_State"]==state]
        #Convert to numpy array for training
        train = df_train2.values
        #Separate the features and labels
        X_train, y_train = train[:,:-2], train[:,-2:]
        #model1 for predicting Confirmed Cases
        model1 = XGBRegressor(n_estimators=1000)
        model1.fit(X_train, y_train[:,0])
        #model2 for predicting Fatalities
        model2 = XGBRegressor(n_estimators=1000)
        model2.fit(X_train, y_train[:,1])
        #Get the test data for that particular country and state
        df_test1 = df_test[(df_test["Country_Region"]==country) & (df_test["Province_State"] == state)]
        #Store the ForecastId separately
        ForecastId = df_test1.ForecastId.values
        #Remove the unwanted columns
        df_test2 = df_test1[columns]
        #Get the predictions
        y_pred1 = model1.predict(df_test2.values)
        y_pred2 = model2.predict(df_test2.values)
        #Append the predicted values to submission list
        for i in range(len(y_pred1)):
            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':y_pred1[i], 'Fatalities':y_pred2[i]}
            submission.append(d)


# In[60]:


df_submit = pd.DataFrame(submission)


# In[61]:


df_submit.to_csv(r'submission.csv', index=False)


# In[62]:


df_forcast = pd.concat([df_test,df_submit.iloc[:,1:]], axis=1)
df_world_f = df_forcast.copy()
df_world_f = df_world_f.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_world_f = add_daily_measures(df_world_f)


# In[63]:


df_world = avoid_data_leakage(df_world)


# In[64]:


fig = go.Figure(data=[
    go.Bar(name='Total Cases', x=df_world['Date'], y=df_world['ConfirmedCases']),
    go.Bar(name='Total Cases Forecasted', x=df_world_f['Date'], y=df_world_f['ConfirmedCases'])
])
# Change the bar mode
fig.update_layout(barmode='group', title='Worldwide Confirmed Cases + Forcasted Cases')
fig.show()


# In[65]:


fig = go.Figure(data=[
    go.Bar(name='Total Deaths', x=df_world['Date'], y=df_world['Fatalities']),
    go.Bar(name='Total Deaths Forecasted', x=df_world_f['Date'], y=df_world_f['Fatalities'])
])
# Change the bar mode
fig.update_layout(barmode='group', title='Worldwide Deaths + Forcasted Deaths')
fig.show()

