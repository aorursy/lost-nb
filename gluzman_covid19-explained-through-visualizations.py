#!/usr/bin/env python
# coding: utf-8







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




df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')




display(df_train.head())
display(df_train.describe())
display(df_train.info())




df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')
df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')




train_date_min = df_train['Date'].min()
train_date_max = df_train['Date'].max()
print('Minimum date from training set: {}'.format(train_date_min))
print('Maximum date from training set: {}'.format(train_date_max))




test_date_min = df_test['Date'].min()
test_date_max = df_test['Date'].max()
print('Minimum date from test set: {}'.format(test_date_min))
print('Maximum date from test set: {}'.format(test_date_max))




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




px.line(df_trend, x='Date', y='ln(Cases)', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries(Logarithmic Scale)')




px.line(df_trend, x='Date', y='ln(Deaths)', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries(Logarithmic Scale)')




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




df_ind_cases = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
df_ind_cases.dropna(how='all',inplace=True)
df_ind_cases['DateTime'] = pd.to_datetime(df_ind_cases['Date'], format = '%d/%m/%y')




r = requests.get(url='https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson')
geojson = r.json()




def change_state_name(state):
    if state == 'Odisha':
        return 'Orissa'
    elif state == 'Telengana':
        return 'Telangana'
    return state




df_ind_cases['State/UnionTerritory'] = df_ind_cases.apply(lambda x: change_state_name(x['State/UnionTerritory']), axis=1)
last_date = df_ind_cases.DateTime.max()
df_ind_cases = df_ind_cases[df_ind_cases['DateTime']==last_date]




columns = ['State/UnionTerritory', 'Cured', 'Deaths','Confirmed']
df_ind_cases = df_ind_cases[columns]
df_ind_cases.sort_values('Confirmed',inplace=True, ascending=False)
df_ind_cases.reset_index(drop=True,inplace=True)
df_ind_cases.style.background_gradient(cmap='viridis')




fig = px.choropleth(df_ind_cases, geojson=geojson, color="Confirmed",
                    locations="State/UnionTerritory", featureidkey="properties.NAME_1",
                    hover_data=['Cured','Deaths'],
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title='India: Total Current cases per state'
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=600,margin={"r":0,"t":30,"l":0,"b":30})
fig.show()




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




df_ita_cases = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_region.csv')
df_ita_cases['Date'] = pd.to_datetime(df_ita_cases['Date'], format='%Y-%m-%d')
df_ita_cases['Date'] = [d.date() for d in df_ita_cases['Date']]




df_ita_group = df_ita_cases.groupby(['Date','RegionCode','RegionName','Latitude','Longitude'],as_index=False)['HospitalizedPatients','IntensiveCarePatients','TotalHospitalizedPatients','HomeConfinement','CurrentPositiveCases' ,'NewPositiveCases','Recovered','Deaths','TotalPositiveCases','TestsPerformed'].sum()




from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("mapboxtoken")




curr_date = df_ita_group.Date.max()
df_ita_curr = df_ita_cases[df_ita_cases['Date']==curr_date]




columns=['RegionName','TotalHospitalizedPatients','Recovered', 'Deaths', 'TotalPositiveCases','TestsPerformed']
df_ita_temp = df_ita_curr[columns]
df_ita_temp.sort_values('TotalPositiveCases',inplace=True, ascending=False)
df_ita_temp.reset_index(drop=True,inplace=True)
df_ita_temp.style.background_gradient(cmap='Blues_r')




px.set_mapbox_access_token(secret_value_0)
fig = px.scatter_mapbox(df_ita_curr,
                        lat="Latitude",
                        lon="Longitude",
                        size="TotalPositiveCases",
                        color='TestsPerformed',
                        color_continuous_scale=px.colors.sequential.Blues_r,
                        hover_name='RegionName',
                        hover_data=['HospitalizedPatients','IntensiveCarePatients','TotalHospitalizedPatients','HomeConfinement','CurrentPositiveCases'\
 ,'NewPositiveCases','Recovered','Deaths'],
                        zoom=4,
                        size_max=50,
                        title= 'Italy:COVID19 Cases and Test performed')
fig.show()




region_colors = {
    'Abruzzo': 'skyblue', 
    'Basilicata': 'gold', 
    'P.A. Bolzano': 'lightseagreen', 
    'Calabria': 'black', 
    'Campania': 'crimson',
    'Emilia-Romagna': 'darkred', 
    'Friuli Venezia Giulia': 'mistyrose', 
    'Lazio': 'lavender', 
    'Liguria': 'wheat',
    'Lombardia': 'red', 
    'Marche': 'green', 
    'Molise': 'yellow', 
    'Piemonte': 'blue', 
    'Puglia': 'purple', 
    'Sardegna': 'peachpuff',
    'Sicilia': 'lightcoral', 
    'Toscana': 'lightpink', 
    'P.A. Trento': 'fuchsia', 
    'Umbria': 'firebrick', 
    "Valle d'Aosta": 'cyan',
    'Veneto': 'darkblue'
} 




fig = make_subplots(
    rows=5, cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.009,
    x_title='Date'
)

for region in df_ita_group.RegionName.unique():
    df_region = df_ita_group[df_ita_group['RegionName']==region]
    #Add Traces
    fig.add_trace(go.Scatter(name=region,x=df_region['Date'], y=df_region['TotalHospitalizedPatients'], legendgroup=region, line=dict(color=region_colors[region])), row=1, col=1)
    fig.add_trace(go.Scatter(name=region,x=df_region['Date'], y=df_region['Recovered'], legendgroup=region, showlegend=False, line=dict(color=region_colors[region])), row=2, col=1)
    fig.add_trace(go.Scatter(name=region,x=df_region['Date'], y=df_region['Deaths'], legendgroup=region, showlegend=False, line=dict(color=region_colors[region])), row=3, col=1)
    fig.add_trace(go.Scatter(name=region,x=df_region['Date'], y=df_region['TotalPositiveCases'], legendgroup=region, showlegend=False, line=dict(color=region_colors[region])), row=4, col=1)
    fig.add_trace(go.Scatter(name=region,x=df_region['Date'], y=df_region['TestsPerformed'], legendgroup=region, showlegend=False, line=dict(color=region_colors[region])), row=5, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Hospitalizations", row=1, col=1)
fig.update_yaxes(title_text="Recoveries", row=2, col=1)
fig.update_yaxes(title_text="Deaths", row=3, col=1)
fig.update_yaxes(title_text="Positive Cases", row=4, col=1)
fig.update_yaxes(title_text="Tests Performed", row=5, col=1)

# Update title and height
fig.update_layout(title_text="Italy: Daily Metrics per Region", height=1000, showlegend=True)

fig.show()




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




df_ch = df_plot.query("Country_Region=='China'")
px.line(df_ch, x='Date', y='ConfirmedCases', color='Province_State', title='Total Cases growth for China')




px.line(df_ch, x='Date', y='Fatalities', color='Province_State', title='Total Deaths growth for China')




r = requests.get(url='https://raw.githubusercontent.com/deldersveld/topojson/master/countries/china/china-provinces.json')
topology = r.json()




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




df_ch = df_ch[df_ch['Date']==train_date_max]




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




df_sk = df_plot.query("Country_Region=='Korea, South'")
df_sk.reset_index(inplace = True)
df_sk = add_daily_measures(df_sk)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_sk['Date'], y=df_sk['Daily Cases']),
    go.Bar(name='Deaths', x=df_sk['Date'], y=df_sk['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(South Korea)')
fig.show()




df_sk_cases = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')




df_hotspots = df_sk_cases[(df_sk_cases['group']==True)&(df_sk_cases['city']!='from other city')]
df_hotspots['latitude'] = df_hotspots.apply(lambda x: float(x['latitude']), axis=1)
df_hotspots['longitude'] = df_hotspots.apply(lambda x: float(x['longitude']), axis=1)




px.set_mapbox_access_token(secret_value_0)
fig = px.scatter_mapbox(df_hotspots,
                        lat="latitude",
                        lon="longitude",
                        size="confirmed",
                        hover_data=['infection_case','city','province'],
                        zoom=5,
                        size_max=50,
                        title= 'COVID19 Hotspots in South Korea')
fig.show()




df_sk = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')
df_sk.age.replace('66s','60s', inplace=True)




df_sk_age = df_sk.groupby(['age','sex','country','province','city','infection_case'], as_index=False)['patient_id'].count()




# Initialize figure with subplots
fig = make_subplots(
    rows=2, cols=2, subplot_titles=("Age-wise distribution of Cases", "Province-wise distribution of Cases"\
                                    , "Infection Origin", "Gender-wise distribution of Patient Statuses"),
    specs=[[{"type": "bar"}, {"type": "pie"}],
           [{"type": "pie"}, {"type": "bar"}]],
)

# Add traces
# trace-1
df_males = df_sk_age[df_sk_age['sex']=='male'].groupby('age',as_index=False)['patient_id'].sum()
df_females = df_sk_age[df_sk_age['sex']=='female'].groupby('age',as_index=False)['patient_id'].sum()
fig.add_trace(go.Bar(name='Males', x=df_males['age'], y=df_males['patient_id']), row=1, col=1)
fig.add_trace(go.Bar(name='Females', x=df_females['age'], y=df_females['patient_id']), row=1, col=1)
# trace-2
df_province = df_sk_age.groupby('province',as_index=False)['patient_id'].sum()
fig.add_trace(go.Pie(labels=df_province['province'], values=df_province['patient_id'], hole=0.3), row=1, col=2)
# trace-3
df_inf_case = df_sk_age.groupby('infection_case', as_index=False)['patient_id'].sum()
df_inf_case.loc[df_inf_case['patient_id'] < 50, 'infection_case'] = 'etc'
fig.add_trace(go.Pie(labels=df_inf_case['infection_case'],values=df_inf_case['patient_id'], hole=0.3), row=2, col=1)
# trace-4
df_males = df_sk[df_sk['sex']=='male'].groupby('state',as_index=False)['patient_id'].sum()
df_females = df_sk[df_sk['sex']=='female'].groupby('state',as_index=False)['patient_id'].sum()
fig.add_trace(go.Bar(name='Males', x=df_males['state'], y=df_males['patient_id']), row=2, col=2)
fig.add_trace(go.Bar(name='Females', x=df_females['state'], y=df_females['patient_id']), row=2, col=2)

# Update xaxis properties
fig.update_xaxes(title_text="Age", row=1, col=1)
fig.update_xaxes(title_text="Status", row=2, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Confirmed Cases", row=1, col=1)
fig.update_yaxes(title_text="Total number of People", row=2, col=2)

# Update title and height
fig.update_layout(title_text="South Korea: Some more visualizations", height=700, showlegend=False)

fig.show()




df_pd = pd.read_csv('/kaggle/input/countries-dataset-2020/Pupulation density by countries.csv') 
df_pd['iso_code3'] = df_pd.apply(lambda x: obj.fetch_iso3(x['Country (or dependent territory)'].strip()), axis=1)
df = df_train[df_train['Date']==train_date_max]
#df = df_train.copy()
df = df.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()
df['iso_code3'] = df.apply(lambda x:obj.fetch_iso3(x['Country_Region']), axis=1)
df = df.merge(df_pd, how='left', on='iso_code3')




def convert(pop):
    if pop == float('nan'):
        return 0.0
    return float(pop.replace(',',''))

df['Population'].fillna('0', inplace=True)
df['Population'] = df.apply(lambda x: convert(x['Population']),axis=1)
df['Density pop./km2'].fillna('0', inplace=True)
df['Density pop./km2'] = df.apply(lambda x: convert(x['Density pop./km2']),axis=1)




q3 = np.percentile(df.ConfirmedCases,75)
q1 = np.percentile(df.ConfirmedCases,25)
IQR = q3-q1
low = q1 - 1.5*IQR
high = q3 + 1.3*IQR
df = df[(df['ConfirmedCases']>low) & (df['ConfirmedCases']>high)]
df['continent'] = df.apply(lambda x: obj.fetch_continent(x['Country_Region']), axis=1)




df['Date_x'] = df['Date_x'].astype(str)




px.scatter(df,x='ConfirmedCases',y='Density pop./km2', size = 'Population', color='continent',hover_data=['Country_Region'], title='Variation of Population density wrt Confirmed Cases')




px.scatter(df,x='Fatalities',y='Density pop./km2', size = 'Population', color='continent',hover_data=['Country_Region'],title='Variation of Population density wrt Fatalities')




df.corr()




#Add continent column to training set
df_train['Continent'] = df_train.apply(lambda X: obj.fetch_continent(X['Country_Region']), axis=1)




def categoricalToInteger(df):
    #convert NaN Province State values to a string
    df.Province_State.fillna('NaN', inplace=True)
    #Define Ordinal Encoder Model
    oe = OrdinalEncoder()
    df[['Province_State','Country_Region','Continent']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region','Continent']])
    return df




def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df




def train_dev_split(df, days):
    #Last days data as dev set
    date = df['Date'].max() - dt.timedelta(days=days)
    return df[df['Date'] <= date], df[df['Date'] > date]




def avoid_data_leakage(df, date=test_date_min):
    return df[df['Date']<date]




df_train = avoid_data_leakage(df_train)
df_train = categoricalToInteger(df_train)
df_train = create_features(df_train)




df_train, df_dev = train_dev_split(df_train,0)




columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','Continent','ConfirmedCases','Fatalities']
df_train = df_train[columns]
df_dev = df_dev[columns]




#Apply the same transformation to test set that were applied to the training set
df_test['Continent'] = df_test.apply(lambda X: obj.fetch_continent(X['Country_Region']), axis=1)
df_test = categoricalToInteger(df_test)
df_test = create_features(df_test)
#Columns to select
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','Continent']




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




df_submit = pd.DataFrame(submission)




df_submit.to_csv(r'submission.csv', index=False)




df_forcast = pd.concat([df_test,df_submit.iloc[:,1:]], axis=1)
df_world_f = df_forcast.copy()
df_world_f = df_world_f.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_world_f = add_daily_measures(df_world_f)




df_world = avoid_data_leakage(df_world)




fig = go.Figure(data=[
    go.Bar(name='Total Cases', x=df_world['Date'], y=df_world['ConfirmedCases']),
    go.Bar(name='Total Cases Forecasted', x=df_world_f['Date'], y=df_world_f['ConfirmedCases'])
])
# Change the bar mode
fig.update_layout(barmode='group', title='Worldwide Confirmed Cases + Forcasted Cases')
fig.show()




fig = go.Figure(data=[
    go.Bar(name='Total Deaths', x=df_world['Date'], y=df_world['Fatalities']),
    go.Bar(name='Total Deaths Forecasted', x=df_world_f['Date'], y=df_world_f['Fatalities'])
])
# Change the bar mode
fig.update_layout(barmode='group', title='Worldwide Deaths + Forcasted Deaths')
fig.show()

