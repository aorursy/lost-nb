#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install pycountry-convert')
get_ipython().system('pip install country-converter')
get_ipython().system('pip install plotly')
get_ipython().system('pip install plotly_express')




# import libararies
import numpy as np
import pandas as pd

import datetime as dt
import os
import requests

import country_converter as coco
import pycountry as pyco
import pycountry_convert as pc

import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

get_ipython().run_line_magic('matplotlib', 'inline')




# global constants
BASE_DATASET_DIR = "/kaggle/input/covid19-global-forecasting-week-4"




TRAIN_DF = pd.read_csv(os.path.join(BASE_DATASET_DIR, "train.csv"))
TEST_DF = pd.read_csv(os.path.join(BASE_DATASET_DIR, "test.csv"))

TRAIN = "train"
TEST = "test"

BASE_DFS = {TRAIN: TRAIN_DF, TEST: TEST_DF}




display(TRAIN_DF.head())
display(TRAIN_DF.describe())
display(TRAIN_DF.info())
display(TRAIN_DF.dtypes)




# reformat the 'Date' field first
for df in BASE_DFS.values():
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')




min_train_date = BASE_DFS[TRAIN]["Date"].min()
max_train_date = BASE_DFS[TRAIN]["Date"].max()

print(f"Our training dataset ranges from {min_train_date} to {max_train_date}")
print("Total of %d days" % (max_train_date - min_train_date).days)




min_test_date = BASE_DFS[TEST]["Date"].min()
max_test_date = BASE_DFS[TEST]["Date"].max()

print(f"Our testing dataset ranges from {min_test_date} to {max_test_date}")
print("Total of %d days" % (max_test_date - min_test_date).days)




for df in BASE_DFS.values():
    id_col = df.columns[0] # makes the assumption that id is 1st col
    df.set_index(id_col)




def co_to_continent(alpha_2):
    """
    Converts country (ISO-2) to continent (name).
    
    Arguments:
        country
    """
    
    if len(alpha_2) != 2:
        return "UNKNOWN"
    try:
        continent_code = pc.country_alpha2_to_continent_code(alpha_2)
    except KeyError:
        return "UNKNOWN"
    
    return pc.convert_continent_code_to_continent_name(continent_code)

def co_to_country(country_name):
    """
    Converts country to Country class.
    
    Arguments:
        country_name
    """
    
    result = pyco.countries.get(name=country_name)
    
    if result is None:
        try:
            result = (pyco.countries.search_fuzzy(country_name))[0]
        except LookupError:
            result = None
        
    return result




cc = coco.CountryConverter()

for df in BASE_DFS.values():
    # saves repeatedly searching unnecessarily
    unique_countries = df["Country_Region"].unique()
    conv_countries = [co_to_country(country) for country in unique_countries]
    country_dict = dict(zip(unique_countries, conv_countries))
    
    df["country_iso2"] = df["Country_Region"].apply(lambda x: country_dict[x].alpha_2 
                                                    if country_dict[x] is not None else "")
    
    df["country_iso3"] = df["Country_Region"].apply(lambda x: country_dict[x].alpha_3 
                                                    if country_dict[x] is not None else "")
    
    df["Country_Region"] = df["Country_Region"].apply(lambda x: country_dict[x].name 
                                                      if country_dict[x] is not None else x)
    
    df["continent"] = df["country_iso2"].apply(co_to_continent)




# replace the NaN values
for df in BASE_DFS.values():
    df["Province_State"].fillna("NaN", inplace=True)




df_temp = BASE_DFS[TRAIN].copy()
df_temp["Province_State"] = df_temp["Province_State"].apply(lambda x: "" if x == "NaN" else x)

# we want to see data filtered on the current date (can change)
date = df_temp.Date.max()
df_temp = df_temp[df_temp['Date']==date]

df_temp["world"] = "World"
fig = px.treemap(df_temp, path=['world', 'continent', 'Country_Region','Province_State'], 
                 values='ConfirmedCases', color='ConfirmedCases', hover_data=['Country_Region'],
                 color_continuous_scale='haline_r', title='Current distribution of Global COVID-19 Cases')
fig.show()




df_temp = BASE_DFS[TRAIN].copy()
df_temp["Province_State"] = df_temp["Province_State"].apply(lambda x: "" if x == "NaN" else x)

# we want to see data filtered on the current date (can change)
date = df_temp.Date.max()
df_temp = df_temp[df_temp['Date']==date]

df_temp["world"] = "World"
fig = px.treemap(df_temp, path=['world', 'continent', 'Country_Region','Province_State'], values='Fatalities',
                  color='Fatalities', hover_data=['Country_Region'],
                  color_continuous_scale='magma_r', title='Current distribution of Global COVID-19 Deaths')
fig.show()




df_world = (BASE_DFS[TRAIN].copy()).groupby(['Date']).sum()

fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_world.index, y=df_world['ConfirmedCases']),
    go.Bar(name='Fatalities', x=df_world.index, y=df_world['Fatalities'])
])

fig.update_layout(barmode='overlay', title='Global Cumulative Confirmed Cases and Fatailities')
fig.show()




print("Current global fatality rate: %.2f%%" 
      % (df_world['Fatalities'].max() 
         / df_world['ConfirmedCases'].max() * 100))




def filter_down(df, filtered_df, parent_col, parent_val, child_cols, val_col, suffix):
    """
    Filters down a dataframe until it's completely filtered.
    Then applies the discrete growth formula.
    
    Arguments:
        df
        parent_col
        parent_val
        child_cols
        val_col
        suffix
    """
    filtered_df = filtered_df[filtered_df[parent_col] == parent_val]
    if len(child_cols):
        new_parent_col = child_cols[0]
        unique_filter_vals = filtered_df[new_parent_col].unique()
        for value in unique_filter_vals:
            filter_down(df, filtered_df, new_parent_col, value, child_cols[1:], val_col, suffix)
    else:
        index_vals = list(filtered_df.index)

        first_index = index_vals.pop(0)
        df.loc[first_index, val_col+suffix] = 0

        for i in index_vals:
            df.loc[i, val_col+suffix]                     = df.loc[i, val_col] - df.loc[i-1, val_col] 

def discrete_growth(df, date_col, filter_cols, cum_sum_col, suffix="_discrete"):
    """
    Determines the discrete growth for each filter column.
    
    Arguments:
        df
        filter_columns
        cum_sum_column
    """
    new_df = df.copy()
    new_df = new_df.sort_values(by=filter_cols+[date_col])
    new_df.head()
    column = filter_cols.pop(0)
    unique_filter_vals = new_df[column].unique()

    for value in unique_filter_vals:
        filter_down(new_df, new_df, column, value, filter_cols, cum_sum_col, suffix)
    
    return new_df




dg_df = BASE_DFS[TRAIN].copy()
dg_df = discrete_growth(dg_df, "Date", ["continent", "Country_Region", "Province_State"], "ConfirmedCases")
dg_df = discrete_growth(dg_df, "Date", ["continent", "Country_Region", "Province_State"], "Fatalities")




dg_df_world = dg_df.groupby(['Date']).sum()
dates = list(dg_df_world.index)
prev_date = dates.pop(0)

# calculate growth rates
for date in dates:
    dg_df_world.loc[date, "ConfirmedCases_GrowthRate"]             = dg_df_world.loc[date, "ConfirmedCases_discrete"] / dg_df_world.loc[prev_date, "ConfirmedCases"]
    dg_df_world.loc[date, "Fatalities_GrowthRate"]             = dg_df_world.loc[date, "Fatalities_discrete"] / dg_df_world.loc[prev_date, "Fatalities"]
    prev_date = date

fig = go.Figure(data=[
    go.Bar(name='Cases', x=dg_df_world.index, 
           y=dg_df_world['ConfirmedCases_discrete'], yaxis="y1", opacity=0.5),
    go.Bar(name='Fatalities', x=dg_df_world.index, 
           y=dg_df_world['Fatalities_discrete'], yaxis="y1", opacity=0.5),
    go.Line(name="Cases_GrowthRate", x=dg_df_world.index, 
            y=dg_df_world["ConfirmedCases_GrowthRate"], yaxis="y2", line_color="forestgreen"),
    go.Line(name="Fatalities_GrowthRate", x=dg_df_world.index, 
            y=dg_df_world["Fatalities_GrowthRate"], yaxis="y2", line_color="crimson")
])

fig.update_layout(barmode='overlay', title='Global Daily Confirmed Cases and Fatalities',
                 yaxis=dict(title="Cases and Fatalities"),
                 yaxis2=dict(title="Growth Rate %", overlaying='y',side='right'),
                 yaxis2_tickformat = '%')
fig.show()




map_df = BASE_DFS[TRAIN].copy()
map_df['Date'] = map_df['Date'].astype(str)
map_df = map_df.groupby(['Date', 'Country_Region', 'country_iso3'], 
                        as_index=False)['ConfirmedCases', 'Fatalities'].sum()

map_df['ln(ConfirmedCases)'] = np.log(map_df.ConfirmedCases + 1)
map_df['ln(Fatalities)'] = np.log(map_df.Fatalities + 1)




px.choropleth(map_df, 
              locations="country_iso3", 
              color="ln(ConfirmedCases)", 
              hover_name="Country_Region", 
              hover_data=["ConfirmedCases"] ,
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.Purples, 
              title='Total Confirmed Cases (Ln Scale) by Country/Region Filtered by Date')




px.choropleth(map_df, 
              locations="country_iso3", 
              color="ln(Fatalities)", 
              hover_name="Country_Region", 
              hover_data=["Fatalities"] ,
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.OrRd, 
              title='Total Fatalities (Ln Scale) by Country/Region Filtered by Date')




r = requests.get(url="https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.geojson")
topology = r.json()




aus_df = (BASE_DFS[TRAIN].copy())
aus_df = aus_df[aus_df["Country_Region"] == "Australia"]
aus_df['Date'] = aus_df['Date'].astype(str)

fig = px.choropleth(pd.DataFrame((aus_df.groupby(["Province_State"])).max()).reset_index(),
                    geojson=topology,
                    locations='Province_State',
                    featureidkey="properties.STATE_NAME",
                    color_continuous_scale=px.colors.sequential.matter,
                    hover_name='Province_State',
                    color='ConfirmedCases',
                    title='Australia: Total Cases per State by Date'
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.show()




px.line(aus_df, x='Date', y='ConfirmedCases', color='Province_State', 
        title='Australia: Total Cases by State and Date').show()
px.line(aus_df, x='Date', y='Fatalities', color='Province_State', 
        title='Australia: Total Fatalities by State and Date').show()




aus_df['ln(ConfirmedCases)'] = np.log(aus_df.ConfirmedCases + 1)
aus_df['ln(Fatalities)'] = np.log(aus_df.Fatalities + 1)




px.line(aus_df, x='Date', y='ln(ConfirmedCases)', color='Province_State', 
        title='Australia: Total ln(ConfirmedCases) by State and Date').show()
px.line(aus_df, x='Date', y='ln(Fatalities)', color='Province_State', 
        title='Australia: Total ln(Fatalities) by State and Date').show()




gth_aus_df = discrete_growth(aus_df, "Date", 
                             ["continent", "Country_Region", "Province_State"], 
                             "ConfirmedCases")
gth_aus_df = discrete_growth(gth_aus_df, "Date", 
                             ["continent", "Country_Region", "Province_State"], 
                             "Fatalities")




dg_df_world = gth_aus_df.groupby(['Date']).sum()
dates = list(dg_df_world.index)
prev_date = dates.pop(0)

# calculate growth rates
for date in dates:
    dg_df_world.loc[date, "ConfirmedCases_GrowthRate"]             = dg_df_world.loc[date, "ConfirmedCases_discrete"] / dg_df_world.loc[prev_date, "ConfirmedCases"]
    dg_df_world.loc[date, "Fatalities_GrowthRate"]             = dg_df_world.loc[date, "Fatalities_discrete"] / dg_df_world.loc[prev_date, "Fatalities"]
    prev_date = date

fig = go.Figure(data=[
    go.Bar(name='Cases', x=dg_df_world.index, 
           y=dg_df_world['ConfirmedCases_discrete'], yaxis="y1", opacity=0.5),
    go.Bar(name='Fatalities', x=dg_df_world.index, 
           y=dg_df_world['Fatalities_discrete'], yaxis="y1", opacity=0.5),
    go.Line(name="Cases_GrowthRate", x=dg_df_world.index, 
            y=dg_df_world["ConfirmedCases_GrowthRate"], yaxis="y2", line_color="forestgreen"),
    go.Line(name="Fatalities_GrowthRate", x=dg_df_world.index, 
            y=dg_df_world["Fatalities_GrowthRate"], yaxis="y2", line_color="crimson")
])

fig.update_layout(barmode='overlay', title='Global Daily Confirmed Cases and Fatalities',
                 yaxis=dict(title="Cases and Fatalities"),
                 yaxis2=dict(title="Growth Rate %", overlaying='y',side='right'),
                 yaxis2_tickformat = '%')
fig.show()




grped_aus_df = aus_df.groupby('Date').sum()

print("Current Austrailian fatality rate: %.2f%%" 
      % (grped_aus_df['Fatalities'].max() 
         / grped_aus_df['ConfirmedCases'].max() * 100))






