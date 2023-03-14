#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<iframe src="https://ourworldindata.org/grapher/covid-confirmed-cases-since-100th-case?country=CAN+MEX+PAN" style="width: 100%; height: 600px; border: 0px none;"></iframe>')


# In[2]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# Plotly installation: https://plot.ly/python/getting-started/#jupyterlab-support-python-35

DATA_FOLDER = "../input/covid19-global-forecasting-week-1/"
DATA_ALT_FOLDER = "../input/novel-corona-virus-2019-dataset/"


# In[3]:


df = pd.read_csv(DATA_FOLDER + "train.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.shape


# In[4]:


df_alt = pd.read_csv(DATA_ALT_FOLDER + "time_series_covid_19_confirmed.csv")
df_alt.shape


# In[5]:


df_country_stats = df.groupby(["Country/Region", "Date"])[["ConfirmedCases", "Fatalities"]].sum().reset_index()
print("# of Entries:", df_country_stats.shape[0])
print("# of Non-Zero Entries:", df_country_stats[df_country_stats.ConfirmedCases > 0].shape[0])
print("# of Countries:", df_country_stats["Country/Region"].nunique())
print("# of Countries with confirmed cases:", df_country_stats[df_country_stats.ConfirmedCases > 0]["Country/Region"].nunique())
df_country_stats.sample(5)


# In[6]:


# Only keep entries with ConfirmedCases > 0
df_country_stats = df_country_stats[df_country_stats.ConfirmedCases > 0].reset_index(drop=True)
df_country_stats[df_country_stats["Country/Region"] == "US"].tail()


# In[7]:


df_tmp = df_alt.drop(["Lat", "Long"], axis=1).groupby(["Country/Region"]).sum()
df_country_stats_alt = df_tmp.reset_index().melt(["Country/Region"], var_name="Date", value_name="ConfirmedCases")
df_country_stats_alt["Date"] = pd.to_datetime(df_country_stats_alt["Date"], format="%m/%d/%y")
print("# of Entries:", df_country_stats_alt.shape[0])
print("# of Non-Zero Entries:", df_country_stats_alt[df_country_stats_alt.ConfirmedCases > 0].shape[0])
print("# of Countries:", df_country_stats_alt["Country/Region"].nunique())
print("# of Countries with confirmed cases:", df_country_stats_alt[df_country_stats_alt.ConfirmedCases > 0]["Country/Region"].nunique())
df_country_stats_alt.sample(5)


# In[8]:


# Only keep entries with ConfirmedCases > 0
df_country_stats_alt = df_country_stats_alt[df_country_stats_alt.ConfirmedCases > 0].reset_index(drop=True)
df_country_stats_alt[df_country_stats_alt["Country/Region"] == "US"].tail()


# In[9]:


df_tmp = df_country_stats_alt[df_country_stats_alt.Date <= "2020-03-18"].merge(
    df_country_stats[df_country_stats.Date <= "2020-03-18"], on=["Country/Region", "Date"], how="inner"
)
np.array_equal(df_tmp["ConfirmedCases_x"].values.astype("int"), df_tmp["ConfirmedCases_y"].values.astype("int"))
df_tmp[df_tmp["ConfirmedCases_x"] != df_tmp["ConfirmedCases_y"]]


# In[10]:


df_country_stats = df_country_stats_alt


# In[11]:


SAMPLED_COUNTRIES = ["Iran", "Korea, South", "Japan", "US", "Italy", "France", "Germany", "UK", "Spain"]
# Reference: https://plot.ly/python/time-series/
fig = go.Figure(
    [
        go.Scatter(
            x=df_country_stats[df_country_stats["Country/Region"] == name]['Date'], 
            y=df_country_stats[df_country_stats["Country/Region"] == name]['ConfirmedCases'], 
            name=name,
            mode='lines',
            line=dict(width=2)
        ) for name in SAMPLED_COUNTRIES
    ],
    layout_title_text="Cumulative number of cases in Selected Countries"
)
fig.update_layout(
    yaxis_type="log",
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white")
fig.show()


# In[12]:


fig = go.Figure(
    [
        go.Scatter(
            x=np.arange(df_country_stats[df_country_stats["Country/Region"] == name].shape[0]), 
            y=df_country_stats[df_country_stats["Country/Region"] == name]['ConfirmedCases'], 
            name=name,
            mode='lines',
            line=dict(width=2)
        ) for name in SAMPLED_COUNTRIES
    ],
    layout_title_text="Cumulative number of cases, by number of days since first case"    
)
fig.update_layout(
    yaxis_type="log",
    xaxis_title="Day",
    margin=dict(l=20, r=20, t=50, b=50),
    template="plotly_white")
fig.show()


# In[13]:


fig = go.Figure(
    [
        go.Scatter(
            x=np.arange(df_country_stats[
                (df_country_stats["Country/Region"] == name) & (df_country_stats.ConfirmedCases >= 100)
            ].shape[0]), 
            y=df_country_stats[
                (df_country_stats["Country/Region"] == name) & (df_country_stats.ConfirmedCases >= 100)
            ]['ConfirmedCases'], 
            mode='lines',
            name=name,
            line=dict(width=2)
        ) for name in SAMPLED_COUNTRIES
    ],
    layout_title_text="Cumulative number of cases, by number of days since the 100th case"
)
fig.update_layout(
    yaxis_type="log",
    xaxis_title="Day",
    margin=dict(l=20, r=20, t=50, b=50),
    width=800, height=500,
    template="plotly_white")
fig.add_shape(
    type="line",
#     xref="x",
#     yref="y",
    x0=0,
    y0=100,
    x1=30,
    y1=799,
    line=dict(
        color="Grey",
        width=1,
        dash="dot"
    ),
)
fig.add_shape(
    type="line",
    x0=0,
    y0=100,
    x1=30,
    y1=6400,
    line=dict(
        color="Grey",
        width=1,
        dash="dot"
    ),
)
fig.add_shape(
    type="line",
    x0=0,
    y0=100,
    x1=30,
    y1=102400,
    line=dict(
        color="Grey",
        width=1,
        dash="dot"
    ),
)
fig.add_shape(
    type="line",
    x0=0,
    y0=100,
    x1=20,
    y1=102400,
    line=dict(
        color="Grey",
        width=1,
        dash="dot"
    ),
)
fig.add_annotation(
    xref="x",
    yref="y",    
    x=18,
    y=4.63,
    showarrow=False,
    text="doubling every 2 days",
    textangle=-43,
    font={
        "color": "Grey"
    }
)
fig.add_annotation(
    xref="x",
    yref="y",    
    x=27,
    y=4.63,
    showarrow=False,
    text="doubling every 3 days",
    textangle=-33,
    font={
        "color": "Grey"
    }
)
fig.add_annotation(
    xref="x",
    yref="y",    
    x=28,
    y=3.63,
    showarrow=False,
    text="doubling every 5 days",
    textangle=-17,
    font={
        "color": "Grey"
    }
)
fig.add_annotation(
    xref="x",
    yref="y",    
    x=28,
    y=2.78,
    showarrow=False,
    text="doubling every 10 days",
    textangle=-8,
    font={
        "color": "Grey"
    }
)
fig.show()


# In[ ]:




