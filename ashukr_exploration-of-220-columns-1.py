#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls ../input/')


# In[ ]:


from scipy import stats


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[ ]:


from plotly import tools


# In[ ]:


init_notebook_mode(connected=True)


# In[ ]:


def bar_hor(df, col, title, color, w=None, h=None, lm=0, limit=100, return_trace=False, rev=False, xlb = False):
    cnt_srs = df[col].value_counts()
    yy = cnt_srs.head(limit).index[::-1] 
    xx = cnt_srs.head(limit).values[::-1] 
    if rev:
        yy = cnt_srs.tail(limit).index[::-1] 
        xx = cnt_srs.tail(limit).values[::-1] 
    if xlb:
        trace = go.Bar(y=xlb, x=xx, orientation = 'h', marker=dict(color=color))
    else:
        trace = go.Bar(y=yy, x=xx, orientation = 'h', marker=dict(color=color))
    if return_trace:
        return trace 
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


# In[ ]:


def gp(col, title):
    df1 = application[application["TARGET"] == 1]
    df0 = application[application["TARGET"] == 0]
    a1 = df1[col].value_counts()
    b1 = df0[col].value_counts()
    
    total = dict(application[col].value_counts())
    x0 = a1.index
    x1 = b1.index
    
    y0 = [float(x)*100 / total[x0[i]] for i,x in enumerate(a1.values)]
    y1 = [float(x)*100 / total[x1[i]] for i,x in enumerate(b1.values)]

    trace1 = go.Bar(x=a1.index, y=y0, name='Target : 1', marker=dict(color="#96D38C"))
    trace2 = go.Bar(x=b1.index, y=y1, name='Target : 0', marker=dict(color="#FEBFB3"))
    return trace1, trace2 


# In[ ]:


def exploreCat(col):
    t = application[col].value_counts()
    labels = t.index
    values = t.values
    colors = ['#96D38C','#FEBFB3']
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo="all", textinfo='value',
                   textfont=dict(size=12),
                   marker=dict(colors=colors,
                               line=dict(color='#fff', width=2)))
    layout = go.Layout(title=col, height=400)
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)


#the realation between the categorical column and the target 
def catAndTrgt(col):
    tr0 = bar_hor(application, col, "Distribution of "+col ,"#f975ae", w=700, lm=100, return_trace= True)
    tr1, tr2 = gp(col, 'Distribution of Target with ' + col)

    fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles = [col +" Distribution" , "% Rpyment difficulty by "+col ,"% of otherCases by "+col])
    fig.append_trace(tr0, 1, 1);
    fig.append_trace(tr1, 1, 2);
    fig.append_trace(tr2, 1, 3);
    fig['layout'].update(height=350, showlegend=False, margin=dict(l=50));
    iplot(fig);


# In[ ]:


def numeric(col):
    plt.figure(figsize=(12,5))
    plt.title("Distribution of "+col)
    ax = sns.distplot(application[col].dropna())


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


application = pd.read_csv("../input/application_train.csv")


# In[ ]:


application.head()


# In[ ]:


application.shape


# In[ ]:


bar_hor(application, "TARGET", "Distribution of Target Variable" , ["#44ff54", '#ff4444'], h=350, w=600, lm=200, xlb = ['Target : 1','Target : 0'])


# In[ ]:


exploreCat("NAME_CONTRACT_TYPE")


# In[ ]:


catAndTrgt("NAME_CONTRACT_TYPE")


# In[ ]:


exploreCat("CODE_GENDER")


# In[ ]:


catAndTrgt("CODE_GENDER")


# In[ ]:


exploreCat("FLAG_OWN_CAR")


# In[ ]:


catAndTrgt("FLAG_OWN_CAR")


# In[ ]:


exploreCat("FLAG_OWN_REALTY")


# In[ ]:


catAndTrgt("FLAG_OWN_REALTY")


# In[ ]:


exploreCat("CNT_CHILDREN")


# In[ ]:


catAndTrgt("CNT_CHILDREN")


# In[ ]:


application["AMT_INCOME_TOTAL"].dtype


# In[ ]:


numeric("AMT_INCOME_TOTAL")


# In[ ]:


numeric("AMT_CREDIT")


# In[ ]:


application["AMT_ANNUITY"].dtype


# In[ ]:


numeric("AMT_ANNUITY")


# In[ ]:


numeric("AMT_GOODS_PRICE")


# In[ ]:


exploreCat("NAME_TYPE_SUITE")


# In[ ]:


catAndTrgt("NAME_TYPE_SUITE")


# In[ ]:


application["NAME_INCOME_TYPE"].dtype


# In[ ]:


exploreCat("NAME_INCOME_TYPE")


# In[ ]:


catAndTrgt("NAME_INCOME_TYPE")


# In[ ]:


application["NAME_EDUCATION_TYPE"].dtype


# In[ ]:



exploreCat("NAME_EDUCATION_TYPE")


# In[ ]:


application["NAME_FAMILY_STATUS"].dtype


# In[ ]:



exploreCat("NAME_FAMILY_STATUS")


# In[ ]:



catAndTrgt("NAME_FAMILY_STATUS")


# In[ ]:


application["NAME_HOUSING_TYPE"].dtype


# In[ ]:



exploreCat("NAME_HOUSING_TYPE")


# In[ ]:



catAndTrgt("NAME_HOUSING_TYPE")


# In[ ]:


application["REGION_POPULATION_RELATIVE"].dtype


# In[ ]:



numeric("REGION_POPULATION_RELATIVE")


# In[ ]:


application["REGION_POPULATION_RELATIVE"].describe()


# In[ ]:


application["DAYS_BIRTH"].dtype


# In[ ]:



numeric("DAYS_BIRTH")


# In[ ]:



application["DAYS_BIRTH"].describe()


# In[ ]:


application["DAYS_EMPLOYED"].dtype


# In[ ]:


numeric("DAYS_EMPLOYED")


# In[ ]:


application["DAYS_EMPLOYED"].describe()


# In[ ]:


application["DAYS_REGISTRATION"].dtype


# In[ ]:


numeric("DAYS_REGISTRATION")


# In[ ]:


application["DAYS_REGISTRATION"].describe()


# In[ ]:


application["OWN_CAR_AGE"].dtype


# In[ ]:


numeric("OWN_CAR_AGE")


# In[ ]:


application["OWN_CAR_AGE"].describe()


# In[ ]:


exploreCat("FLAG_MOBIL")


# In[ ]:



catAndTrgt("FLAG_MOBIL")


# In[ ]:


exploreCat("FLAG_EMP_PHONE")


# In[ ]:



catAndTrgt("FLAG_EMP_PHONE")


# In[ ]:


exploreCat("FLAG_WORK_PHONE")


# In[ ]:


catAndTrgt("FLAG_WORK_PHONE")


# In[ ]:


exploreCat("FLAG_CONT_MOBILE")


# In[ ]:


catAndTrgt("FLAG_CONT_MOBILE")


# In[ ]:


exploreCat("FLAG_PHONE")


# In[ ]:


catAndTrgt("FLAG_PHONE")


# In[ ]:


exploreCat("FLAG_EMAIL")


# In[ ]:


catAndTrgt("FLAG_EMAIL")


# In[ ]:


application["OCCUPATION_TYPE"].dtype


# In[ ]:



exploreCat("OCCUPATION_TYPE")


# In[ ]:



catAndTrgt("OCCUPATION_TYPE")


# In[ ]:


exploreCat("CNT_FAM_MEMBERS")


# In[ ]:


catAndTrgt("CNT_FAM_MEMBERS")


# In[ ]:


exploreCat("REGION_RATING_CLIENT")


# In[ ]:



catAndTrgt("REGION_RATING_CLIENT")


# In[ ]:


exploreCat("REGION_RATING_CLIENT_W_CITY")


# In[ ]:


catAndTrgt("REGION_RATING_CLIENT_W_CITY")


# In[ ]:


exploreCat("REGION_RATING_CLIENT_W_CITY")


# In[ ]:



catAndTrgt("REGION_RATING_CLIENT_W_CITY")


# In[ ]:


exploreCat("HOUR_APPR_PROCESS_START")


# In[ ]:


catAndTrgt("HOUR_APPR_PROCESS_START")


# In[ ]:


exploreCat("REG_REGION_NOT_LIVE_REGION")


# In[ ]:


catAndTrgt("REG_REGION_NOT_LIVE_REGION")


# In[ ]:


exploreCat("REG_REGION_NOT_WORK_REGION")


# In[ ]:



catAndTrgt("REG_REGION_NOT_WORK_REGION")


# In[ ]:


exploreCat("LIVE_REGION_NOT_WORK_REGION")


# In[ ]:


catAndTrgt("LIVE_REGION_NOT_WORK_REGION")


# In[ ]:


exploreCat("REG_CITY_NOT_LIVE_CITY")


# In[ ]:



catAndTrgt("REG_CITY_NOT_LIVE_CITY")


# In[ ]:


exploreCat("REG_CITY_NOT_LIVE_CITY")


# In[ ]:



catAndTrgt("REG_CITY_NOT_LIVE_CITY")


# In[ ]:


exploreCat("LIVE_CITY_NOT_WORK_CITY")


# In[ ]:



catAndTrgt("LIVE_CITY_NOT_WORK_CITY")


# In[ ]:


exploreCat("ORGANIZATION_TYPE")


# In[ ]:


catAndTrgt("ORGANIZATION_TYPE")


# In[ ]:


application["EXT_SOURCE_1"].dtype


# In[ ]:


numeric("EXT_SOURCE_1")


# In[ ]:


application["EXT_SOURCE_2"].dtype


# In[ ]:



numeric("EXT_SOURCE_2")


# In[ ]:


numeric("EXT_SOURCE_3")


# In[ ]:


application["EXT_SOURCE_3"].describe()


# In[ ]:


numeric("APARTMENTS_AVG")


# In[ ]:


application["APARTMENTS_AVG"].describe()


# In[ ]:


numeric("BASEMENTAREA_AVG")


# In[ ]:


application["BASEMENTAREA_AVG"].describe()


# In[ ]:


numeric("YEARS_BEGINEXPLUATATION_AVG")


# In[ ]:


application["YEARS_BEGINEXPLUATATION_AVG"].describe()


# In[ ]:


numeric("YEARS_BUILD_AVG")


# In[ ]:


application["YEARS_BUILD_AVG"].describe()


# In[ ]:


numeric("COMMONAREA_AVG")


# In[ ]:


application["COMMONAREA_AVG"].describe()


# In[ ]:


numeric("ELEVATORS_AVG")


# In[ ]:



numeric("ENTRANCES_AVG")


# In[ ]:


application["ENTRANCES_AVG"].describe()


# In[ ]:



numeric("FLOORSMAX_AVG")


# In[ ]:


application["FLOORSMAX_AVG"].describe()


# In[ ]:


numeric("FLOORSMIN_AVG")


# In[ ]:


application["FLOORSMIN_AVG"].describe()


# In[ ]:


numeric("LANDAREA_AVG")


# In[ ]:


application["LANDAREA_AVG"].describe()


# In[ ]:


numeric("LIVINGAPARTMENTS_AVG")


# In[ ]:


application["LIVINGAPARTMENTS_AVG"].describe()


# In[ ]:


numeric("LIVINGAREA_AVG")


# In[ ]:


application["LIVINGAREA_AVG"].describe()


# In[ ]:


numeric("NONLIVINGAPARTMENTS_AVG")


# In[ ]:


application["NONLIVINGAPARTMENTS_AVG"].describe()


# In[ ]:


numeric("NONLIVINGAREA_AVG")


# In[ ]:


application["NONLIVINGAPARTMENTS_AVG"].describe()


# In[ ]:


numeric("APARTMENTS_MODE")


# In[ ]:


application["APARTMENTS_MODE"].describe()


# In[ ]:


numeric("BASEMENTAREA_MODE")


# In[ ]:



application["BASEMENTAREA_MODE"].describe()


# In[ ]:


numeric("YEARS_BEGINEXPLUATATION_MODE")


# In[ ]:


application["YEARS_BEGINEXPLUATATION_MODE"].describe()


# In[ ]:


numeric("YEARS_BUILD_MODE")


# In[ ]:



application["YEARS_BUILD_MODE"].describe()


# In[ ]:


numeric("COMMONAREA_MODE")


# In[ ]:



application["COMMONAREA_MODE"].describe()


# In[ ]:



numeric("ELEVATORS_MODE")


# In[ ]:



application["ELEVATORS_MODE"].describe()


# In[ ]:



numeric("ENTRANCES_MODE")


# In[ ]:



application["ENTRANCES_MODE"].describe()


# In[ ]:



application["FLOORSMAX_MODE"].describe()


# In[ ]:


numeric("FLOORSMAX_MODE")


# In[ ]:


numeric("FLOORSMIN_MODE")


# In[ ]:



application["FLOORSMIN_MODE"].describe()


# In[ ]:



numeric("LANDAREA_MODE")


# In[ ]:



application["LANDAREA_MODE"].describe()


# In[ ]:


numeric("LIVINGAPARTMENTS_MODE")


# In[ ]:



application["LIVINGAPARTMENTS_MODE"].describe()


# In[ ]:



numeric("LIVINGAREA_MODE")


# In[ ]:


application["LIVINGAREA_MODE"].describe()


# In[ ]:


numeric("NONLIVINGAPARTMENTS_MODE")


# In[ ]:



application["NONLIVINGAPARTMENTS_MODE"].describe()


# In[ ]:


numeric("NONLIVINGAREA_MODE")


# In[ ]:



application["NONLIVINGAREA_MODE"].describe()


# In[ ]:



numeric("APARTMENTS_MEDI")


# In[ ]:


application["APARTMENTS_MEDI"].describe()


# In[ ]:


numeric("BASEMENTAREA_MEDI")


# In[ ]:


application["BASEMENTAREA_MEDI"].describe()


# In[ ]:


numeric("YEARS_BEGINEXPLUATATION_MEDI")


# In[ ]:


application["YEARS_BEGINEXPLUATATION_MEDI"].describe()


# In[ ]:


numeric("COMMONAREA_MEDI")


# In[ ]:



application["COMMONAREA_MEDI"].describe()


# In[ ]:


numeric("ELEVATORS_MEDI")


# In[ ]:


application["ELEVATORS_MEDI"].describe()


# In[ ]:


numeric("ENTRANCES_MEDI")


# In[ ]:


application["ENTRANCES_MEDI"].describe()


# In[ ]:


numeric("FLOORSMAX_MEDI")


# In[ ]:


application["FLOORSMAX_MEDI"].describe()


# In[ ]:


numeric("FLOORSMIN_MEDI")


# In[ ]:


application["FLOORSMIN_MEDI"].describe()


# In[ ]:


numeric("LANDAREA_MEDI")


# In[ ]:


application["LANDAREA_MEDI"].describe()


# In[ ]:


numeric("LIVINGAPARTMENTS_MEDI")


# In[ ]:


application["LIVINGAPARTMENTS_MEDI"].describe()


# In[ ]:


numeric("LIVINGAREA_MEDI")


# In[ ]:


application["LIVINGAREA_MEDI"].describe()


# In[ ]:


numeric("NONLIVINGAPARTMENTS_MEDI")


# In[ ]:


application["NONLIVINGAPARTMENTS_MEDI"].describe()


# In[ ]:


numeric("NONLIVINGAREA_MEDI")


# In[ ]:


application["NONLIVINGAREA_MEDI"].describe()


# In[ ]:


application["FONDKAPREMONT_MODE"].dtype


# In[ ]:


exploreCat("FONDKAPREMONT_MODE")


# In[ ]:


catAndTrgt("FONDKAPREMONT_MODE")


# In[ ]:


application["HOUSETYPE_MODE"].dtype


# In[ ]:


exploreCat("HOUSETYPE_MODE")


# In[ ]:


exploreCat("WALLSMATERIAL_MODE")


# In[ ]:



catAndTrgt("WALLSMATERIAL_MODE")


# In[ ]:


exploreCat("EMERGENCYSTATE_MODE")


# In[ ]:


catAndTrgt("EMERGENCYSTATE_MODE")


# In[ ]:


application["OBS_30_CNT_SOCIAL_CIRCLE"].dtype


# In[ ]:


numeric("OBS_30_CNT_SOCIAL_CIRCLE")


# In[ ]:


application["OBS_30_CNT_SOCIAL_CIRCLE"].describe()


# In[ ]:


exploreCat("DEF_30_CNT_SOCIAL_CIRCLE")


# In[ ]:


catAndTrgt("DEF_30_CNT_SOCIAL_CIRCLE")


# In[ ]:


exploreCat("OBS_60_CNT_SOCIAL_CIRCLE")


# In[ ]:


catAndTrgt("OBS_60_CNT_SOCIAL_CIRCLE")


# In[ ]:


exploreCat("DEF_60_CNT_SOCIAL_CIRCLE")


# In[ ]:


catAndTrgt("DEF_60_CNT_SOCIAL_CIRCLE")


# In[ ]:


application["DAYS_LAST_PHONE_CHANGE"].dtype


# In[ ]:


numeric("DAYS_LAST_PHONE_CHANGE")


# In[ ]:


application["DAYS_LAST_PHONE_CHANGE"].describe()


# In[ ]:


exploreCat("FLAG_DOCUMENT_2")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_2")


# In[ ]:


exploreCat("FLAG_DOCUMENT_3")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_3")


# In[ ]:


exploreCat("FLAG_DOCUMENT_4")


# In[ ]:



catAndTrgt("FLAG_DOCUMENT_4")


# In[ ]:


exploreCat("FLAG_DOCUMENT_5")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_5")


# In[ ]:


exploreCat("FLAG_DOCUMENT_6")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_6")


# In[ ]:


exploreCat("FLAG_DOCUMENT_7")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_7")


# In[ ]:


exploreCat("FLAG_DOCUMENT_8")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_8")


# In[ ]:


exploreCat("FLAG_DOCUMENT_9")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_9")


# In[ ]:


exploreCat("FLAG_DOCUMENT_10")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_10")


# In[ ]:


exploreCat("FLAG_DOCUMENT_11")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_11")


# In[ ]:


exploreCat("FLAG_DOCUMENT_12")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_12")


# In[ ]:


exploreCat("FLAG_DOCUMENT_13")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_13")


# In[ ]:


exploreCat("FLAG_DOCUMENT_14")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_14")


# In[ ]:


exploreCat("FLAG_DOCUMENT_15")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_15")


# In[ ]:


exploreCat("FLAG_DOCUMENT_16")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_16")


# In[ ]:


exploreCat("FLAG_DOCUMENT_17")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_17")


# In[ ]:


exploreCat("FLAG_DOCUMENT_18")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_18")


# In[ ]:


exploreCat("FLAG_DOCUMENT_19")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_19")


# In[ ]:


exploreCat("FLAG_DOCUMENT_20")


# In[ ]:


catAndTrgt("FLAG_DOCUMENT_21")


# In[ ]:


application["AMT_REQ_CREDIT_BUREAU_HOUR"].dtype


# In[ ]:


numeric("AMT_REQ_CREDIT_BUREAU_HOUR")


# In[ ]:


application["AMT_REQ_CREDIT_BUREAU_HOUR"].describe()


# In[ ]:


application["AMT_REQ_CREDIT_BUREAU_DAY"].dtype


# In[ ]:


numeric("AMT_REQ_CREDIT_BUREAU_DAY")


# In[ ]:


application["AMT_REQ_CREDIT_BUREAU_DAY"].dtype


# In[ ]:


numeric("AMT_REQ_CREDIT_BUREAU_DAY")


# In[ ]:


(application["AMT_REQ_CREDIT_BUREAU_DAY"].describe())


# In[ ]:


application["AMT_REQ_CREDIT_BUREAU_WEEK"].dtype


# In[ ]:


numeric("AMT_REQ_CREDIT_BUREAU_WEEK")


# In[ ]:


application["AMT_REQ_CREDIT_BUREAU_WEEK"].describe()


# In[ ]:


application["AMT_REQ_CREDIT_BUREAU_MON"].describe()


# In[ ]:


numeric("AMT_REQ_CREDIT_BUREAU_MON")


# In[ ]:


application["AMT_REQ_CREDIT_BUREAU_QRT"].describe()


# In[ ]:


numeric("AMT_REQ_CREDIT_BUREAU_QRT")


# In[ ]:


application["AMT_REQ_CREDIT_BUREAU_YEAR"].describe()


# In[ ]:


numeric("AMT_REQ_CREDIT_BUREAU_YEAR")


# In[ ]:


bureau = pd.read_csv("../input/bureau.csv")


# In[ ]:


bureau.head()


# In[ ]:


bureau["SK_ID_CURR"].head()


# In[ ]:


bureau["SK_ID_CURR"].describe()


# In[ ]:


BNumeric("SK_ID_CURR")


# In[ ]:


bureau["SK_ID_BUREAU"].head()


# In[ ]:


bureau["SK_ID_BUREAU"].describe()


# In[ ]:


bureau["CREDIT_ACTIVE"].describe()


# In[ ]:


def BExpCat(col):
    t = bureau[col].value_counts()
    labels = t.index
    values = t.values
    colors = ['#96D38C','#FEBFB3']
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo="all", textinfo='value',
                   textfont=dict(size=12),
                   marker=dict(colors=colors,
                               line=dict(color='#fff', width=2)))
    layout = go.Layout(title=col, height=400)
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
def BNumeric(col):
    plt.figure(figsize=(12,5))
    plt.title("Distribution of "+col)
    ax = sns.distplot(bureau[col].dropna())


# In[ ]:


BExpCat("CREDIT_ACTIVE")


# In[ ]:


bureau["CREDIT_CURRENCY"].describe()


# In[ ]:


BExpCat("CREDIT_CURRENCY")


# In[ ]:


bureau["DAYS_CREDIT"].describe()


# In[ ]:


BNumeric("DAYS_CREDIT")


# In[ ]:


bureau["CREDIT_DAY_OVERDUE"].describe()


# In[ ]:


BNumeric("CREDIT_DAY_OVERDUE")


# In[ ]:


bureau["DAYS_CREDIT_ENDDATE"].describe()


# In[ ]:


BNumeric("DAYS_CREDIT_ENDDATE")


# In[ ]:


bureau["DAYS_ENDDATE_FACT"].describe()


# In[ ]:


BNumeric("DAYS_ENDDATE_FACT")


# In[ ]:


bureau["AMT_CREDIT_MAX_OVERDUE"].describe()


# In[ ]:


BNumeric("AMT_CREDIT_MAX_OVERDUE")


# In[ ]:



bureau["CNT_CREDIT_PROLONG"].describe()


# In[ ]:


BNumeric("CNT_CREDIT_PROLONG")


# In[ ]:


bureau["AMT_CREDIT_SUM"].describe()


# In[ ]:


BNumeric("AMT_CREDIT_SUM")


# In[ ]:


bureau["AMT_CREDIT_SUM_DEBT"].describe()


# In[ ]:


BNumeric("AMT_CREDIT_SUM_DEBT")


# In[ ]:


bureau["AMT_CREDIT_SUM_LIMIT"].describe()


# In[ ]:


BNumeric("AMT_CREDIT_SUM_LIMIT")


# In[ ]:


bureau["AMT_CREDIT_SUM_OVERDUE"].describe()


# In[ ]:


BNumeric("AMT_CREDIT_SUM_OVERDUE")


# In[ ]:


bureau["AMT_CREDIT_SUM_OVERDUE"].describe()


# In[ ]:


BNumeric("AMT_CREDIT_SUM_OVERDUE")


# In[ ]:


bureau["CREDIT_TYPE"].describe()


# In[ ]:


BExpCat("CREDIT_TYPE")


# In[ ]:


bureau["DAYS_CREDIT_UPDATE"].describe()


# In[ ]:


BNumeric("DAYS_CREDIT_UPDATE")


# In[ ]:


bureau["AMT_ANNUITY"].describe()


# In[ ]:


BNumeric("AMT_ANNUITY")


# In[ ]:


bb = pd.read_csv("../input/bureau_balance.csv")


# In[ ]:


bb.head()


# In[ ]:


bb["SK_ID_BUREAU"].describe()


# In[ ]:


def BBExpCat(col):
    t = bb[col].value_counts()
    labels = t.index
    values = t.values
    colors = ['#96D38C','#FEBFB3']
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo="all", textinfo='value',
                   textfont=dict(size=12),
                   marker=dict(colors=colors,
                               line=dict(color='#fff', width=2)))
    layout = go.Layout(title=col, height=400)
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
def BBNumeric(col):
    plt.figure(figsize=(12,5))
    plt.title("Distribution of "+col)
    ax = sns.distplot(bb[col].dropna())


# In[ ]:


BBNumeric("SK_ID_BUREAU")


# In[ ]:


bb["MONTHS_BALANCE"].describe()


# In[ ]:


BBNumeric("MONTHS_BALANCE")


# In[ ]:


bb["STATUS"].describe()


# In[ ]:


BBExpCat("STATUS")


# In[ ]:


PC = pd.read_csv("../input/POS_CASH_balance.csv")


# In[ ]:


def PCExpCat(col):
    t = PC[col].value_counts()
    labels = t.index
    values = t.values
    colors = ['#96D38C','#FEBFB3']
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo="all", textinfo='value',
                   textfont=dict(size=12),
                   marker=dict(colors=colors,
                               line=dict(color='#fff', width=2)))
    layout = go.Layout(title=col, height=400)
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
def PCNumeric(col):
    plt.figure(figsize=(12,5))
    plt.title("Distribution of "+col)
    ax = sns.distplot(PC[col].dropna())


# In[ ]:


PC.head()


# In[ ]:


PC["SK_ID_PREV"].describe()


# In[ ]:


PCNumeric("SK_ID_PREV")


# In[ ]:


PC["SK_ID_CURR"].describe()


# In[ ]:


PCNumeric("SK_ID_CURR")


# In[ ]:



PCNumeric("MONTHS_BALANCE")


# In[ ]:


PC["MONTHS_BALANCE"].describe()


# In[ ]:


PC["CNT_INSTALMENT"].describe()


# In[ ]:


PCNumeric("CNT_INSTALMENT")


# In[ ]:


PCNumeric("CNT_INSTALMENT_FUTURE")


# In[ ]:


PC["CNT_INSTALMENT_FUTURE"].describe()


# In[ ]:


PC["CNT_INSTALMENT_FUTURE"].describe()


# In[ ]:


PCNumeric("CNT_INSTALMENT_FUTURE")


# In[ ]:


PC["NAME_CONTRACT_STATUS"].describe()


# In[ ]:


PCExpCat("NAME_CONTRACT_STATUS")


# In[ ]:


PC["SK_DPD"].describe()


# In[ ]:


PCNumeric("SK_DPD")


# In[ ]:


PC["SK_DPD_DEF"].describe()


# In[ ]:


PCNumeric("SK_DPD_DEF")


# In[ ]:


CC = pd.read_csv("../input/credit_card_balance.csv")


# In[ ]:


def CCExpCat(col):
    t = CC[col].value_counts()
    labels = t.index
    values = t.values
    colors = ['#96D38C','#FEBFB3']
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo="all", textinfo='value',
                   textfont=dict(size=12),
                   marker=dict(colors=colors,
                               line=dict(color='#fff', width=2)))
    layout = go.Layout(title=col, height=400)
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
def CCNumeric(col):
    plt.figure(figsize=(12,5))
    plt.title("Distribution of "+col)
    ax = sns.distplot(CC[col].dropna())


# In[ ]:


CC.describe()


# In[ ]:


CC["SK_ID_PREV"].describe()


# In[ ]:


CCNumeric("SK_ID_PREV")


# In[ ]:


CC["SK_ID_CURR"].describe()


# In[ ]:


CCNumeric('SK_ID_CURR')


# In[ ]:


CC["MONTHS_BALANCE"].describe()


# In[ ]:


CCNumeric("MONTHS_BALANCE")


# In[ ]:


CC["AMT_BALANCE"].describe()


# In[ ]:


CCNumeric("AMT_BALANCE")


# In[ ]:


CC["AMT_CREDIT_LIMIT_ACTUAL"].describe()


# In[ ]:


CCNumeric("AMT_CREDIT_LIMIT_ACTUAL")


# In[ ]:


CC["AMT_DRAWINGS_ATM_CURRENT"].describe()


# In[ ]:


CCNumeric("AMT_DRAWINGS_ATM_CURRENT")


# In[ ]:


CC["AMT_DRAWINGS_CURRENT"].describe()


# In[ ]:


CCNumeric("AMT_DRAWINGS_CURRENT")


# In[ ]:


CC["AMT_DRAWINGS_OTHER_CURRENT"].describe()


# In[ ]:


CCNumeric("AMT_DRAWINGS_OTHER_CURRENT")


# In[ ]:


CC["AMT_DRAWINGS_POS_CURRENT"].describe()


# In[ ]:


CCNumeric("AMT_DRAWINGS_POS_CURRENT")


# In[ ]:


CC["AMT_INST_MIN_REGULARITY"].describe()


# In[ ]:


CCNumeric('AMT_INST_MIN_REGULARITY')


# In[ ]:


CC["AMT_PAYMENT_CURRENT"].describe()


# In[ ]:


CCNumeric("AMT_PAYMENT_CURRENT")


# In[ ]:


CC["AMT_PAYMENT_TOTAL_CURRENT"].describe()


# In[ ]:


CCNumeric("AMT_PAYMENT_TOTAL_CURRENT")


# In[ ]:


CC["AMT_RECEIVABLE_PRINCIPAL"].describe()


# In[ ]:


CCNumeric("AMT_RECEIVABLE_PRINCIPAL")


# In[ ]:


CC["AMT_RECIVABLE"].describe()


# In[ ]:


CCNumeric("AMT_RECIVABLE")


# In[ ]:


CC["AMT_TOTAL_RECEIVABLE"].describe()


# In[ ]:


CCNumeric("AMT_TOTAL_RECEIVABLE")


# In[ ]:


CC["CNT_DRAWINGS_ATM_CURRENT"].describe()


# In[ ]:


CCNumeric("CNT_DRAWINGS_ATM_CURRENT")


# In[ ]:


CC["CNT_DRAWINGS_CURRENT"].describe()


# In[ ]:


CCNumeric("CNT_DRAWINGS_CURRENT")


# In[ ]:


CC["CNT_DRAWINGS_OTHER_CURRENT"].describe()


# In[ ]:


CCNumeric("CNT_DRAWINGS_OTHER_CURRENT")


# In[ ]:


CC["CNT_DRAWINGS_POS_CURRENT"].describe()


# In[ ]:


CCNumeric("CNT_DRAWINGS_POS_CURRENT")


# In[ ]:


CC["CNT_INSTALMENT_MATURE_CUM"].describe()


# In[ ]:


CCNumeric("CNT_INSTALMENT_MATURE_CUM")


# In[ ]:


CC["NAME_CONTRACT_STATUS"].describe()


# In[ ]:


CCExpCat("NAME_CONTRACT_STATUS")


# In[ ]:


CC["SK_DPD"].describe()


# In[ ]:


CCNumeric("SK_DPD")


# In[ ]:


CC["SK_DPD_DEF"].describe()


# In[ ]:


CCNumeric("SK_DPD_DEF")


# In[ ]:


Pre = pd.read_csv("../input/previous_application.csv")


# In[ ]:


def PreExpCat(col):
    t = Pre[col].value_counts()
    labels = t.index
    values = t.values
    colors = ['#96D38C','#FEBFB3']
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo="all", textinfo='value',
                   textfont=dict(size=12),
                   marker=dict(colors=colors,
                               line=dict(color='#fff', width=2)))
    layout = go.Layout(title=col, height=400)
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
def PreNumeric(col):
    plt.figure(figsize=(12,5))
    plt.title("Distribution of "+col)
    ax = sns.distplot(Pre[col].dropna())


# In[ ]:


Pre.head()


# In[ ]:


Pre["SK_ID_PREV"].describe()


# In[ ]:


PreNumeric("SK_ID_PREV")


# In[ ]:


Pre["SK_ID_CURR"].describe()


# In[ ]:


PreNumeric("SK_ID_CURR")


# In[ ]:


Pre["NAME_CONTRACT_TYPE"].describe()


# In[ ]:


PreExpCat('NAME_CONTRACT_TYPE')


# In[ ]:


Pre["AMT_ANNUITY"].describe()


# In[ ]:


PreNumeric("AMT_ANNUITY")


# In[ ]:


Pre["AMT_APPLICATION"].describe()


# In[ ]:


PreNumeric("AMT_APPLICATION")


# In[ ]:


Pre["AMT_CREDIT"].describe()


# In[ ]:


PreNumeric("AMT_CREDIT")


# In[ ]:


Pre["AMT_DOWN_PAYMENT"].describe()


# In[ ]:


PreNumeric("AMT_DOWN_PAYMENT")


# In[ ]:


Pre["AMT_GOODS_PRICE"].describe()


# In[ ]:


PreNumeric("AMT_GOODS_PRICE")


# In[ ]:


Pre["WEEKDAY_APPR_PROCESS_START"].describe()


# In[ ]:


PreExpCat("WEEKDAY_APPR_PROCESS_START")


# In[ ]:


Pre["HOUR_APPR_PROCESS_START"].describe()


# In[ ]:


PreNumeric("HOUR_APPR_PROCESS_START")


# In[ ]:


Pre["FLAG_LAST_APPL_PER_CONTRACT"].describe()


# In[ ]:


PreExpCat("FLAG_LAST_APPL_PER_CONTRACT")


# In[ ]:


Pre["NFLAG_LAST_APPL_IN_DAY"].describe()


# In[ ]:


PreNumeric("NFLAG_LAST_APPL_IN_DAY")


# In[ ]:


Pre["RATE_DOWN_PAYMENT"].describe()


# In[ ]:


PreNumeric("RATE_DOWN_PAYMENT")


# In[ ]:


Pre["RATE_INTEREST_PRIMARY"].describe()


# In[ ]:


PreNumeric("RATE_INTEREST_PRIMARY")


# In[ ]:


Pre["RATE_INTEREST_PRIVILEGED"].describe()


# In[ ]:


PreNumeric("RATE_INTEREST_PRIVILEGED")


# In[ ]:


Pre["NAME_CASH_LOAN_PURPOSE"].describe()


# In[ ]:


PreExpCat("NAME_CASH_LOAN_PURPOSE")


# In[ ]:


Pre["NAME_CONTRACT_STATUS"].describe()


# In[ ]:


PreExpCat("NAME_CONTRACT_STATUS")


# In[ ]:


Pre["DAYS_DECISION"].describe()


# In[ ]:


PreNumeric("DAYS_DECISION")


# In[ ]:


Pre['NAME_PAYMENT_TYPE'].describe()


# In[ ]:


PreExpCat("NAME_PAYMENT_TYPE")


# In[ ]:


Pre["CODE_REJECT_REASON"].describe()


# In[ ]:


PreExpCat("CODE_REJECT_REASON")


# In[ ]:


Pre["NAME_TYPE_SUITE"].describe()


# In[ ]:


PreExpCat("NAME_TYPE_SUITE")


# In[ ]:


Pre["NAME_CLIENT_TYPE"].describe()


# In[ ]:


PreExpCat("NAME_CLIENT_TYPE")


# In[ ]:


Pre["NAME_GOODS_CATEGORY"].describe()


# In[ ]:


PreExpCat("NAME_GOODS_CATEGORY")


# In[ ]:


Pre["NAME_PORTFOLIO"].describe()


# In[ ]:


PreExpCat("NAME_PORTFOLIO")


# In[ ]:


Pre["NAME_PRODUCT_TYPE"].describe()


# In[ ]:


PreExpCat("NAME_PRODUCT_TYPE")


# In[ ]:


Pre["CHANNEL_TYPE"].describe()


# In[ ]:


PreExpCat("CHANNEL_TYPE")


# In[ ]:


Pre["SELLERPLACE_AREA"].describe()


# In[ ]:


PreNumeric("SELLERPLACE_AREA")


# In[ ]:


Pre["NAME_SELLER_INDUSTRY"].describe()


# In[ ]:


PreExpCat("NAME_SELLER_INDUSTRY")


# In[ ]:


Pre["CNT_PAYMENT"].describe()


# In[ ]:


PreNumeric("CNT_PAYMENT")


# In[ ]:


Pre["NAME_YIELD_GROUP"].describe()


# In[ ]:


PreExpCat("NAME_YIELD_GROUP")


# In[ ]:


Pre["PRODUCT_COMBINATION"].describe()


# In[ ]:


PreExpCat("PRODUCT_COMBINATION")


# In[ ]:


Pre["DAYS_FIRST_DRAWING"].describe()


# In[ ]:


PreNumeric("DAYS_FIRST_DRAWING")


# In[ ]:


Pre["DAYS_FIRST_DUE"].describe()


# In[ ]:


PreNumeric("DAYS_FIRST_DUE")


# In[ ]:


Pre["DAYS_LAST_DUE_1ST_VERSION"].describe()


# In[ ]:


PreNumeric("DAYS_LAST_DUE_1ST_VERSION")


# In[ ]:


Pre["DAYS_LAST_DUE"].describe()


# In[ ]:


PreNumeric("DAYS_LAST_DUE")


# In[ ]:


Pre["DAYS_TERMINATION"].describe()


# In[ ]:


PreNumeric("DAYS_TERMINATION")


# In[ ]:


Pre["NFLAG_INSURED_ON_APPROVAL"].describe()


# In[ ]:


PreNumeric("NFLAG_INSURED_ON_APPROVAL")


# In[ ]:


ip = pd.read_csv("../input/installments_payments.csv")


def ipExpCat(col):
    t = ip[col].value_counts()
    labels = t.index
    values = t.values
    colors = ['#96D38C','#FEBFB3']
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo="all", textinfo='value',
                   textfont=dict(size=12),
                   marker=dict(colors=colors,
                               line=dict(color='#fff', width=2)))
    layout = go.Layout(title=col, height=400)
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
def ipNumeric(col):
    plt.figure(figsize=(12,5))
    plt.title("Distribution of "+col)
    ax = sns.distplot(ip[col].dropna())


# In[ ]:


ip["SK_ID_PREV"].describe()


# In[ ]:


ipNumeric("SK_ID_PREV")


# In[ ]:


ip["SK_ID_CURR"].describe()


# In[ ]:


ipNumeric("SK_ID_CURR")


# In[ ]:


ip["NUM_INSTALMENT_VERSION"].describe()


# In[ ]:


ipNumeric("NUM_INSTALMENT_VERSION")


# In[ ]:


ip["NUM_INSTALMENT_NUMBER"].describe()


# In[ ]:


ipNumeric("NUM_INSTALMENT_NUMBER")


# In[ ]:


ip["DAYS_INSTALMENT"].describe()


# In[ ]:


ipNumeric("DAYS_INSTALMENT")


# In[ ]:


ip["DAYS_ENTRY_PAYMENT"].describe()


# In[ ]:


ipNumeric("DAYS_ENTRY_PAYMENT")


# In[ ]:


ip["AMT_INSTALMENT"].describe()


# In[ ]:


ipNumeric("AMT_INSTALMENT")


# In[ ]:


ip["AMT_PAYMENT"].describe()


# In[ ]:


ipNumeric("AMT_PAYMENT")

