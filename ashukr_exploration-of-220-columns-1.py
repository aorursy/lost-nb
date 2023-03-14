#!/usr/bin/env python
# coding: utf-8



get_ipython().system('ls')




get_ipython().system('ls ../input/')




from scipy import stats




import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
get_ipython().run_line_magic('matplotlib', 'inline')




import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




from plotly import tools




init_notebook_mode(connected=True)




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




def numeric(col):
    plt.figure(figsize=(12,5))
    plt.title("Distribution of "+col)
    ax = sns.distplot(application[col].dropna())




get_ipython().system('ls ../input')




application = pd.read_csv("../input/application_train.csv")




application.head()




application.shape




bar_hor(application, "TARGET", "Distribution of Target Variable" , ["#44ff54", '#ff4444'], h=350, w=600, lm=200, xlb = ['Target : 1','Target : 0'])




exploreCat("NAME_CONTRACT_TYPE")




catAndTrgt("NAME_CONTRACT_TYPE")




exploreCat("CODE_GENDER")




catAndTrgt("CODE_GENDER")




exploreCat("FLAG_OWN_CAR")




catAndTrgt("FLAG_OWN_CAR")




exploreCat("FLAG_OWN_REALTY")




catAndTrgt("FLAG_OWN_REALTY")




exploreCat("CNT_CHILDREN")




catAndTrgt("CNT_CHILDREN")




application["AMT_INCOME_TOTAL"].dtype




numeric("AMT_INCOME_TOTAL")




numeric("AMT_CREDIT")




application["AMT_ANNUITY"].dtype




numeric("AMT_ANNUITY")




numeric("AMT_GOODS_PRICE")




exploreCat("NAME_TYPE_SUITE")




catAndTrgt("NAME_TYPE_SUITE")




application["NAME_INCOME_TYPE"].dtype




exploreCat("NAME_INCOME_TYPE")




catAndTrgt("NAME_INCOME_TYPE")




application["NAME_EDUCATION_TYPE"].dtype





exploreCat("NAME_EDUCATION_TYPE")




application["NAME_FAMILY_STATUS"].dtype





exploreCat("NAME_FAMILY_STATUS")





catAndTrgt("NAME_FAMILY_STATUS")




application["NAME_HOUSING_TYPE"].dtype





exploreCat("NAME_HOUSING_TYPE")





catAndTrgt("NAME_HOUSING_TYPE")




application["REGION_POPULATION_RELATIVE"].dtype





numeric("REGION_POPULATION_RELATIVE")




application["REGION_POPULATION_RELATIVE"].describe()




application["DAYS_BIRTH"].dtype





numeric("DAYS_BIRTH")





application["DAYS_BIRTH"].describe()




application["DAYS_EMPLOYED"].dtype




numeric("DAYS_EMPLOYED")




application["DAYS_EMPLOYED"].describe()




application["DAYS_REGISTRATION"].dtype




numeric("DAYS_REGISTRATION")




application["DAYS_REGISTRATION"].describe()




application["OWN_CAR_AGE"].dtype




numeric("OWN_CAR_AGE")




application["OWN_CAR_AGE"].describe()




exploreCat("FLAG_MOBIL")





catAndTrgt("FLAG_MOBIL")




exploreCat("FLAG_EMP_PHONE")





catAndTrgt("FLAG_EMP_PHONE")




exploreCat("FLAG_WORK_PHONE")




catAndTrgt("FLAG_WORK_PHONE")




exploreCat("FLAG_CONT_MOBILE")




catAndTrgt("FLAG_CONT_MOBILE")




exploreCat("FLAG_PHONE")




catAndTrgt("FLAG_PHONE")




exploreCat("FLAG_EMAIL")




catAndTrgt("FLAG_EMAIL")




application["OCCUPATION_TYPE"].dtype





exploreCat("OCCUPATION_TYPE")





catAndTrgt("OCCUPATION_TYPE")




exploreCat("CNT_FAM_MEMBERS")




catAndTrgt("CNT_FAM_MEMBERS")




exploreCat("REGION_RATING_CLIENT")





catAndTrgt("REGION_RATING_CLIENT")




exploreCat("REGION_RATING_CLIENT_W_CITY")




catAndTrgt("REGION_RATING_CLIENT_W_CITY")




exploreCat("REGION_RATING_CLIENT_W_CITY")





catAndTrgt("REGION_RATING_CLIENT_W_CITY")




exploreCat("HOUR_APPR_PROCESS_START")




catAndTrgt("HOUR_APPR_PROCESS_START")




exploreCat("REG_REGION_NOT_LIVE_REGION")




catAndTrgt("REG_REGION_NOT_LIVE_REGION")




exploreCat("REG_REGION_NOT_WORK_REGION")





catAndTrgt("REG_REGION_NOT_WORK_REGION")




exploreCat("LIVE_REGION_NOT_WORK_REGION")




catAndTrgt("LIVE_REGION_NOT_WORK_REGION")




exploreCat("REG_CITY_NOT_LIVE_CITY")





catAndTrgt("REG_CITY_NOT_LIVE_CITY")




exploreCat("REG_CITY_NOT_LIVE_CITY")





catAndTrgt("REG_CITY_NOT_LIVE_CITY")




exploreCat("LIVE_CITY_NOT_WORK_CITY")





catAndTrgt("LIVE_CITY_NOT_WORK_CITY")




exploreCat("ORGANIZATION_TYPE")




catAndTrgt("ORGANIZATION_TYPE")




application["EXT_SOURCE_1"].dtype




numeric("EXT_SOURCE_1")




application["EXT_SOURCE_2"].dtype





numeric("EXT_SOURCE_2")




numeric("EXT_SOURCE_3")




application["EXT_SOURCE_3"].describe()




numeric("APARTMENTS_AVG")




application["APARTMENTS_AVG"].describe()




numeric("BASEMENTAREA_AVG")




application["BASEMENTAREA_AVG"].describe()




numeric("YEARS_BEGINEXPLUATATION_AVG")




application["YEARS_BEGINEXPLUATATION_AVG"].describe()




numeric("YEARS_BUILD_AVG")




application["YEARS_BUILD_AVG"].describe()




numeric("COMMONAREA_AVG")




application["COMMONAREA_AVG"].describe()




numeric("ELEVATORS_AVG")





numeric("ENTRANCES_AVG")




application["ENTRANCES_AVG"].describe()





numeric("FLOORSMAX_AVG")




application["FLOORSMAX_AVG"].describe()




numeric("FLOORSMIN_AVG")




application["FLOORSMIN_AVG"].describe()




numeric("LANDAREA_AVG")




application["LANDAREA_AVG"].describe()




numeric("LIVINGAPARTMENTS_AVG")




application["LIVINGAPARTMENTS_AVG"].describe()




numeric("LIVINGAREA_AVG")




application["LIVINGAREA_AVG"].describe()




numeric("NONLIVINGAPARTMENTS_AVG")




application["NONLIVINGAPARTMENTS_AVG"].describe()




numeric("NONLIVINGAREA_AVG")




application["NONLIVINGAPARTMENTS_AVG"].describe()




numeric("APARTMENTS_MODE")




application["APARTMENTS_MODE"].describe()




numeric("BASEMENTAREA_MODE")





application["BASEMENTAREA_MODE"].describe()




numeric("YEARS_BEGINEXPLUATATION_MODE")




application["YEARS_BEGINEXPLUATATION_MODE"].describe()




numeric("YEARS_BUILD_MODE")





application["YEARS_BUILD_MODE"].describe()




numeric("COMMONAREA_MODE")





application["COMMONAREA_MODE"].describe()





numeric("ELEVATORS_MODE")





application["ELEVATORS_MODE"].describe()





numeric("ENTRANCES_MODE")





application["ENTRANCES_MODE"].describe()





application["FLOORSMAX_MODE"].describe()




numeric("FLOORSMAX_MODE")




numeric("FLOORSMIN_MODE")





application["FLOORSMIN_MODE"].describe()





numeric("LANDAREA_MODE")





application["LANDAREA_MODE"].describe()




numeric("LIVINGAPARTMENTS_MODE")





application["LIVINGAPARTMENTS_MODE"].describe()





numeric("LIVINGAREA_MODE")




application["LIVINGAREA_MODE"].describe()




numeric("NONLIVINGAPARTMENTS_MODE")





application["NONLIVINGAPARTMENTS_MODE"].describe()




numeric("NONLIVINGAREA_MODE")





application["NONLIVINGAREA_MODE"].describe()





numeric("APARTMENTS_MEDI")




application["APARTMENTS_MEDI"].describe()




numeric("BASEMENTAREA_MEDI")




application["BASEMENTAREA_MEDI"].describe()




numeric("YEARS_BEGINEXPLUATATION_MEDI")




application["YEARS_BEGINEXPLUATATION_MEDI"].describe()




numeric("COMMONAREA_MEDI")





application["COMMONAREA_MEDI"].describe()




numeric("ELEVATORS_MEDI")




application["ELEVATORS_MEDI"].describe()




numeric("ENTRANCES_MEDI")




application["ENTRANCES_MEDI"].describe()




numeric("FLOORSMAX_MEDI")




application["FLOORSMAX_MEDI"].describe()




numeric("FLOORSMIN_MEDI")




application["FLOORSMIN_MEDI"].describe()




numeric("LANDAREA_MEDI")




application["LANDAREA_MEDI"].describe()




numeric("LIVINGAPARTMENTS_MEDI")




application["LIVINGAPARTMENTS_MEDI"].describe()




numeric("LIVINGAREA_MEDI")




application["LIVINGAREA_MEDI"].describe()




numeric("NONLIVINGAPARTMENTS_MEDI")




application["NONLIVINGAPARTMENTS_MEDI"].describe()




numeric("NONLIVINGAREA_MEDI")




application["NONLIVINGAREA_MEDI"].describe()




application["FONDKAPREMONT_MODE"].dtype




exploreCat("FONDKAPREMONT_MODE")




catAndTrgt("FONDKAPREMONT_MODE")




application["HOUSETYPE_MODE"].dtype




exploreCat("HOUSETYPE_MODE")




exploreCat("WALLSMATERIAL_MODE")





catAndTrgt("WALLSMATERIAL_MODE")




exploreCat("EMERGENCYSTATE_MODE")




catAndTrgt("EMERGENCYSTATE_MODE")




application["OBS_30_CNT_SOCIAL_CIRCLE"].dtype




numeric("OBS_30_CNT_SOCIAL_CIRCLE")




application["OBS_30_CNT_SOCIAL_CIRCLE"].describe()




exploreCat("DEF_30_CNT_SOCIAL_CIRCLE")




catAndTrgt("DEF_30_CNT_SOCIAL_CIRCLE")




exploreCat("OBS_60_CNT_SOCIAL_CIRCLE")




catAndTrgt("OBS_60_CNT_SOCIAL_CIRCLE")




exploreCat("DEF_60_CNT_SOCIAL_CIRCLE")




catAndTrgt("DEF_60_CNT_SOCIAL_CIRCLE")




application["DAYS_LAST_PHONE_CHANGE"].dtype




numeric("DAYS_LAST_PHONE_CHANGE")




application["DAYS_LAST_PHONE_CHANGE"].describe()




exploreCat("FLAG_DOCUMENT_2")




catAndTrgt("FLAG_DOCUMENT_2")




exploreCat("FLAG_DOCUMENT_3")




catAndTrgt("FLAG_DOCUMENT_3")




exploreCat("FLAG_DOCUMENT_4")





catAndTrgt("FLAG_DOCUMENT_4")




exploreCat("FLAG_DOCUMENT_5")




catAndTrgt("FLAG_DOCUMENT_5")




exploreCat("FLAG_DOCUMENT_6")




catAndTrgt("FLAG_DOCUMENT_6")




exploreCat("FLAG_DOCUMENT_7")




catAndTrgt("FLAG_DOCUMENT_7")




exploreCat("FLAG_DOCUMENT_8")




catAndTrgt("FLAG_DOCUMENT_8")




exploreCat("FLAG_DOCUMENT_9")




catAndTrgt("FLAG_DOCUMENT_9")




exploreCat("FLAG_DOCUMENT_10")




catAndTrgt("FLAG_DOCUMENT_10")




exploreCat("FLAG_DOCUMENT_11")




catAndTrgt("FLAG_DOCUMENT_11")




exploreCat("FLAG_DOCUMENT_12")




catAndTrgt("FLAG_DOCUMENT_12")




exploreCat("FLAG_DOCUMENT_13")




catAndTrgt("FLAG_DOCUMENT_13")




exploreCat("FLAG_DOCUMENT_14")




catAndTrgt("FLAG_DOCUMENT_14")




exploreCat("FLAG_DOCUMENT_15")




catAndTrgt("FLAG_DOCUMENT_15")




exploreCat("FLAG_DOCUMENT_16")




catAndTrgt("FLAG_DOCUMENT_16")




exploreCat("FLAG_DOCUMENT_17")




catAndTrgt("FLAG_DOCUMENT_17")




exploreCat("FLAG_DOCUMENT_18")




catAndTrgt("FLAG_DOCUMENT_18")




exploreCat("FLAG_DOCUMENT_19")




catAndTrgt("FLAG_DOCUMENT_19")




exploreCat("FLAG_DOCUMENT_20")




catAndTrgt("FLAG_DOCUMENT_21")




application["AMT_REQ_CREDIT_BUREAU_HOUR"].dtype




numeric("AMT_REQ_CREDIT_BUREAU_HOUR")




application["AMT_REQ_CREDIT_BUREAU_HOUR"].describe()




application["AMT_REQ_CREDIT_BUREAU_DAY"].dtype




numeric("AMT_REQ_CREDIT_BUREAU_DAY")




application["AMT_REQ_CREDIT_BUREAU_DAY"].dtype




numeric("AMT_REQ_CREDIT_BUREAU_DAY")




(application["AMT_REQ_CREDIT_BUREAU_DAY"].describe())




application["AMT_REQ_CREDIT_BUREAU_WEEK"].dtype




numeric("AMT_REQ_CREDIT_BUREAU_WEEK")




application["AMT_REQ_CREDIT_BUREAU_WEEK"].describe()




application["AMT_REQ_CREDIT_BUREAU_MON"].describe()




numeric("AMT_REQ_CREDIT_BUREAU_MON")




application["AMT_REQ_CREDIT_BUREAU_QRT"].describe()




numeric("AMT_REQ_CREDIT_BUREAU_QRT")




application["AMT_REQ_CREDIT_BUREAU_YEAR"].describe()




numeric("AMT_REQ_CREDIT_BUREAU_YEAR")




bureau = pd.read_csv("../input/bureau.csv")




bureau.head()




bureau["SK_ID_CURR"].head()




bureau["SK_ID_CURR"].describe()




BNumeric("SK_ID_CURR")




bureau["SK_ID_BUREAU"].head()




bureau["SK_ID_BUREAU"].describe()




bureau["CREDIT_ACTIVE"].describe()




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




BExpCat("CREDIT_ACTIVE")




bureau["CREDIT_CURRENCY"].describe()




BExpCat("CREDIT_CURRENCY")




bureau["DAYS_CREDIT"].describe()




BNumeric("DAYS_CREDIT")




bureau["CREDIT_DAY_OVERDUE"].describe()




BNumeric("CREDIT_DAY_OVERDUE")




bureau["DAYS_CREDIT_ENDDATE"].describe()




BNumeric("DAYS_CREDIT_ENDDATE")




bureau["DAYS_ENDDATE_FACT"].describe()




BNumeric("DAYS_ENDDATE_FACT")




bureau["AMT_CREDIT_MAX_OVERDUE"].describe()




BNumeric("AMT_CREDIT_MAX_OVERDUE")





bureau["CNT_CREDIT_PROLONG"].describe()




BNumeric("CNT_CREDIT_PROLONG")




bureau["AMT_CREDIT_SUM"].describe()




BNumeric("AMT_CREDIT_SUM")




bureau["AMT_CREDIT_SUM_DEBT"].describe()




BNumeric("AMT_CREDIT_SUM_DEBT")




bureau["AMT_CREDIT_SUM_LIMIT"].describe()




BNumeric("AMT_CREDIT_SUM_LIMIT")




bureau["AMT_CREDIT_SUM_OVERDUE"].describe()




BNumeric("AMT_CREDIT_SUM_OVERDUE")




bureau["AMT_CREDIT_SUM_OVERDUE"].describe()




BNumeric("AMT_CREDIT_SUM_OVERDUE")




bureau["CREDIT_TYPE"].describe()




BExpCat("CREDIT_TYPE")




bureau["DAYS_CREDIT_UPDATE"].describe()




BNumeric("DAYS_CREDIT_UPDATE")




bureau["AMT_ANNUITY"].describe()




BNumeric("AMT_ANNUITY")




bb = pd.read_csv("../input/bureau_balance.csv")




bb.head()




bb["SK_ID_BUREAU"].describe()




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




BBNumeric("SK_ID_BUREAU")




bb["MONTHS_BALANCE"].describe()




BBNumeric("MONTHS_BALANCE")




bb["STATUS"].describe()




BBExpCat("STATUS")




PC = pd.read_csv("../input/POS_CASH_balance.csv")




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




PC.head()




PC["SK_ID_PREV"].describe()




PCNumeric("SK_ID_PREV")




PC["SK_ID_CURR"].describe()




PCNumeric("SK_ID_CURR")





PCNumeric("MONTHS_BALANCE")




PC["MONTHS_BALANCE"].describe()




PC["CNT_INSTALMENT"].describe()




PCNumeric("CNT_INSTALMENT")




PCNumeric("CNT_INSTALMENT_FUTURE")




PC["CNT_INSTALMENT_FUTURE"].describe()




PC["CNT_INSTALMENT_FUTURE"].describe()




PCNumeric("CNT_INSTALMENT_FUTURE")




PC["NAME_CONTRACT_STATUS"].describe()




PCExpCat("NAME_CONTRACT_STATUS")




PC["SK_DPD"].describe()




PCNumeric("SK_DPD")




PC["SK_DPD_DEF"].describe()




PCNumeric("SK_DPD_DEF")




CC = pd.read_csv("../input/credit_card_balance.csv")




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




CC.describe()




CC["SK_ID_PREV"].describe()




CCNumeric("SK_ID_PREV")




CC["SK_ID_CURR"].describe()




CCNumeric('SK_ID_CURR')




CC["MONTHS_BALANCE"].describe()




CCNumeric("MONTHS_BALANCE")




CC["AMT_BALANCE"].describe()




CCNumeric("AMT_BALANCE")




CC["AMT_CREDIT_LIMIT_ACTUAL"].describe()




CCNumeric("AMT_CREDIT_LIMIT_ACTUAL")




CC["AMT_DRAWINGS_ATM_CURRENT"].describe()




CCNumeric("AMT_DRAWINGS_ATM_CURRENT")




CC["AMT_DRAWINGS_CURRENT"].describe()




CCNumeric("AMT_DRAWINGS_CURRENT")




CC["AMT_DRAWINGS_OTHER_CURRENT"].describe()




CCNumeric("AMT_DRAWINGS_OTHER_CURRENT")




CC["AMT_DRAWINGS_POS_CURRENT"].describe()




CCNumeric("AMT_DRAWINGS_POS_CURRENT")




CC["AMT_INST_MIN_REGULARITY"].describe()




CCNumeric('AMT_INST_MIN_REGULARITY')




CC["AMT_PAYMENT_CURRENT"].describe()




CCNumeric("AMT_PAYMENT_CURRENT")




CC["AMT_PAYMENT_TOTAL_CURRENT"].describe()




CCNumeric("AMT_PAYMENT_TOTAL_CURRENT")




CC["AMT_RECEIVABLE_PRINCIPAL"].describe()




CCNumeric("AMT_RECEIVABLE_PRINCIPAL")




CC["AMT_RECIVABLE"].describe()




CCNumeric("AMT_RECIVABLE")




CC["AMT_TOTAL_RECEIVABLE"].describe()




CCNumeric("AMT_TOTAL_RECEIVABLE")




CC["CNT_DRAWINGS_ATM_CURRENT"].describe()




CCNumeric("CNT_DRAWINGS_ATM_CURRENT")




CC["CNT_DRAWINGS_CURRENT"].describe()




CCNumeric("CNT_DRAWINGS_CURRENT")




CC["CNT_DRAWINGS_OTHER_CURRENT"].describe()




CCNumeric("CNT_DRAWINGS_OTHER_CURRENT")




CC["CNT_DRAWINGS_POS_CURRENT"].describe()




CCNumeric("CNT_DRAWINGS_POS_CURRENT")




CC["CNT_INSTALMENT_MATURE_CUM"].describe()




CCNumeric("CNT_INSTALMENT_MATURE_CUM")




CC["NAME_CONTRACT_STATUS"].describe()




CCExpCat("NAME_CONTRACT_STATUS")




CC["SK_DPD"].describe()




CCNumeric("SK_DPD")




CC["SK_DPD_DEF"].describe()




CCNumeric("SK_DPD_DEF")




Pre = pd.read_csv("../input/previous_application.csv")




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




Pre.head()




Pre["SK_ID_PREV"].describe()




PreNumeric("SK_ID_PREV")




Pre["SK_ID_CURR"].describe()




PreNumeric("SK_ID_CURR")




Pre["NAME_CONTRACT_TYPE"].describe()




PreExpCat('NAME_CONTRACT_TYPE')




Pre["AMT_ANNUITY"].describe()




PreNumeric("AMT_ANNUITY")




Pre["AMT_APPLICATION"].describe()




PreNumeric("AMT_APPLICATION")




Pre["AMT_CREDIT"].describe()




PreNumeric("AMT_CREDIT")




Pre["AMT_DOWN_PAYMENT"].describe()




PreNumeric("AMT_DOWN_PAYMENT")




Pre["AMT_GOODS_PRICE"].describe()




PreNumeric("AMT_GOODS_PRICE")




Pre["WEEKDAY_APPR_PROCESS_START"].describe()




PreExpCat("WEEKDAY_APPR_PROCESS_START")




Pre["HOUR_APPR_PROCESS_START"].describe()




PreNumeric("HOUR_APPR_PROCESS_START")




Pre["FLAG_LAST_APPL_PER_CONTRACT"].describe()




PreExpCat("FLAG_LAST_APPL_PER_CONTRACT")




Pre["NFLAG_LAST_APPL_IN_DAY"].describe()




PreNumeric("NFLAG_LAST_APPL_IN_DAY")




Pre["RATE_DOWN_PAYMENT"].describe()




PreNumeric("RATE_DOWN_PAYMENT")




Pre["RATE_INTEREST_PRIMARY"].describe()




PreNumeric("RATE_INTEREST_PRIMARY")




Pre["RATE_INTEREST_PRIVILEGED"].describe()




PreNumeric("RATE_INTEREST_PRIVILEGED")




Pre["NAME_CASH_LOAN_PURPOSE"].describe()




PreExpCat("NAME_CASH_LOAN_PURPOSE")




Pre["NAME_CONTRACT_STATUS"].describe()




PreExpCat("NAME_CONTRACT_STATUS")




Pre["DAYS_DECISION"].describe()




PreNumeric("DAYS_DECISION")




Pre['NAME_PAYMENT_TYPE'].describe()




PreExpCat("NAME_PAYMENT_TYPE")




Pre["CODE_REJECT_REASON"].describe()




PreExpCat("CODE_REJECT_REASON")




Pre["NAME_TYPE_SUITE"].describe()




PreExpCat("NAME_TYPE_SUITE")




Pre["NAME_CLIENT_TYPE"].describe()




PreExpCat("NAME_CLIENT_TYPE")




Pre["NAME_GOODS_CATEGORY"].describe()




PreExpCat("NAME_GOODS_CATEGORY")




Pre["NAME_PORTFOLIO"].describe()




PreExpCat("NAME_PORTFOLIO")




Pre["NAME_PRODUCT_TYPE"].describe()




PreExpCat("NAME_PRODUCT_TYPE")




Pre["CHANNEL_TYPE"].describe()




PreExpCat("CHANNEL_TYPE")




Pre["SELLERPLACE_AREA"].describe()




PreNumeric("SELLERPLACE_AREA")




Pre["NAME_SELLER_INDUSTRY"].describe()




PreExpCat("NAME_SELLER_INDUSTRY")




Pre["CNT_PAYMENT"].describe()




PreNumeric("CNT_PAYMENT")




Pre["NAME_YIELD_GROUP"].describe()




PreExpCat("NAME_YIELD_GROUP")




Pre["PRODUCT_COMBINATION"].describe()




PreExpCat("PRODUCT_COMBINATION")




Pre["DAYS_FIRST_DRAWING"].describe()




PreNumeric("DAYS_FIRST_DRAWING")




Pre["DAYS_FIRST_DUE"].describe()




PreNumeric("DAYS_FIRST_DUE")




Pre["DAYS_LAST_DUE_1ST_VERSION"].describe()




PreNumeric("DAYS_LAST_DUE_1ST_VERSION")




Pre["DAYS_LAST_DUE"].describe()




PreNumeric("DAYS_LAST_DUE")




Pre["DAYS_TERMINATION"].describe()




PreNumeric("DAYS_TERMINATION")




Pre["NFLAG_INSURED_ON_APPROVAL"].describe()




PreNumeric("NFLAG_INSURED_ON_APPROVAL")




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




ip["SK_ID_PREV"].describe()




ipNumeric("SK_ID_PREV")




ip["SK_ID_CURR"].describe()




ipNumeric("SK_ID_CURR")




ip["NUM_INSTALMENT_VERSION"].describe()




ipNumeric("NUM_INSTALMENT_VERSION")




ip["NUM_INSTALMENT_NUMBER"].describe()




ipNumeric("NUM_INSTALMENT_NUMBER")




ip["DAYS_INSTALMENT"].describe()




ipNumeric("DAYS_INSTALMENT")




ip["DAYS_ENTRY_PAYMENT"].describe()




ipNumeric("DAYS_ENTRY_PAYMENT")




ip["AMT_INSTALMENT"].describe()




ipNumeric("AMT_INSTALMENT")




ip["AMT_PAYMENT"].describe()




ipNumeric("AMT_PAYMENT")

