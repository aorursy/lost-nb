#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def show(_df, count=2):
    print(_df.shape)
    size = _df.shape[0]
    count = count if size > count else size
    display(_df.head(count))
    
    
    display(_df.sample(count))
    display(_df.tail(count))


# In[3]:


import numpy as np
import seaborn as sns
import pandas as pd
import array as arr
import pandas_profiling
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[4]:


test_file = '/test.csv'
train_file = '/train.csv'
train_df = pd.read_csv(dirname + train_file)
test_df = pd.read_csv(dirname + test_file)


# In[5]:


#define variables 
predCol = 'TargetValue'
idCol = 'Id'
fidCol = 'ForecastId'
ignoreCols = [predCol, idCol, fidCol, 'Date']
ctry = 'Country_Region'
ste = 'Province_State'
cty = 'County'
lcn='Location'


# In[6]:



d2 = pd.to_datetime('2020-12-31')
lr = test_df[test_df[ctry]=='India'].tail(1)[['ForecastId','Date']].values[0]
d1 = pd.to_datetime(lr[1]) + timedelta(1)
id = lr[0]
print(d1)
print (d2)
ppd = pd.DataFrame(
    [[d]
     for d  in pd.date_range(d1, d2, freq='D')])
ppd.tail(2)


# In[7]:




def getid(id1):
    global id
    id1 = id + 1
    id = id1
    return id1;

print(id)
cc= pd.DataFrame(
    [[getid(id), np.NaN,np.NaN, 'India',1295210000,0.04766, d, 'ConfirmedCases']
     for d  in pd.date_range(d1, d2, freq='D')],
    columns=['ForecastId', cty, ste, ctry, 'Population', 'Weight', 'Date', 'Target']
)
print(cc.tail(2))

ff= pd.DataFrame(
    [[getid(id), np.NaN,np.NaN, 'India',1295210000,0.47660, d, 'Fatalities']
     for d in pd.date_range(d1, d2, freq='D')],
    columns=['ForecastId', cty, ste, ctry, 'Population', 'Weight', 'Date', 'Target']
)
print(ff.head(2))


moretest = pd.concat([cc,ff])
#moretest['ForecastId'] = moretest.apply(lambda x: getid(id))['ForecastId']

moretest[(moretest[ctry]=='India') & (moretest['Date']== '2020-06-10')]


# In[8]:


test_df =pd.concat([test_df, moretest])
#test_df.drop_duplicates()


# In[9]:


test_df[(test_df[ctry]=='India') & (test_df['Date']== '2020-06-10')]


# In[ ]:





# In[ ]:





# In[10]:


show(test_df)


# In[11]:


df_all = pd.concat([train_df, test_df], sort=False)
show(df_all, 5)


# In[12]:


df_all[lcn] = df_all[ctry] + '_' +  df_all[ste].fillna('NA') + '_' + df_all[cty].fillna('NA')


# In[13]:


country = ["US", "India","China", "Spain", "Italy", "Pakistan", "Mexico", "United Kingdom", "France", "Japan", "South Korea", "Russia", "Cananda", "Peru", "Turkey"]
country =[x + "_NA_NA" for x in country]
print(country)


# In[14]:



if len(country) >0 :
    df_all = df_all[df_all[lcn].isin(country )]

show(df_all, 2)


# In[15]:


locMin = df_all[df_all[predCol]>0 | df_all[predCol].notnull()].groupby([lcn])['Date'].min()
locMin


# In[16]:


def replace_startdate(_ctry, _dt):
    idex = locMin[locMin.index.str.startswith(_ctry)].index
    print(idex)
    _dt = pd.to_datetime(_dt, infer_datetime_format=True)
    print(_dt.strftime("%d-%m"))
    if len(idex) ==1:
        print(locMin.at[idex[0]])
        locMin.at[idex[0]] = _dt
        print(locMin.at[idex[0]])


replace_startdate('Russia', '03-05-2020')
replace_startdate('Italy', '02-20-2020')
replace_startdate('Germany', '02-24-2020')
replace_startdate('Turkey', '03-15-2020')
replace_startdate('Canada_NA_NA', '02-05-2020')


# In[17]:


df_all= pd.merge(df_all, locMin, on=[lcn,lcn])
show(df_all)


# In[18]:


import time
df_all['Date_x'] = pd.to_datetime(df_all['Date_x'], infer_datetime_format=True)
df_all['Date_y'] = pd.to_datetime(df_all['Date_y'], infer_datetime_format=True)


# In[19]:


df_all = df_all[df_all['Date_x']>=df_all['Date_y']]


# In[20]:


df_all['Days'] = df_all.apply(lambda x: ((x['Date_x'] -  x['Date_y']).days +1) ,axis=1)
df_all


# In[21]:


from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]
def makeMonotoic(df):
    state = df - df.shift(1)
    state = state.fillna(0)
    for st in state:
        if (st < 0):
            df = df.shift(1);

def topCharts(_dg, _n=[0,1], _s=0, _d=0, _w=18, _h=5, _m=400,_dif=2, _l=''):
    
    dg = _dg[_dg[predCol].notnull()]
    
    if _l != '':
        dg = dg[dg[lcn].str.startswith(_l)]
    dg = dg.groupby([lcn, 'Target' ])         .sum()         .sort_values(by=predCol, ascending=False)         .reset_index()
    dg1 = dg[(dg['Target']=='ConfirmedCases') | (dg['Target'] == 1)]
    dg1 = dg1.groupby([lcn]).sum()
    dg1 = dg1.sort_values(predCol, ascending=False)
    top_c = dg1[_n[0]:_n[0] + _n[1]]
    if top_c.shape[0] == 0 :
        print("Nothing to plot for any country")
        return
    
  
    _c = 4 if _n[1] > 4 else _n[1] 
    #show(top_c,1)
    top_10 = top_c.index
    #vfunc = np.vectorize(visualize, excluded=['_dg','_days'])
    vt = 4
    rows = len(top_10) // _c + 1
    figsize = (_w,_h*rows)
    fig ,axes= plt.subplots(rows, _c, figsize=figsize, constrained_layout=True)
    plt.grid(True)
    axes = trim_axs(axes, len(top_10))
    for ax, _ctry in zip(axes, top_10):
        ax2 = ax.twinx()
   
        tr = _dg[_dg[lcn]==_ctry]
        if _d > 0 :
            tr = tr.iloc[_s:_d*2]
        tr = tr.sort_values(by='Days', ascending= True)
        tr = tr[tr[predCol].notnull()]

        cases = tr[(tr['Target']=='ConfirmedCases') | (tr['Target'] == 1)]
        cases_dates = np.array(cases.apply(lambda x:  x['Date_x'].strftime("%d-%m") , axis=1).unique())
        days = np.array(tr.apply(lambda x:  x['Days'], axis=1).unique())
        confirmedCases = np.array(cases[predCol]).cumsum()
        
        
        
        fatals = tr[(tr['Target']=='Fatalities')  | (tr['Target'] == 2)]
        fatals_dates = np.array(fatals.apply(lambda x:  x['Date_x'].strftime("%d-%m") , axis=1).unique())
        Fatalities = np.array(fatals[predCol]).cumsum()


        
        #print(cc05)
     
        #print(Fatalities)

        #low = np.ma.masked_where(confirmedCases<=10, confirmedCases)
        #high = np.ma.masked_where(confirmedCases>10, confirmedCases)
        #ax.plot(low, high)
        
                    
        ax.plot(cases_dates, confirmedCases, '-b.', color='purple')
        qf= 'q95'
        qf1 = 'q05'
        if 'q05' in cases:
            cc05 = np.array( (cases[predCol] - cases[qf1]).cumsum())
            ax.plot(cases_dates, cc05, color='purple', linestyle='--')
            cc95 = np.array( (cases[predCol] + cases[qf]).cumsum())
            ax.plot(cases_dates, cc95, color='purple', linestyle='--')
            
                    
        if 'q05' in fatals:
            ff = fatals[predCol] - fatals[qf1]
            # print(fatals[predCol])
            #print(fatals[qf])
            # print(ff)
            ff05 = np.array( ff.cumsum() )
            ax2.plot(fatals_dates, ff05, color='orange', linestyle='--')
            ff95 = np.array((fatals[predCol] + fatals[qf]).cumsum())
            ax2.plot(fatals_dates, ff95, color='orange', linestyle='--')
        
        #if 'cc05' in locals() and 'cc95' in locals():
            #ax.bar(cases_dates, cc05,  color='')
            #ax.bar(cases_dates, cc95, bottom=cc05, color='grey')
            
       # if 'ff05' in locals() and 'ff95' in locals():
           
            #ax2.bar(fatals_dates, ff05,  color='pink')
            #ax2.bar(fatals_dates, ff95, bottom=ff05, color='red')
        
        ax.set_ylabel("Cases",fontsize=14,color='blue')
        
        #ax.text(days, confirmedCases, str(confirmedCases))
        

        ax2.plot(fatals_dates, Fatalities, '-b.', color='orange', linestyle='-')

        flim = ax2.get_ylim()
        ax2.set_ylim([-10, flim[1]*1.3])
        vt = int(len(fatals_dates)*_c/50 + 1)
        ax2.set_ylabel("Fatal",fontsize=14,color='blue')
        ax.set_title(_ctry + '(' + tr['Date_y'].iloc[0].strftime("%d-%m") + ')')
        ax.set_xticklabels(fatals_dates[0::vt], rotation=90)
        plt.xticks(fatals_dates[0::vt], rotation=90)
        #ax2.text(dates, Fatalities)
        ax.grid(True)
        ax2.grid(True)
        secax = ax.secondary_xaxis('top')
        
        secax.set_xlabel('days')
        
        lbly = (ax.get_ylim()[1]/_h) ;
        p = 0
        if _dif > 0:
            for i, v in enumerate(confirmedCases):
                if(v < _m and v - p >= _dif) :
                    ax.text(days[i] -1, v ,  str(v.astype(int)) , rotation=90, ha='left')
                    p = v

    
#topCharts(dfv1, _dif=10000, _l='India')


# In[22]:


topCharts(df_all, _n=[0,2], _d=10)


# In[23]:


topCharts(df_all, _n=[2,1], _d=90)


# In[24]:


topCharts(df_all, _n=[3,1], _d=40)


# In[25]:


topCharts(df_all, _n=[4,1], _d=40)


# In[26]:


topCharts(df_all, _n=[5,1], _d=20, _dif=1)


# In[27]:


topCharts(df_all, _n=[6,4], _d=15,_dif=0)


# In[28]:


topCharts(df_all, _n=[10,4], _d=15,_dif=0)


# In[29]:


topCharts(df_all, _n=[14,1], _d=30,_dif=0)


# In[30]:


topCharts(df_all, _n=[16,1], _d=30,_dif=0)


# In[31]:


topCharts(df_all, _n=[17,2], _d=30,_dif=0)


# In[32]:


topCharts(df_all, _n=[19,2], _d=30,_dif=1)


# In[33]:


topCharts(df_all, _n=[21,4], _d=10,_dif=5)


# In[34]:


topCharts(df_all, _n=[25,4], _d=10,_dif=5)


# In[35]:


topCharts(df_all, _n=[29,4], _d=10,_dif=0)


# In[36]:


topCharts(df_all, _n=[33,16], _d=30,_dif=0)


# In[37]:


ecol = df_all.isna().sum().sort_values(ascending=False)[df_all.isna().sum()>0].index;
print(ecol)


# In[38]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
dflcn = df_all[[lcn, ctry, cty, ste]]

df_all[lcn] = encoder.fit_transform(df_all[lcn])
dflcn['code'] = df_all[lcn]
#pd.get_dummies(df, columns=['Location'], drop_first=True)

show(dflcn)


# In[39]:


df_all['Target'] = df_all['Target'].apply(lambda s: 1 if s== "ConfirmedCases" else 2 )


# In[40]:


from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor
from sklearn.linear_model.theil_sen import TheilSenRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor


# In[41]:


models = [
    ExtraTreesRegressor(n_estimators=500,n_jobs=-1,verbose=1),
    XGBRegressor(n_estimators = 2300 , alpha = 0, gamma = 0, learning_rate = 0.04,  random_state = 42 , max_depth = 23),
    #LGBMRegressor(),
    #RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',max_depth=None, max_features='auto', max_leaf_nodes=None,max_samples=None, min_impurity_decrease=0.0,                      min_impurity_split=None, min_samples_leaf=1,                      min_samples_split=2, min_weight_fraction_leaf=0.0,                      n_estimators=100, n_jobs=None, oob_score=False,                      random_state=None, verbose=0, warm_start=False),
    #KNeighborsRegressor(),
    #AdaBoostRegressor(),
    #PassiveAggressiveRegressor(),
    #TheilSenRegressor()
]


# In[42]:



train = df_all[df_all[predCol].notnull()].drop(columns=['ForecastId'])

test = df_all[df_all[predCol].isna()].drop([predCol], axis=1).drop(columns=['Id'])

test.rename(columns={'ForecastId':'Id'}, inplace=True)


# In[43]:


cols = ['Id', 'Population', 'Weight', 'Location', 'Days', 'Target']

test= test[cols]
train= train[cols + ['TargetValue']]


# In[44]:


Xtrn, Xtest, Ytrn, Ytest = train_test_split(train.drop([predCol], axis=1), train[[predCol]], test_size=0.2, random_state=42)


# In[45]:


def handle_predictions (predictions):
    predictions[predictions < 0] = 0   
    return predictions


# In[46]:


TestModels = pd.DataFrame()
tmp = {}
 
for model in models:
    # get model name
    
    tmp['Model'] = str(model)
    # fit model on training dataset
    model.fit(Xtrn, Ytrn[predCol])
    pred= model.predict(Xtest)
    pred = handle_predictions(pred)

   
    act = pred
    targ = handle_predictions(Ytest[predCol])
    
    tmp['accuracy'] = r2_score(targ, act)
    tmp['rmsle'] = (mean_squared_log_error(targ,act))
    tmp['rmse'] = (mean_squared_error(targ, act))
    # write obtained data
    TestModels = TestModels.append([tmp])
        
TestModels.set_index('Model', inplace=True)
TestModels


# In[47]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import make_interp_spline, BSpline
import math 

scaler = StandardScaler()
scaled_data = scaler.fit_transform(TestModels)
print(scaled_data.min())
scaled_data = scaled_data + abs(scaled_data.min())+1
accuracy = scaled_data[0:,0:1]
rmsle = scaled_data[0:,1:2]
rmse =   scaled_data[0:,2:]
allmet =  1/accuracy

print(scaled_data)
print(1/accuracy)
fig, axes = plt.subplots(figsize=(15, 6))
axes.plot(range(len(accuracy)),accuracy, color='blue' , marker='.')
axes.plot(range(len(accuracy)),rmsle, color='green' , marker='.')
axes.plot(range(len(accuracy)),rmse, color='orange' , marker='.')

axes.plot(range(len(accuracy)),allmet, color='red' , marker='.')

plt.show()


# In[48]:


bestModel = models[allmet.argmin()]
TestModels.iloc[allmet.argmin():allmet.argmin()+1].index[0]


# In[49]:


bestModelName = TestModels.iloc[allmet.argmin():allmet.argmin()+1].index[0]
bestScore = allmet[allmet.argmin()]      
print('Best Score : ' + str(bestScore))
selectedModel = models[allmet.argmin()]


# In[50]:


show(test)


# In[51]:


selectedModel.fit(train.drop([predCol], axis=1), train[predCol])
prediction =selectedModel.predict(test)


# In[52]:


test[predCol] = prediction
test['istrain'] = 0
train['istrain'] = 1
print(test.shape)
print(train.shape)


# In[53]:


cols = [lcn, 'Days','Target']
#print(train[(train['Date_x']=='2020-05-17') & (train[ctry]=='India')])
train1=train.rename(columns={'Id':'tid'})
#print(train1[(train1['Date_x']=='2020-05-17') & (train1[ctry]=='India')])

intersect = pd.merge(train1[ [predCol, 'tid'] + cols], test[cols + ['Id']], how='inner')#.set_index('tid')
tidx = intersect['tid']
train = train[~train['Id'].isin(tidx)]
#show(intersect,2)
#print(intersect[(intersect['Days']==77) & (intersect[lcn]==142) ])
intersect.drop(columns=['tid'],axis=1, inplace=True)

intersect= intersect.set_index('Id')
tt = test#[(test['Date_x']=='2020-05-17') & (test[ctry]=='India')]
tt = tt.drop(columns=[predCol])#.set_index('Id')
#print(tt[(tt['Days']==77) & (tt[lcn]==142)])
#print(intersect[(intersect['Days']==77) & (intersect[lcn]==142)])

intersect = pd.merge(tt, intersect, how='inner')

test = test[~test['Id'].isin(intersect['Id'])]
#print(intersect)
#print()
print(intersect[(intersect['Days']==77) & (intersect[lcn]==142)])
test1 = test
test1 = pd.concat([intersect, test1], sort=False)
#print(test1[(test1['Date_x']=='2020-05-17') & (test1[ctry]=='India')])
test = test1.drop_duplicates()
#print(test1[(test1['Date_x']=='2020-05-17') & (test1[ctry]=='India')])


# In[54]:


dflcn = dflcn.drop_duplicates()

dfv = pd.concat([train, test ], sort=False)

dfv = pd.merge(dfv, dflcn, left_on=lcn, right_on='code',how='left')
dfv.rename(columns={'Location_y':lcn}, inplace=True)


# In[55]:


dfv['Date_x'] = dfv.apply(lambda x: pd.to_datetime(locMin.at[x[lcn],]) + timedelta(x['Days']) , axis=1)
dfv['Date_y'] = dfv.apply(lambda x: pd.to_datetime(locMin.at[x[lcn],]) , axis=1)
dfv


# In[56]:


dfv['Target'] = dfv['Target'].apply(lambda s: "ConfirmedCases" if s == 1 else "Fatalities" )
dfv


# In[57]:


dfv1= dfv

winsize = len(dfv1[lcn].unique())
print (winsize)
dfv1['q05'] = dfv1.groupby([lcn, 'Target'])[predCol].apply(lambda x: x.shift().rolling(min_periods=1,window=winsize).quantile(0.05)).reset_index(name='q05')['q05']
dfv1['q95'] = dfv1.groupby([lcn, 'Target'])[predCol].apply(lambda x: x.shift().rolling(min_periods=1,window=winsize).quantile(0.95)).reset_index(name='q95')['q95']
dfv1['q50'] = dfv1.groupby([lcn, 'Target'])[predCol].apply(lambda x: x.shift().rolling(min_periods=1,window=winsize).quantile(0.5)).reset_index(name='q50')['q50']

#dfv1['q05'] = dfv1.groupby(['Id'])[predCol].quantile(0.05).reset_index(name='q05')['q05']
#dfv1['q95'] = dfv1.groupby(['Id'])[predCol].quantile(0.95).reset_index(name='q95')['q95']
#dfv1['q50'] = dfv1.groupby(['Id'])[predCol].quantile(0.5).reset_index(name='q50')['q50']

#dfv1[predCol] = dfv1['q50']

dfv1[dfv1['Target']=='ConfirmedCases'].tail(20)


# In[58]:


dfv1[dfv1[lcn]=='India_NA_NA']


# In[59]:


from datetime import date

today = pd.to_datetime(date.today())
offset = today - timedelta (1)

dfv1.loc[(dfv1['Date_x'] < offset), 'q05'] = 0
dfv1.loc[(dfv1['Date_x'] < offset), 'q50'] = 0
dfv1.loc[(dfv1['Date_x'] < offset), 'q95'] = 0


# In[60]:


test_r = dfv1[dfv1['istrain']==0]
#test_r['q50'] = test_r[predCol]
sub=pd.melt(test_r, id_vars=['Id'], value_vars=['q05','q50','q95'])
sub['variable']=sub['variable'].str.replace("q","0.", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(int).astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub


# In[61]:


dfc = dfv[dfv[lcn]=='Italy_NA_NA']
dfc[dfc['Date_x']=='2020-04-28']
#dfc = dfc[dfc['Target']=='Fatalities']
#dfc = dfc.groupby('Date_x').count()
#dfc[dfc['code']!=1]


# In[62]:


topCharts(dfv1, _n=[0,1], _s=10, _d=180, _dif=0)


# In[63]:


topCharts(dfv1, _n=[1,1])


# In[64]:


topCharts(dfv1, _n=[2,1], _s=10, _d=180, _dif=0)


# In[65]:


topCharts(dfv1, _n=[3,4], _s=10, _d=180, _dif=0)


# In[66]:


topCharts(dfv1, _n=[7,3], _s=10, _d=180, _dif=0)


# In[67]:


topCharts(dfv1, _n=[10,3], _s=10, _d=180, _dif=0)


# In[68]:


topCharts(dfv1, _n=[13,3], _s=10, _d=180, _dif=0)


# In[69]:


topCharts(dfv1, _n=[16,3], _s=10, _d=180, _dif=0)


# In[70]:


topCharts(dfv1, _n=[19,3], _s=10, _d=180, _dif=0)


# In[71]:


topCharts(dfv1, _n=[22,3], _s=10, _d=180, _dif=0)


# In[72]:


topCharts(dfv1, _n=[25,3], _s=10, _d=180, _dif=0)


# In[ ]:





# In[73]:


topCharts(dfv1, _l='US')


# In[74]:


topCharts(dfv1, _l='Bra',  _dif=10000)


# In[75]:


topCharts(dfv1, _dif=10000, _l='India')


# In[76]:


topCharts(dfv1, _dif=10000, _l='Russia')


# In[77]:


topCharts(dfv1, _dif=10000, _l='United K')


# In[78]:


topCharts(dfv1, _dif=10000, _l='Spain')


# In[79]:


topCharts(dfv1, _dif=10000, _l='Italy')


# In[80]:


topCharts(dfv1, _dif=10000, _l='China')


# In[81]:


topCharts(dfv1, _dif=10000, _l='Mexi')


# In[82]:


topCharts(dfv1, _dif=10000, _l='Pak')


# In[83]:


topcc = pd.DataFrame(dfv[(dfv[ste].isnull()) & (dfv['Target']=='ConfirmedCases')].groupby(lcn)[predCol].sum())
#result['countries'] =.index
topcc['fatals'] = dfv[(dfv[ste].isnull()) & (dfv['Target']=='Fatalities')].groupby(lcn)[predCol].sum().values
topcc.sort_values(by=predCol, ascending=False).head(20)


# In[ ]:




