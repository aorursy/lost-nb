#!/usr/bin/env python
# coding: utf-8



###############################################################################
## Imports
###############################################################################

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from google.cloud import bigquery
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error

###############################################################################
## Functions
###############################################################################
def create_time_features(df):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    return X

def get_wg( country = 'US', state   = 'New York' ):
    booll   = (train['Country/Region']==country) & (train['Province/State']==state) & (train['ConfirmedCases']>0)
    narf = train[ booll ].copy().reset_index()
    narf['ts'] = narf.index+1
    
    if narf.shape[0] == 0:
        return
    
    r  = 8.171
    a  = 4449661.60968839
    c  = 4.83006577124696
    hc = .998
    
    r,a,c,hc = get_params(booll
                          , [r*.75, 8.171, r*1.25]
                          , [a*.75, 4449661.60968839, a*1.25]
                          , [c*.75, 4.83006577124696, c*1.25]
                          , [hc*.75, .998, .999])
    
    narf['pt']  = narf.ts.apply(lambda x: (1-(a/(a+x**c))**r)*(1-hc))
    
    temp = narf.copy()
    temp['ts'] = temp.index
    temp = temp[['ts','ConfirmedCases','pt']]
    temp.columns = ['ts','incr', 'ptr']
    
    if temp.shape[0] == 0:
        return
    
    narf = pd.merge(narf, temp, on='ts', how='left')
    narf['incr'] = narf['incr'] - narf['ConfirmedCases']
    narf['ptr']  = narf['ptr'] - narf['pt']
    
    index = narf.shape[0] - 1
    pop   = narf.iloc[index]['pop']
    
    if np.isnan(pop):
        return
    
    cc    = narf.iloc[index]['ConfirmedCases']
    dfj   = narf.iloc[0]['day_from_jan_first']
    pt    = (pop-cc) * np.log(1-narf.iloc[index]['pt'])
    
    narf['LL'] = narf.apply(lambda x: np.log(x.ptr) * x.incr, axis=1)
    
    narf.loc[index,'LL'] = pt
    
    LL = np.sum( narf.LL ) 
    
    bool2 = (test['Country/Region']==country) & (test['Province/State']==state)
    narft = test[bool2].copy().reset_index()
    narft['ts'] = narft.day_from_jan_first - dfj
    
    narft['pt'] = narft.ts.apply(lambda x: (1-(a/(a+x**c))**r)*(1-hc))
    narft['wg'] = narft.pt * pop
    
    test.loc[bool2, 'wg'] = narft.wg.values

print('done')


def get_params(booll, rl = [1], al = [1], cl = [1], hcl = [.5]):
    r_s,a_s,c_s,hc_s = rl[0],al[0],cl[0], hcl[0]
    LL       = float('-inf')
    for r in rl:
        for a in al:
            for c in cl:
                for hc in hcl:
                    narf = train[ booll ].copy().reset_index()
                    narf['ts'] = narf.index+1
                    
                    if narf.shape[0] == 0:
                        return r_s, a_s, c_s, hc_s
                    
                    narf['pt']  = narf.ts.apply(lambda x: (1-(a/(a+x**c))**r)*(1-hc))
                    
                    temp = narf.copy()
                    temp['ts'] = temp.index
                    temp = temp[['ts','ConfirmedCases','pt']]
                    temp.columns = ['ts','incr', 'ptr']
                    
                    if temp.shape[0] == 0:
                        return r_s, a_s, c_s, hc_s
                    
                    narf = pd.merge(narf, temp, on='ts', how='left')
                    narf['incr'] = narf['incr'] - narf['ConfirmedCases']
                    narf['ptr']  = narf['ptr'] - narf['pt']
                    
                    index = narf.shape[0] - 1
                    pop   = narf.iloc[index]['pop']
                    
                    if np.isnan(pop):
                        return r_s, a_s, c_s, hc_s
                    
                    cc    = narf.iloc[index]['ConfirmedCases']
                    dfj   = narf.iloc[0]['day_from_jan_first']
                    pt    = (pop-cc) * np.log(1-narf.iloc[index]['pt'])
                    
                    narf['LL'] = narf.apply(lambda x: np.log(x.ptr) * x.incr, axis=1)
                    
                    narf.loc[index,'LL'] = pt
                    
                    ll = np.sum( narf.LL )
                    
                    if ll > LL:
                        r_s, a_s, c_s, hc_s = r, a, c, hc
                        LL = ll
        return r_s, a_s, c_s, hc_s




###############################################################################
## Read Data
###############################################################################

PATH = '/kaggle/input/covid19-global-forecasting-week-1/'
train  = pd.read_csv(PATH + 'train.csv')
test  = pd.read_csv(PATH + 'test.csv')

dfp  = pd.read_csv('/kaggle/input/population/' + 'population.csv')

print(train.shape, dfp.shape)




train.loc[train['Province/State'].isnull(), 'Province/State'] = 'NARF'
test.loc[test['Province/State'].isnull(), 'Province/State']   = 'NARF'

n = train.shape[0]
train = pd.merge(train, dfp, on=['Country/Region','Province/State'], how='left')
assert train.shape[0] == n

n = test.shape[0]
test = pd.merge(test, dfp, on=['Country/Region','Province/State'], how='left')
assert test.shape[0] == n

train.loc[train['pop'].isnull(),'pop'] = 0
test.loc[test['pop'].isnull(),'pop'] = 0

mo = train['Date'].apply(lambda x: x[5:7])
da = train['Date'].apply(lambda x: x[8:10])
train['day_from_jan_first'] = (da.apply(int)
                               + 31*(mo=='02') 
                               + 60*(mo=='03')
                               + 91*(mo=='04')  
                              )

mo = test['Date'].apply(lambda x: x[5:7])
da = test['Date'].apply(lambda x: x[8:10])
test['day_from_jan_first'] = (da.apply(int)
                               + 31*(mo=='02') 
                               + 60*(mo=='03')
                               + 91*(mo=='04')  
                              )
print('done')




test['wg'] = np.NaN

for country in test['Country/Region'].unique():
    booll = test['Country/Region']==country
    for state in test[booll]['Province/State'].unique():
        print( country, state)
        get_wg(country, state)

print('done', train.shape)




###############################################################################
## Weather Data
###############################################################################

client = bigquery.Client()
dataset_ref = client.dataset("noaa_gsod", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)

tables = list(client.list_tables(dataset))

table_ref = dataset_ref.table("stations")
table = client.get_table(table_ref)
stations_df = client.list_rows(table).to_dataframe()

table_ref = dataset_ref.table("gsod2020")
table = client.get_table(table_ref)
twenty_twenty_df = client.list_rows(table).to_dataframe()

stations_df['STN'] = stations_df['usaf'] + '-' + stations_df['wban']
twenty_twenty_df['STN'] = twenty_twenty_df['stn'] + '-' + twenty_twenty_df['wban']

cols_1 = ['STN', 'mo', 'da', 'temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']
cols_2 = ['STN', 'country', 'state', 'call', 'lat', 'lon', 'elev']
weather_df = twenty_twenty_df[cols_1].join(stations_df[cols_2].set_index('STN'), on='STN')

weather_df.tail(10)




###############################################################################
## Join df w/ Weather Data
###############################################################################
weather_df['day_from_jan_first'] = (weather_df['da'].apply(int)
                                   + 31*(weather_df['mo']=='02') 
                                   + 60*(weather_df['mo']=='03')
                                   + 91*(weather_df['mo']=='04')  
                                   )

C = []
for j in train.index:
    df = train.iloc[j:(j+1)]
    mat = cdist(df[['Lat','Long', 'day_from_jan_first']],
                weather_df[['lat','lon', 'day_from_jan_first']], 
                metric='euclidean')
    new_df = pd.DataFrame(mat, index=df.Id, columns=weather_df.index)
    arr = new_df.values
    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)
    L = [i[i.astype(bool)].tolist()[0] for i in new_close]
    C.append(L[0])
    
train['closest_station'] = C

train = train.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)
train.sort_values(by=['Id'], inplace=True)
train.head()




###############################################################################
## Join dft w/ Weather Data
###############################################################################

weather_df['day_from_jan_first'] = (weather_df['da'].apply(int)
                                   + 31*(weather_df['mo']=='02') 
                                   + 60*(weather_df['mo']=='03')
                                   + 91*(weather_df['mo']=='04')  
                                   )

C = []
for j in test.index:
    df = test.iloc[j:(j+1)]
    mat = cdist(df[['Lat','Long', 'day_from_jan_first']],
                weather_df[['lat','lon', 'day_from_jan_first']], 
                metric='euclidean')
    new_df = pd.DataFrame(mat, index=df.ForecastId, columns=weather_df.index)
    arr = new_df.values
    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)
    L = [i[i.astype(bool)].tolist()[0] for i in new_close]
    C.append(L[0])
    
test['closest_station'] = C

test = test.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)
test.sort_values(by=['ForecastId'], inplace=True)
test.head()




###############################################################################
## Cleaning
###############################################################################
train["wdsp"] = pd.to_numeric(train["wdsp"])
test["wdsp"] = pd.to_numeric(test["wdsp"])
train["fog"] = pd.to_numeric(train["fog"])
test["fog"] = pd.to_numeric(test["fog"])

print('done')




train.loc[train['Province/State']=='NARF', 'Province/State'] = np.NaN
test.loc[  test['Province/State']=='NARF', 'Province/State'] = np.NaN




X_train = train.drop(["ConfirmedCases"], axis=1)
countries = X_train["Country/Region"]

X_train = X_train.drop(["Id"], axis=1)
X_test = test.drop(["ForecastId"], axis=1)

# Change the Date column to be a datetime
X_train['Date']= pd.to_datetime(X_train['Date']) 
X_test['Date']= pd.to_datetime(X_test['Date']) 

#Set the index to the date
X_train = X_train.set_index(['Date'])
X_test = X_test.set_index(['Date'])

#Create time features
create_time_features(X_train)
create_time_features(X_test)

X_train.drop("date", axis=1, inplace=True)
X_test.drop("date", axis=1, inplace=True)
print(X_train.shape, X_test.shape)


X_train.drop(["Fatalities"], axis=1, inplace=True)

print('done', X_train.shape)
"""
dfw = X_train[[ 'Country/Region', 'Lat', 'Long','dayofyear','Fatalities']].copy()
dfw.columns = ['Country/Region', 'Lat', 'Long','first_death','Fatalities']
dfw = dfw[dfw.Fatalities>0]
dfw.drop('Fatalities', inplace=True, axis=1)
dfw.reset_index(drop=True, inplace=True)
dfw.drop_duplicates(subset=['Country/Region', 'Lat', 'Long'], inplace=True)


X_train = pd.merge( X_train, dfw, on=['Country/Region', 'Lat', 'Long'], how='left')

X_train['first_death'] = X_train.dayofyear - X_train.first_death

X_train.loc[X_train.dayofyear<X_train.first_death, 'first_death'] = np.NaN
X_test  = pd.merge( X_test,  dfw, on=['Country/Region', 'Lat', 'Long'], how='left')
"""
print('done', X_train.shape)




#One hot encode the Provice/State and the Country/Region columns
X_train = pd.concat([X_train,pd.get_dummies(X_train['Province/State'], prefix='ps')],axis=1)
X_train.drop(['Province/State'],axis=1, inplace=True)
X_test = pd.concat([X_test,pd.get_dummies(X_test['Province/State'], prefix='ps')],axis=1)
X_test.drop(['Province/State'],axis=1, inplace=True)

X_train = pd.concat([X_train,pd.get_dummies(X_train['Country/Region'], prefix='cr')],axis=1)
X_train.drop(['Country/Region'],axis=1, inplace=True)
X_test = pd.concat([X_test,pd.get_dummies(X_test['Country/Region'], prefix='cr')],axis=1)
X_test.drop(['Country/Region'],axis=1, inplace=True)

print(X_train.shape, X_test.shape)




###############################################################################
## Modeling
###############################################################################




y_train    = train["Fatalities"]
y_train_cc = train["ConfirmedCases"]




X_TRAIN = X_train.values

params_xgb = {}
params_xgb['n_estimators']       = 1100
params_xgb['max_depth']          = 9
params_xgb['seed']               = 2020
params_xgb['colsample_bylevel']  = 1
params_xgb['colsample_bytree']   = 1
params_xgb['learning_rate']      = 0.300000012
params_xgb['reg_alpha']          = 0
params_xgb['reg_lambda']         = 1
params_xgb['subsample']          = 1

isTraining = False

if isTraining:
    kf      = KFold(n_splits = 5, shuffle = True, random_state=2020)
    acc     = []

    for tr_idx, val_idx in kf.split(X_TRAIN, y_train_cc):
        ## Set up XY train/validation
        X_tr, X_vl = X_TRAIN[tr_idx], X_TRAIN[val_idx, :]
        y_tr, y_vl = y_train_cc[tr_idx], y_train_cc[val_idx]
        print(X_tr.shape)

        model_xgb_cc = xgb.XGBRegressor(**params_xgb)
        model_xgb_cc.fit(X_tr, y_tr, verbose=True)
        y_hat = model_xgb_cc.predict(X_vl)

        print('xgb mae :', mean_absolute_error(  y_vl, y_hat) )
        acc.append(mean_absolute_error( y_vl, y_hat) )


    print('done', np.mean(acc))# Best run: 168.26412715647604 #30.2019242771957




print('done', np.mean(acc))# Best run: 168.26412715647604 #30.2019242771957




## Fit fatalities
params_xgb = {}
params_xgb['n_estimators']       = 1100
params_xgb['max_depth']          = 9
params_xgb['seed']               = 2020
params_xgb['colsample_bylevel']  = 1
params_xgb['colsample_bytree']   = 1
params_xgb['learning_rate']      = 0.300000012
params_xgb['reg_alpha']          = 0
params_xgb['reg_lambda']         = 1
params_xgb['subsample']          = 1

model_xgb_f = xgb.XGBRegressor(**params_xgb)
model_xgb_f.fit(X_train, y_train, verbose=True)

y_hat_xgb_f = model_xgb_f.predict(X_test.drop('wg',axis=1))
print(np.mean(y_hat_xgb_f))




## Fit confirmed cases
params_xgb = {}
params_xgb['n_estimators']       = 1100
params_xgb['max_depth']          = 9
params_xgb['seed']               = 2020
params_xgb['colsample_bylevel']  = 1
params_xgb['colsample_bytree']   = 1
params_xgb['learning_rate']      = 0.300000012
params_xgb['reg_alpha']          = 0
params_xgb['reg_lambda']         = 1
params_xgb['subsample']          = 1

model_xgb_cc = xgb.XGBRegressor(**params_xgb)
model_xgb_cc.fit(X_train, y_train_cc, verbose=True)

y_hat_xgb_cc = model_xgb_cc.predict(X_test.drop('wg',axis=1))

print(np.mean(y_hat_xgb_cc))




###############################################################################
## Feature Importantce
###############################################################################

plot = plot_importance(model_xgb_cc, height=0.9, max_num_features=20)




test[test.wg.isnull()].shape




test['y_hat_cc']                = y_hat_xgb_cc
test.loc[test.wg.isnull(),'wg'] = test[test.wg.isnull()].y_hat_cc
test['y_hat_ens']   = .75 * test.y_hat_cc + .25 * test.wg

print('done')




test['y_hat_ens'] = test.y_hat_ens.astype(float)









print(test[test.wg.isnull()].shape, np.mean(test.wg), np.mean(y_hat_xgb_cc), np.mean(test.y_hat_ens))




###############################################################################
## Submision
###############################################################################




submissionOrig = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")
submissionOrig["ConfirmedCases"]= pd.Series( test.y_hat_ens)#pd.Series(y_hat_xgb_cc)
submissionOrig["Fatalities"]    = pd.Series(y_hat_xgb_f)




submissionOrig.to_csv('submission.csv',index=False)

