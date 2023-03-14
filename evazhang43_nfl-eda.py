#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




import datetime
from kaggle.competitions import nflrush
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2OXGBoostEstimator
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations 
from h2o.estimators.glm import H2OGeneralizedLinearEstimator




env = nflrush.make_env()




def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df




h2o.init()









train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)




def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans




train_df["GameClock"]= train_df["GameClock"].apply(strtoseconds)
train_df["TimeDelta"]= pd.to_datetime(train_df["TimeHandoff"]) - pd.to_datetime(train_df["TimeSnap"])
train_df = train_df.drop(["TimeHandoff","TimeSnap"], axis=1)
train_df=train_df.drop(["NflId", "DisplayName","JerseyNumber","Location"],axis=1)




train_df= reduce_mem_usage(train_df)




arr = [ t.split(", ") for t in train_df["OffensePersonnel"]]




for i in arr[1]:
    if "RB" in i:
        print(int(i[0]))




train_df["RB-Defense"] = 0




for i in arr[1:10]:
    if "RB" in i :
        print(i)




train_df["RB-Defense"] = [[i[0] if "RB" in i else 0 for i in line] for line in arr]




train_df["RB-Defense"].sum()




train_df["RB-Defense"]




train_df["OffensePersonnel"]




train_df["OffensePersonnel"].value_counts().index.sort_values()




trian_df_h2o= h2o.H2OFrame(train_df)




h2o_tree = H2ORandomForestEstimator(ntrees = 50, max_depth = 20, nfolds =10)
#train the model,if x not specify,model will use all x except the y column
h2o_tree.train(y = 'Yards', training_frame = trian_df_h2o)
#print variable importance
h2o_tree_df = h2o_tree._model_json['output']['variable_importances'].as_data_frame()
#visualize the importance




h2o_tree_df




'''param = {
    
      "ntrees" : 100
    , "learn_rate" : 0.1
    , "max_depth" : 20
    , "sample_rate" : 0.7
    , "col_sample_rate_per_tree" : 0.9
    , "min_rows" : 5
    , "seed": 4241
    , "score_tree_interval": 100
    ,  'nfolds': 10
    , "stopping_metric": "MSE"
}
XGmodel = H2OXGBoostEstimator(**param)
XGmodel.train(y = 'Yards', training_frame = trian_df_h2o)
h2o_XGboost_df =XGmodel._model_json['output']['variable_importances'].as_data_frame()
'''




lm_model = H2OGeneralizedLinearEstimator(family= "multinomial", lambda_ = 0)




trian_df_h2o['Yards'] = trian_df_h2o['Yards'].asfactor()




lm_model.train(y = 'Yards', training_frame = trian_df_h2o)









iter_test = env.iter_test()




(test_df, sample_prediction_df) = next(iter_test)




sample_prediction_df[:20]




test_df["GameClock"]= test_df["GameClock"].apply(strtoseconds)
test_df["TimeDelta"]= pd.to_datetime(test_df["TimeHandoff"]) - pd.to_datetime(test_df["TimeSnap"])
test_df = test_df.drop(["TimeHandoff","TimeSnap"], axis=1)
test_df=test_df.drop(["NflId", "DisplayName","JerseyNumber","Location"],axis=1)




test_df.shape




test_df_h2o= h2o.H2OFrame(test_df)




prediction_Df = lm_model.predict(test_df_h2o)




prediction_Df1 =prediction_Df.as_data_frame(use_pandas=True, header=True)









colnames = list(map(lambda st: str.replace(st, "p", "Yards"), prediction_Df.columns))




prediction_Df1['Yardsredict']




prediction_Df1.columns = colnames




test = pd.concat([sample_prediction_df,prediction_Df1],axis =0,sort=False)









sample_prediction_df.columns




test = test.drop('Yardsredict',axis =1)




test = test[1:23]




test = test.fillna(0)




test = test.drop('PlayId',axis =1)




env.predict(test)




for (test_df, test) in iter_test:
    env.predict(test)




env.write_submission_file()




import os
print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])




test[1:23]




prediction_Df.




colnames = []




sample_prediction_df.conca




h2o_XGboost_df




h2o_XGboost_df.to_csv("FeatureImportance.csv")









outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor', 
           'Outside', 'Outddors','Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']

indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 'Retractable Roof',
                 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']

indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']
dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']
dome_open     = ['Domed, Open', 'Domed, open']




train_df['StadiumType'] = train_df['StadiumType'].replace(outdoor,'outdoor')
train_df['StadiumType'] = train_df['StadiumType'].replace(indoor_closed,'indoor_closed')
train_df['StadiumType'] = train_df['StadiumType'].replace(indoor_open,'indoor_open')
train_df['StadiumType'] = train_df['StadiumType'].replace(dome_closed,'dome_closed')
train_df['StadiumType'] = train_df['StadiumType'].replace(dome_open,'dome_open')
train_df['StadiumType'] = train_df['StadiumType'].replace(np.nan, "Missing")




def anova_test_mean(column):
    formular = "Yards~"+column
    model = ols(formular,data = train_df).fit()
    anova_result = sm.stats.anova_lm(model, typ=2)
    print(anova_result)
def Ks_test_Cat(column):
    list_of_cat = train_df[column].unique()
    comb = combinations(list_of_cat, 2) 
    test = list(comb)
    NoVariance = []
    for i in list(test):
        result = ks_2samp(train_df[train_df[column]==i[0]]["Yards"],train_df[train_df[column]==i[1]]["Yards"])
        if result[1]>0.01:
            NoVariance.append(i)
    print(NoVariance)
    




Ks_test_Cat("JerseyNumber")




Ks_test_Cat("WindDirection")




ks_2samp(train_df[train_df["WindDirection"]=="S"]["Yards"],train_df[train_df["WindDirection"]=="N"]["Yards"])




Ks_test_Cat("OffensePersonnel")




ks_2samp(train_df[train_df["JerseyNumber"]==29]["Yards"],train_df[train_df["JerseyNumber"]==23]["Yards"])[1]




test_mean('Location')




train_df['StadiumType'].value_counts().index.sort_values()




train_df[train_df['StadiumType']== 'Cloudy']["Stadium"]




train_df['Stadium'].value_counts()




train_df[train_df['Stadium']=='Paul Brown Stdium']




#Data Preprocessing:
#Stadium Type 




Mean_Std_Count("GameId")




Mean_Std_Count("PlayId")




train_df=train_df.drop(["NflId", "DisplayName","JerseyNumber","Location"],axis=1)









kf=KFold(n_splits = 5)
resu1 = 0
impor1 = 0
resu2_cprs = 0
resu3_mae=0
##y_pred = 0
stack_train = np.zeros([X_train.shape[0],])
models = []
for train_index, test_index in kf.split(X_train, y_train):
    X_train2= X_train.iloc[train_index,:]
    y_train2= y_train.iloc[train_index]
    X_test2= X_train.iloc[test_index,:]
    y_test2= y_train.iloc[test_index]




train_data = lgb.Dataset(train_df.drop(["Yards"], axis=1))




y_train = lgb.Dataset(train_df["Yards"])




clf = lgb.LGBMRegressor(n_estimators=10000, random_state=47,learning_rate=0.005,importance_type = 'gain',
                     n_jobs = -1,metric='mae')




clf.fit(train_data,y_train)




train_df=train_df.dropna(how = "any")




train_frame = train_df.sample(frac=0.5, replace=True, random_state=1)
val_frame = train_df.sample(frac=0.5, replace=True, random_state=1)




#prepare H2O Frame 
train_frame_H2O = h2o.H2OFrame(train_frame)
val_frame_H2O = h2o.H2OFrame(val_frame)




# select parameter
"""
hyper_params = {'max_depth' : [4,6,8,12,16,20]
               ,"learn_rate" : [0.1, 0.01, 0.0001] 
               }
param_grid = {
      "ntrees" : 50
    , "sample_rate" : 0.7
    , "col_sample_rate_per_tree" : 0.9
    , "min_rows" : 5
    , "seed": 4241
    , "score_tree_interval": 100
    ,  'nfolds': 10
    , "stopping_metric" : "AUC",
    "categorical_encoding":"enum_limited"
}
model_grid = H2OXGBoostEstimator(**param_grid)
grid = H2OGridSearch(model_grid,hyper_params,
                         grid_id = 'depth_grid',
                         search_criteria = {'strategy': "Cartesian"})
grid.train(y='Yards',training_frame = train_frame_H2O, validation_frame=val_frame_H2O)
gb_gridperf = grid.get_grid(sort_by='mse', decreasing=True)
"""




#gb_gridperf




best_param = {
      "ntrees" : 100
    , "learn_rate" : 0.1
    , "max_depth" : 20
    , "sample_rate" : 0.7
    , "col_sample_rate_per_tree" : 0.9
    , "min_rows" : 5
    , "seed": 4241
    , "score_tree_interval": 100
    ,  'nfolds': 10
    ,"categorical_encoding":"enum_limited"
    , "stopping_metric" : "AUC"
}

best_model = H2OXGBoostEstimator(**best_param)
best_model.train(y = 'Yards', training_frame = train_frame_H2O)




best_metrics = best_model.model_performance(test_data=val_frame_H2O) 
best_metrics




def plot_bar_x(DataSeries, xLabel, yLabel):
    # this is for plotting a specific bar chart for the Series 
    # input a Series 
    # output a vertical bar chart for a Series 
    index = np.arange(len(DataSeries))
    plt.bar(index, DataSeries[yLabel])
    #plt.xlabel(xLabel, fontsize=10)
    #plt.ylabel(yLabel, fontsize=10)
    plt.xticks(index, DataSeries[xLabel], fontsize=10, rotation=30)
    return plt
def Draw_Cat_Var(column, dataset,target):
    DATA = dataset[[column, target]].fillna("NA")
    Cat_EDA = DATA.groupby(column).mean()[target]
    plot_bar_x(Cat_EDA, column,target)









#exam the missing values
(train_df.isnull().sum()/len(train_df)).sort_values(ascending = False)




def Plot_Team():
    plot1 = train_df[train_df["Team"]=="away"]["Yards"]
    plot2 = train_df[train_df["Team"]=="home"]["Yards"]
    bins = numpy.linspace(0, 100, 100)
    plt.hist(plot1, bins, alpha=0.5, label='away')
    plt.hist(plot2, bins, alpha=0.5, label='home')
    plt.legend(loc='upper right')
    plt.show()




train_df.columns




train_df["Yards"].hist()




'''
#light Xgboost
train = train_df.sample(frac=0.5, replace=True, random_state=1)
Validate = train_df.sample(frac=0.5, replace=True, random_state=1)
X_train_lgbm = lgb.Dataset(train.drop("Yards", axis =1), label = train["Yards"])
X_validate_lgbm = lgb.Dataset(Validate.drop("Yards", axis =1), label = Validate["Yards"])

evals_result = {}

base_params = {
        "objective" : "multiclass",
    "num_class":94,
        "metric" : "auc",
        "is_unbalance" : True,
        "tree_learner": "voting",  
        'max_bin': 255,
        'max_depth': -1,
        "min_child_samples" : 100,
        'verbose_eval': 10,
        'num_boost_round': 170,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 1234,
        'boosting_type':'gbdt'
    }
base_lgb_model = lgb.train(base_params, train_set=X_train_lgbm,
                           valid_sets=X_validate_lgbm,
                           evals_result = evals_result
                           )
'''




def Mean_Std_Count(Column):
    Try1 = train_df[[Column, "Yards"]].groupby(Column).mean().sort_values(by = "Yards").reset_index()
    Try2 = train_df[[Column, "Yards"]].groupby(Column).std().sort_values(by = "Yards").reset_index()
    Try3 = train_df[[Column, "Yards"]].groupby(Column).count().sort_values(by = "Yards").reset_index()
    fig = plt.figure(figsize=(12,4), dpi=100)
    fig.suptitle( Column+" Features", fontsize=16)
    plt.subplot(131)
    plot_bar_x(Try1, Column, "Yards")
    plt.xlabel("Mean")
    plt.subplot(132)
    plot_bar_x(Try2, Column, "Yards")
    plt.xlabel("std")
    plt.subplot(133)
    plot_bar_x(Try3,  Column, "Yards")
    plt.xlabel("Count")




Mean_Std_Count("NflId")




train_df['WindDirection'].value_counts()




Mean_Std_Count('WindDirection')




Mean_Std_Count('WindSpeed')




Mean_Std_Count('Humidity')




Mean_Std_Count('Temperature')




Mean_Std_Count('GameWeather')




Mean_Std_Count('StadiumType')




Mean_Std_Count('Location')




Mean_Std_Count('Stadium')




Mean_Std_Count('PlayerBirthDate')




Mean_Std_Count('PlayerWeight')




Mean_Std_Count('PlayerHeight')




Mean_Std_Count('PlayerCollegeName')




Mean_Std_Count("NflId")




Mean_Std_Count('JerseyNumber')




Mean_Std_Count('DisplayName')




Try1 = train_df[['Season', "Yards"]].groupby('Season').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try1, 'Season', "Yards")




Try = train_df[['Turf', "Yards"]].groupby('Turf').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'Turf', "Yards")




Try = train_df[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'Week', "Yards")




Try = train_df[['VisitorTeamAbbr', "Yards"]].groupby('VisitorTeamAbbr').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'VisitorTeamAbbr', "Yards")




Try = train_df[['HomeTeamAbbr', "Yards"]].groupby('HomeTeamAbbr').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'HomeTeamAbbr', "Yards")




Try = train_df[['Position', "Yards"]].groupby('Position').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'Position', "Yards")




Try = train_df[['PlayDirection', "Yards"]].groupby('PlayDirection').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'PlayDirection', "Yards")




Try = train_df[['DefensePersonnel', "Yards"]].groupby('DefensePersonnel').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'DefensePersonnel', "Yards")




Try = train_df[['DefendersInTheBox', "Yards"]].groupby('DefendersInTheBox').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'DefendersInTheBox', "Yards")




Try = train_df[['OffensePersonnel', "Yards"]].groupby('OffensePersonnel').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'OffensePersonnel', "Yards")




Try = train_df[['OffenseFormation', "Yards"]].groupby('OffenseFormation').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'OffenseFormation', "Yards")




Try = train_df[['NflIdRusher', "Yards"]].groupby('NflIdRusher').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'NflIdRusher', "Yards")




Try = train_df[['VisitorScoreBeforePlay', "Yards"]].groupby('VisitorScoreBeforePlay').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'VisitorScoreBeforePlay', "Yards")




Try = train_df[['HomeScoreBeforePlay', "Yards"]].groupby('HomeScoreBeforePlay').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'HomeScoreBeforePlay', "Yards")




Try = train_df[['FieldPosition', "Yards"]].groupby('FieldPosition').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'FieldPosition', "Yards")




Try = train_df[['Distance', "Yards"]].groupby('Distance').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'Distance', "Yards")




Try = train_df[['Down', "Yards"]].groupby('Down').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'Down', "Yards")




Try = train_df[["PossessionTeam", "Yards"]].groupby("PossessionTeam").mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, "PossessionTeam", "Yards")




Try = train_df[["GameClock", "Yards"]].groupby("GameClock").mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, "GameClock", "Yards")




Try = train_df[["GameId", "Yards"]].groupby("GameId").mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, "GameId", "Yards")




Try = train_df[['FieldPosition', "Yards"]].groupby('FieldPosition').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'FieldPosition', "Yards")




Try = train_df[['Season', "Yards"]].groupby('Season').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'Season', "Yards")




Try = train_df[['YardLine', "Yards"]].groupby('YardLine').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'YardLine', "Yards")




Try = train_df[['Quarter', "Yards"]].groupby('Quarter').mean().sort_values(by = "Yards").reset_index()
plot_bar_x(Try, 'Quarter', "Yards")

