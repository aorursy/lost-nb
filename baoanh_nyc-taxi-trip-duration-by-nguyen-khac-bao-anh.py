#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualiser les données
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings('ignore')




get_ipython().run_line_magic('matplotlib', 'inline')
sns.set({'figure.figsize':(16,8)})




train = pd.read_csv("../input/nyc-taxi-duration-eda-by-nguyen-khac-bao-anh/training_data.csv")
test = pd.read_csv("../input/nyc-taxi-duration-eda-by-nguyen-khac-bao-anh/testing_data.csv")




print(f"shape of training set{train.shape}")
print(f"shape of testing set{test.shape}")




train.head()




test.head()




col_diff = list(set(train.columns).difference(set(test.columns)))
print(f"La différence de la variable entre data training et data testing:{set(train.columns).difference(set(test.columns))}")




xtrain = train.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'log_trip_duration'], axis = 1).as_matrix()
xtest = test.drop(['id', 'pickup_datetime', ], axis = 1).as_matrix()
y = train['log_trip_duration'].values
del(train, test)




from sklearn.model_selection import train_test_split, cross_val_score




X_train, X_valid, y_train, y_valid = train_test_split(xtrain,y, test_size=0.2, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape




#from sklearn.ensemble import RandomForestRegressor




#rf_defaut = RandomForestRegressor()
# crosse validation pour tester si le model est stable 
#rf_cv = cross_val_score(rf_defaut, X_train, y_train, cv=5)
#rf_cv




#plt.plot(range(1,len(rf_cv)+1), rf_cv)
#plt.ylim(np.min(rf_cv)-0.1,np.max(rf_cv)+0.1)
#plt.xlabel("nombre de fold")
#plt.ylabel("score du model Random Forest Regressor");




# la fonction permet de nous donner un score qui est le règle de cette compétition
# (root mean squared log error)
#from sklearn.metrics import mean_squared_log_error, mean_squared_error
def rmse(y,pred):
    return np.sqrt(np.mean(np.square(np.log(np.exp(y))-np.log(np.exp(pred)))))




#rf_defaut = RandomForestRegressor()
#rf_defaut.fit(X_train, y_train)




#y_pred = rf_defaut.predict(X_valid)




#print(rmse(y_valid,y_pred))




#from sklearn.model_selection import GridSearchCV




# n_estimators et max_depth pour fitter bien le model mais causer overfitting
# par contre, min_samples_leaf et min_samples_split nous permets de éviter overfitting en donnant la valeur
# plus grand
#params = {
#    'n_estimators' : [10, 15, 20],
#    'max_depth': [30, 50, 100],
#    'min_samples_leaf': [100],
#    'min_samples_split': [150]
#}
#rf2 = RandomForestRegressor()
#gs_rf2 = GridSearchCV(rf2, param_grid=params, scoring='neg_mean_squared_error',cv=3, verbose=10, n_jobs=-1)
#gs_rf2.fit(X_train, y_train)
#gs_rf2.best_score_
#best_rf2 = gs_rf2.best_estimator_




#rf2 = RandomForestRegressor(n_estimators=10,min_samples_leaf=100, min_samples_split=150)




#rf2_cv = cross_val_score(rf2, X_train, y_train, cv=5)
#rf2_cv




#plt.plot(range(1,len(rf2_cv)+1), rf2_cv)
#plt.ylim(np.min(rf2_cv)-0.1,np.max(rf2_cv)+0.1)
#plt.xlabel("nombre de fold")
#plt.ylabel("score du model Random Forest Regressor avec hyperparameters");




#rf2.fit(X_train, y_train)




#pred = rf2.predict(X_valid)




#print(rmse(y_valid,pred))




import lightgbm as lgb




#lgb_train = lgb.Dataset(X_train, y_train)
#lgb_valid = lgb.Dataset(X_valid, y_valid)
# training all dataset
dtrain = lgb.Dataset(xtrain,y)
del(X_train, y_train, X_valid, y_valid, xtrain,y)




lgb_params = {
    'learning_rate': 0.1,
    'max_depth': 8,
    'num_leaves': 55, 
    'objective': 'regression',
    #'metric': {'rmse'},
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    #'bagging_freq': 5,
    'max_bin': 300}       # 1000




#cv_result_lgb = lgb.cv(lgb_params,
#                       lgb_train, 
#                       num_boost_round=1000, 
#                       nfold=3,
#                       early_stopping_rounds=50, 
#                       verbose_eval=100, 
#                       show_stdv=True,stratified=False)




#n_rounds = len(cv_result_lgb['rmsle-mean'])
#print('num_boost_rounds_lgb=' + str(n_rounds))




# visualisation des résultat dans cv
# CV scores
#train_scores = np.array(cv_result_lgb['rmsle-mean'])
#train_stds = np.array(cv_result_lgb['rmsle-stdv'])
#plt.plot(train_scores, color='violet')
#plt.fill_between(range(len(cv_result_lgb['rmsle-mean'])), 
#                 train_scores - train_stds, train_scores + train_stds, 
#                 alpha=0.1, color='violet')
#plt.title('LightGMB CV-results')
#plt.xlabel("number of rounds")
#plt.ylabel("score");




# Train a model
#model_lgb = lgb.train(lgb_params, 
#                      dtrain, 
#                      feval=lgb_rmsle_score, 
#                      num_boost_round=n_rounds)




## Predict on train
#y_train_pred = model_lgb.predict(X_train)
#print('RMSLE on train = {}'.format(rmse(y_train_pred, y_train)))
## Predict on validation
#y_valid_pred = model_lgb.predict(X_valid)
#print('RMSLE on valid = {}'.format(rmse(y_valid_pred, y_valid)))




# Train a model
model_lgb = lgb.train(lgb_params, 
                      dtrain,
                      num_boost_round=1500)




submit = pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.csv')
submit.head()




pred_test = np.exp(model_lgb.predict(xtest))




submit['trip_duration'] = pred_test




submit.head()




submit.to_csv("submit_file.csv", index=False)






