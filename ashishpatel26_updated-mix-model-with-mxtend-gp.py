#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('html', '', '<h1><b>LIVE EARTHQUACK MAP</b></h1>\n<iframe width="800" height="600" src="https://ds.iris.edu/seismon/" allowfullscreen style="align:center;"></iframe>')




get_ipython().run_cell_magic('html', '', '\n<h1> <strong><span style="color:blue;">Can We Predict Earthquakes?</span></strong></h1>\n<iframe width="800" height="400" src="https://www.youtube.com/embed/uUEzGcRJIZE" style="align:center;" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import gc
import os
print(os.listdir("../input"))
from mlxtend.regressor import stacking_regression
# pandas doesn't show us all the decimals
pd.options.display.precision = 15

# Any results you write to the current directory are saved as output.




import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("fivethirtyeight")
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR,LinearSVR, SVR
from sklearn.metrics import mean_absolute_error,r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, BayesianRidge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score




get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})\ngc.collect()")




plt.figure(figsize=(20, 5))
plt.plot(train['acoustic_data'].values[::100], color='blue', label='Acoustic Data')
plt.legend()
plt.ylabel("Acoustic Data value")
plt.xlabel("Total Value Count")
plt.title('Acoustic Data')
plt.show()
plt.figure(figsize=(20, 5))
plt.plot(train['time_to_failure'].values[::100], color='red', label='Time_to_failure')
plt.legend()
plt.ylabel("Time Data value")
plt.xlabel("Time Value Count")
plt.title('Time to failure')
plt.show()




def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())




gc.collect()




# Create a training file with simple derived features
# Feature Engineering : https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction?scriptVersionId=9550007

rows = 150_000
segments = int(np.floor(train.shape[0] / rows))

X_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min','q95','q99', 'q05','q01',
                                'abs_max', 'abs_mean', 'abs_std', 'trend', 'abs_trend'])
y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

for segment in tqdm(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    y_train.loc[segment, 'time_to_failure'] = y
    
    X_train.loc[segment, 'ave'] = x.mean()
    X_train.loc[segment, 'std'] = x.std()
    X_train.loc[segment, 'max'] = x.max()
    X_train.loc[segment, 'min'] = x.min()
    X_train.loc[segment, 'q95'] = np.quantile(x,0.95)
    X_train.loc[segment, 'q99'] = np.quantile(x,0.99)
    X_train.loc[segment, 'q05'] = np.quantile(x,0.05)
    X_train.loc[segment, 'q01'] = np.quantile(x,0.01)
    
    X_train.loc[segment, 'abs_max'] = np.abs(x).max()
    X_train.loc[segment, 'abs_mean'] = np.abs(x).mean()
    X_train.loc[segment, 'abs_std'] = np.abs(x).std()
    X_train.loc[segment, 'trend'] = add_trend_feature(x)
    X_train.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)
    
    
display(X_train.head())
gc.collect()




get_ipython().run_cell_magic('time', '', "axs = pd.scatter_matrix(X_train[::100], figsize=(20,12), diagonal='kde')\ndisplay(X_train[::100].corr())\ngc.collect()")




get_ipython().run_cell_magic('time', '', 'scaler = StandardScaler()\nscaler.fit(X_train)\nX_train_scaled = scaler.transform(X_train)\ngc.collect()')




get_ipython().run_cell_magic('time', '', "axs = pd.scatter_matrix(X_train[::100], figsize=(20,12), diagonal='kde')\ndisplay(X_train[::100].corr())\ngc.collect()")




def nusvr_code(NuSVR,X_train_scaled,y_train):
    svm1 = NuSVR(nu=0.95, gamma=0.62,C=2.45)
    svm1.fit(X_train_scaled, y_train.values.flatten())
    y_pred = svm1.predict(X_train_scaled)
    plt.figure(figsize=(20, 6))
    plt.scatter(y_train.values.flatten(), y_pred)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
    plt.title('NuSVR')
    plt.show()
    score = rmse(y_train.values.flatten(), y_pred)
    print(f'RMSE Score: {score:0.3f}')
    score = r2_score(y_train.values.flatten(), y_pred)
    print(f'Score: {score:0.3f}')
    return (y_pred,score,svm1)
    
y_pred_nusvr, score, svm1 = nusvr_code(NuSVR,X_train_scaled,y_train)




def svr_code(SVR,X_train_scaled,y_train):
    svm3 = SVR(C=1000, verbose = 1)
    svm3.fit(X_train_scaled, y_train.values.flatten())
    y_pred = svm3.predict(X_train_scaled)
    plt.figure(figsize=(20, 6))
    plt.scatter(y_train.values.flatten(), y_pred)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
    plt.title('SVR')
    plt.show()
    score = rmse(y_train.values.flatten(), y_pred)
    print(f'RMSE Score: {score:0.3f}')
    score = r2_score(y_train.values.flatten(), y_pred)
    print(f'Score: {score:0.3f}')
    return (y_pred,score, svm3)
    
y_pred_SVR, score, svm3 = svr_code(SVR,X_train_scaled,y_train)




def br_code(BayesianRidge,X_train_scaled,y_train):
    svm2 = KernelRidge(kernel='rbf',alpha = 0.05, gamma = 0.06)
    svm2.fit(X_train_scaled, y_train.values.flatten())
    y_pred = svm2.predict(X_train_scaled)
    plt.figure(figsize=(20, 6))
    plt.scatter(y_train.values.flatten(), y_pred)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
    plt.title('Kernel Ridge Regression')
    plt.show()
    score = rmse(y_train.values.flatten(), y_pred)
    print(f'RMSE Score: {score:0.3f}')
    score = r2_score(y_train.values.flatten(), y_pred)
    print(f'Score: {score:0.3f}')
    return (y_pred,score, svm2)
    
y_pred_Bayesian, score, svm2 = br_code(BayesianRidge,X_train_scaled,y_train)




svm5 = LGBMRegressor(num_leaves=31, max_depth=-1, learning_rate=0.01, n_estimators=1000)
svm5.fit(X_train_scaled, y_train.values.flatten())
y_pred_lgb = svm5.predict(X_train_scaled)
plt.figure(figsize=(20, 6))
plt.scatter(y_train.values.flatten(), y_pred_lgb)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.title('Light GBM')
score = rmse(y_train.values.flatten(), y_pred_lgb)
print(f'RMSE Score: {score:0.3f}')
score = r2_score(y_train.values.flatten(), y_pred_lgb)
print(f'Score: {score:0.3f}')




def cat_code(CatBoostRegressor,X_train_scaled,y_train):
    svm4 = CatBoostRegressor(depth=8)
    svm4.fit(X_train_scaled, y_train.values.flatten())
    y_pred = svm4.predict(X_train_scaled)
    plt.figure(figsize=(20, 6))
    plt.scatter(y_train.values.flatten(), y_pred)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
    plt.title('CATBOOST')
    plt.show()
    score = rmse(y_train.values.flatten(), y_pred)
    print(f'RMSE Score: {score:0.3f}')
    score = r2_score(y_train.values.flatten(), y_pred)
    print(f'Score: {score:0.3f}')
    return (y_pred,score, svm4)
    
y_pred_cat, score, svm4 = cat_code(CatBoostRegressor,X_train_scaled,y_train)




submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)




for seg_id in tqdm(X_test.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
    X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)
    X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)
    X_test.loc[seg_id, 'q05'] = np.quantile(x,0.05)
    X_test.loc[seg_id, 'q01'] = np.quantile(x,0.01)
    
    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()
    X_test.loc[seg_id, 'abs_mean'] = np.abs(x).mean()
    X_test.loc[seg_id, 'abs_std'] = np.abs(x).std()
    X_test.loc[seg_id, 'trend'] = add_trend_feature(x)
    X_test.loc[seg_id, 'abs_trend'] = add_trend_feature(x, abs_values=True)    




f = [y_pred_nusvr,y_pred_SVR,y_pred_cat,y_pred_lgb,y_pred_Bayesian]




f = np.transpose(f)




f.shape




svm6 = LGBMRegressor()
svm6.fit(f, y_train.values.flatten())
y_pred_stack = svm6.predict(f)
plt.figure(figsize=(20, 6))
plt.scatter(y_train.values.flatten(), y_pred_lgb)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.title('Light GBM Stacking')
score = rmse(y_train.values.flatten(), y_pred_stack)
print(f'RMSE Score: {score:0.3f}')
score = r2_score(y_train.values.flatten(), y_pred_lgb)
print(f'Score: {score:0.3f}')




d = {'Model': ['nuSVR', 'SVR','Kernel Ridge','Lightgbm','Catboost','Stacking'], 'RMSE': [2.647,2.635,2.653,2.029,2.491,1.20],'F1_Score': [0.48,0.485,0.478,0.695,0.54,0.695]}
analysis_df = pd.DataFrame(d)
# display(analysis_df)

analysis_df.index = analysis_df.Model
del analysis_df['Model']
display(analysis_df)




plt.figure(figsize=(20,8))
plt.plot(analysis_df.index,analysis_df.RMSE,'mD-',animated=True)
plt.scatter(analysis_df.index, analysis_df.RMSE,s=y*50, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title("RMSE by Model")
plt.show()




plt.figure(figsize=(20,8))
plt.plot(analysis_df.index,analysis_df.F1_Score,'rD-',animated=True)
plt.scatter(analysis_df.index, analysis_df.F1_Score,s=y*50, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)
plt.xlabel('Model')
plt.ylabel('F1_Score')
plt.title("F1_Score by Model")
plt.show()




svm5




from mlxtend.regressor import StackingRegressor
sclf = StackingRegressor(regressors=[svm1,svm2,svm3,svm4,svm5,svm6], 
                          meta_regressor=SVR())

sclf.fit(X_train_scaled, y_train.values.flatten())




y_pred_final = sclf.predict(X_train_scaled)
plt.figure(figsize=(20, 6))
plt.scatter(y_train.values.flatten(), y_pred_final)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.title('Light GBM')
score = rmse(y_train.values.flatten(), y_pred_final)
print(f'RMSE Score: {score:0.3f}')
score = r2_score(y_train.values.flatten(), y_pred_final)
print(f'Score: {score:0.3f}')




X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = sclf.predict(X_test_scaled)
# submission['time_to_failure1'] = svm1.predict(X_test_scaled)
# submission['time_to_failure2'] = svm2.predict(X_test_scaled)
# submission['time_to_failure3'] = svm3.predict(X_test_scaled)
# submission['time_to_failure4'] = svm4.predict(X_test_scaled)
# submission['time_to_failure5'] = svm5.predict(X_test_scaled)
# submission['time_to_failure'] = (submission['time_to_failure1']+submission['time_to_failure2']+submission['time_to_failure3']+submission['time_to_failure4']+submission['time_to_failure5'])/5

# del submission['time_to_failure1'],submission['time_to_failure2'],submission['time_to_failure3'],submission['time_to_failure4'],submission['time_to_failure5']




submission.to_csv("Advance_stack.csv")




submission1 = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
submission1['time_to_failure'] = svm5.predict(X_test_scaled)
submission1.to_csv("submission_lgbbestmodel.csv")




submission1 = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
submission1['time_to_failure'] = svm3.predict(X_test_scaled)
submission1.to_csv("submission_svrbestmodel.csv")




submission1 = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
submission1['time_to_failure'] = svm1.predict(X_test_scaled)
submission1.to_csv("submission_nusvrbestmodel.csv")

